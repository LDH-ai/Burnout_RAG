from __future__ import annotations

import hashlib
import json
import os
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from typing import Literal, Optional

import numpy as np

# langchain_community sunset 경고만 억제
warnings.filterwarnings(
    "ignore",
    message=".*langchain-community.*is being sunset.*",
    category=DeprecationWarning,
)

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory

from prompt_config import (
    BASE_SYSTEM as PROMPT_BASE_SYSTEM,
    FEW_SHOT_EXAMPLES as PROMPT_FEW_SHOT_EXAMPLES,
    INFORMATIONAL_DIRECTIVE as PROMPT_INFORMATIONAL_DIRECTIVE,
    MODE_DIRECTIVES as PROMPT_MODE_DIRECTIVES,
    RESPONSE_MODE_EXPLORE,
    RESPONSE_MODE_GUIDE,
    RESPONSE_MODE_INFO,
    RESPONSE_MODE_SUPPORT,
    RISK_DIRECTIVES as PROMPT_RISK_DIRECTIVES,
    RISK_HIGH,
    RISK_MID,
    RISK_LOW,
)

# ── 의도 상수 ──────────────────────────────────────────────────────────────
INTENT_INFORMATIONAL = "informational"
INTENT_PERSONAL      = "personal"

# ── 위기 상담 자원 ─────────────────────────────────────────────────────────
CRISIS_LINE_SUICIDE: str = "109"
CRISIS_LINE_MENTAL:  str = "1577-0199"

CRISIS_GUIDE_TEXT: str = (
    f"힘든 마음이 클 때는 혼자 견디지 않으셔도 괜찮아요. "
    f"학교 상담센터나 자살예방 상담전화({CRISIS_LINE_SUICIDE}), "
    f"정신건강 위기상담전화({CRISIS_LINE_MENTAL})로 24시간 도움을 받을 수 있어요."
)

# ── 안전 키워드 ────────────────────────────────────────────────────────────
# 1인칭 위기 표현: 정보성 질의 여부와 무관하게 항상 개인 위기 호소로 간주 (안전 우선)
# 학술 주제어("자살" 등)는 정보성 질의에서 고위험 게이트를 과발동하지 않도록 분리
PERSONAL_DISTRESS_EXPRESSIONS: tuple[str, ...] = (
    "죽고 싶", "사라지고 싶", "없어지고 싶", "더는 못 버티",
    "버티기 싫", "끝내고 싶", "살기 싫", "다 그만두고 싶",
    "사라졌으면", "안 깼으면", "이대로 끝났으면", "그만 살고 싶",
    "눈뜨기 싫", "존재하기 싫", "내가 없어도", "다 놓고 싶", "버틸 이유가 없",
)

# 위 표현 + 학술·객관적 위기 주제어 (개인 호소가 아닌 설명 맥락에도 등장 가능)
SAFETY_KEYWORDS_HIGH: tuple[str, ...] = PERSONAL_DISTRESS_EXPRESSIONS + (
    "자살", "스스로 목숨", "극단적 선택", "생을 마감",
)

SAFETY_KEYWORDS_MID: tuple[str, ...] = (
    "무기력", "절망", "아무 의미 없", "다 포기하고 싶",
    "너무 힘들어서 모르겠", "그냥 다 그만두고 싶",
)

# ── 의도 분류 마커 ─────────────────────────────────────────────────────────
INFORMATIONAL_MARKERS: tuple[str, ...] = (
    "설명", "무엇인가요", "무엇인지", "무엇입니까",
    "요인은", "요인과", "개념", "정의", "차이",
    "효과는", "역할", "영향을", "특성", "비교",
    "정리해", "요약", "알려주세요", "알려줘",
)

# 실천·가이드 마커: INFORMATIONAL_MARKERS보다 우선 검사
GUIDE_MARKERS: tuple[str, ...] = (
    "루틴", "전략", "관리법", "방법", "완화", "예방",
    "실천", "대처", "가이드", "체크리스트", "오늘 바로",
    "계획", "뭐부터", "세워줘", "어디서부터", "시작하",
)

# classify_response_mode 전용 — GUIDE_MARKERS + 요청·해결 의도 단어
_RESPONSE_GUIDE_MARKERS: tuple[str, ...] = (
    "어떻게", "어쩌지", "뭐라고", "추천", "도와줘",
    "해야 해", "하면 좋을까", "해결", "조언",
) + GUIDE_MARKERS

# ── 청킹 설정 ──────────────────────────────────────────────────────────────
# Context Precision 유지를 위해 chunk_size·overlap을 카테고리별로 조정
CHUNK_CONFIG: dict[str, dict] = {
    "safety":       {"chunk_size": 380, "chunk_overlap": 120},
    "burnout":      {"chunk_size": 550, "chunk_overlap": 150},
    "recovery":     {"chunk_size": 550, "chunk_overlap": 150},
    "sleep_stress": {"chunk_size": 480, "chunk_overlap": 130},
}

RISK_PRIORITY: dict[str, int] = {
    "safety": 1, "burnout": 2, "recovery": 3, "sleep_stress": 4,
}

DEFAULT_CATEGORY        = "burnout"
DOCUMENT_GLOB           = "**/*.txt"
TEXT_ENCODING           = "utf-8"
VECTORSTORE_CONFIG_FILE = ".config.json"
VECTORSTORE_INDEX_FILE  = "index.faiss"

DEFAULT_MIND_TEMPERATURE = 65.0
MID_RISK_TEMPERATURE     = 47.0
HIGH_RISK_TEMPERATURE    = 20.0
PAUSE_AFTER_USER_TURNS   = 3


@dataclass(frozen=True)
class QueryProfile:
    risk_level: str
    intent: str
    response_mode: str
    directive: str
    mode_directive: str


@dataclass(frozen=True)
class AnswerParts:
    answer: str
    core_answer: str
    caveat: str
    safety_note: str


def clamp_temperature(value: float) -> float:
    return round(max(0.0, min(100.0, float(value))), 1)


class BaseRAGPipeline(ABC):
    DATA_DIR:  str = "./data"
    FAISS_DIR: str = "./faiss_db"
    ENV_PATH:  str = "./data/.env"

    CHUNK_SIZE:    int = 500
    CHUNK_OVERLAP: int = 80

    EMBEDDING_MODEL: str   = "text-embedding-3-large"
    LLM_MODEL:       str   = "gpt-4o-mini"
    TEMPERATURE:     float = 0.35

    CATEGORY_DIRS: dict[str, str] = {
        "safety":       "P0_safety",
        "burnout":      "P1_burnout",
        "recovery":     "P2_recovery",
        "sleep_stress": "P3_sleep_stress",
    }

    MAX_HISTORY_TURNS: int = 4
    # 마음 온도 임계값 (0~100, 낮을수록 지쳐 있음) — 톤 선택용, 의학적 진단 아님
    THRESHOLD_HIGH: float  = 35.0
    THRESHOLD_MID:  float  = 60.0

    TEMP_HISTORY_PATH: str = "./data/temperature_history.json"

    def __init__(self, **overrides) -> None:
        for key, value in overrides.items():
            if hasattr(self, key):
                setattr(self, key, value)
        load_dotenv(self.ENV_PATH)
        self.embeddings     = OpenAIEmbeddings(model=self.EMBEDDING_MODEL)
        self.llm            = ChatOpenAI(model=self.LLM_MODEL, temperature=self.TEMPERATURE)
        self.vectorstore:   Optional[FAISS]                      = None
        self.retriever:     Optional[Runnable]                   = None
        self._chain:        Optional[Runnable]                   = None
        self._splits:       Optional[list[Document]]             = None
        self._histories:    dict[str, InMemoryChatMessageHistory] = {}
        self._temp_history: dict[str, list[dict]]                = {}
        self._load_temp_history()

    # --- 마음 온도 이력 (파일 기반 영속화) ---

    def _load_temp_history(self) -> None:
        if not os.path.exists(self.TEMP_HISTORY_PATH):
            return
        try:
            with open(self.TEMP_HISTORY_PATH, "r", encoding=TEXT_ENCODING) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            self._temp_history = {}
            return
        self._temp_history = data if isinstance(data, dict) else {}

    def _save_temp_history(self) -> None:
        history_dir = os.path.dirname(self.TEMP_HISTORY_PATH)
        if history_dir:
            os.makedirs(history_dir, exist_ok=True)
        with open(self.TEMP_HISTORY_PATH, "w", encoding=TEXT_ENCODING) as f:
            json.dump(self._temp_history, f, ensure_ascii=False, indent=2)

    def record_temperature(self, session_id: str, temperature: float) -> None:
        """같은 날 재접속하면 마지막 값으로 덮어쓴다."""
        today = date.today().isoformat()
        safe_temperature = clamp_temperature(temperature)
        records = self._temp_history.setdefault(session_id, [])
        if records and records[-1]["date"] == today:
            records[-1]["temperature"] = safe_temperature
        else:
            records.append({"date": today, "temperature": safe_temperature})
        self._save_temp_history()

    def get_temperature_history(self, session_id: str) -> list[dict]:
        return self._temp_history.get(session_id, [])

    # --- 인덱싱 ---

    def _category_path(self, folder: str) -> str:
        return os.path.join(self.DATA_DIR, folder)

    def _vectorstore_config_path(self) -> str:
        return os.path.join(self.FAISS_DIR, VECTORSTORE_CONFIG_FILE)

    def _vectorstore_index_path(self) -> str:
        return os.path.join(self.FAISS_DIR, VECTORSTORE_INDEX_FILE)

    def _missing_documents_error(self, purpose: str = "문서 로딩") -> FileNotFoundError:
        folders = ", ".join(self.CATEGORY_DIRS.values())
        return FileNotFoundError(
            f"{purpose}에 사용할 txt 문서를 찾지 못했습니다. "
            f"{self.DATA_DIR} 아래 카테고리 폴더({folders})에 .txt 파일을 배치해 주세요."
        )

    def load_documents(self) -> list[Document]:
        all_docs: list[Document] = []
        for category, folder in self.CATEGORY_DIRS.items():
            path = self._category_path(folder)
            if not os.path.isdir(path):
                continue
            loader = DirectoryLoader(
                path,
                glob=DOCUMENT_GLOB,
                loader_cls=TextLoader,
                loader_kwargs={"encoding": TEXT_ENCODING},
                show_progress=False,
            )
            docs = loader.load()
            priority = RISK_PRIORITY.get(category, 9)
            for d in docs:
                d.metadata["category"]      = category
                d.metadata["risk_priority"] = priority
                d.metadata["source_folder"] = folder
            all_docs.extend(docs)
        return all_docs

    def split_documents(self, docs: list[Document]) -> list[Document]:
        buckets: dict[str, list[Document]] = {}
        for doc in docs:
            cat = doc.metadata.get("category", DEFAULT_CATEGORY)
            buckets.setdefault(cat, []).append(doc)

        splits: list[Document] = []
        for cat, cat_docs in buckets.items():
            cfg = CHUNK_CONFIG.get(cat, {"chunk_size": self.CHUNK_SIZE, "chunk_overlap": self.CHUNK_OVERLAP})
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=cfg["chunk_size"],
                chunk_overlap=cfg["chunk_overlap"],
                separators=["\n\n", "\n", "。", ". ", " ", ""],
            )
            cat_splits = splitter.split_documents(cat_docs)
            for s in cat_splits:
                s.metadata.setdefault("category", cat)
                s.metadata.setdefault("risk_priority", RISK_PRIORITY.get(cat, 9))
            splits.extend(cat_splits)
        return splits

    # --- FAISS 설정 해시 (청크 설정 변경 시 자동 재빌드) ---

    @staticmethod
    def _chunk_config_hash() -> str:
        return hashlib.md5(
            json.dumps(CHUNK_CONFIG, sort_keys=True).encode()
        ).hexdigest()

    def _is_vectorstore_config_valid(self) -> bool:
        config_path = self._vectorstore_config_path()
        if not os.path.exists(config_path):
            return False
        try:
            with open(config_path, encoding="utf-8") as f:
                return json.load(f).get("chunk_hash") == self._chunk_config_hash()
        except Exception:
            return False

    def _save_vectorstore_config(self) -> None:
        config_path = self._vectorstore_config_path()
        os.makedirs(self.FAISS_DIR, exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump({"chunk_hash": self._chunk_config_hash()}, f)

    def build_vectorstore(self, rebuild: bool = False) -> FAISS:
        index_exists = os.path.exists(self._vectorstore_index_path())
        config_valid = self._is_vectorstore_config_valid()

        if index_exists and not rebuild and config_valid:
            self.vectorstore = FAISS.load_local(
                self.FAISS_DIR,
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True,
            )
        else:
            docs = self.load_documents()
            if not docs:
                raise self._missing_documents_error("FAISS 인덱싱")
            splits = self.split_documents(docs)
            self._splits = splits
            self.vectorstore = FAISS.from_documents(splits, self.embeddings)
            self.vectorstore.save_local(self.FAISS_DIR)
            self._save_vectorstore_config()
        return self.vectorstore

    def _get_corpus_documents(self) -> list[Document]:
        """BM25 인덱싱용 split 문서를 반환한다. FAISS 로컬 로드 시 docstore에서 복원."""
        if self._splits:
            return self._splits
        if self.vectorstore is not None:
            try:
                docs = list(self.vectorstore.docstore._dict.values())
                if docs:
                    self._splits = docs
                    return docs
            except AttributeError:
                pass
        raw = self.load_documents()
        if not raw:
            raise self._missing_documents_error("BM25 코퍼스 생성")
        self._splits = self.split_documents(raw)
        return self._splits

    # --- 리트리버 (추상) ---

    @abstractmethod
    def build_retriever(self, vectorstore: FAISS) -> Runnable:
        raise NotImplementedError

    # --- 위험도 분류 ---

    @staticmethod
    def _detect_safety_keyword(text: str) -> str | None:
        for kw in SAFETY_KEYWORDS_HIGH:
            if kw in text:
                return RISK_HIGH
        for kw in SAFETY_KEYWORDS_MID:
            if kw in text:
                return RISK_MID
        return None

    @staticmethod
    def _has_personal_distress(text: str) -> bool:
        """1인칭 위기 표현 감지. 정보성 질의라도 이 표현이 있으면 개인 위기 호소로 간주."""
        return any(kw in text for kw in PERSONAL_DISTRESS_EXPRESSIONS)

    @classmethod
    def classify_intent(cls, query: str) -> str:
        """질의 의도를 informational / personal 로 분류한다.

        우선순위: 1인칭 위기 표현 → 실천·가이드 마커 → 학술·설명 마커 → 기본값(personal)
        """
        if cls._has_personal_distress(query):
            return INTENT_PERSONAL
        if any(m in query for m in GUIDE_MARKERS):
            return INTENT_PERSONAL
        if any(m in query for m in INFORMATIONAL_MARKERS):
            return INTENT_INFORMATIONAL
        return INTENT_PERSONAL

    @staticmethod
    def compute_mind_temperature(checkin: dict[str, int]) -> float:
        """체크인 점수(각 1~5)로 마음 온도(0~100)를 계산한다. 의학적 진단이 아님."""
        def g(k: str) -> float:
            return float(checkin.get(k, 3))
        positive = g("sleep") * 0.20 + g("energy") * 0.15 + g("recovery") * 0.15 + g("mood") * 0.10
        negative = g("stress") * 0.20 + g("fatigue") * 0.20
        index    = (positive - negative + 1.4) / 4.0 * 100
        return round(max(0.0, min(100.0, index)), 1)

    def classify_risk(
        self,
        query: str,
        checkin: Optional[dict[str, int]] = None,
        session_id: str = "default",
    ) -> str:
        """위험도(high/mid/low) 판정.

        우선순위: 1인칭 위기 표현 → 정보성 질의(체크인만 반영) → 키워드 게이트 → 체크인 → 기본값(low)
        정보성 질의에서는 학술 키워드만으로 고위험 게이트를 발동하지 않는다.
        """
        # 1) 1인칭 위기 표현 → 의도 무관, 항상 최우선 안전 게이트
        if self._has_personal_distress(query):
            return RISK_HIGH

        intent = self.classify_intent(query)

        # 2) 정보성 질의: 학술 키워드 제외, 체크인만 반영
        if intent == INTENT_INFORMATIONAL:
            if checkin is not None:
                temp = self.compute_mind_temperature(checkin)
                if temp < self.THRESHOLD_HIGH:
                    return RISK_HIGH
                if temp < self.THRESHOLD_MID:
                    return RISK_MID
            return RISK_LOW

        # 3) 개인 질의: 키워드 게이트
        keyword_risk = self._detect_safety_keyword(query)
        if keyword_risk is not None:
            return keyword_risk

        # 4) 체크인 마음 온도
        if checkin is not None:
            temp = self.compute_mind_temperature(checkin)
            if temp < self.THRESHOLD_HIGH:
                return RISK_HIGH
            if temp < self.THRESHOLD_MID:
                return RISK_MID

        return RISK_LOW

    # --- 프롬프트 정책 ---

    BASE_SYSTEM             = PROMPT_BASE_SYSTEM
    INFORMATIONAL_DIRECTIVE = PROMPT_INFORMATIONAL_DIRECTIVE
    RISK_DIRECTIVES         = PROMPT_RISK_DIRECTIVES
    MODE_DIRECTIVES         = PROMPT_MODE_DIRECTIVES

    @classmethod
    def classify_response_mode(cls, query: str) -> str:
        q = query.strip()
        if cls.classify_intent(q) == INTENT_INFORMATIONAL:
            return RESPONSE_MODE_INFO
        if any(m in q for m in _RESPONSE_GUIDE_MARKERS):
            return RESPONSE_MODE_GUIDE
        support_markers = (
            "후회", "억울", "불안", "무서", "외로", "속상", "슬프",
            "죄책감", "자책", "혼란",
        )
        if any(m in q for m in support_markers):
            return RESPONSE_MODE_SUPPORT
        if len(q) <= 30:
            return RESPONSE_MODE_EXPLORE
        return RESPONSE_MODE_SUPPORT

    def build_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", self.BASE_SYSTEM),
            MessagesPlaceholder(variable_name="few_shot_examples"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "[참고 문서]\n{context}\n\n[질문]\n{input}"),
        ])

    @staticmethod
    def _build_few_shot_messages(mode: str, risk: str) -> list:
        messages = []
        for ex in PROMPT_FEW_SHOT_EXAMPLES:
            if ex["mode"] == mode and ex["risk"] == risk:
                messages.append(HumanMessage(content=ex["user"]))
                messages.append(AIMessage(content=ex["assistant"]))
        return messages

    # --- 대화 기반 마음 온도 분석 ---

    def analyze_conversation_temperature(self, query: str, messages: list) -> float:
        """LLM이 대화 전체를 읽고 마음 온도(0~100)를 추정한다. 실패 시 키워드 휴리스틱으로 폴백."""
        if not messages:
            return self._fallback_temperature(query)

        recent = messages[-6:]
        conv = "\n".join(
            f"{'사용자' if isinstance(m, HumanMessage) else '봇'}: {m.content[:300]}"
            for m in recent
        )
        prompt_text = (
            "아래 상담 대화를 읽고 사용자의 심리적 안정도(마음 온도)를 0~100 점수로 추정하세요.\n"
            "0~35: 번아웃 위험·위기 신호\n"
            "36~60: 피로·스트레스 주의\n"
            "61~100: 비교적 안정\n\n"
            f"대화:\n{conv}\n\n"
            f"현재 메시지: {query}"
        )
        try:
            result = self.llm.with_structured_output(_ConvTempOutput).invoke(prompt_text)
            return clamp_temperature(result.temperature)
        except Exception:
            return self._fallback_temperature(query)

    def _fallback_temperature(self, query: str) -> float:
        if self._has_personal_distress(query):
            return HIGH_RISK_TEMPERATURE
        if self._detect_safety_keyword(query) == RISK_MID:
            return MID_RISK_TEMPERATURE
        return DEFAULT_MIND_TEMPERATURE

    # --- Self-RAG 검증 훅 (SelfRAGPipeline에서 오버라이드) ---

    def verify_answer(self, question: str, answer: str, context: str) -> tuple[bool, str]:
        return True, answer

    SAFETY_FALLBACK = (
        f"지금 많이 힘드시다면 혼자 견디지 않으셔도 괜찮아요. "
        f"학교 상담센터나 자살예방 상담전화({CRISIS_LINE_SUICIDE})에 연락해 보시길 권해요. "
        f"24시간 운영되며, 정신건강 위기상담전화({CRISIS_LINE_MENTAL})도 이용할 수 있어요. "
        f"제가 도울 수 있는 부분이 있다면 더 이야기해 주세요."
    )

    # 위기 자원 안내 블록 — 본문(core_answer)과 분리해 후처리로만 덧붙임 (Faithfulness 평가 대상 제외)
    SAFETY_BLOCK: str = CRISIS_GUIDE_TEXT

    # --- 체인 조립 ---

    @staticmethod
    def _format_docs(docs: list[Document]) -> str:
        return "\n\n".join(d.page_content for d in docs)

    def _query_profile(
        self,
        question: str,
        checkin: Optional[dict[str, int]],
        session_id: str,
    ) -> QueryProfile:
        risk_level    = self.classify_risk(question, checkin, session_id)
        intent        = self.classify_intent(question)
        response_mode = self.classify_response_mode(question)
        directive = (
            self.INFORMATIONAL_DIRECTIVE
            if intent == INTENT_INFORMATIONAL
            else self.RISK_DIRECTIVES.get(risk_level, self.RISK_DIRECTIVES[RISK_LOW])
        )
        mode_directive = self.MODE_DIRECTIVES.get(response_mode, self.MODE_DIRECTIVES[RESPONSE_MODE_SUPPORT])
        return QueryProfile(
            risk_level=risk_level,
            intent=intent,
            response_mode=response_mode,
            directive=directive,
            mode_directive=mode_directive,
        )

    def _retrieve_context_docs(self, question: str) -> list[Document]:
        if self.retriever is None:
            raise RuntimeError("retriever가 초기화되지 않았습니다. build_chain()을 먼저 호출하세요.")
        return self.retriever.invoke(question)

    def _generate_core_answer(
        self,
        question: str,
        profile: QueryProfile,
        history: InMemoryChatMessageHistory,
    ) -> str:
        if self._chain is None:
            raise RuntimeError("chain이 초기화되지 않았습니다. build_chain()을 먼저 호출하세요.")
        return self._chain.invoke({
            "input":             question,
            "risk_directive":    profile.directive,
            "mode_directive":    profile.mode_directive,
            "pause_directive":   self._pause_directive(history),
            "chat_history":      history.messages,
            "few_shot_examples": self._build_few_shot_messages(profile.response_mode, profile.risk_level),
        })

    def _pause_directive(self, history: InMemoryChatMessageHistory) -> str:
        current_user_turn = self._user_turn_count(history) + 1
        if current_user_turn < PAUSE_AFTER_USER_TURNS:
            return ""
        if current_user_turn % PAUSE_AFTER_USER_TURNS != 0:
            return ""
        return (
            "[대화 휴식 지침] 이번 답변은 새로운 질문으로 끝내지 마세요. "
            "사용자가 대화를 이어가야 한다는 압박을 느끼지 않도록, "
            "'여기서 잠깐 쉬어도 괜찮다'는 문장으로 부드럽게 마무리하세요."
        )

    @staticmethod
    def _user_turn_count(history: InMemoryChatMessageHistory) -> int:
        return sum(1 for message in history.messages if isinstance(message, HumanMessage))

    def _extract_verifiable_text(self, answer: str, profile: QueryProfile) -> tuple[str, str]:
        """공감 블록(검증 제외)과 검증 대상 텍스트를 분리한다.

        informational 또는 단일 문단이면 전체를 검증 대상으로 반환한다.
        복수 문단이면 첫 문단을 공감 블록으로, 나머지를 검증 대상으로 분리한다.
        """
        if profile.intent == INTENT_INFORMATIONAL:
            return "", answer
        paragraphs = [p.strip() for p in answer.split("\n\n") if p.strip()]
        if len(paragraphs) <= 1:
            return "", answer
        return paragraphs[0], "\n\n".join(paragraphs[1:])

    def _rewrite_grounded_answer_conservatively(
        self, question: str, grounded_text: str, context: str
    ) -> str:
        """기본 구현은 원본 반환. SelfRAGPipeline에서 문맥 기반 재작성으로 오버라이드."""
        return grounded_text

    def _compose_answer_parts(
        self,
        core_answer: str,
        supported: bool,
        profile: QueryProfile,
    ) -> AnswerParts:
        caveat      = "" if supported else getattr(self, "LOW_SUPPORT_NOTE", "").strip()
        safety_note = ""
        is_genuine_crisis = profile.risk_level == RISK_HIGH and profile.intent == INTENT_PERSONAL

        if is_genuine_crisis:
            if not supported:
                core_answer = self.SAFETY_FALLBACK
                caveat      = ""
            else:
                safety_note = self.SAFETY_BLOCK

        parts = [p for p in (core_answer.strip(), caveat, safety_note.strip()) if p]
        return AnswerParts(
            answer="\n\n".join(parts),
            core_answer=core_answer,
            caveat=caveat,
            safety_note=safety_note,
        )

    @staticmethod
    def _response_dict(profile: QueryProfile, answer_parts: AnswerParts, mind_temp: float) -> dict:
        return {
            "answer":           answer_parts.answer,
            "core_answer":      answer_parts.core_answer,
            "safety_note":      answer_parts.safety_note,
            "caveat":           answer_parts.caveat,
            "risk_level":       profile.risk_level,
            "query_intent":     profile.intent,
            "response_mode":    profile.response_mode,
            "mind_temperature": mind_temp,
        }

    def _get_session_history(self, session_id: str) -> InMemoryChatMessageHistory:
        """세션 히스토리 반환. MAX_HISTORY_TURNS 초과분은 앞에서 잘라낸다."""
        history  = self._histories.setdefault(session_id, InMemoryChatMessageHistory())
        max_msgs = self.MAX_HISTORY_TURNS * 2
        if len(history.messages) > max_msgs:
            history.messages = history.messages[-max_msgs:]
        return history

    def build_chain(self) -> Runnable:
        if self.vectorstore is None:
            self.build_vectorstore()
        if self.retriever is None:
            self.retriever = self.build_retriever(self.vectorstore)
        # chat_history는 ask()에서 직접 주입 — RunnableWithMessageHistory 불필요
        self._chain = (
            RunnablePassthrough.assign(
                context=lambda x: self._format_docs(self.retriever.invoke(x["input"])),
            )
            | self.build_prompt()
            | self.llm
            | StrOutputParser()
        )
        return self._chain

    # --- 공개 진입점 ---

    def ask(
        self,
        question: str,
        session_id: str = "default",
        checkin: Optional[dict[str, int]] = None,
    ) -> dict:
        if self._chain is None:
            self.build_chain()

        profile = self._query_profile(question, checkin, session_id)
        history = self._get_session_history(session_id)

        # 1: 마음 온도 — AI 답변 추가 전 사용자 중심 이력으로 분석
        mind_temp = self.analyze_conversation_temperature(question, history.messages)
        self.record_temperature(session_id, mind_temp)

        # 2: 문맥 근거 본문 생성
        core_answer = self._generate_core_answer(question, profile, history)

        # 3: 대화 이력 저장 (마음 온도 분석 이후)
        history.add_messages([HumanMessage(content=question), AIMessage(content=core_answer)])

        # 4: Self-RAG 검증 — 공감 블록 제외, 정보 블록만 검증
        context_docs = self._retrieve_context_docs(question)
        context_str  = self._format_docs(context_docs)
        empathy_part, verifiable_text = self._extract_verifiable_text(core_answer, profile)
        ok, _ = self.verify_answer(question, verifiable_text, context_str)

        # 5: personal 의도 응답은 caveat 면제 — 공감 답변은 학술 문서 검증 대상이 아님
        if not ok and profile.intent == INTENT_PERSONAL:
            if empathy_part:
                rewritten = self._rewrite_grounded_answer_conservatively(
                    question, verifiable_text, context_str
                )
                core_answer = empathy_part + "\n\n" + rewritten
            ok = True

        # 6: 공감·면책·안전 안내 블록 조립
        answer_parts = self._compose_answer_parts(core_answer, ok, profile)
        return self._response_dict(profile, answer_parts, mind_temp)


# ===========================================================================
# BM25 한국어 전처리 헬퍼 (kiwipiepy 미설치 시 공백 분리로 폴백)
# ===========================================================================
_KIWI_INSTANCE = None


def _import_ensemble_retriever():
    for module_path in (
        "langchain_community.retrievers",
        "langchain.retrievers",
        "langchain_classic.retrievers",
    ):
        try:
            module = __import__(module_path, fromlist=["EnsembleRetriever"])
            if hasattr(module, "EnsembleRetriever"):
                return module.EnsembleRetriever
        except ImportError:
            continue
    raise ImportError(
        "EnsembleRetriever 를 찾을 수 없습니다. langchain-classic / langchain-community 설치를 확인하세요."
    )


def make_kiwi_tokenizer():
    global _KIWI_INSTANCE
    try:
        from kiwipiepy import Kiwi
        if _KIWI_INSTANCE is None:
            _KIWI_INSTANCE = Kiwi()
        kiwi = _KIWI_INSTANCE
        return lambda text: [token.form for token in kiwi.tokenize(text)]
    except ImportError:
        return str.split


# ===========================================================================
# Self-RAG 검증 스키마
# ===========================================================================
class _ConvTempOutput(BaseModel):
    """대화 분석 마음 온도 출력 스키마."""
    temperature: float = Field(
        description="사용자 심리 안정도. 0~35=번아웃 위험, 36~60=피로 주의, 61~100=비교적 안정",
        ge=0.0, le=100.0,
    )


class SupportEval(BaseModel):
    reasoning: str = Field(description="판단 근거를 한국어로 간단히")
    issup: Literal["Fully supported", "Partially supported", "No support"] = Field(
        description="문맥에 의한 답변 지원 정도"
    )


class UtilityEval(BaseModel):
    reasoning: str = Field(description="판단 근거를 한국어로 간단히")
    isuse: int = Field(description="1(최저)~5(최고) 유용성 점수", ge=1, le=5)


# ===========================================================================
# SelfRAGPipeline — Hybrid Search + 세션 위험도 추적 + Self-RAG 검증
# ===========================================================================
class SelfRAGPipeline(BaseRAGPipeline):
    TOP_K:        int   = 3
    FETCH_K:      int   = 12
    BM25_WEIGHT:  float = 0.5
    FAISS_WEIGHT: float = 0.5

    HIGH_RISK_STREAK_THRESHOLD: int = 2

    MIN_UTILITY:          int  = 4
    REQUIRE_FULL_SUPPORT: bool = False

    LOW_SUPPORT_NOTE = (
        "참고: 위 내용은 제공된 자료로 충분히 뒷받침되지 않을 수 있어요. "
        "중요한 결정이나 증상 판단은 전문가와 상의해 주세요."
    )

    _SUPPORT_PROMPT = ChatPromptTemplate.from_messages([
        ("system",
         "당신은 RAG 답변 검증자입니다. [답변]의 각 문장이 [문맥]에 의해 "
         "사실적으로 뒷받침되는지 문장 단위로 점검하세요. 문맥에 없는 주장(외부 지식·"
         "추측·일반 상식)이 하나라도 있으면 지원 정도를 낮게 매기세요. "
         "위로·안내 같은 비사실 문장은 사실성 판단에서 제외하세요."),
        ("human", "[질문]\n{question}\n\n[답변]\n{answer}\n\n[문맥]\n{context}"),
    ])

    _UTILITY_PROMPT = ChatPromptTemplate.from_messages([
        ("system",
         "당신은 RAG 답변 평가자입니다. [답변]이 [질문]에 대해 얼마나 "
         "유용한지 1~5점으로 평가하세요. 핵심을 짚고 실질적 도움이 되면 높게, "
         "겉돌거나 회피적이면 낮게 매기세요."),
        ("human", "[질문]\n{question}\n\n[답변]\n{answer}"),
    ])

    _REWRITE_PROMPT = ChatPromptTemplate.from_messages([
        ("system",
         "당신은 상담형 AI 답변 수정자입니다. "
         "[원본 답변]이 [문맥]에 충분히 근거하지 않는 것으로 판단되었습니다. "
         "다음 원칙에 따라 수정하세요:\n"
         "- 문맥에 없는 사실을 추가하지 말 것\n"
         "- 확실하지 않은 내용은 단정하지 말 것\n"
         "- 문서에서 확인 가능한 내용만 짧게 유지할 것\n"
         "- 공감적 어조는 유지하되, 사실처럼 단정하지 말 것\n"
         "- 바로 실천할 수 있는 행동을 1~2개만 제안할 것\n"
         "수정된 답변만 출력하고, 설명이나 메타 코멘트는 생략하세요."),
        ("human",
         "[원본 답변]\n{grounded_text}\n\n[문맥]\n{context}\n\n[질문]\n{question}"),
    ])

    def __init__(self, **overrides) -> None:
        super().__init__(**overrides)
        self._high_risk_streak: dict[str, int] = {}
        self._support_chain: Optional[Runnable] = None
        self._utility_chain: Optional[Runnable] = None
        self.last_critique:  Optional[dict]     = None

    # --- 세션 위험도 연속 추적 ---

    def classify_risk(
        self,
        query: str,
        checkin: Optional[dict[str, int]] = None,
        session_id: str = "default",
    ) -> str:
        base_risk = super().classify_risk(query, checkin, session_id)
        streak    = self._high_risk_streak.get(session_id, 0)
        if base_risk == RISK_HIGH:
            self._high_risk_streak[session_id] = streak + 1
        else:
            self._high_risk_streak[session_id] = max(0, streak - 1)
        if self._high_risk_streak[session_id] >= self.HIGH_RISK_STREAK_THRESHOLD:
            return RISK_HIGH
        return base_risk

    # --- Hybrid 검색 (BM25 + FAISS 앙상블 + 임베딩 리랭킹) ---

    def _rerank_by_embedding_similarity(
        self,
        query: str,
        docs: list[Document],
        top_k: int,
    ) -> list[Document]:
        if len(docs) <= top_k:
            return docs
        q_emb   = np.array(self.embeddings.embed_query(query))
        d_embs  = np.array(self.embeddings.embed_documents([d.page_content for d in docs]))
        q_norm  = q_emb  / (np.linalg.norm(q_emb)                         + 1e-10)
        d_norms = d_embs / (np.linalg.norm(d_embs, axis=1, keepdims=True) + 1e-10)
        scores  = d_norms @ q_norm
        return [docs[i] for i in np.argsort(scores)[::-1][:top_k]]

    def build_retriever(self, vectorstore: FAISS) -> Runnable:
        from langchain_community.retrievers import BM25Retriever
        EnsembleRetriever = _import_ensemble_retriever()

        corpus = self._get_corpus_documents()
        bm25   = BM25Retriever.from_documents(corpus, preprocess_func=make_kiwi_tokenizer())
        bm25.k = self.FETCH_K

        faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": self.FETCH_K})
        ensemble = EnsembleRetriever(
            retrievers=[bm25, faiss_retriever],
            weights=[self.BM25_WEIGHT, self.FAISS_WEIGHT],
        )
        top_k = self.TOP_K

        def _rerank(query: str) -> list[Document]:
            docs = ensemble.invoke(query)
            return self._rerank_by_embedding_similarity(query, docs, top_k)

        return RunnableLambda(_rerank)

    # --- Self-RAG 검증 ---

    def _build_critique_chains(self) -> None:
        self._support_chain = self._SUPPORT_PROMPT | self.llm.with_structured_output(SupportEval)
        self._utility_chain = self._UTILITY_PROMPT | self.llm.with_structured_output(UtilityEval)

    def verify_answer(self, question: str, answer: str, context: str) -> tuple[bool, str]:
        if self._support_chain is None or self._utility_chain is None:
            self._build_critique_chains()
        try:
            support: SupportEval = self._support_chain.invoke(
                {"question": question, "answer": answer, "context": context}
            )
            utility: UtilityEval = self._utility_chain.invoke(
                {"question": question, "answer": answer}
            )
        except Exception:
            self.last_critique = {"error": "critique_failed"}
            return True, answer

        self.last_critique = {
            "issup":             support.issup,
            "isuse":             utility.isuse,
            "support_reasoning": support.reasoning,
            "utility_reasoning": utility.reasoning,
        }
        supported = (
            support.issup == "Fully supported"
            if self.REQUIRE_FULL_SUPPORT
            else support.issup != "No support"
        )
        ok = supported and utility.isuse >= self.MIN_UTILITY
        return ok, answer

    # --- 검증 실패 시 정보 블록 보수적 재작성 ---

    def _rewrite_grounded_answer_conservatively(
        self, question: str, grounded_text: str, context: str
    ) -> str:
        """검증 실패한 정보 블록을 문맥 기반으로 보수적으로 재작성한다. 최대 1회 시도."""
        try:
            chain = self._REWRITE_PROMPT | self.llm | StrOutputParser()
            return chain.invoke({
                "question":      question,
                "grounded_text": grounded_text,
                "context":       context,
            })
        except Exception:
            return grounded_text
