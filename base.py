from __future__ import annotations

import hashlib
import json
import os
import warnings
from abc import ABC, abstractmethod
from datetime import date
from typing import Literal, Optional

import numpy as np

# langchain_community는 sunset 예정이지만 FAISS·DirectoryLoader·TextLoader·BM25Retriever의
# standalone 패키지가 아직 없어 계속 사용해야 한다. 해당 경고만 선택적으로 억제한다.
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


RISK_HIGH = "high"
RISK_MID  = "mid"
RISK_LOW  = "low"

# 자살예방 상담전화는 109로 통합됨 (2024.1~)
CRISIS_LINE_SUICIDE: str = "109"
CRISIS_LINE_MENTAL:  str = "1577-0199"

CRISIS_GUIDE_TEXT: str = (
    f"힘든 마음이 클 때는 혼자 견디지 않으셔도 괜찮아요. "
    f"학교 상담센터나 자살예방 상담전화({CRISIS_LINE_SUICIDE}), "
    f"정신건강 위기상담전화({CRISIS_LINE_MENTAL})로 24시간 도움을 받을 수 있어요."
)

SAFETY_KEYWORDS_HIGH: tuple[str, ...] = (
    "죽고 싶",
    "사라지고 싶",
    "없어지고 싶",
    "더는 못 버티",
    "버티기 싫",
    "끝내고 싶",
    "자살",
    "스스로 목숨",
    "극단적 선택",
    "생을 마감",
    "살기 싫",
)

SAFETY_KEYWORDS_MID: tuple[str, ...] = (
    "무기력",
    "절망",
    "아무 의미 없",
    "다 포기하고 싶",
    "너무 힘들어서 모르겠",
    "그냥 다 그만두고 싶",
)

# 카테고리별 청킹 설정
# RAGAS 평가(Context Precision 0.52)에서 청크 크기 과대가 정밀도 저하 원인으로 진단됨.
# chunk_size는 ~30% 축소, chunk_overlap은 유지(Context Recall 0.85 보호).
CHUNK_CONFIG: dict[str, dict] = {
    "safety":       {"chunk_size": 300, "chunk_overlap": 80},
    "burnout":      {"chunk_size": 450, "chunk_overlap": 100},
    "recovery":     {"chunk_size": 450, "chunk_overlap": 100},
    "sleep_stress": {"chunk_size": 380, "chunk_overlap": 80},
}

RISK_PRIORITY: dict[str, int] = {
    "safety": 1,
    "burnout": 2,
    "recovery": 3,
    "sleep_stress": 4,
}


class BaseRAGPipeline(ABC):
    DATA_DIR:  str = "./data"
    FAISS_DIR: str = "./faiss_db"
    ENV_PATH:  str = "./data/.env"

    CHUNK_SIZE:    int = 500
    CHUNK_OVERLAP: int = 80

    EMBEDDING_MODEL: str   = "text-embedding-3-large"
    LLM_MODEL:       str   = "gpt-4o-mini"
    TEMPERATURE:     float = 0.2

    CATEGORY_DIRS: dict[str, str] = {
        "safety":       "P0_safety",
        "burnout":      "P1_burnout",
        "recovery":     "P2_recovery",
        "sleep_stress": "P3_sleep_stress",
    }

    MAX_HISTORY_TURNS: int = 4

    # 마음 온도 임계값 (0~100, 낮을수록 지쳐 있음) — 의학적 진단 아님, 톤 선택용
    THRESHOLD_HIGH: float = 35.0
    THRESHOLD_MID:  float = 60.0

    TEMP_HISTORY_PATH: str = "./data/temperature_history.json"

    def __init__(self, **overrides) -> None:
        for key, value in overrides.items():
            if hasattr(self, key):
                setattr(self, key, value)

        load_dotenv(self.ENV_PATH)

        self.embeddings = OpenAIEmbeddings(model=self.EMBEDDING_MODEL)
        self.llm        = ChatOpenAI(model=self.LLM_MODEL, temperature=self.TEMPERATURE)

        self.vectorstore:   Optional[FAISS]            = None
        self.retriever:     Optional[Runnable]          = None
        self._chain:        Optional[Runnable]          = None
        self._splits:       Optional[list[Document]]   = None
        self._histories:    dict[str, InMemoryChatMessageHistory] = {}
        self._temp_history: dict[str, list[dict]]      = {}
        self._load_temp_history()

    # ------------------------------------------------------------------
    # 마음 온도 이력 (파일 기반 영속화)
    # ------------------------------------------------------------------
    def _load_temp_history(self) -> None:
        if os.path.exists(self.TEMP_HISTORY_PATH):
            with open(self.TEMP_HISTORY_PATH, "r", encoding="utf-8") as f:
                self._temp_history = json.load(f)

    def _save_temp_history(self) -> None:
        os.makedirs(os.path.dirname(self.TEMP_HISTORY_PATH), exist_ok=True)
        with open(self.TEMP_HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(self._temp_history, f, ensure_ascii=False, indent=2)

    def record_temperature(self, session_id: str, temperature: float) -> None:
        """같은 날 재접속하면 마지막 값으로 덮어쓴다."""
        today   = date.today().isoformat()
        records = self._temp_history.setdefault(session_id, [])
        if records and records[-1]["date"] == today:
            records[-1]["temperature"] = temperature
        else:
            records.append({"date": today, "temperature": temperature})
        self._save_temp_history()

    def get_temperature_history(self, session_id: str) -> list[dict]:
        return self._temp_history.get(session_id, [])

    # ------------------------------------------------------------------
    # 인덱싱
    # ------------------------------------------------------------------
    def load_documents(self) -> list[Document]:
        all_docs: list[Document] = []
        for category, folder in self.CATEGORY_DIRS.items():
            path = os.path.join(self.DATA_DIR, folder)
            if not os.path.isdir(path):
                continue
            loader = DirectoryLoader(
                path,
                glob="**/*.txt",
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"},
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
            cat = doc.metadata.get("category", "burnout")
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

    # ------------------------------------------------------------------
    # FAISS 설정 해시 — 청크 설정 변경 시 자동 재빌드
    # ------------------------------------------------------------------
    @staticmethod
    def _chunk_config_hash() -> str:
        return hashlib.md5(
            json.dumps(CHUNK_CONFIG, sort_keys=True).encode()
        ).hexdigest()

    def _is_vectorstore_config_valid(self) -> bool:
        config_path = os.path.join(self.FAISS_DIR, ".config.json")
        if not os.path.exists(config_path):
            return False
        try:
            with open(config_path, encoding="utf-8") as f:
                return json.load(f).get("chunk_hash") == self._chunk_config_hash()
        except Exception:
            return False

    def _save_vectorstore_config(self) -> None:
        config_path = os.path.join(self.FAISS_DIR, ".config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump({"chunk_hash": self._chunk_config_hash()}, f)

    def build_vectorstore(self, rebuild: bool = False) -> FAISS:
        index_exists   = os.path.exists(os.path.join(self.FAISS_DIR, "index.faiss"))
        config_valid   = self._is_vectorstore_config_valid()

        if index_exists and not rebuild and config_valid:
            self.vectorstore = FAISS.load_local(
                self.FAISS_DIR,
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True,
            )
        else:
            docs = self.load_documents()
            if not docs:
                raise FileNotFoundError(
                    f"{self.DATA_DIR} 의 카테고리 폴더에서 PDF를 찾지 못했습니다."
                )
            splits       = self.split_documents(docs)
            self._splits = splits
            self.vectorstore = FAISS.from_documents(splits, self.embeddings)
            self.vectorstore.save_local(self.FAISS_DIR)
            self._save_vectorstore_config()

        return self.vectorstore

    def _get_corpus_documents(self) -> list[Document]:
        """BM25 인덱싱용 split 문서를 반환한다.

        FAISS는 디스크에 저장되지만 BM25 인덱스는 매 실행마다 메모리에서
        재구성해야 하므로, 동일 코퍼스를 일관되게 공급하는 것이 중요하다.
        """
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
            raise FileNotFoundError(
                f"{self.DATA_DIR} 에서 BM25 코퍼스를 만들 PDF를 찾지 못했습니다."
            )
        self._splits = self.split_documents(raw)
        return self._splits

    # ------------------------------------------------------------------
    # 리트리버 (추상)
    # ------------------------------------------------------------------
    @abstractmethod
    def build_retriever(self, vectorstore: FAISS) -> Runnable:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # 위험도 분류
    # ------------------------------------------------------------------
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
        """위험도(high/mid/low)를 판정한다.

        우선순위: 1) 안전 키워드 텍스트 감지  2) 체크인 마음 온도  3) 기본값 low
        안전 키워드 감지는 체크인 없이도 즉시 고위험 판정하는 안전 게이트다.
        """
        keyword_risk = self._detect_safety_keyword(query)
        if keyword_risk is not None:
            return keyword_risk

        if checkin is not None:
            temp = self.compute_mind_temperature(checkin)
            if temp < self.THRESHOLD_HIGH:
                return RISK_HIGH
            if temp < self.THRESHOLD_MID:
                return RISK_MID

        return RISK_LOW

    # ------------------------------------------------------------------
    # 프롬프트
    # ------------------------------------------------------------------
    # Answer Relevancy 개선: 직접 답변 지시 + 핵심→근거→권고 구조 + 분량 제한
    # Faithfulness 유지: 환각 방지 강화 ("반드시 문맥만 근거")
    BASE_SYSTEM = (
        "당신은 번아웃과 정서 회복을 돕는 따뜻한 상담 도우미입니다. "
        "반드시 아래 [참고 문서]에 근거해서만 답하고, 문서에 없는 내용은 '자료에서 확인되지 않아요'라고 솔직히 말하세요. "
        "질문에 직접 답하고, 문맥 중 질문과 무관한 내용은 사용하지 마세요. "
        "답변은 [핵심 답변 → 근거 → 권고(필요 시)] 순서로 3~5문장 이내로 간결하게 작성하세요. "
        "진단·단정 표현은 피하고, 사용자를 평가하지 마세요. "
        "의료적 판단이 필요하면 전문가 상담을 권유하세요.\n{risk_directive}"
    )

    RISK_DIRECTIVES: dict[str, str] = {
        RISK_HIGH: (
            "[톤 지침] 사용자가 많이 지쳐 있을 수 있습니다. '고위험' 같은 진단 표현 대신 "
            "'최근 많이 힘드셨겠어요' 처럼 부드럽게 공감하세요. "
            "반드시 학교 상담센터·자살예방 상담전화(109)·정신건강 위기상담전화(1577-0199) "
            "연계를 자연스럽게 안내하고, 혼자 감당하지 않아도 됨을 전달하세요. "
            "삶의 이유(가족, 친구, 미래 목표 등)를 함께 확인하는 질문을 건네세요."
        ),
        RISK_MID: (
            "[톤 지침] '최근 조금 지쳐 보여요' 정도의 부담 없는 표현을 쓰세요. "
            "작게 실천할 수 있는 회복 행동과 학업·업무 조절을 한두 가지 제안하세요. "
            "신뢰할 수 있는 주변 사람이나 상담센터 방문을 부드럽게 권유하세요."
        ),
        RISK_LOW: (
            "[톤 지침] 안정적인 상태로 보입니다. 예방 관점의 가벼운 가이드와 "
            "긍정적 강화를 중심으로 답하세요."
        ),
    }

    def build_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", self.BASE_SYSTEM),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "[참고 문서]\n{context}\n\n[질문]\n{input}"),
        ])

    # ------------------------------------------------------------------
    # 대화 기반 마음 온도 분석
    # ------------------------------------------------------------------
    def analyze_conversation_temperature(
        self, query: str, messages: list
    ) -> float:
        """LLM이 대화 전체를 읽고 마음 온도(0~100)를 추정한다.

        슬라이더 체크인 없이 대화 톤·내용만으로 심리적 안정도를 산출한다.
        키워드 감지가 먼저 적용되며, LLM 호출 실패 시 키워드 기반 휴리스틱으로 폴백.
        """
        # 첫 메시지: 히스토리 없음 → 키워드 감지만
        if not messages:
            risk = self._detect_safety_keyword(query)
            return {RISK_HIGH: 20.0, RISK_MID: 47.0}.get(risk, 65.0)

        # 최근 3턴(6 메시지) 대화 요약
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
            return round(max(0.0, min(100.0, result.temperature)), 1)
        except Exception:
            risk = self._detect_safety_keyword(query)
            return {RISK_HIGH: 20.0, RISK_MID: 47.0}.get(risk, 65.0)

    # ------------------------------------------------------------------
    # Self-RAG 검증 훅 (Phase3SelfRAGPipeline에서 오버라이드)
    # ------------------------------------------------------------------
    def verify_answer(self, question: str, answer: str, context: str) -> tuple[bool, str]:
        return True, answer

    SAFETY_FALLBACK = (
        f"지금 많이 힘드시다면 혼자 견디지 않으셔도 괜찮아요. "
        f"학교 상담센터나 자살예방 상담전화({CRISIS_LINE_SUICIDE})에 연락해 보시길 권해요. "
        f"24시간 운영되며, 정신건강 위기상담전화({CRISIS_LINE_MENTAL})도 이용할 수 있어요. "
        f"제가 도울 수 있는 부분이 있다면 더 이야기해 주세요."
    )

    # ------------------------------------------------------------------
    # 체인 조립
    # ------------------------------------------------------------------
    @staticmethod
    def _format_docs(docs: list[Document]) -> str:
        return "\n\n".join(d.page_content for d in docs)

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

        # chat_history 는 ask() 에서 직접 주입 — RunnableWithMessageHistory 불필요
        self._chain = (
            RunnablePassthrough.assign(
                context=lambda x: self._format_docs(self.retriever.invoke(x["input"])),
            )
            | self.build_prompt()
            | self.llm
            | StrOutputParser()
        )
        return self._chain

    # ------------------------------------------------------------------
    # 공개 진입점
    # ------------------------------------------------------------------
    def ask(
        self,
        question: str,
        session_id: str = "default",
        checkin: Optional[dict[str, int]] = None,
    ) -> dict:
        if self._chain is None:
            self.build_chain()

        risk_level     = self.classify_risk(question, checkin, session_id)
        risk_directive = self.RISK_DIRECTIVES.get(risk_level, self.RISK_DIRECTIVES[RISK_LOW])

        history = self._get_session_history(session_id)

        answer = self._chain.invoke({
            "input":          question,
            "risk_directive": risk_directive,
            "chat_history":   history.messages,
        })

        # 대화 이력 업데이트 (RunnableWithMessageHistory 대체)
        history.add_messages([HumanMessage(content=question), AIMessage(content=answer)])

        context_docs = self.retriever.invoke(question)
        ok, answer   = self.verify_answer(question, answer, self._format_docs(context_docs))

        if not ok and risk_level == RISK_HIGH:
            answer = self.SAFETY_FALLBACK

        # 자살 키워드 감지 시 안전 안내를 답변 뒤에 항상 덧붙임
        if risk_level == RISK_HIGH and self._detect_safety_keyword(question) == RISK_HIGH:
            if self.SAFETY_FALLBACK not in answer:
                answer = answer + "\n\n" + self.SAFETY_FALLBACK

        # 대화 내용으로 마음 온도를 추정 (슬라이더 체크인 불필요)
        mind_temp = self.analyze_conversation_temperature(question, history.messages)
        self.record_temperature(session_id, mind_temp)

        return {
            "answer":           answer,
            "risk_level":       risk_level,
            "mind_temperature": mind_temp,
        }


# ===========================================================================
# Phase 1 — 단일 FAISS 리트리버
# ===========================================================================
class Phase1FaissPipeline(BaseRAGPipeline):
    TOP_K: int = 3  # RAGAS: Context Precision 0.52 → Top-K 축소로 노이즈 감소

    def build_retriever(self, vectorstore: FAISS) -> Runnable:
        return vectorstore.as_retriever(search_kwargs={"k": self.TOP_K})


# ===========================================================================
# Phase 2 — 세션별 위험도 누적 추적
#   고위험 키워드가 연속 N회 이상 감지되면 체크인과 무관하게 RISK_HIGH 유지.
# ===========================================================================
class Phase2MemoryPipeline(Phase1FaissPipeline):
    HIGH_RISK_STREAK_THRESHOLD: int = 2

    def __init__(self, **overrides) -> None:
        super().__init__(**overrides)
        self._high_risk_streak: dict[str, int] = {}

    def classify_risk(
        self,
        query: str,
        checkin: Optional[dict[str, int]] = None,
        session_id: str = "default",
    ) -> str:
        base_risk = super().classify_risk(query, checkin)

        streak = self._high_risk_streak.get(session_id, 0)
        if base_risk == RISK_HIGH:
            self._high_risk_streak[session_id] = streak + 1
        else:
            self._high_risk_streak[session_id] = max(0, streak - 1)

        if self._high_risk_streak[session_id] >= self.HIGH_RISK_STREAK_THRESHOLD:
            return RISK_HIGH

        return base_risk


# ===========================================================================
# Kiwi 형태소 토크나이저 (BM25 한국어 전처리)
#   kiwipiepy 미설치 시 공백 분리로 폴백.
# ===========================================================================
_KIWI_INSTANCE = None


def _import_ensemble_retriever():
    for module_path in (
        "langchain_classic.retrievers",    # langchain 1.x
        "langchain_community.retrievers",  # langchain 0.x
        "langchain.retrievers",
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
# Phase 3-A — BM25 단독 (희소 검색, 주로 평가·비교용)
# ===========================================================================
class Phase3BM25Pipeline(Phase2MemoryPipeline):
    TOP_K: int = 3

    def build_retriever(self, vectorstore: FAISS) -> Runnable:
        from langchain_community.retrievers import BM25Retriever

        corpus = self._get_corpus_documents()
        bm25   = BM25Retriever.from_documents(corpus, preprocess_func=make_kiwi_tokenizer())
        bm25.k = self.TOP_K
        return bm25


# ===========================================================================
# Phase 3-B — BM25 + FAISS 앙상블 (Hybrid Search)
# ===========================================================================
class Phase3HybridPipeline(Phase2MemoryPipeline):
    TOP_K:        int   = 3   # 최종 반환 문서 수
    FETCH_K:      int   = 5   # 앙상블 1차 후보 수 (BM25 키워드 커버리지 확보 후 리랭킹으로 필터)
    BM25_WEIGHT:  float = 0.5
    FAISS_WEIGHT: float = 0.5

    def build_retriever(self, vectorstore: FAISS) -> Runnable:
        from langchain_community.retrievers import BM25Retriever
        EnsembleRetriever = _import_ensemble_retriever()

        corpus = self._get_corpus_documents()
        bm25   = BM25Retriever.from_documents(corpus, preprocess_func=make_kiwi_tokenizer())
        bm25.k = self.FETCH_K  # BM25는 넓게 수집 후 리랭킹으로 필터

        faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": self.TOP_K})

        ensemble = EnsembleRetriever(
            retrievers=[bm25, faiss_retriever],
            weights=[self.BM25_WEIGHT, self.FAISS_WEIGHT],
        )

        # 임베딩 기반 리랭커: BM25 노이즈 문서를 걸러내 Context Precision 향상
        embeddings = self.embeddings
        top_k      = self.TOP_K

        def _rerank(query: str) -> list[Document]:
            docs = ensemble.invoke(query)
            if len(docs) <= top_k:
                return docs
            q_emb   = np.array(embeddings.embed_query(query))
            d_embs  = np.array(embeddings.embed_documents([d.page_content for d in docs]))
            q_norm  = q_emb  / (np.linalg.norm(q_emb) + 1e-10)
            d_norms = d_embs / (np.linalg.norm(d_embs, axis=1, keepdims=True) + 1e-10)
            scores  = d_norms @ q_norm
            return [docs[i] for i in np.argsort(scores)[::-1][:top_k]]

        return RunnableLambda(_rerank)


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
# Phase 3 (최종형) — Hybrid 검색 + Self-RAG 자체 검증
#   근거 부족 시 투명하게 고지하는 것이 P0 안전 도메인의 신뢰도 핵심.
# ===========================================================================
class Phase3SelfRAGPipeline(Phase3HybridPipeline):
    MIN_UTILITY: int = 3

    LOW_SUPPORT_NOTE = (
        "\n\n참고: 위 내용은 제공된 자료로 충분히 뒷받침되지 않을 수 있어요. "
        "중요한 결정이나 증상 판단은 전문가와 상의해 주세요."
    )

    def __init__(self, **overrides) -> None:
        super().__init__(**overrides)
        self._support_chain: Optional[Runnable] = None
        self._utility_chain: Optional[Runnable] = None
        self.last_critique:  Optional[dict]     = None

    def _build_critique_chains(self) -> None:
        support_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "당신은 RAG 답변 검증자입니다. 주어진 [답변]이 [문맥]에 의해 "
             "사실적으로 뒷받침되는 정도를 평가하세요. 문맥에 없는 내용을 답변이 "
             "주장하면 지원 정도를 낮게 매기세요."),
            ("human", "[질문]\n{question}\n\n[답변]\n{answer}\n\n[문맥]\n{context}"),
        ])
        utility_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "당신은 RAG 답변 평가자입니다. [답변]이 [질문]에 대해 얼마나 "
             "유용한지 1~5점으로 평가하세요. 핵심을 짚고 실질적 도움이 되면 높게, "
             "겉돌거나 회피적이면 낮게 매기세요."),
            ("human", "[질문]\n{question}\n\n[답변]\n{answer}"),
        ])
        self._support_chain = support_prompt | self.llm.with_structured_output(SupportEval)
        self._utility_chain = utility_prompt | self.llm.with_structured_output(UtilityEval)

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
            # 검증 실패 시 통과 — 안전 게이트는 ask()의 키워드 감지가 별도 담당
            self.last_critique = {"error": "critique_failed"}
            return True, answer

        self.last_critique = {
            "issup":              support.issup,
            "isuse":              utility.isuse,
            "support_reasoning":  support.reasoning,
            "utility_reasoning":  utility.reasoning,
        }

        ok = (support.issup != "No support") and (utility.isuse >= self.MIN_UTILITY)

        if not ok and self.LOW_SUPPORT_NOTE not in answer:
            answer = answer + self.LOW_SUPPORT_NOTE

        return ok, answer
