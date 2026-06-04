"""base.py — 마음 회복 RAG 파이프라인 (단일 통합형)

설계 변경 요약
  - Phase 1/2/3 분리를 폐기하고, 강의에서 배운 모든 기법을 하나의 파이프라인에 통합한다.
      · 인덱싱 : DirectoryLoader + PyPDFLoader → 카테고리별 RecursiveCharacterTextSplitter → FAISS 저장
      · 검색   : BM25(Kiwi 형태소) + FAISS 앙상블(Hybrid Search)
      · 메모리 : RunnableWithMessageHistory (세션별 대화 누적)
      · 검증   : Self-RAG (Support / Utility 이중 평가)
  - 마음 온도는 "온보딩 설문값"으로 1회 초기화되고, 이후에는 사용자가 직접 조절하지 못한다.
    대신 매 대화마다 **별도 분류 체인(Mood Delta Chain)** 이 대화 내용을 분석해 온도를 자동 조정한다.
  - 고위험 안전 키워드는 온도·검증과 무관하게 즉시 안전 안내로 전환되는 하드 게이트다.

⚠️ 본 서비스는 의료 행위가 아니며, 마음 온도는 톤 선택용 휴리스틱일 뿐 의학적 진단이 아니다.
"""

from __future__ import annotations

import json
import os
import warnings
from datetime import date
from typing import Literal, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# FAISS, DirectoryLoader, PyPDFLoader, BM25Retriever 은 아직 standalone 패키지가 없어 community 유지
warnings.filterwarnings("ignore", message=".*langchain-community.*is being sunset.*", category=DeprecationWarning)
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader  # noqa: E402
from langchain_community.vectorstores import FAISS  # noqa: E402
from langchain_community.retrievers import BM25Retriever  # noqa: E402
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.chat_history import InMemoryChatMessageHistory

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser


# ===========================================================================
# 상수
# ===========================================================================
RISK_HIGH = "high"
RISK_MID  = "mid"
RISK_LOW  = "low"

# 자살예방 상담전화는 109로 통합됨 (2024.1~)
CRISIS_LINE_SUICIDE: str = "109"
CRISIS_LINE_MENTAL:  str = "1577-0199"

SAFETY_KEYWORDS_HIGH: tuple[str, ...] = (
    "죽고 싶", "사라지고 싶", "없어지고 싶", "더는 못 버티", "버티기 싫",
    "끝내고 싶", "자살", "스스로 목숨", "극단적 선택", "생을 마감", "살기 싫", "자해",
)

SAFETY_KEYWORDS_MID: tuple[str, ...] = (
    "무기력", "절망", "아무 의미 없", "다 포기하고 싶",
    "너무 힘들어서 모르겠", "그냥 다 그만두고 싶",
)

# 카테고리별 청킹 — safety는 작게(위험 키워드 경계 분리 방지), 나머지는 맥락 유지
CHUNK_CONFIG: dict[str, dict] = {
    "safety":       {"chunk_size": 400, "chunk_overlap": 80},
    "burnout":      {"chunk_size": 600, "chunk_overlap": 100},
    "recovery":     {"chunk_size": 600, "chunk_overlap": 100},
    "sleep_stress": {"chunk_size": 500, "chunk_overlap": 80},
}
RISK_PRIORITY: dict[str, int] = {"safety": 1, "burnout": 2, "recovery": 3, "sleep_stress": 4}


# ===========================================================================
# Self-RAG 검증 스키마
# ===========================================================================
class SupportEval(BaseModel):
    reasoning: str = Field(description="판단 근거를 한국어로 간단히")
    issup: Literal["Fully supported", "Partially supported", "No support"] = Field(
        description="문맥에 의한 답변 지원 정도"
    )


class UtilityEval(BaseModel):
    reasoning: str = Field(description="판단 근거를 한국어로 간단히")
    isuse: int = Field(description="1(최저)~5(최고) 유용성 점수", ge=1, le=5)


# ===========================================================================
# 대화 기반 온도 조절 스키마 (★ 신설된 별도 분류 체인)
# ===========================================================================
class MoodDeltaEval(BaseModel):
    """사용자의 최근 발화가 마음 상태를 얼마나 변화시키는지 분류한다."""
    reasoning: str = Field(description="판단 근거를 한국어 한 문장으로")
    direction: Literal["improving", "worsening", "stable"] = Field(
        description="대화 흐름상 사용자의 상태 방향"
    )
    delta: int = Field(
        description="마음 온도 변화량. 호전 +, 악화 -, 변화 없음 0. -8~+8 범위.",
        ge=-8, le=8,
    )


# ===========================================================================
# 마음 회복 RAG 파이프라인 (통합 단일 클래스)
# ===========================================================================
class MindCareRAGPipeline:
    # 경로
    DATA_DIR:  str = "./data"
    FAISS_DIR: str = "./faiss_db"
    ENV_PATH:  str = "./data/.env"
    TEMP_HISTORY_PATH: str = "./data/temperature_history.json"

    # 모델
    EMBEDDING_MODEL: str   = "text-embedding-3-large"
    LLM_MODEL:       str   = "gpt-4o-mini"
    TEMPERATURE:     float = 0.2

    # 검색
    TOP_K:        int   = 4
    BM25_WEIGHT:  float = 0.5
    FAISS_WEIGHT: float = 0.5

    # 메모리
    MAX_HISTORY_TURNS: int = 4

    # 위험도/온도 (0~100, 낮을수록 지쳐 있음) — 의학적 진단 아님, 톤 선택용
    THRESHOLD_HIGH: float = 35.0
    THRESHOLD_MID:  float = 60.0
    DEFAULT_TEMP:   float = 50.0
    HIGH_KEYWORD_DROP: float = 15.0   # 고위험 키워드 감지 시 온도 급락폭
    HIGH_KEYWORD_FLOOR: float = 20.0  # 급락 후 상한(이 값 이하로 유지)
    HIGH_RISK_STREAK_THRESHOLD: int = 2

    # Self-RAG
    MIN_UTILITY: int = 3
    LOW_SUPPORT_NOTE = (
        "\n\n참고: 위 내용은 제공된 자료로 충분히 뒷받침되지 않을 수 있어요. "
        "중요한 결정이나 증상 판단은 전문가와 상의해 주세요."
    )

    CATEGORY_DIRS: dict[str, str] = {
        "safety":       "P0_safety",
        "burnout":      "P1_burnout",
        "recovery":     "P2_recovery",
        "sleep_stress": "P3_sleep_stress",
    }

    # -------------------------------------------------------------------
    def __init__(self, **overrides) -> None:
        for key, value in overrides.items():
            if hasattr(self, key):
                setattr(self, key, value)

        load_dotenv(".env")
        load_dotenv(self.ENV_PATH)

        self.embeddings = OpenAIEmbeddings(model=self.EMBEDDING_MODEL)
        self.llm        = ChatOpenAI(model=self.LLM_MODEL, temperature=self.TEMPERATURE)

        self.vectorstore: Optional[FAISS]    = None
        self.retriever:   Optional[Runnable] = None
        self._chain:      Optional[Runnable] = None
        self._splits:     Optional[list[Document]] = None

        self._histories:     dict[str, InMemoryChatMessageHistory] = {}
        self._current_temp:  dict[str, float]              = {}   # 세션별 "현재" 마음 온도(러닝)
        self._high_streak:   dict[str, int]                = {}
        self._temp_history:  dict[str, list[dict]]         = {}
        self.last_critique:  Optional[dict]                = None

        # 검증/온도 체인은 최초 사용 시 지연 생성
        self._support_chain: Optional[Runnable] = None
        self._utility_chain: Optional[Runnable] = None
        self._mood_chain:    Optional[Runnable] = None

        self._load_temp_history()

    # ===================================================================
    # 마음 온도 영속화 + 러닝 상태
    # ===================================================================
    def _load_temp_history(self) -> None:
        if os.path.exists(self.TEMP_HISTORY_PATH):
            try:
                with open(self.TEMP_HISTORY_PATH, "r", encoding="utf-8") as f:
                    self._temp_history = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._temp_history = {}

    def _save_temp_history(self) -> None:
        directory = os.path.dirname(self.TEMP_HISTORY_PATH)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.TEMP_HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(self._temp_history, f, ensure_ascii=False, indent=2)

    def record_temperature(self, session_id: str, temperature: float) -> None:
        """같은 날 재호출하면 마지막 값으로 덮어쓴다(추이 그래프용)."""
        today   = date.today().isoformat()
        records = self._temp_history.setdefault(session_id, [])
        if records and records[-1]["date"] == today:
            records[-1]["temperature"] = temperature
        else:
            records.append({"date": today, "temperature": temperature})
        self._save_temp_history()

    def get_temperature_history(self, session_id: str) -> list[dict]:
        return self._temp_history.get(session_id, [])

    @staticmethod
    def compute_mind_temperature(checkin: dict[str, int]) -> float:
        """온보딩 설문 점수(각 1~5)로 초기 마음 온도(0~100)를 계산한다."""
        def g(k: str) -> float:
            return float(checkin.get(k, 3))

        positive = g("sleep") * 0.20 + g("energy") * 0.15 + g("recovery") * 0.15 + g("mood") * 0.10
        negative = g("stress") * 0.20 + g("fatigue") * 0.20
        index    = (positive - negative + 1.4) / 4.0 * 100
        return round(max(0.0, min(100.0, index)), 1)

    def set_initial_temperature(self, session_id: str, checkin: dict[str, int]) -> float:
        """온보딩 설문 완료 시 1회 호출 — 이후 온도는 대화로만 변동된다."""
        temp = self.compute_mind_temperature(checkin)
        self._current_temp[session_id] = temp
        self.record_temperature(session_id, temp)
        return temp

    def get_current_temperature(self, session_id: str) -> float:
        return self._current_temp.get(session_id, self.DEFAULT_TEMP)

    def _adjust_temperature(self, session_id: str, delta: float, floor_high: bool = False) -> float:
        cur = self.get_current_temperature(session_id)
        new = cur + delta
        if floor_high:
            new = min(new, self.HIGH_KEYWORD_FLOOR)
        new = round(max(0.0, min(100.0, new)), 1)
        self._current_temp[session_id] = new
        self.record_temperature(session_id, new)
        return new

    # ===================================================================
    # 인덱싱
    # ===================================================================
    def load_documents(self) -> list[Document]:
        all_docs: list[Document] = []
        for category, folder in self.CATEGORY_DIRS.items():
            path = os.path.join(self.DATA_DIR, folder)
            if not os.path.isdir(path):
                continue
            loader = DirectoryLoader(path, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=False)
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
            buckets.setdefault(doc.metadata.get("category", "burnout"), []).append(doc)

        splits: list[Document] = []
        for cat, cat_docs in buckets.items():
            cfg = CHUNK_CONFIG.get(cat, {"chunk_size": 500, "chunk_overlap": 80})
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=cfg["chunk_size"], chunk_overlap=cfg["chunk_overlap"],
                separators=["\n\n", "\n", "。", ". ", " ", ""],
            )
            cat_splits = splitter.split_documents(cat_docs)
            for s in cat_splits:
                s.metadata.setdefault("category", cat)
                s.metadata.setdefault("risk_priority", RISK_PRIORITY.get(cat, 9))
            splits.extend(cat_splits)
        return splits

    def build_vectorstore(self, rebuild: bool = False) -> FAISS:
        index_exists = os.path.exists(os.path.join(self.FAISS_DIR, "index.faiss"))
        if index_exists and not rebuild:
            self.vectorstore = FAISS.load_local(
                self.FAISS_DIR, embeddings=self.embeddings,
                allow_dangerous_deserialization=True,
            )
        else:
            docs = self.load_documents()
            if not docs:
                raise FileNotFoundError(f"{self.DATA_DIR} 의 카테고리 폴더에서 PDF를 찾지 못했습니다.")
            splits       = self.split_documents(docs)
            self._splits = splits
            self.vectorstore = FAISS.from_documents(splits, self.embeddings)
            self.vectorstore.save_local(self.FAISS_DIR)
        return self.vectorstore

    def _get_corpus_documents(self) -> list[Document]:
        """BM25 인덱싱용 split 문서. FAISS는 디스크 영속화되지만 BM25는 매 실행 메모리 재구성."""
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
            raise FileNotFoundError(f"{self.DATA_DIR} 에서 BM25 코퍼스를 만들 PDF를 찾지 못했습니다.")
        self._splits = self.split_documents(raw)
        return self._splits

    # ===================================================================
    # 검색 — BM25(Kiwi) + FAISS 앙상블 (Hybrid)
    # ===================================================================
    def build_retriever(self, vectorstore: FAISS) -> Runnable:
        faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": self.TOP_K})

        try:
            EnsembleRetriever = _import_ensemble_retriever()
            corpus = self._get_corpus_documents()
            bm25 = BM25Retriever.from_documents(corpus, preprocess_func=make_kiwi_tokenizer())
            bm25.k = self.TOP_K

            return EnsembleRetriever(
                retrievers=[bm25, faiss_retriever],
                weights=[self.BM25_WEIGHT, self.FAISS_WEIGHT],
            )
        except Exception as e:
            # rank_bm25, kiwipiepy, langchain-classic 등이 빠진 환경에서도 앱이 완전히 죽지 않도록
            # FAISS 단독 검색으로 폴백합니다. 발표/가점용 앙상블을 쓰려면 requirements를 설치하세요.
            warnings.warn(f"BM25 앙상블 검색을 만들지 못해 FAISS 검색만 사용합니다: {e}")
            return faiss_retriever

    # ===================================================================
    # 위험도 분류 — 키워드 게이트 + 러닝 온도 + 연속 누적
    # ===================================================================
    @staticmethod
    def _detect_safety_keyword(text: str) -> Optional[str]:
        for kw in SAFETY_KEYWORDS_HIGH:
            if kw in text:
                return RISK_HIGH
        for kw in SAFETY_KEYWORDS_MID:
            if kw in text:
                return RISK_MID
        return None

    def classify_risk(self, query: str, session_id: str = "default") -> str:
        """우선순위: 1) 안전 키워드  2) 현재 러닝 마음 온도  3) 연속 고위험 누적."""
        keyword_risk = self._detect_safety_keyword(query)

        if keyword_risk == RISK_HIGH:
            base_risk = RISK_HIGH
        else:
            temp = self.get_current_temperature(session_id)
            if temp < self.THRESHOLD_HIGH:
                base_risk = RISK_HIGH
            elif temp < self.THRESHOLD_MID:
                base_risk = RISK_MID
            else:
                base_risk = RISK_LOW
            if keyword_risk == RISK_MID and base_risk == RISK_LOW:
                base_risk = RISK_MID   # 중위험 키워드는 최소 mid 보장

        # 연속 고위험 누적 — 한 번 high가 2회 이상 누적되면 잠시 high 유지
        streak = self._high_streak.get(session_id, 0)
        if base_risk == RISK_HIGH:
            self._high_streak[session_id] = streak + 1
        else:
            self._high_streak[session_id] = max(0, streak - 1)
        if self._high_streak[session_id] >= self.HIGH_RISK_STREAK_THRESHOLD:
            return RISK_HIGH
        return base_risk

    # ===================================================================
    # 프롬프트
    # ===================================================================
    BASE_SYSTEM = (
        "당신은 번아웃과 정서 회복을 돕는 따뜻한 상담 도우미입니다. "
        "아래 [참고 문서]에 근거해서만 답하고, 문서에 없으면 모른다고 솔직히 말하세요. "
        "진단·단정 표현은 피하고, 사용자를 평가하지 마세요. "
        "의료적 판단이 필요하면 전문가 상담을 권유하세요.\n{risk_directive}"
    )

    RISK_DIRECTIVES: dict[str, str] = {
        RISK_HIGH: (
            "[톤 지침] 사용자가 많이 지쳐 있을 수 있습니다. '고위험' 같은 진단 표현 대신 "
            "'최근 많이 힘드셨겠어요' 처럼 부드럽게 공감하세요. "
            f"반드시 학교 상담센터·자살예방 상담전화({CRISIS_LINE_SUICIDE})·"
            f"정신건강 위기상담전화({CRISIS_LINE_MENTAL}) 연계를 자연스럽게 안내하고, "
            "혼자 감당하지 않아도 됨을 전달하세요. 삶의 이유(가족, 친구, 미래 목표 등)를 함께 확인하는 질문을 건네세요."
        ),
        RISK_MID: (
            "[톤 지침] '최근 조금 지쳐 보여요' 정도의 부담 없는 표현을 쓰세요. "
            "작게 실천할 수 있는 회복 행동과 학업·업무 조절을 한두 가지 제안하세요. "
            "신뢰할 수 있는 주변 사람이나 상담센터 방문을 부드럽게 권유하세요."
        ),
        RISK_LOW: (
            "[톤 지침] 안정적인 상태로 보입니다. 예방 관점의 가벼운 가이드와 긍정적 강화를 중심으로 답하세요."
        ),
    }

    def build_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", self.BASE_SYSTEM),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "[참고 문서]\n{context}\n\n[질문]\n{input}"),
        ])

    SAFETY_FALLBACK = (
        f"지금 많이 힘드시다면 혼자 견디지 않으셔도 괜찮아요. "
        f"학교 상담센터나 자살예방 상담전화({CRISIS_LINE_SUICIDE})에 연락해 보시길 권해요. "
        f"24시간 운영되며, 정신건강 위기상담전화({CRISIS_LINE_MENTAL})도 이용할 수 있어요. "
        f"제가 도울 수 있는 부분이 있다면 더 이야기해 주세요."
    )

    # ===================================================================
    # 체인 조립 (메모리 포함)
    # ===================================================================
    @staticmethod
    def _format_docs(docs: list[Document]) -> str:
        return "\n\n".join(d.page_content for d in docs)

    def _get_session_history(self, session_id: str) -> InMemoryChatMessageHistory:
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

        core_chain = (
            RunnablePassthrough.assign(
                context=lambda x: self._format_docs(self.retriever.invoke(x["input"])),
            )
            | self.build_prompt()
            | self.llm
            | StrOutputParser()
        )

        self._chain = RunnableWithMessageHistory(
            core_chain,
            self._get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )
        return self._chain

    # ===================================================================
    # Self-RAG 검증 체인
    # ===================================================================
    def _build_critique_chains(self) -> None:
        support_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "당신은 RAG 답변 검증자입니다. 주어진 [답변]이 [문맥]에 의해 사실적으로 "
             "뒷받침되는 정도를 평가하세요. 문맥에 없는 내용을 답변이 주장하면 지원 정도를 낮게 매기세요."),
            ("human", "[질문]\n{question}\n\n[답변]\n{answer}\n\n[문맥]\n{context}"),
        ])
        utility_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "당신은 RAG 답변 평가자입니다. [답변]이 [질문]에 대해 얼마나 유용한지 1~5점으로 "
             "평가하세요. 핵심을 짚고 실질적 도움이 되면 높게, 겉돌거나 회피적이면 낮게 매기세요."),
            ("human", "[질문]\n{question}\n\n[답변]\n{answer}"),
        ])
        self._support_chain = support_prompt | self.llm.with_structured_output(SupportEval)
        self._utility_chain = utility_prompt | self.llm.with_structured_output(UtilityEval)

    def verify_answer(self, question: str, answer: str, context: str) -> tuple[bool, str]:
        if self._support_chain is None or self._utility_chain is None:
            self._build_critique_chains()
        try:
            support: SupportEval = self._support_chain.invoke(
                {"question": question, "answer": answer, "context": context})
            utility: UtilityEval = self._utility_chain.invoke(
                {"question": question, "answer": answer})
        except Exception:
            self.last_critique = {"error": "critique_failed"}
            return True, answer

        self.last_critique = {
            "issup": support.issup, "isuse": utility.isuse,
            "support_reasoning": support.reasoning, "utility_reasoning": utility.reasoning,
        }
        ok = (support.issup != "No support") and (utility.isuse >= self.MIN_UTILITY)
        if not ok and self.LOW_SUPPORT_NOTE not in answer:
            answer = answer + self.LOW_SUPPORT_NOTE
        return ok, answer

    # ===================================================================
    # ★ 대화 기반 마음 온도 분류 체인 (신설)
    #   사용자의 최근 발화를 분석해 마음 온도 변화량(delta)을 산출한다.
    #   온도는 설문값으로 초기화된 뒤, 이 체인의 출력으로만 움직인다.
    # ===================================================================
    def _build_mood_chain(self) -> None:
        mood_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "당신은 번아웃 상담 대화의 정서 변화 분석가입니다. [최근 대화]와 [사용자 발화]를 보고 "
             "사용자의 마음 상태가 직전 대비 호전됐는지/악화됐는지/변화 없는지 판단하세요.\n"
             "- 휴식·수면 개선, 희망·의욕 회복, 작은 실천 성공, 긍정 정서 → improving(+)\n"
             "- 피로·무기력 심화, 수면 악화, 절망·압박감, 부정 정서 심화 → worsening(-)\n"
             "- 단순 정보 질문이나 중립적 발화 → stable(0)\n"
             "delta 는 변화 강도에 비례해 -8~+8 정수로 정하세요. 과민 반응을 피하고 보수적으로 매기세요."),
            ("human", "[최근 대화]\n{history}\n\n[사용자 발화]\n{message}"),
        ])
        self._mood_chain = mood_prompt | self.llm.with_structured_output(MoodDeltaEval)

    def assess_mood_delta(self, session_id: str, message: str) -> MoodDeltaEval:
        if self._mood_chain is None:
            self._build_mood_chain()
        history = self._get_session_history(session_id)
        recent  = "\n".join(f"{m.type}: {m.content}" for m in history.messages[-4:]) or "(대화 없음)"
        try:
            return self._mood_chain.invoke({"history": recent, "message": message})
        except Exception:
            # 분류 실패 시 온도 변화 없음으로 폴백 (안전)
            return MoodDeltaEval(reasoning="분류 실패 — 변화 없음 처리", direction="stable", delta=0)

    # ===================================================================
    # 공개 진입점
    # ===================================================================
    def ask(self, question: str, session_id: str = "default") -> dict:
        """온도는 내부 러닝 상태로 관리한다. (설문값 초기화 후 대화로만 변동)"""
        if self._chain is None:
            self.build_chain()

        # 1) 고위험 키워드 하드 게이트 — 온도 급락 + 즉시 안전 안내
        if self._detect_safety_keyword(question) == RISK_HIGH:
            self._high_streak[session_id] = self._high_streak.get(session_id, 0) + 1
            new_temp = self._adjust_temperature(session_id, -self.HIGH_KEYWORD_DROP, floor_high=True)
            return {
                "answer":           self.SAFETY_FALLBACK,
                "risk_level":       RISK_HIGH,
                "mind_temperature": new_temp,
                "temp_delta":       None,
                "delta_reason":     "고위험 표현 감지 — 안전 게이트 작동",
                "safety_triggered": True,
                "critique":         None,
            }

        # 2) 위험군 판정(현재 온도 기준) → 톤 분기
        risk_level     = self.classify_risk(question, session_id)
        risk_directive = self.RISK_DIRECTIVES.get(risk_level, self.RISK_DIRECTIVES[RISK_LOW])

        # 3) RAG 답변 생성 (메모리 자동 누적)
        answer = self._chain.invoke(
            {"input": question, "risk_directive": risk_directive},
            config={"configurable": {"session_id": session_id}},
        )

        # 4) Self-RAG 자체 검증
        context_docs = self.retriever.invoke(question)
        ok, answer   = self.verify_answer(question, answer, self._format_docs(context_docs))

        # 5) ★ 대화 기반 온도 조절 (별도 분류 체인)
        mood = self.assess_mood_delta(session_id, question)
        new_temp = self._adjust_temperature(session_id, float(mood.delta))

        # 6) 온도 변동으로 위험군이 바뀌었으면 재판정(연속 누적 갱신 없이 조회만)
        if new_temp < self.THRESHOLD_HIGH:
            risk_level = RISK_HIGH

        return {
            "answer":           answer,
            "risk_level":       risk_level,
            "mind_temperature": new_temp,
            "temp_delta":       mood.delta,
            "delta_reason":     mood.reasoning,
            "safety_triggered": False,
            "critique":         self.last_critique,
        }


# ===========================================================================
# Kiwi 형태소 토크나이저 (BM25 한국어 전처리) — 미설치 시 공백 분리 폴백
# ===========================================================================
_KIWI_INSTANCE = None


def make_kiwi_tokenizer():
    global _KIWI_INSTANCE
    try:
        from kiwipiepy import Kiwi
        if _KIWI_INSTANCE is None:
            _KIWI_INSTANCE = Kiwi()
        kiwi = _KIWI_INSTANCE
        return lambda text: [t.form for t in kiwi.tokenize(text)]
    except ImportError:
        return str.split


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
