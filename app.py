from __future__ import annotations

import os
import uuid

import pandas as pd
import streamlit as st

if "OPENAI_API_KEY" not in os.environ:
    try:
        if "OPENAI_API_KEY" in st.secrets:
            os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass

from base import (
    Phase1FaissPipeline,
    Phase2MemoryPipeline,
    Phase3HybridPipeline,
    Phase3SelfRAGPipeline,
    RISK_HIGH,
    CRISIS_LINE_SUICIDE,
    CRISIS_LINE_MENTAL,
)

st.set_page_config(page_title="🌱번아웃 예방 및 관리 RAG🌱", page_icon="🌱", layout="centered")

# ── Design tokens ─────────────────────────────────────────────────────────────
C_BG      = "#F8F4EF"
C_CARD    = "#FFFFFF"
C_BLUE    = "#6B9EC7"
C_MINT    = "#7CC8A0"
C_YELLOW  = "#F5C842"
C_CORAL   = "#F4856A"
C_TEXT    = "#2D3436"
C_SUBTEXT = "#8A9BA8"
C_BORDER  = "#E8E2DA"

st.markdown(f"""
<style>
    .stApp {{ background-color: {C_BG} !important; }}

    section[data-testid="stSidebar"] {{
        background: {C_CARD} !important;
        border-right: 1px solid {C_BORDER};
    }}

    .main .block-container {{ max-width: 760px; padding-top: 1.5rem; }}

    [data-testid="stChatMessage"] {{
        border-radius: 16px !important;
        border: 1px solid {C_BORDER} !important;
        background: {C_CARD} !important;
        margin-bottom: 6px !important;
    }}

    [data-testid="stChatInput"] textarea {{
        border-radius: 24px !important;
        background: {C_CARD} !important;
        border: 1.5px solid {C_BORDER} !important;
    }}
    [data-testid="stChatInput"] textarea:focus {{
        border-color: {C_BLUE} !important;
        box-shadow: 0 0 0 2px {C_BLUE}22 !important;
    }}

    .stButton > button {{
        border-radius: 20px !important;
        border: 1.5px solid {C_BORDER} !important;
        background: {C_CARD} !important;
        color: {C_TEXT} !important;
        font-size: 0.85rem !important;
        transition: border-color 0.15s, color 0.15s !important;
        white-space: normal !important;
        height: auto !important;
    }}
    .stButton > button:hover {{
        border-color: {C_BLUE} !important;
        color: {C_BLUE} !important;
    }}

    footer, #MainMenu, header {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)


# ── Constants ─────────────────────────────────────────────────────────────────
PHASE_OPTIONS = {
    "Phase 2 · 메모리(권장)":       Phase2MemoryPipeline,
    "Phase 1 · 기본 FAISS":         Phase1FaissPipeline,
    "Phase 3 · Hybrid(BM25+FAISS)": Phase3HybridPipeline,
    "Phase 3 · Hybrid+Self-RAG":    Phase3SelfRAGPipeline,
}

SUGGESTED_QUESTIONS = [
    "요즘 너무 지쳐있어요. 번아웃인가요?",
    "번아웃을 회복하는 방법이 궁금해요",
    "잠을 잘 못 자고 있어요",
    "스트레스를 줄이는 방법을 알려주세요",
]


@st.cache_resource(show_spinner="잠깐, 문서를 불러오고 있어요 ✨")
def get_pipeline(phase_label: str):
    rag = PHASE_OPTIONS[phase_label]()
    rag.build_chain()
    return rag


# ── Session state ─────────────────────────────────────────────────────────────
if "session_id"      not in st.session_state:
    st.session_state.session_id      = f"user-{uuid.uuid4().hex[:8]}"
if "messages"        not in st.session_state:
    st.session_state.messages        = []
if "checkin"         not in st.session_state:
    st.session_state.checkin         = None
if "phase_label"     not in st.session_state:
    st.session_state.phase_label     = "Phase 2 · 메모리(권장)"
if "pending_message" not in st.session_state:
    st.session_state.pending_message = None


# ── Temperature gauge ─────────────────────────────────────────────────────────
def _temp_gauge(temp: float, thr_high: float, thr_mid: float) -> str:
    if temp < thr_high:
        color, label, desc = C_CORAL,  "번아웃 위험", "많이 힘드셨겠어요. 혼자 견디지 않아도 괜찮아요."
    elif temp < thr_mid:
        color, label, desc = C_YELLOW, "피로 주의",   "조금 지쳐 보여요. 작은 회복부터 찾아봐요."
    else:
        color, label, desc = C_MINT,   "안정",        "비교적 안정적이에요. 지금 리듬을 잘 지켜봐요."

    # Clamp marker position so it never clips the edge
    pct = min(max(temp, 3), 97)

    return f"""
    <div style="padding:0.5rem 0;">
        <div style="text-align:center;margin-bottom:0.75rem;">
            <span style="font-size:2.4rem;font-weight:700;color:{color};line-height:1;">{temp}</span>
            <span style="font-size:0.9rem;color:{C_SUBTEXT};">&thinsp;/ 100</span>
            <span style="background:{color}25;color:{color};font-size:0.72rem;font-weight:600;
                         padding:2px 9px;border-radius:10px;margin-left:6px;vertical-align:middle;">
                {label}
            </span>
        </div>
        <div style="position:relative;height:10px;border-radius:5px;
                    background:linear-gradient(to right,
                        {C_CORAL} 0%, {C_YELLOW} 35%, {C_MINT} 60%, {C_BLUE} 100%);">
            <div style="position:absolute;left:calc({pct}% - 10px);top:-5px;
                        width:20px;height:20px;border-radius:50%;
                        background:white;border:3px solid {color};
                        box-shadow:0 2px 6px rgba(0,0,0,0.15);"></div>
        </div>
        <p style="font-size:0.77rem;color:{C_SUBTEXT};text-align:center;
                  margin:0.55rem 0 0;line-height:1.5;">{desc}</p>
    </div>
    """


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f"<h2 style='font-size:1.25rem;color:{C_TEXT};font-weight:700;margin:0;'>🌱 마음 회복</h2>"
        f"<p style='color:{C_SUBTEXT};font-size:0.78rem;margin:2px 0 1rem;'>AI 심리 회복 도우미</p>",
        unsafe_allow_html=True,
    )

    with st.expander("⚙️ 검색 엔진", expanded=False):
        phase_label = st.selectbox(
            "파이프라인",
            list(PHASE_OPTIONS.keys()),
            index=list(PHASE_OPTIONS.keys()).index(st.session_state.phase_label),
            label_visibility="collapsed",
            help="Phase 3는 kiwipiepy·rank_bm25 설치 필요",
        )
        if phase_label != st.session_state.phase_label:
            st.session_state.phase_label  = phase_label
            st.session_state.messages     = []
            st.session_state.session_id   = f"user-{uuid.uuid4().hex[:8]}"
            st.rerun()

    st.markdown(
        f"<p style='font-size:0.88rem;font-weight:600;color:{C_TEXT};margin:0.75rem 0 0.2rem;'>"
        f"🌡️ 오늘의 마음 체크인</p>"
        f"<p style='font-size:0.76rem;color:{C_SUBTEXT};margin-bottom:0.5rem;'>"
        f"진단이 아니라, 요즘 내 상태를 돌아보는 용도예요.</p>",
        unsafe_allow_html=True,
    )

    use_checkin = st.toggle("체크인 반영하기", value=True)

    mood     = st.slider("기분",     1, 5, 3, help="1=가라앉음 · 5=좋음")
    energy   = st.slider("에너지",   1, 5, 3, help="1=바닥남 · 5=활기참")
    sleep    = st.slider("수면",     1, 5, 3, help="1=거의 못 잠 · 5=푹 잠")
    recovery = st.slider("회복감",   1, 5, 3, help="1=회복 안 됨 · 5=잘 회복됨")
    stress   = st.slider("스트레스", 1, 5, 3, help="1=거의 없음 · 5=아주 큼")
    fatigue  = st.slider("피로감",   1, 5, 3, help="1=거의 없음 · 5=아주 큼")

    checkin = {
        "mood": mood, "energy": energy, "sleep": sleep,
        "recovery": recovery, "stress": stress, "fatigue": fatigue,
    }
    st.session_state.checkin = checkin if use_checkin else None


rag = get_pipeline(st.session_state.phase_label)


with st.sidebar:
    if use_checkin:
        temp = rag.compute_mind_temperature(checkin)
        st.markdown(_temp_gauge(temp, rag.THRESHOLD_HIGH, rag.THRESHOLD_MID), unsafe_allow_html=True)

        history = rag.get_temperature_history(st.session_state.session_id)
        if len(history) > 1:
            st.markdown(
                f"<p style='font-size:0.8rem;color:{C_SUBTEXT};margin:0.5rem 0 0.25rem;'>📈 마음 온도 추이</p>",
                unsafe_allow_html=True,
            )
            df = pd.DataFrame(history).set_index("date")
            st.area_chart(df["temperature"], height=120, color=C_BLUE)

    st.divider()
    if st.button("🗑️ 대화 초기화", use_container_width=True):
        st.session_state.messages   = []
        st.session_state.session_id = f"user-{uuid.uuid4().hex[:8]}"
        st.rerun()


# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown(
    f"<h1 style='font-size:1.6rem;font-weight:700;color:{C_TEXT};margin-bottom:0;'>🌱 번아웃 예방 및 관리 도우미🌱</h1>"
    f"<p style='color:{C_SUBTEXT};font-size:0.82rem;margin:4px 0 1.25rem;'>"
    f"번아웃·수면·스트레스 회복을 돕는 AI 상담 도우미예요. "
    f"의료 행위가 아니며, 위급할 때는 전문기관에 연락해 주세요.</p>",
    unsafe_allow_html=True,
)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Suggested questions — only shown before the first message
if not st.session_state.messages:
    st.markdown(
        f"<p style='font-size:0.82rem;color:{C_SUBTEXT};margin-bottom:0.4rem;'>자주 묻는 질문</p>",
        unsafe_allow_html=True,
    )
    cols = st.columns(2)
    for i, q in enumerate(SUGGESTED_QUESTIONS):
        if cols[i % 2].button(q, key=f"sq_{i}", use_container_width=True):
            st.session_state.pending_message = q
            st.rerun()

# Resolve prompt: typed or suggested-question click
prompt = st.chat_input("요즘 어떤 점이 힘드세요?")
if st.session_state.pending_message:
    prompt                           = st.session_state.pending_message
    st.session_state.pending_message = None

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner(""):
            result = rag.ask(
                prompt,
                session_id=st.session_state.session_id,
                checkin=st.session_state.checkin,
            )
        st.markdown(result["answer"])

        if result["risk_level"] == RISK_HIGH:
            st.markdown(
                f"""<div style="background:#FFF0EE;border-left:3px solid {C_CORAL};
                               border-radius:0 8px 8px 0;padding:0.7rem 1rem;
                               margin-top:0.75rem;font-size:0.83rem;
                               color:{C_TEXT};line-height:1.8;">
                    <strong>힘든 마음이 클 때는 전문가의 도움이 큰 힘이 돼요.</strong><br>
                    📞 자살예방 상담전화 <strong>{CRISIS_LINE_SUICIDE}</strong>&nbsp;(24시간)<br>
                    📞 정신건강 위기상담전화 <strong>{CRISIS_LINE_MENTAL}</strong>&nbsp;(24시간)<br>
                    🏫 학교 상담센터
                </div>""",
                unsafe_allow_html=True,
            )

    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
    st.rerun()
