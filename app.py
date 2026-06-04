"""app.py — 번아웃 예방 및 관리 RAG 챗봇 UI 개선본

핵심 흐름
  1) 온보딩 설문 6문항으로 초기 마음 온도 계산
  2) 채팅 단계에서 RAG 답변 생성
  3) 마음 온도, 위험도, 체크인 점수를 대시보드 형태로 표시
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

st.set_page_config(
    page_title="마음온도 RAG",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# 환경 변수 로딩
# -----------------------------------------------------------------------------
load_dotenv(".env")
load_dotenv("./data/.env")

if "OPENAI_API_KEY" not in os.environ:
    try:
        if "OPENAI_API_KEY" in st.secrets:
            os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass

if not os.environ.get("OPENAI_API_KEY"):
    st.error("🔑 OPENAI_API_KEY가 설정되지 않았습니다.")
    st.info(
        "로컬 실행: 프로젝트 폴더의 `.env` 또는 `data/.env`에 `OPENAI_API_KEY=키값`을 넣어주세요.\n\n"
        "Streamlit Cloud: Settings → Secrets에 `OPENAI_API_KEY`를 추가하세요."
    )
    st.stop()

from base import (  # noqa: E402
    CRISIS_LINE_MENTAL,
    CRISIS_LINE_SUICIDE,
    RISK_HIGH,
    RISK_MID,
    MindCareRAGPipeline,
)

# -----------------------------------------------------------------------------
# 세션 상태
# -----------------------------------------------------------------------------
def init_state() -> None:
    defaults: dict[str, Any] = {
        "session_id": f"user-{uuid.uuid4().hex[:8]}",
        "stage": "onboarding",
        "survey_step": 0,
        "answers": {},
        "messages": [],
        "mind_temp": MindCareRAGPipeline.DEFAULT_TEMP,
        "risk_level": "low",
        "pending_message": None,
        "show_debug": False,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

    if "theme" not in st.session_state:
        param = st.query_params.get("theme", "light")
        if isinstance(param, list):
            param = param[0] if param else "light"
        st.session_state.theme = param if param in ("light", "dark") else "light"


init_state()

IS_DARK = st.session_state.theme == "dark"

# -----------------------------------------------------------------------------
# 디자인 토큰
# -----------------------------------------------------------------------------
if IS_DARK:
    BG = "#141714"
    BG_SOFT = "#1D211D"
    CARD = "#222820"
    CARD_2 = "#1B201B"
    TEXT = "#F0F3EC"
    MUTED = "#A7B0A3"
    BORDER = "rgba(255,255,255,.11)"
    INPUT = "#202620"
    SHADOW = "0 18px 60px rgba(0,0,0,.32)"
    GRADIENT = "radial-gradient(circle at top left, rgba(126,200,160,.20), transparent 35%), linear-gradient(135deg, #141714 0%, #20271E 100%)"
    BOTTOM_BG = "rgba(20,23,20,.94)"
    BOTTOM_CARD = "#1B201B"
else:
    BG = "#F7F4EC"
    BG_SOFT = "#EFE9DC"
    CARD = "#FFFFFF"
    CARD_2 = "#FBF8F1"
    TEXT = "#2F332D"
    MUTED = "#7F887A"
    BORDER = "rgba(65,78,55,.13)"
    INPUT = "#FFFFFF"
    SHADOW = "0 18px 55px rgba(89,94,72,.14)"
    GRADIENT = "radial-gradient(circle at top left, rgba(126,200,160,.30), transparent 33%), linear-gradient(135deg, #F7F4EC 0%, #ECF4E9 100%)"
    BOTTOM_BG = "rgba(247,244,236,.82)"
    BOTTOM_CARD = "#FFFFFF"

PRIMARY = "#70A77B"
MINT = "#7CC8A0"
YELLOW = "#F3C54B"
CORAL = "#E68070"
BLUE = "#79A7D8"


def color_for_temp(temp: float) -> tuple[str, str, str]:
    if temp >= MindCareRAGPipeline.THRESHOLD_MID:
        return MINT, "안정", "☀️"
    if temp >= MindCareRAGPipeline.THRESHOLD_HIGH:
        return YELLOW, "주의", "🌤️"
    return CORAL, "위험", "🌧️"


def risk_label(risk: str) -> tuple[str, str]:
    if risk == RISK_HIGH:
        return CORAL, "많이 지쳐 보여요"
    if risk == RISK_MID:
        return YELLOW, "조금 지쳐 보여요"
    return PRIMARY, "안정적이에요"


# -----------------------------------------------------------------------------
# CSS
# -----------------------------------------------------------------------------
# 주의: 아래 CSS는 반드시 <style>...</style> 안에 있어야 화면에 코드가 노출되지 않습니다.
st.markdown(
    f"""
<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css');
* {{ box-sizing: border-box; }}
html, body, .stApp {{ background: {GRADIENT} !important; color: {TEXT} !important; font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important; }}
body, p, div, span, button, input, textarea, label, h1, h2, h3, h4, h5, h6, li {{ font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important; }}
/* Streamlit 채팅 아바타는 Material Symbols 폰트의 ligature를 쓰기 때문에, 전역 폰트가 덮어쓰면 smart_toy 같은 글자가 노출됨 */
.material-icons, .material-icons-outlined, .material-icons-round, .material-icons-sharp,
.material-symbols-outlined, .material-symbols-rounded, .material-symbols-sharp,
[data-testid="stChatMessageAvatarAssistant"] span,
[data-testid="stChatMessageAvatarUser"] span,
[data-testid="chatAvatarIcon-assistant"],
[data-testid="chatAvatarIcon-user"] {{
    font-family: 'Material Symbols Rounded', 'Material Icons' !important;
    font-weight: normal !important;
    font-style: normal !important;
    line-height: 1 !important;
    text-transform: none !important;
    letter-spacing: normal !important;
    white-space: nowrap !important;
    word-wrap: normal !important;
    direction: ltr !important;
    -webkit-font-feature-settings: 'liga' !important;
    -webkit-font-smoothing: antialiased !important;
}}
[data-testid="stAppViewContainer"] {{ background: transparent !important; }}
.main .block-container {{ max-width: 1180px !important; padding-top: 1.5rem !important; padding-bottom: 6.5rem !important; }}

h1, h2, h3, h4, h5, h6, p, li, span, label {{ color: {TEXT} !important; }}
[data-testid="stCaptionContainer"] p, .muted {{ color: {MUTED} !important; }}

/* sidebar */
section[data-testid="stSidebar"] {{ background: {CARD_2} !important; border-right: 1px solid {BORDER} !important; }}
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{ gap: .55rem !important; }}

/* buttons */
.stButton > button {{
    border-radius: 16px !important;
    border: 1px solid {BORDER} !important;
    background: {CARD} !important;
    color: {TEXT} !important;
    min-height: 42px !important;
    font-weight: 650 !important;
    box-shadow: 0 8px 22px rgba(0,0,0,.04) !important;
    transition: all .18s ease !important;
}}
.stButton > button:hover {{
    border-color: {PRIMARY} !important;
    color: {PRIMARY} !important;
    transform: translateY(-1px);
    box-shadow: 0 12px 28px rgba(112,167,123,.20) !important;
}}
.stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, {PRIMARY}, {MINT}) !important;
    color: white !important;
    border: none !important;
}}

/* chat */
[data-testid="stChatMessage"] {{
    border-radius: 22px !important;
    border: 1px solid {BORDER} !important;
    background: {CARD} !important;
    box-shadow: 0 8px 28px rgba(0,0,0,.055) !important;
    margin-bottom: 14px !important;
}}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {{
    background: linear-gradient(135deg, rgba(112,167,123,.18), rgba(124,200,160,.10)) !important;
}}
[data-testid="stChatMessage"] p {{ line-height: 1.78 !important; }}

/* chat input */
[data-testid="stBottom"],
[data-testid="stBottomBlockContainer"],
[data-testid="stBottom"] > div,
[data-testid="stBottomBlockContainer"] > div {{
    background: {BOTTOM_BG} !important;
    background-color: {BOTTOM_BG} !important;
    backdrop-filter: blur(18px) saturate(160%) !important;
    -webkit-backdrop-filter: blur(18px) saturate(160%) !important;
    border-top: 1px solid {BORDER} !important;
    box-shadow: 0 -10px 34px rgba(0,0,0,.18) !important;
}}
[data-testid="stChatInput"],
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] > div > div,
[data-testid="stChatInputContainer"],
div[class*="stChatInput"] {{
    background: transparent !important;
    background-color: transparent !important;
}}
[data-testid="stChatInput"] textarea {{
    background: {INPUT} !important;
    color: {TEXT} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 999px !important;
    min-height: 48px !important;
    box-shadow: none !important;
}}
[data-testid="stChatInput"] textarea::placeholder {{ color: {MUTED} !important; }}
[data-testid="stChatInput"] button {{
    background: {BOTTOM_CARD} !important;
    color: {TEXT} !important;
    border-radius: 50% !important;
}}

/* streamlit progress */
[data-testid="stProgress"] > div > div > div > div {{ background: linear-gradient(90deg, {PRIMARY}, {MINT}) !important; }}
hr {{ border-color: {BORDER} !important; opacity: .55; }}
footer, #MainMenu {{ visibility: hidden; }}
header {{ visibility: hidden; height: 0 !important; }}
[data-testid="collapsedControl"] {{ visibility: visible !important; display: block !important; }}

.app-shell {{ animation: fadeUp .38s ease both; }}
.hero {{
    background: {CARD}; border: 1px solid {BORDER}; border-radius: 30px;
    box-shadow: {SHADOW}; padding: 28px; margin-bottom: 18px;
    position: relative; overflow: hidden;
}}
.hero::after {{
    content: ''; position: absolute; right: -80px; top: -90px; width: 250px; height: 250px;
    border-radius: 50%; background: rgba(112,167,123,.18);
}}
.hero-eyebrow {{ color: {PRIMARY} !important; font-size: 13px; font-weight: 800; letter-spacing: .08em; margin-bottom: 8px; }}
.hero-title {{ font-size: 34px; font-weight: 900; letter-spacing: -.055em; line-height: 1.16; margin: 0 0 10px; }}
.hero-sub {{ color: {MUTED} !important; font-size: 15px; line-height: 1.7; max-width: 620px; margin: 0; }}

.soft-card {{
    background: {CARD}; border: 1px solid {BORDER}; border-radius: 24px;
    padding: 18px; box-shadow: 0 10px 32px rgba(0,0,0,.055); margin-bottom: 14px;
}}
.pill {{
    display: inline-flex; align-items: center; gap: 7px; padding: 7px 13px;
    border-radius: 999px; font-size: 12px; font-weight: 800; border: 1px solid {BORDER};
    background: {CARD_2}; color: {TEXT};
}}
.metric-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin: 16px 0 8px; }}
.metric-card {{ background: {CARD_2}; border: 1px solid {BORDER}; border-radius: 20px; padding: 15px; }}
.metric-label {{ color: {MUTED} !important; font-size: 12px; font-weight: 700; margin-bottom: 8px; }}
.metric-value {{ font-size: 24px; font-weight: 900; letter-spacing: -.03em; }}
.temp-ring {{
    width: 132px; height: 132px; border-radius: 50%; display: grid; place-items: center;
    background: conic-gradient(var(--ring-color) calc(var(--temp) * 1%), rgba(127,136,122,.18) 0);
    box-shadow: inset 0 0 0 12px {CARD_2}, 0 10px 30px rgba(0,0,0,.08);
}}
.temp-ring-inner {{
    width: 96px; height: 96px; border-radius: 50%; background: {CARD}; display: grid; place-items: center;
    border: 1px solid {BORDER};
}}
.temp-number {{ font-size: 27px; font-weight: 950; line-height: 1; }}
.temp-text {{ color: {MUTED} !important; font-size: 11px; font-weight: 800; margin-top: 4px; }}
.check-row {{ margin: 10px 0; }}
.check-top {{ display:flex; justify-content:space-between; font-size:12px; font-weight:750; margin-bottom:6px; }}
.check-bar {{ height: 8px; border-radius: 999px; background: rgba(127,136,122,.18); overflow: hidden; }}
.check-fill {{ height: 100%; border-radius: 999px; background: linear-gradient(90deg, {PRIMARY}, {MINT}); }}
.tip-box {{ background: rgba(112,167,123,.12); border: 1px solid rgba(112,167,123,.28); border-radius: 18px; padding: 14px; }}
.suggest-title {{ color: {MUTED} !important; font-size: 13px; font-weight: 800; margin: 18px 0 10px; }}
@keyframes fadeUp {{ from {{ opacity:0; transform: translateY(10px); }} to {{ opacity:1; transform: translateY(0); }} }}
@media (max-width: 800px) {{
    .hero-title {{ font-size: 27px; }}
    .metric-grid {{ grid-template-columns: 1fr; }}
}}


/* v5 fixes: prevent chat input clipping and remove white containers in dark mode */
[data-testid="stBottom"] *,
[data-testid="stBottomBlockContainer"] *,
[data-testid="stChatInput"] *,
[data-testid="stChatInputContainer"] * {{
    box-shadow: none !important;
}}
[data-testid="stChatInput"] [data-baseweb="textarea"],
[data-testid="stChatInput"] [data-baseweb="base-input"],
[data-testid="stChatInput"] div {{
    background: transparent !important;
    background-color: transparent !important;
}}
[data-testid="stChatInput"] textarea {{
    min-height: 54px !important;
    height: 54px !important;
    padding: 15px 18px !important;
    line-height: 1.35 !important;
    overflow-y: hidden !important;
}}
[data-testid="stChatInput"] > div {{
    border-radius: 24px !important;
    background: transparent !important;
}}
[data-testid="stBottom"] button,
[data-testid="stBottomBlockContainer"] button {{
    background: {BOTTOM_CARD} !important;
    color: {TEXT} !important;
}}
[data-testid="stBottom"] svg,
[data-testid="stBottomBlockContainer"] svg {{
    color: {TEXT} !important;
}}
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 설문/추천 질문
# -----------------------------------------------------------------------------
SURVEY = [
    ("sleep", "🌙", "요즘 잠은 잘 주무시나요?", True, ["거의 못 잠", "뒤척임", "보통", "잘 잠", "아주 푹 잠"]),
    ("energy", "⚡", "하루 동안 에너지는 어떤가요?", True, ["바닥남", "많이 없음", "보통", "괜찮음", "활기참"]),
    ("recovery", "🌿", "쉬고 나면 회복이 잘 되시나요?", True, ["전혀", "조금", "보통", "잘 됨", "충분히"]),
    ("mood", "😊", "전반적인 기분은 어떠세요?", True, ["매우 가라앉음", "우울함", "보통", "좋음", "아주 좋음"]),
    ("stress", "🔥", "요즘 스트레스 정도는 어떤가요?", False, ["거의 없음", "약간", "보통", "심함", "매우 심함"]),
    ("fatigue", "😴", "몸과 마음의 피로감은요?", False, ["거의 없음", "약간", "보통", "피곤함", "완전 지침"]),
]

SUGGESTED_QUESTIONS = [
    "요즘 너무 지쳐있어요. 번아웃인가요?",
    "수면이 안 좋아서 집중이 안 돼요.",
    "회복하려면 오늘 뭘 하면 좋을까요?",
    "스트레스를 줄이는 방법을 알려주세요.",
]


CATEGORY_FOLDERS = ("P0_safety", "P1_burnout", "P2_recovery", "P3_sleep_stress")


def _count_pdfs(base_dir: str) -> int:
    base = Path(base_dir)
    total = 0
    for folder in CATEGORY_FOLDERS:
        target = base / folder
        if target.is_dir():
            total += len(list(target.rglob("*.pdf")))
    return total


def detect_data_dir() -> str:
    """
    base.py는 기본적으로 ./data/P0_safety 같은 구조를 찾는다.
    그런데 현재 사용자 폴더처럼 P0_safety, P1_burnout 폴더가 프로젝트 루트에 바로 있는 경우도 자동으로 인식한다.
    """
    data_pdf_count = _count_pdfs("./data")
    root_pdf_count = _count_pdfs(".")

    if data_pdf_count > 0:
        return "./data"
    if root_pdf_count > 0:
        return "."

    # PDF가 아직 없더라도 폴더가 루트에 있으면 루트 구조로 안내한다.
    if any((Path(".") / folder).is_dir() for folder in CATEGORY_FOLDERS):
        return "."
    return "./data"


@st.cache_resource(show_spinner="문서 검색기를 준비하고 있어요 🌿")
def get_pipeline(data_dir: str) -> MindCareRAGPipeline:
    return MindCareRAGPipeline(DATA_DIR=data_dir)


# -----------------------------------------------------------------------------
# UI helper
# -----------------------------------------------------------------------------
def toggle_theme() -> None:
    st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"
    st.query_params["theme"] = st.session_state.theme
    st.rerun()


def badge(text: str, color: str, icon: str = "") -> str:
    icon_part = f"{icon} " if icon else ""
    return (
        f'<span class="pill" style="border-color:{color}55; background:{color}18; color:{color} !important;">'
        f'{icon_part}{text}</span>'
    )


def temp_ring(temp: float) -> str:
    color, label, emoji = color_for_temp(temp)
    return f"""
    <div style="display:flex;align-items:center;gap:18px;">
      <div class="temp-ring" style="--temp:{max(0, min(100, temp))};--ring-color:{color};">
        <div class="temp-ring-inner">
          <div style="text-align:center;">
            <div class="temp-number" style="color:{color} !important;">{temp}°</div>
            <div class="temp-text">{emoji} {label}</div>
          </div>
        </div>
      </div>
      <div>
        <div style="font-size:13px;font-weight:900;color:{TEXT};margin-bottom:6px;">마음 온도</div>
        <div style="font-size:12px;color:{MUTED};line-height:1.65;">대화 내용을 바탕으로<br>자동으로 변화해요.</div>
      </div>
    </div>
    """


def header_actions() -> None:
    left, right = st.columns([1, 1])
    with left:
        st.markdown(badge("RAG 기반 멘탈케어", PRIMARY, "🌱"), unsafe_allow_html=True)
    with right:
        label = "🌙 다크" if not IS_DARK else "☀️ 라이트"
        if st.button(label, key=f"theme_{st.session_state.stage}", use_container_width=False):
            toggle_theme()


def onboarding_hero() -> None:
    st.markdown(
        """
        <div class="hero app-shell">
            <div class="hero-eyebrow">MIND TEMPERATURE CHECK-IN</div>
            <div class="hero-title">지금 마음 상태를<br>가볍게 확인해볼게요.</div>
            <p class="hero-sub">
                6개의 짧은 질문으로 초기 마음 온도를 계산하고, 이후 대화에서는 RAG가 신뢰 자료를 참고해 답변합니다.
                진단이 아니라 현재 상태를 이해하기 위한 체크인입니다.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def check_bar(label: str, value: int, icon: str, color: str = PRIMARY) -> str:
    pct = max(0, min(100, value / 5 * 100))
    # 한 줄 HTML로 반환해야 사이드바에서 <div> 코드가 그대로 노출되지 않음
    return (
        f'<div class="check-row">'
        f'<div class="check-top"><span>{icon} {label}</span><span style="color:{color} !important;">{value}/5</span></div>'
        f'<div class="check-bar"><div class="check-fill" style="width:{pct}%;background:linear-gradient(90deg,{color},{color}99);"></div></div>'
        f'</div>'
    )


def recovery_tip(answers: dict[str, int]) -> str:
    sleep = answers.get("sleep", 3)
    energy = answers.get("energy", 3)
    recovery = answers.get("recovery", 3)
    weak = min({"sleep": sleep, "energy": energy, "recovery": recovery}, key={"sleep": sleep, "energy": energy, "recovery": recovery}.get)
    tips = {
        "sleep": ("🌙", "오늘은 잠을 먼저 챙겨봐요", "자기 전 30분만 화면을 멀리하고, 가벼운 스트레칭으로 몸을 낮춰보세요."),
        "energy": ("⚡", "에너지를 작게 회복해봐요", "큰 계획보다 10분 산책이나 물 한 컵처럼 바로 가능한 행동부터 시작해보세요."),
        "recovery": ("🌿", "회복 시간을 따로 떼어놔요", "쉬어도 쉬는 것 같지 않다면, 오늘은 할 일을 하나 줄이는 것도 회복 행동이에요."),
    }
    icon, title, body = tips[weak]
    return f"""
    <div class="tip-box">
      <div style="font-weight:900;color:{PRIMARY};font-size:13px;margin-bottom:5px;">{icon} 오늘의 회복 팁</div>
      <div style="font-weight:850;font-size:13px;margin-bottom:4px;">{title}</div>
      <div style="color:{MUTED};font-size:12px;line-height:1.65;">{body}</div>
    </div>
    """


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            f"""
            <div class="soft-card">
              <div style="font-size:20px;font-weight:950;letter-spacing:-.04em;margin-bottom:4px;">🌱 마음온도 RAG</div>
              <div style="color:{MUTED};font-size:12px;line-height:1.55;">Hybrid Search · Memory · Self-RAG</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(temp_ring(st.session_state.mind_temp), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        color, risk_text = risk_label(st.session_state.risk_level)
        st.markdown(
            f"<div class='soft-card'><div style='color:{MUTED};font-size:12px;font-weight:800;margin-bottom:8px;'>현재 상태</div>"
            f"{badge(risk_text, color, '●')}"
            f"</div>",
            unsafe_allow_html=True,
        )

        answers = st.session_state.get("answers", {})
        if answers:
            st.markdown(
                "<div class='soft-card'><div style='font-size:13px;font-weight:900;margin-bottom:10px;'>체크인 요약</div>"
                + check_bar("수면", answers.get("sleep", 3), "🌙", BLUE)
                + check_bar("에너지", answers.get("energy", 3), "⚡", YELLOW)
                + check_bar("회복력", answers.get("recovery", 3), "🌿", PRIMARY)
                + check_bar("기분", answers.get("mood", 3), "😊", MINT)
                + check_bar("스트레스 낮음", 6 - answers.get("stress", 3), "🔥", CORAL)
                + check_bar("피로도 낮음", 6 - answers.get("fatigue", 3), "😴", CORAL)
                + "</div>",
                unsafe_allow_html=True,
            )
            st.markdown(recovery_tip(answers), unsafe_allow_html=True)

        st.divider()
        st.caption(f"위급하거나 자해 위험이 있다면 자살예방상담전화 {CRISIS_LINE_SUICIDE} 또는 정신건강 위기상담전화 {CRISIS_LINE_MENTAL}에 연락하세요.")
        st.session_state.show_debug = st.toggle("개발자 오류 정보 보기", value=st.session_state.show_debug)
        if st.session_state.show_debug:
            st.caption(f"현재 RAG 데이터 기준 폴더: {detect_data_dir()} / PDF 수: {_count_pdfs(detect_data_dir())}개")


def render_onboarding() -> None:
    render_sidebar()
    header_actions()
    onboarding_hero()

    step = st.session_state.survey_step
    total = len(SURVEY)

    if step < total:
        key, icon, question, _positive, scale = SURVEY[step]
        progress = step / total
        st.progress(progress, text=f"체크인 {step + 1} / {total}")

        st.markdown(
            f"""
            <div class="soft-card app-shell">
              <div style="font-size:44px;margin-bottom:8px;">{icon}</div>
              <div style="font-size:25px;font-weight:950;letter-spacing:-.045em;margin-bottom:8px;">{question}</div>
              <div style="color:{MUTED};font-size:13px;margin-bottom:18px;">가장 가까운 답을 하나 골라주세요.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        cols = st.columns([1, 1, 1, 1, 1])
        for idx, option in enumerate(scale):
            with cols[idx]:
                if st.button(f"{idx + 1}\n\n{option}", key=f"survey_{key}_{idx}", use_container_width=True):
                    st.session_state.answers[key] = idx + 1
                    st.session_state.survey_step += 1
                    st.rerun()

        nav_l, nav_r = st.columns([1, 5])
        with nav_l:
            if step > 0 and st.button("← 이전", use_container_width=True):
                st.session_state.survey_step -= 1
                st.rerun()
        with nav_r:
            st.caption("응답은 앱 안에서 현재 세션의 맞춤 답변을 위해 사용됩니다.")
        return

    filled = {k: st.session_state.answers.get(k, 3) for (k, *_rest) in SURVEY}
    temp = rag.set_initial_temperature(st.session_state.session_id, filled)
    if temp < MindCareRAGPipeline.THRESHOLD_HIGH:
        risk = RISK_HIGH
        result_title = "지금은 회복을 먼저 챙겨야 할 수 있어요."
        result_body = "많이 지쳐 있는 상태로 보입니다. 대화에서는 부담을 낮추고 안전한 도움 연결을 우선으로 안내할게요."
    elif temp < MindCareRAGPipeline.THRESHOLD_MID:
        risk = RISK_MID
        result_title = "조금 지쳐 있는 상태예요."
        result_body = "작게 실천할 수 있는 회복 행동과 학업·일상 조절 방법을 중심으로 안내할게요."
    else:
        risk = "low"
        result_title = "비교적 안정적인 상태예요."
        result_body = "현재 상태를 유지하고 번아웃을 예방하는 방향으로 도와드릴게요."

    color, label, emoji = color_for_temp(temp)
    risk_color, risk_text = risk_label(risk)
    st.markdown(
        f"""
        <div class="hero app-shell" style="text-align:center;">
            <div class="hero-eyebrow">CHECK-IN RESULT</div>
            <div style="display:flex;justify-content:center;margin:12px 0 20px;">{temp_ring(temp)}</div>
            <div style="margin-bottom:12px;">{badge(risk_text, risk_color, emoji)}</div>
            <div style="font-size:25px;font-weight:950;letter-spacing:-.045em;margin-bottom:8px;">{result_title}</div>
            <p class="hero-sub" style="margin:0 auto;">{result_body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("상담 시작하기 →", type="primary", use_container_width=True):
        st.session_state.mind_temp = temp
        st.session_state.risk_level = risk
        greet = (
            f"체크인 결과 마음 온도가 {temp}°로 나왔어요. 지금 많이 지쳐 계실 수 있으니, 가장 부담되는 부분부터 천천히 이야기해 주세요."
            if risk == RISK_HIGH
            else f"마음 온도는 {temp}°예요. 조금 지쳐 있는 상태로 보여요. 오늘 가장 신경 쓰이는 것부터 가볍게 이야기해볼까요?"
            if risk == RISK_MID
            else f"마음 온도는 {temp}°예요. 비교적 안정적인 상태네요. 번아웃 예방이나 회복에 대해 궁금한 점을 편하게 물어보세요."
        )
        st.session_state.messages = [{"role": "assistant", "content": greet}]
        st.session_state.stage = "chat"
        st.rerun()

    st.caption("본 서비스는 의료 행위가 아니며, 마음 온도는 진단이 아닌 참고용 지표입니다.")


def run_rag(prompt: str) -> dict[str, Any]:
    try:
        return rag.ask(prompt, session_id=st.session_state.session_id)
    except FileNotFoundError as e:
        answer = (
            "문서 PDF를 찾지 못해서 RAG 답변을 만들 수 없어요.\n\n"
            "현재 앱은 `data/P0_safety...` 구조와 프로젝트 바로 아래 `P0_safety...` 구조를 모두 자동으로 확인합니다. "
            "각 카테고리 폴더 안에 PDF가 들어있는지 확인하고, 기존 `faiss_db` 폴더가 있으면 삭제한 뒤 다시 실행해 주세요."
        )
        if st.session_state.show_debug:
            answer += f"\n\n오류 내용: `{e}`"
    except Exception as e:
        answer = "실행 중 오류가 났어요. API 키, 패키지 설치, data 폴더 구조를 확인해 주세요."
        if st.session_state.show_debug:
            answer += f"\n\n오류 내용: `{type(e).__name__}: {e}`"
    else:
        return answer  # type: ignore[return-value]

    return {
        "answer": answer,
        "mind_temperature": st.session_state.mind_temp,
        "risk_level": st.session_state.risk_level,
        "temp_delta": None,
        "delta_reason": "",
        "critique": None,
    }


def render_chat() -> None:
    render_sidebar()
    header_actions()

    color, label, emoji = color_for_temp(st.session_state.mind_temp)
    risk_color, risk_text = risk_label(st.session_state.risk_level)
    st.markdown(
        f"""
        <div class="hero app-shell">
            <div class="hero-eyebrow">BURNOUT CARE CHATBOT</div>
            <div style="display:flex;align-items:center;justify-content:space-between;gap:18px;flex-wrap:wrap;">
                <div>
                    <div class="hero-title">번아웃 예방 및 관리 RAG</div>
                    <p class="hero-sub">신뢰할 수 있는 문서를 검색하고, 현재 마음 온도와 대화 맥락을 반영해 답변합니다.</p>
                </div>
                <div style="display:flex;gap:8px;flex-wrap:wrap;justify-content:flex-end;">
                    {badge(f'{emoji} {st.session_state.mind_temp}°', color, '🌡️')}
                    {badge(risk_text, risk_color, '●')}
                </div>
            </div>
            <div class="metric-grid">
                <div class="metric-card"><div class="metric-label">검색 방식</div><div class="metric-value">BM25 + FAISS</div></div>
                <div class="metric-card"><div class="metric-label">대화 기억</div><div class="metric-value">Memory</div></div>
                <div class="metric-card"><div class="metric-label">검증</div><div class="metric-value">Self-RAG</div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for msg in st.session_state.messages:
        avatar = "🌱" if msg["role"] == "assistant" else "🙂"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    if len(st.session_state.messages) <= 1:
        st.markdown("<div class='suggest-title'>바로 물어볼 수 있는 질문</div>", unsafe_allow_html=True)
        cols = st.columns(2)
        for i, question in enumerate(SUGGESTED_QUESTIONS):
            with cols[i % 2]:
                if st.button(question, key=f"suggest_{i}", use_container_width=True):
                    st.session_state.pending_message = question
                    st.rerun()

    if st.session_state.pending_message:
        prompt = st.session_state.pending_message
        st.session_state.pending_message = None
    else:
        prompt = st.chat_input("요즘 어떤 점이 가장 힘드세요?")

    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🙂"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🌱"):
        with st.spinner("관련 문서를 찾고 답변을 정리하고 있어요... 🌿"):
            result = run_rag(prompt)
        st.markdown(result["answer"])

        st.session_state.mind_temp = result.get("mind_temperature", st.session_state.mind_temp)
        st.session_state.risk_level = result.get("risk_level", st.session_state.risk_level)

        delta = result.get("temp_delta")
        if delta:
            d_color = PRIMARY if delta > 0 else CORAL
            arrow = "▲" if delta > 0 else "▼"
            st.markdown(
                f"<div class='tip-box' style='margin-top:10px;border-color:{d_color}44;background:{d_color}12;'>"
                f"<span style='color:{d_color} !important;font-weight:900;'>{arrow} 마음 온도 {'+' if delta > 0 else ''}{delta}°</span>"
                f"<span style='color:{MUTED} !important;'> → {result['mind_temperature']}° · {result.get('delta_reason', '')}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        if result.get("risk_level") == RISK_HIGH:
            st.markdown(
                f"""
                <div class="tip-box" style="margin-top:10px;border-color:{CORAL}55;background:{CORAL}14;">
                    <div style="font-weight:950;color:{CORAL} !important;margin-bottom:5px;">긴급 도움 안내</div>
                    <div style="font-size:13px;line-height:1.8;">
                    힘든 마음이 클 때는 혼자 감당하지 않아도 괜찮아요.<br>
                    📞 자살예방상담전화 <b>{CRISIS_LINE_SUICIDE}</b> · 📞 정신건강 위기상담전화 <b>{CRISIS_LINE_MENTAL}</b> · 🏫 학교 상담센터
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
    st.rerun()


# -----------------------------------------------------------------------------
# 라우팅
# -----------------------------------------------------------------------------
DATA_DIR_USED = detect_data_dir()
rag = get_pipeline(DATA_DIR_USED)

if st.session_state.stage == "onboarding":
    render_onboarding()
else:
    render_chat()
