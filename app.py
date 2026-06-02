"""app.py — 마음 회복 RAG 챗봇 (Streamlit)

흐름
  1) 신규 이용자 온보딩 설문(6문항) → 위험군 판별 → 초기 마음 온도 설정
  2) 채팅: 위험군에 따라 답변 톤이 분기되고, 마음 온도는 대화로만 자동 조정된다.
  3) 마음 온도는 우상단 배지 + 사이드바 세로 온도계로 표시(읽기 전용)
"""

from __future__ import annotations
import os
import uuid
import streamlit as st

if "OPENAI_API_KEY" not in os.environ:
    try:
        if "OPENAI_API_KEY" in st.secrets:
            os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass

from base import (
    MindCareRAGPipeline,
    RISK_HIGH, RISK_MID,
    CRISIS_LINE_SUICIDE, CRISIS_LINE_MENTAL,
)

st.set_page_config(
    page_title="🌱 번아웃 예방 및 관리 RAG",
    page_icon="🌱",
    layout="centered",
)

# ── Session state 초기화 (theme 포함 — CSS 주입 전에 먼저) ─────────────────────
if "session_id"      not in st.session_state: st.session_state.session_id      = f"user-{uuid.uuid4().hex[:8]}"
if "stage"           not in st.session_state: st.session_state.stage           = "onboarding"
if "survey_step"     not in st.session_state: st.session_state.survey_step     = 0
if "answers"         not in st.session_state: st.session_state.answers         = {}
if "messages"        not in st.session_state: st.session_state.messages        = []
if "mind_temp"       not in st.session_state: st.session_state.mind_temp       = MindCareRAGPipeline.DEFAULT_TEMP
if "risk_level"      not in st.session_state: st.session_state.risk_level      = "low"
if "pending_message" not in st.session_state: st.session_state.pending_message = None
if "theme"           not in st.session_state: st.session_state.theme           = st.query_params.get("theme", "light")

# ── Design tokens ──────────────────────────────────────────────────────────────
_is_dark = (st.session_state.theme == "dark")

if _is_dark:
    C_BG           = "#1C1815"
    C_BG_GRAD      = "linear-gradient(160deg,#1C1815 0%,#231E18 100%)"
    C_CARD         = "#272017"
    C_CARD_AI      = "#272017"
    C_CARD_USER    = "#1A2E1D"
    C_USER_BORDER  = "#2C4A32"
    C_SIDEBAR      = "#1E1A14"
    C_TEXT         = "#EDE5D8"
    C_SUBTEXT      = "#9A8878"
    C_BORDER       = "#3A322A"
    C_INPUT        = "#2A2218"
    C_BOTTOM_GLASS = "rgba(28,24,21,0.94)"
    C_BOTTOM_BORDER= "rgba(58,50,42,0.8)"
    C_ALERT_BG     = "#3A1A15"
    C_HERO_GRAD_H  = "#3A1A15"
    C_HERO_GRAD_M  = "#2A2A10"
    C_HERO_GRAD_L  = "#1A2E1D"
else:
    C_BG           = "#FAF7F2"
    C_BG_GRAD      = "linear-gradient(160deg,#FAF7F2 0%,#F4EDE3 100%)"
    C_CARD         = "#FFFFFF"
    C_CARD_AI      = "#FFFFFF"
    C_CARD_USER    = "#EDF8EE"
    C_USER_BORDER  = "#C8E8CC"
    C_SIDEBAR      = "#F5F1EA"
    C_TEXT         = "#3A3330"
    C_SUBTEXT      = "#9A8878"
    C_BORDER       = "#E5DDD4"
    C_INPUT        = "#FFFFFF"
    C_BOTTOM_GLASS = "rgba(250,247,242,0.92)"
    C_BOTTOM_BORDER= "rgba(229,221,212,0.7)"
    C_ALERT_BG     = "#FFF0EE"
    C_HERO_GRAD_H  = "#FEE4DC"
    C_HERO_GRAD_M  = "#FFFBE6"
    C_HERO_GRAD_L  = "#EDF8EE"

# 공통 accent — 라이트/다크 동일
C_PRIMARY = "#6EA97A"   # sage green
C_ACCENT  = "#E78A74"   # warm coral
C_BLUE    = "#7AA6D6"   # secondary blue
C_MINT    = "#7CC8A0"
C_YELLOW  = "#F5C842"
C_CORAL   = "#E78A74"


def _toggle_theme() -> None:
    """세션 유지 + URL 파라미터 동기화 테마 전환 (페이지 재로드 없음)."""
    new_theme = "dark" if st.session_state.theme == "light" else "light"
    st.session_state.theme = new_theme
    st.query_params["theme"] = new_theme
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# 전역 CSS — Pretendard 폰트 + 레이아웃 + 컴포넌트
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<!-- Pretendard 웹폰트 (한국어 UX 품질 개선) -->
<link rel="preconnect" href="https://cdn.jsdelivr.net" crossorigin>
<link rel="stylesheet"
      href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css">

<style>
/* ══ 0. 폰트 & html/body 배경 (검정 하단 근본 fix) ══════════════════ */
html, body {{
    background: {C_BG} !important;
    background-color: {C_BG} !important;
    font-family: 'Pretendard', -apple-system, BlinkMacSystemFont,
                 'Apple SD Gothic Neo', 'Noto Sans KR', 'Segoe UI',
                 'Apple Color Emoji', 'Segoe UI Emoji', 'Noto Color Emoji',
                 sans-serif !important;
}}
/* 이모지·아이콘 요소는 제외하고 텍스트 요소에만 폰트 적용 */
p, span, div, li, button, input, textarea, label, h1, h2, h3, h4, h5, h6, a {{
    font-family: 'Pretendard', -apple-system, BlinkMacSystemFont,
                 'Apple SD Gothic Neo', 'Noto Sans KR', 'Segoe UI',
                 'Apple Color Emoji', 'Segoe UI Emoji', 'Noto Color Emoji',
                 sans-serif !important;
    box-sizing: border-box;
}}

/* ══ 1. 앱 전체 배경 ════════════════════════════════════════════════ */
.stApp {{
    background: {C_BG_GRAD} !important;
    color: {C_TEXT} !important;
    transition: background .4s ease, color .3s ease;
}}
.main .block-container {{
    max-width: 760px !important;
    padding-top: 1.4rem !important;
    padding-bottom: 6rem !important;   /* 입력창에 가리지 않도록 */
}}

/* ══ 2. 전역 텍스트 ══════════════════════════════════════════════════ */
.stApp p, .stApp span, .stApp li,
.stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span,
[data-testid="stText"], [data-testid="stMarkdownContainer"] p {{
    color: {C_TEXT} !important;
    line-height: 1.75 !important;
}}
.stCaption p, [data-testid="stCaptionContainer"] p {{
    color: {C_SUBTEXT} !important;
    font-size: 0.78rem !important;
}}
h1, h2, h3, h4, h5 {{
    color: {C_TEXT} !important;
    letter-spacing: -0.02em;
}}

/* ══ 3. 사이드바 ═════════════════════════════════════════════════════ */
section[data-testid="stSidebar"] {{
    background: {C_SIDEBAR} !important;
    border-right: 1px solid {C_BORDER} !important;
    transition: background .4s ease;
}}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] li,
section[data-testid="stSidebar"] .stMarkdown p {{
    color: {C_TEXT} !important;
}}
section[data-testid="stSidebar"] .stCaption p,
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {{
    color: {C_SUBTEXT} !important;
}}

/* ══ 4. 채팅 버블 ════════════════════════════════════════════════════ */
/* 기본값 = 사용자 스타일 (:has 미지원 환경 폴백) */
[data-testid="stChatMessage"] {{
    background: {C_CARD_USER} !important;
    border: 1px solid {C_USER_BORDER} !important;
    border-radius: 22px !important;
    margin-bottom: 12px !important;
    padding: 4px 0 !important;
    box-shadow: 0 2px 10px rgba(0,0,0,.05) !important;
    transition: box-shadow .2s ease, transform .2s ease;
    animation: bubbleIn .4s cubic-bezier(.22,.68,0,1.15) both;
}}
[data-testid="stChatMessage"]:hover {{
    box-shadow: 0 4px 18px rgba(0,0,0,.09) !important;
    transform: translateY(-1px);
}}
/* AI 버블 오버라이드 */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {{
    background: {C_CARD_AI} !important;
    border-color: {C_BORDER} !important;
    box-shadow: 0 2px 14px rgba(0,0,0,.06) !important;
}}
/* 사용자 버블 명시 (지원 환경) */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {{
    background: {C_CARD_USER} !important;
    border-color: {C_USER_BORDER} !important;
}}
/* 버블 내 텍스트 */
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] .stMarkdown p,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {{
    color: {C_TEXT} !important;
    line-height: 1.8 !important;
}}

/* ══ 5. 사용자 아바타 (빨간색 제거) ══════════════════════════════════ */
[data-testid="chatAvatarIcon-user"],
[data-testid="stChatMessageAvatarUser"] {{
    background: linear-gradient(135deg, {C_BLUE}, {C_PRIMARY}) !important;
    color: white !important;
    box-shadow: 0 2px 8px {C_BLUE}44 !important;
}}
[data-testid="chatAvatarIcon-user"] svg path,
[data-testid="stChatMessageAvatarUser"] svg path {{
    fill: white !important;
}}

/* ══ 6. 하단 입력 영역 — Glassmorphism (검정 배경 근본 제거) ══════════ */
/* 다중 선택자로 버전 변경에 내성 강화 */
[data-testid="stBottom"],
[data-testid="stBottomBlockContainer"],
[data-testid="stBottom"] > div,
[data-testid="stBottomBlockContainer"] > div,
div[class*="bottom"],
div[class*="Bottom"] {{
    background: {C_BOTTOM_GLASS} !important;
    backdrop-filter: blur(20px) saturate(180%) !important;
    -webkit-backdrop-filter: blur(20px) saturate(180%) !important;
    border-top: 1px solid {C_BOTTOM_BORDER} !important;
    box-shadow: 0 -4px 24px rgba(0,0,0,.06) !important;
}}
[data-testid="stBottom"]::before,
[data-testid="stBottomBlockContainer"]::before {{
    background: linear-gradient(to top, {C_BG} 40%, transparent) !important;
}}
/* 입력창 컨테이너 */
[data-testid="stChatInput"],
[data-testid="stChatInputContainer"],
div[class*="ChatInput"] {{
    background: transparent !important;
}}
[data-testid="stChatInput"] textarea {{
    border-radius: 28px !important;
    background: {C_INPUT} !important;
    border: 1.5px solid {C_BORDER} !important;
    color: {C_TEXT} !important;
    font-size: 0.95rem !important;
    padding: 12px 20px !important;
    box-shadow: 0 2px 12px rgba(0,0,0,.06) !important;
    transition: border-color .2s, box-shadow .2s;
}}
[data-testid="stChatInput"] textarea::placeholder {{ color: {C_SUBTEXT} !important; }}
[data-testid="stChatInput"] textarea:focus {{
    border-color: {C_PRIMARY} !important;
    box-shadow: 0 0 0 3px {C_PRIMARY}30, 0 2px 12px rgba(0,0,0,.08) !important;
    outline: none !important;
}}

/* ══ 7. 버튼 ═════════════════════════════════════════════════════════ */
.stButton > button {{
    border-radius: 14px !important;
    border: 1.5px solid {C_BORDER} !important;
    background: {C_CARD} !important;
    color: {C_TEXT} !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    letter-spacing: -0.01em !important;
    white-space: normal !important;
    height: auto !important;
    padding: 9px 16px !important;
    transition: all .2s cubic-bezier(.22,.68,0,1.2) !important;
}}
.stButton > button:hover {{
    border-color: {C_PRIMARY} !important;
    color: {C_PRIMARY} !important;
    background: {C_PRIMARY}0C !important;
    box-shadow: 0 4px 14px {C_PRIMARY}28 !important;
    transform: translateY(-1px) !important;
}}
.stButton > button:active {{ transform: translateY(0) scale(.98) !important; }}
.stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, {C_PRIMARY} 0%, {C_PRIMARY}CC 100%) !important;
    color: white !important;
    border-color: transparent !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 16px {C_PRIMARY}44 !important;
    letter-spacing: .01em !important;
}}
.stButton > button[kind="primary"]:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px {C_PRIMARY}55 !important;
    color: white !important;
}}

/* ══ 8. Progress bar ══════════════════════════════════════════════════ */
[data-testid="stProgressBar"] > div {{
    background: {C_BORDER} !important;
    border-radius: 999px !important;
    overflow: hidden;
    height: 6px !important;
}}
[data-testid="stProgressBar"] > div > div {{
    background: linear-gradient(90deg, {C_PRIMARY}, {C_MINT}) !important;
    border-radius: 999px !important;
    transition: width .6s cubic-bezier(.22,.68,0,1);
}}
[data-testid="stProgressBar"] p {{ color: {C_SUBTEXT} !important; font-size: 0.78rem !important; }}

/* ══ 9. 구분선 ════════════════════════════════════════════════════════ */
hr {{ border-color: {C_BORDER} !important; opacity: .5; margin: 1rem 0 !important; }}

/* ══ 10. Spinner ══════════════════════════════════════════════════════ */
[data-testid="stSpinner"] p {{ color: {C_SUBTEXT} !important; }}

/* ══ 11. 애니메이션 ══════════════════════════════════════════════════ */
@keyframes bubbleIn {{
    from {{ opacity: 0; transform: translateY(12px) scale(.97); }}
    to   {{ opacity: 1; transform: translateY(0)   scale(1);    }}
}}
@keyframes fadeSlide {{
    from {{ opacity: 0; transform: translateY(8px); }}
    to   {{ opacity: 1; transform: translateY(0);   }}
}}
@keyframes fadeIn {{
    from {{ opacity: 0; }}
    to   {{ opacity: 1; }}
}}
@keyframes scaleIn {{
    from {{ opacity: 0; transform: scale(.95); }}
    to   {{ opacity: 1; transform: scale(1);   }}
}}

footer, #MainMenu, header {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)


# ── 온보딩 설문 정의 ───────────────────────────────────────────────────────────
SURVEY = [
    ("sleep",    "🌙", "요즘 잠은 잘 주무시나요?",        True,  ["거의 못 잠", "뒤척임", "보통", "잘 잠", "아주 푹 잠"]),
    ("energy",   "⚡", "하루 동안 에너지는 어떤가요?",     True,  ["바닥남", "많이 없음", "보통", "괜찮음", "활기참"]),
    ("recovery", "🌿", "쉬고 나면 회복이 잘 되시나요?",    True,  ["전혀", "조금", "보통", "잘 됨", "충분히"]),
    ("mood",     "😊", "전반적인 기분은 어떠세요?",        True,  ["매우 가라앉음", "우울함", "보통", "좋음", "아주 좋음"]),
    ("stress",   "🔥", "요즘 스트레스 정도는 어떤가요?",   False, ["거의 없음", "약간", "보통", "심함", "매우 심함"]),
    ("fatigue",  "😴", "몸과 마음의 피로감은요?",          False, ["거의 없음", "약간", "보통", "피곤함", "완전 지침"]),
]

SUGGESTED_QUESTIONS = [
    "요즘 너무 지쳐있어요. 번아웃인가요?",
    "수면이 안 좋아서 집중이 안 돼요.",
    "회복하려면 어떻게 해야 할까요?",
    "스트레스를 어떻게 풀 수 있을까요?",
]


@st.cache_resource(show_spinner="잠깐, 문서를 불러오고 있어요 ✨")
def get_pipeline():
    rag = MindCareRAGPipeline()
    rag.build_chain()
    return rag


# ── HTML 헬퍼 ──────────────────────────────────────────────────────────────────
def thermometer_html(temp: float, tall: bool = True) -> str:
    if temp >= MindCareRAGPipeline.THRESHOLD_MID:
        color, label, emoji = C_MINT,   "안정", "☀️"
    elif temp >= MindCareRAGPipeline.THRESHOLD_HIGH:
        color, label, emoji = C_YELLOW, "주의", "🌤️"
    else:
        color, label, emoji = C_CORAL,  "위험", "🌧️"
    h    = 150 if tall else 110
    fill = max(0, min(100, temp))
    ticks = "".join(
        f'<div style="position:absolute;right:0;top:{h - m/100*h - 1:.1f}px;'
        f'width:6px;height:1.5px;background:{C_SUBTEXT};opacity:.4;"></div>'
        for m in (0, 25, 50, 75, 100)
    )
    return f"""
    <div style="display:flex;align-items:center;gap:14px;
                background:{C_SIDEBAR};border:1px solid {C_BORDER};
                border-radius:16px;padding:16px 14px;
                box-shadow:0 2px 12px rgba(0,0,0,.06);
                animation:scaleIn .4s ease;">
      <div style="position:relative;width:30px;height:{h+30}px;flex:0 0 auto;">
        <div style="position:absolute;left:9px;top:0;width:12px;height:{h}px;
                    background:{C_BORDER};border-radius:999px;overflow:hidden;
                    box-shadow:inset 0 0 4px rgba(0,0,0,.12);">
          <div style="position:absolute;bottom:0;left:0;width:100%;height:{fill}%;
                      background:linear-gradient(180deg,{color},{color}BB);
                      border-radius:999px;transition:height .8s cubic-bezier(.22,.68,0,1);"></div>
        </div>
        {ticks}
        <div style="position:absolute;left:3px;bottom:0;width:24px;height:24px;
                    border-radius:50%;background:{color};
                    box-shadow:0 0 10px {color}88;"></div>
      </div>
      <div>
        <div style="font-size:11px;color:{C_SUBTEXT};font-weight:600;
                    letter-spacing:.04em;margin-bottom:2px;">🌡️ 마음 온도</div>
        <div style="font-size:30px;font-weight:800;color:{color};
                    line-height:1;letter-spacing:-0.02em;">{emoji} {temp}°</div>
        <div style="display:inline-block;margin-top:7px;font-size:11px;font-weight:700;
                    color:{color};background:{color}20;border:1px solid {color}55;
                    border-radius:999px;padding:2px 11px;letter-spacing:.02em;">{label}</div>
      </div>
    </div>"""


def temp_badge_html(temp: float) -> str:
    if temp >= MindCareRAGPipeline.THRESHOLD_MID:
        color, emoji = C_MINT,   "☀️"
    elif temp >= MindCareRAGPipeline.THRESHOLD_HIGH:
        color, emoji = C_YELLOW, "🌤️"
    else:
        color, emoji = C_CORAL,  "🌧️"
    return (
        f'<span style="display:inline-flex;align-items:center;gap:5px;'
        f'font-size:14px;font-weight:700;letter-spacing:-.01em;'
        f'color:{color};background:{color}20;border:1px solid {color}55;'
        f'padding:5px 14px;border-radius:999px;'
        f'box-shadow:0 2px 8px {color}30;">🌡️ {emoji} {temp}°</span>'
    )


def risk_badge_html(risk: str) -> str:
    m = {
        RISK_HIGH: (C_CORAL,   "⚠️ 많이 지쳐 보여요"),
        RISK_MID:  (C_YELLOW,  "🌤️ 조금 지쳐 보여요"),
        "low":     (C_PRIMARY, "✅ 안정적이에요"),
    }
    color, label = m.get(risk, m["low"])
    return (
        f'<span style="display:inline-flex;align-items:center;gap:5px;'
        f'background:{color}1A;border:1px solid {color}66;border-radius:999px;'
        f'padding:5px 13px;font-size:12px;color:{color};font-weight:600;">{label}</span>'
    )


def hero_card_html() -> str:
    """채팅 화면 상단 컨텍스트 히어로 카드."""
    risk = st.session_state.risk_level
    temp = st.session_state.mind_temp

    if risk == RISK_HIGH:
        grad_a, grad_b = C_HERO_GRAD_H, C_BG
        accent = C_CORAL
        title  = "지금 많이 힘드시죠?"
        sub    = "괜찮아요. 잠깐, 마음의 짐을 함께 내려놔요."
        emoji  = "🌧️"
    elif risk == RISK_MID:
        grad_a, grad_b = C_HERO_GRAD_M, C_BG
        accent = C_YELLOW
        title  = "조금 지쳐 있는 상태예요"
        sub    = "오늘 하루, 작은 것 하나만 챙겨봐요."
        emoji  = "🌤️"
    else:
        grad_a, grad_b = C_HERO_GRAD_L, C_BG
        accent = C_PRIMARY
        title  = "오늘도 안정적이에요"
        sub    = "이 좋은 상태를 함께 유지해봐요."
        emoji  = "☀️"

    badge = temp_badge_html(temp)
    return f"""
    <div style="
        background: linear-gradient(135deg, {grad_a}, {grad_b});
        border: 1px solid {accent}33;
        border-radius: 20px;
        padding: 20px 24px;
        margin-bottom: 18px;
        animation: fadeSlide .5s ease;
        box-shadow: 0 4px 24px {accent}14;
    ">
        <div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap;">
            <div style="font-size:34px;line-height:1;flex-shrink:0;">{emoji}</div>
            <div style="flex:1;min-width:0;">
                <div style="font-size:1rem;font-weight:700;color:{C_TEXT};
                            margin-bottom:3px;letter-spacing:-.01em;">{title}</div>
                <div style="font-size:0.82rem;color:{C_SUBTEXT};line-height:1.55;">{sub}</div>
            </div>
            <div style="flex-shrink:0;">{badge}</div>
        </div>
    </div>"""


def sidebar_stat_bar(label: str, val: int, icon: str, color: str) -> str:
    """사이드바용 미니 스탯 바 (점수 1–5)."""
    pct   = val / 5 * 100
    stars = "●" * val + "○" * (5 - val)
    return f"""
    <div style="margin-bottom:10px;">
        <div style="display:flex;justify-content:space-between;align-items:center;
                    font-size:11px;color:{C_SUBTEXT};margin-bottom:4px;">
            <span>{icon} {label}</span>
            <span style="color:{color};font-weight:600;letter-spacing:.06em;
                         font-size:9px;">{stars}</span>
        </div>
        <div style="height:5px;background:{C_BORDER};border-radius:999px;overflow:hidden;">
            <div style="width:{pct}%;height:100%;
                        background:linear-gradient(90deg,{color},{color}99);
                        border-radius:999px;
                        transition:width .8s cubic-bezier(.22,.68,0,1);"></div>
        </div>
    </div>"""


def recovery_tip_html(answers: dict) -> str:
    """설문 점수 기반 오늘의 회복 팁."""
    scores = {
        "sleep":    answers.get("sleep",    3),
        "energy":   answers.get("energy",   3),
        "recovery": answers.get("recovery", 3),
    }
    inverted = {"sleep": 6 - scores["sleep"],
                "energy": 6 - scores["energy"],
                "recovery": 6 - scores["recovery"]}
    weak = min(inverted, key=inverted.get)

    tips = {
        "sleep":    ("🌙", "수면 팁", "화면을 1시간 일찍 끄고 짧은 스트레칭 해보세요."),
        "energy":   ("⚡", "에너지 팁", "15분 햇빛 산책이 오후 에너지를 올려줘요."),
        "recovery": ("🌿", "회복 팁",  "깊은 호흡 5회로 긴장을 조금씩 풀어보세요."),
    }
    icon, ttl, body = tips[weak]
    return f"""
    <div style="background:{C_PRIMARY}12;border:1px solid {C_PRIMARY}33;
                border-radius:12px;padding:10px 12px;margin-top:4px;
                animation:fadeIn .6s ease;">
        <div style="font-size:11px;font-weight:700;color:{C_PRIMARY};
                    margin-bottom:3px;">{icon} 오늘의 {ttl}</div>
        <div style="font-size:11px;color:{C_SUBTEXT};line-height:1.55;">{body}</div>
    </div>"""


rag = get_pipeline()


# ════════════════════════════════════════════════════════════════════════════
# 1) 온보딩 설문
# ════════════════════════════════════════════════════════════════════════════
def render_onboarding():
    # 우상단 테마 토글
    _, tr = st.columns([7, 1])
    with tr:
        label = "🌙" if not _is_dark else "☀️"
        if st.button(label, key="theme_onboarding", help="라이트/다크 전환"):
            _toggle_theme()

    step  = st.session_state.survey_step
    total = len(SURVEY)

    if step < total:
        key, icon, q, _positive, scale = SURVEY[step]
        st.progress(step / total, text=f"마음 체크인 설문 · {step + 1} / {total}")
        st.markdown(
            f"<div style='font-size:44px;margin:10px 0 2px;animation:fadeSlide .4s ease;'>"
            f"{icon}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='font-size:1.3rem;font-weight:700;color:{C_TEXT};"
            f"letter-spacing:-.02em;margin-bottom:20px;animation:fadeSlide .45s ease;'>"
            f"{q}</div>",
            unsafe_allow_html=True,
        )
        for i, label in enumerate(scale):
            if st.button(f"{i+1}.  {label}", key=f"opt_{step}_{i}", use_container_width=True):
                st.session_state.answers[key] = i + 1
                st.session_state.survey_step  += 1
                st.rerun()
        if step > 0:
            if st.button("← 이전", key="prev"):
                st.session_state.survey_step -= 1
                st.rerun()
        return

    # ── 결과 화면 ──
    filled = {k: st.session_state.answers.get(k, 3) for (k, *_rest) in SURVEY}
    temp   = rag.set_initial_temperature(st.session_state.session_id, filled)
    if temp < MindCareRAGPipeline.THRESHOLD_HIGH:
        risk, desc = RISK_HIGH, "지금 많이 지쳐 계신 것 같아요. 천천히, 함께 이야기 나눠요."
    elif temp < MindCareRAGPipeline.THRESHOLD_MID:
        risk, desc = RISK_MID,  "조금 지쳐 계신 상태예요. 가벼운 것부터 살펴봐요."
    else:
        risk, desc = "low",     "비교적 안정적이에요. 예방 차원에서 함께 점검해 봐요."

    st.markdown(
        f"<p style='text-align:center;color:{C_SUBTEXT};font-size:0.88rem;"
        f"letter-spacing:.02em;margin-bottom:4px;'>설문 결과 · 초기 마음 온도</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div style='display:flex;justify-content:center;margin-bottom:12px;'>"
        f"{thermometer_html(temp)}</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div style='text-align:center;margin-bottom:8px;'>{risk_badge_html(risk)}</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<p style='text-align:center;color:{C_SUBTEXT};font-size:0.9rem;"
        f"line-height:1.7;margin-bottom:20px;'>{desc}</p>",
        unsafe_allow_html=True,
    )
    if st.button("상담 시작하기 →", type="primary", use_container_width=True):
        st.session_state.mind_temp  = temp
        st.session_state.risk_level = risk
        greet = (
            f"체크인 결과 마음 온도가 {temp}°네요. 요즘 정말 많이 지쳐 계신 것 같아요. "
            "어떤 부분이 가장 힘드신지 편하게 말씀해 주시겠어요? 함께 이야기 나눠봐요."
            if risk == RISK_HIGH else
            f"마음 온도 {temp}°, 조금 지쳐 계신 것 같아요. "
            "오늘 가장 신경 쓰이는 점부터 가볍게 시작해 봐요."
            if risk == RISK_MID else
            f"마음 온도 {temp}°, 지금 비교적 안정적이시네요! "
            "번아웃 예방이나 회복에 대해 궁금한 점을 편하게 물어보세요."
        )
        st.session_state.messages = [{"role": "assistant", "content": greet}]
        st.session_state.stage    = "chat"
        st.rerun()

    st.caption("⚠️ 본 서비스는 의료 행위가 아닙니다. 마음 온도는 이후 대화를 통해 자동으로 조정됩니다.")


# ════════════════════════════════════════════════════════════════════════════
# 2) 채팅
# ════════════════════════════════════════════════════════════════════════════
def render_chat():
    answers = st.session_state.get("answers", {})

    # ── 사이드바 대시보드 ─────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            f"<h3 style='color:{C_TEXT};font-size:1.05rem;"
            f"font-weight:700;letter-spacing:-.02em;margin-bottom:2px;'>"
            f"🌱 번아웃 예방 RAG</h3>",
            unsafe_allow_html=True,
        )
        st.caption("Hybrid(BM25+FAISS) · Self-RAG")
        st.markdown(thermometer_html(st.session_state.mind_temp), unsafe_allow_html=True)
        st.markdown(
            f"<p style='font-size:10.5px;color:{C_SUBTEXT};text-align:center;"
            f"line-height:1.65;margin-top:8px;'>"
            "🔒 마음 온도는 직접 조절할 수 없어요.<br>대화를 통해 자동으로 변화합니다.</p>",
            unsafe_allow_html=True,
        )
        st.divider()

        # 현재 위험군
        st.markdown(
            f"<p style='font-size:11px;color:{C_SUBTEXT};font-weight:600;"
            f"letter-spacing:.04em;margin-bottom:6px;'>현재 위험군</p>",
            unsafe_allow_html=True,
        )
        st.markdown(risk_badge_html(st.session_state.risk_level), unsafe_allow_html=True)

        # 미니 스탯 카드 (설문 응답 있을 때)
        if answers:
            st.markdown(
                f"<p style='font-size:11px;color:{C_SUBTEXT};font-weight:600;"
                f"letter-spacing:.04em;margin-top:14px;margin-bottom:8px;'>"
                "체크인 점수</p>",
                unsafe_allow_html=True,
            )
            st.markdown(
                sidebar_stat_bar("수면",   answers.get("sleep",    3), "🌙", C_BLUE)
                + sidebar_stat_bar("에너지", answers.get("energy",   3), "⚡", C_YELLOW)
                + sidebar_stat_bar("회복력", answers.get("recovery", 3), "🌿", C_PRIMARY),
                unsafe_allow_html=True,
            )
            st.markdown(recovery_tip_html(answers), unsafe_allow_html=True)

        st.divider()

        # 테마 토글
        st.markdown(
            f"<p style='font-size:10.5px;color:{C_SUBTEXT};font-weight:600;"
            f"letter-spacing:.04em;margin-bottom:6px;'>🎨 화면 테마</p>",
            unsafe_allow_html=True,
        )
        toggle_label = "🌙  다크 모드로 전환" if not _is_dark else "☀️  라이트 모드로 전환"
        if st.button(toggle_label, key="theme_chat", use_container_width=True):
            _toggle_theme()

        st.caption(
            f"⚠️ 본 서비스는 의료 행위가 아닙니다.\n"
            f"위급 시 자살예방상담전화 {CRISIS_LINE_SUICIDE}."
        )

    # ── 메인 헤더 ─────────────────────────────────────────────────────
    left, right = st.columns([3, 1])
    with left:
        st.markdown(
            f"<h3 style='color:{C_TEXT};margin:0;font-weight:700;"
            f"letter-spacing:-.03em;font-size:1.25rem;'>🌱 번아웃 예방 및 관리 RAG</h3>",
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            f"<div style='text-align:right;padding-top:6px;'>"
            f"{temp_badge_html(st.session_state.mind_temp)}</div>",
            unsafe_allow_html=True,
        )
    st.divider()

    # ── Hero 카드 ─────────────────────────────────────────────────────
    st.markdown(hero_card_html(), unsafe_allow_html=True)

    # ── 대화 렌더링 ──────────────────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── 추천 질문 (첫 화면) ──────────────────────────────────────────
    if len(st.session_state.messages) <= 1:
        cols = st.columns(2)
        for i, q in enumerate(SUGGESTED_QUESTIONS):
            if cols[i % 2].button(q, key=f"sq_{i}", use_container_width=True):
                st.session_state.pending_message = q
                st.rerun()

    # ── 입력 처리 ─────────────────────────────────────────────────────
    prompt = st.chat_input("요즘 어떤 점이 힘드세요?")
    if st.session_state.pending_message:
        prompt = st.session_state.pending_message
        st.session_state.pending_message = None

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("잠깐, 마음의 이야기를 찾고 있어요... 🌿"):
                result = rag.ask(prompt, session_id=st.session_state.session_id)
            st.markdown(result["answer"])

            st.session_state.mind_temp  = result["mind_temperature"]
            st.session_state.risk_level = result["risk_level"]

            d = result.get("temp_delta")
            if d:
                arrow = "▲" if d > 0 else "▼"
                tone  = C_PRIMARY if d > 0 else C_CORAL
                st.markdown(
                    f"<p style='font-size:0.76rem;color:{tone};margin-top:5px;"
                    f"letter-spacing:.01em;'>"
                    f"{arrow} 마음 온도 {'+' if d > 0 else ''}{d}° "
                    f"→ {result['mind_temperature']}° "
                    f"<span style='color:{C_SUBTEXT};'>"
                    f"({result.get('delta_reason', '')})</span></p>",
                    unsafe_allow_html=True,
                )

            if result["risk_level"] == RISK_HIGH:
                st.markdown(
                    f"""<div style="background:{C_ALERT_BG};
                                   border-left:3px solid {C_CORAL};
                                   border-radius:0 12px 12px 0;
                                   padding:0.8rem 1.1rem;margin-top:0.8rem;
                                   font-size:0.83rem;color:{C_TEXT};line-height:1.9;">
                        <strong>힘든 마음이 클 때는 전문가의 도움이 큰 힘이 돼요.</strong><br>
                        📞 자살예방 상담전화 <strong>{CRISIS_LINE_SUICIDE}</strong>&nbsp;(24시간)<br>
                        📞 정신건강 위기상담전화 <strong>{CRISIS_LINE_MENTAL}</strong>&nbsp;(24시간)<br>
                        🏫 학교 상담센터
                    </div>""",
                    unsafe_allow_html=True,
                )

        st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
        st.rerun()


# ── 라우팅 ─────────────────────────────────────────────────────────────────────
if st.session_state.stage == "onboarding":
    render_onboarding()
else:
    render_chat()
