from __future__ import annotations

import html as _html
from typing import Any

import streamlit as st

from base import (
    RISK_HIGH,
    RISK_MID,
    SelfRAGPipeline as MindCareRAGPipeline,
    DEFAULT_MIND_TEMPERATURE,
)


# -----------------------------------------------------------------------------
# 내부 헬퍼 — base.py 수정 없이 set_initial_temperature 대체
# -----------------------------------------------------------------------------
def _set_initial_temperature(
    rag: MindCareRAGPipeline, session_id: str, answers: dict
) -> float:
    temp = rag.compute_mind_temperature(answers)
    rag.record_temperature(session_id, temp)
    return temp


# -----------------------------------------------------------------------------
# 체크인 세션 상태
# -----------------------------------------------------------------------------
def init_checkin_state() -> None:
    defaults: dict[str, Any] = {
        "stage": "onboarding",
        "survey_step": 0,
        "answers": {},
        "messages": [],
        "mind_temp": DEFAULT_MIND_TEMPERATURE,
        "risk_level": "low",
        "user_name": "",
        "user_avatar": "🧑",
        "assistant_avatar": "🌿",
        "avatar_selected": False,
    }

    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


# -----------------------------------------------------------------------------
# 체크인 설문 문항
# positive=True  : 점수가 높을수록 좋은 상태
# positive=False : 점수가 높을수록 나쁜 상태
# -----------------------------------------------------------------------------
SURVEY = [
    (
        "sleep",
        "🌙",
        "요즘 잠은 잘 주무시나요?",
        True,
        ["거의 못 잠", "뒤척임", "보통", "잘 잠", "아주 푹 잠"],
    ),
    (
        "energy",
        "⚡",
        "하루 동안 에너지는 어떤가요?",
        True,
        ["바닥남", "많이 없음", "보통", "괜찮음", "활기참"],
    ),
    (
        "recovery",
        "🌿",
        "쉬고 나면 회복이 잘 되시나요?",
        True,
        ["전혀", "조금", "보통", "잘 됨", "충분히"],
    ),
    (
        "mood",
        "😊",
        "전반적인 기분은 어떠세요?",
        True,
        ["아주 나쁨", "나쁨", "보통", "좋음", "아주 좋음"],
    ),
    (
        "stress",
        "🔥",
        "요즘 스트레스 정도는 어떤가요?",
        False,
        ["거의 없음", "약간", "보통", "심함", "매우 심함"],
    ),
    (
        "fatigue",
        "😴",
        "몸과 마음의 피로감은요?",
        False,
        ["거의 없음", "약간", "보통", "피곤함", "완전 지침"],
    ),
]


# -----------------------------------------------------------------------------
# 색상 / 상태 라벨
# -----------------------------------------------------------------------------
AVATARS = [
    "🧑", "👩", "🧒", "🧔", "🙂", "😊", "🥰", "😎",
    "🌿", "🌸", "🌊", "🌙", "☀️", "🍃", "🌺", "🌻",
    "🐱", "🐶", "🐼", "🦊", "🦋", "🧸", "🐨", "🐧",
]

PRIMARY = "#70A77B"
MINT = "#7CC8A0"
YELLOW = "#F3C54B"
CORAL = "#E68070"
BLUE = "#79A7D8"

TEXT = "#2F332D"
MUTED = "#7F887A"
CARD = "#FFFFFF"
CARD_2 = "#FBF8F1"
BORDER = "rgba(65,78,55,.13)"
SHADOW = "0 18px 55px rgba(89,94,72,.14)"


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
# 체크인 전용 CSS
# -----------------------------------------------------------------------------
def render_checkin_styles() -> None:
    st.markdown(
        f"""
        <style>
        .app-shell {{
            animation: fadeUp .38s ease both;
        }}

        .hero {{
            background: {CARD};
            border: 1px solid {BORDER};
            border-radius: 30px;
            box-shadow: {SHADOW};
            padding: 28px;
            margin-bottom: 18px;
            position: relative;
            overflow: hidden;
        }}

        .hero::after {{
            content: '';
            position: absolute;
            right: -80px;
            top: -90px;
            width: 250px;
            height: 250px;
            border-radius: 50%;
            background: rgba(112,167,123,.18);
        }}

        .hero-eyebrow {{
            color: {PRIMARY} !important;
            font-size: 13px;
            font-weight: 800;
            letter-spacing: .08em;
            margin-bottom: 8px;
        }}

        .hero-title {{
            font-size: 34px;
            font-weight: 900;
            letter-spacing: -.055em;
            line-height: 1.16;
            margin: 0 0 10px;
            color: {TEXT} !important;
        }}

        .hero-sub {{
            color: {MUTED} !important;
            font-size: 15px;
            line-height: 1.7;
            max-width: 620px;
            margin: 0;
        }}

        .soft-card {{
            background: {CARD};
            border: 1px solid {BORDER};
            border-radius: 24px;
            padding: 18px;
            box-shadow: 0 10px 32px rgba(0,0,0,.055);
            margin-bottom: 14px;
        }}

        .pill {{
            display: inline-flex;
            align-items: center;
            gap: 7px;
            padding: 7px 13px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 800;
            border: 1px solid {BORDER};
            background: {CARD_2};
            color: {TEXT};
        }}

        .temp-ring {{
            width: 132px;
            height: 132px;
            border-radius: 50%;
            display: grid;
            place-items: center;
            background: conic-gradient(
                var(--ring-color) calc(var(--temp) * 1%),
                rgba(127,136,122,.18) 0
            );
            box-shadow: inset 0 0 0 12px {CARD_2}, 0 10px 30px rgba(0,0,0,.08);
        }}

        .temp-ring-inner {{
            width: 96px;
            height: 96px;
            border-radius: 50%;
            background: {CARD};
            display: grid;
            place-items: center;
            border: 1px solid {BORDER};
        }}

        .temp-number {{
            font-size: 27px;
            font-weight: 950;
            line-height: 1;
        }}

        .temp-text {{
            color: {MUTED} !important;
            font-size: 11px;
            font-weight: 800;
            margin-top: 4px;
        }}

        .check-row {{
            margin: 10px 0;
        }}

        .check-top {{
            display: flex;
            justify-content: space-between;
            font-size: 12px;
            font-weight: 750;
            margin-bottom: 6px;
        }}

        .check-bar {{
            height: 8px;
            border-radius: 999px;
            background: rgba(127,136,122,.18);
            overflow: hidden;
        }}

        .check-fill {{
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, {PRIMARY}, {MINT});
        }}

        .tip-box {{
            background: rgba(112,167,123,.12);
            border: 1px solid rgba(112,167,123,.28);
            border-radius: 18px;
            padding: 14px;
        }}

        [data-testid="stProgress"] > div > div > div > div {{
            background: linear-gradient(90deg, {PRIMARY}, {MINT}) !important;
        }}

        .stButton > button {{
            border-radius: 16px !important;
            border: 1px solid {BORDER} !important;
            background: {CARD} !important;
            color: {TEXT} !important;
            min-height: 42px !important;
            font-weight: 650 !important;
            transition: all .18s ease !important;
        }}

        .stButton > button:hover {{
            border-color: {PRIMARY} !important;
            color: {PRIMARY} !important;
            transform: translateY(-1px);
            box-shadow: 0 12px 28px rgba(112,167,123,.20) !important;
        }}

        [data-testid="stFormSubmitButton"] > button[kind="primary"] {{
            background: #4F7C5A !important;
            border-color: #4F7C5A !important;
            color: #fff !important;
        }}

        @keyframes fadeUp {{
            from {{
                opacity: 0;
                transform: translateY(10px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# 체크인 UI helper
# -----------------------------------------------------------------------------
def badge(text: str, color: str, icon: str = "") -> str:
    icon_part = f"{icon} " if icon else ""
    return (
        f'<span class="pill" style="border-color:{color}55; '
        f'background:{color}18; color:{color} !important;">'
        f"{icon_part}{text}</span>"
    )


def temp_ring(temp: float) -> str:
    color, label, emoji = color_for_temp(temp)
    pct = max(0, min(100, temp))

    return (
        f'<div style="display:flex;align-items:center;gap:18px;">'
        f'<div style="width:132px;height:132px;border-radius:50%;display:grid;place-items:center;flex-shrink:0;'
        f'background:conic-gradient({color} {pct}%,rgba(127,136,122,.18) 0);'
        f'box-shadow:inset 0 0 0 12px {CARD_2},0 10px 30px rgba(0,0,0,.08);">'
        f'<div style="width:96px;height:96px;border-radius:50%;background:{CARD};'
        f'display:grid;place-items:center;border:1px solid {BORDER};">'
        f'<div style="text-align:center;">'
        f'<div style="font-size:27px;font-weight:950;line-height:1;color:{color};">{temp}°</div>'
        f'<div style="color:{MUTED};font-size:11px;font-weight:800;margin-top:4px;">{emoji} {label}</div>'
        f'</div></div></div>'
        f'<div>'
        f'<div style="font-size:13px;font-weight:900;color:{TEXT};margin-bottom:6px;">마음 온도</div>'
        f'<div style="font-size:12px;color:{MUTED};line-height:1.65;">'
        f'마음의 상태를<br>확인해보세요🌳</div>'
        f'</div></div>'
    )


def onboarding_hero() -> None:
    name = st.session_state.get("user_name", "")
    safe_name = _html.escape(name)
    title = f"{safe_name}님, 지금 마음 상태를<br>가볍게 확인해볼게요." if safe_name else "지금 마음 상태를<br>가볍게 확인해볼게요."
    st.markdown(
        f'<div class="hero app-shell">'
        f'<div class="hero-eyebrow">MIND TEMPERATURE CHECK-IN</div>'
        f'<div class="hero-title">{title}</div>'
        f'<p class="hero-sub">6개의 짧은 질문으로 초기 마음 온도를 계산합니다. 진단이 아니라 현재 상태를 이해하기 위한 체크인입니다.</p>'
        f'</div>',
        unsafe_allow_html=True,
    )


def check_bar(label: str, value: int, icon: str, color: str = PRIMARY) -> str:
    pct = max(0, min(100, value / 5 * 100))
    return (
        f'<div class="check-row">'
        f'<div class="check-top">'
        f'<span>{icon} {label}</span>'
        f'<span style="color:{color};">{value}/5</span>'
        f'</div>'
        f'<div class="check-bar">'
        f'<div class="check-fill" style="width:{pct}%;background:linear-gradient(90deg,{color},{color}99);"></div>'
        f'</div>'
        f'</div>'
    )


def recovery_tip(answers: dict[str, int]) -> str:
    sleep = answers.get("sleep", 3)
    energy = answers.get("energy", 3)
    recovery = answers.get("recovery", 3)
    stress = answers.get("stress", 3)
    fatigue = answers.get("fatigue", 3)

    # 스트레스·피로도는 높을수록 나쁜 상태 → 반전해서 동일 기준으로 비교
    scores = {
        "sleep": sleep,
        "energy": energy,
        "recovery": recovery,
        "stress": 6 - stress,
        "fatigue": 6 - fatigue,
    }
    weak = min(scores, key=scores.get)

    tips = {
        "sleep": (
            "🌙",
            "오늘은 잠을 먼저 챙겨봐요",
            "자기 전 30분만 화면을 멀리하고, 가벼운 스트레칭으로 몸을 낮춰보세요.",
        ),
        "energy": (
            "⚡",
            "에너지를 작게 회복해봐요",
            "큰 계획보다 10분 산책이나 물 한 컵처럼 바로 가능한 행동부터 시작해보세요.",
        ),
        "recovery": (
            "🌿",
            "회복 시간을 따로 떼어놔요",
            "쉬어도 쉬는 것 같지 않다면, 오늘은 할 일을 하나 줄이는 것도 회복 행동이에요.",
        ),
        "stress": (
            "🔥",
            "스트레스를 잠깐 내려놓아요",
            "지금 가장 머릿속을 채우는 걱정 하나를 종이에 써보세요. 쓰는 것만으로도 머리가 가벼워져요.",
        ),
        "fatigue": (
            "😴",
            "몸이 먼저 쉬어야 해요",
            "억지로 버티는 대신 5분이라도 눈을 감고 호흡에 집중해보세요. 짧은 휴식도 충분히 회복이에요.",
        ),
    }

    icon, title, body = tips[weak]

    return (
        f'<div class="tip-box">'
        f'<div style="font-weight:900;color:{PRIMARY};font-size:13px;margin-bottom:5px;">{icon} 오늘의 회복 팁</div>'
        f'<div style="font-weight:850;font-size:13px;margin-bottom:4px;">{title}</div>'
        f'<div style="color:{MUTED};font-size:12px;line-height:1.65;">{body}</div>'
        f'</div>'
    )


def get_personal_message(checkin_data: dict) -> str:
    stress = checkin_data.get("stress", 3)
    fatigue = checkin_data.get("fatigue", 3)
    mood = checkin_data.get("mood", 3)

    if stress >= 4 or fatigue >= 4:
        return "오늘 많이 지치셨군요. 잠깐 멈추고 스스로를 돌보는 시간을 가져보세요. 당신의 페이스로 가셔도 괜찮아요."
    if mood <= 2:
        return "기분이 좋지 않은 날에는 작은 위안이 큰 힘이 됩니다. 오늘 하루도 수고하셨어요."
    return "지금 이 순간 자신의 상태를 돌아보는 것만으로도 충분히 잘하고 있는 거예요."


def get_recovery_action(checkin_data: dict) -> str:
    sleep = checkin_data.get("sleep", 3)
    energy = checkin_data.get("energy", 3)
    recovery = checkin_data.get("recovery", 3)

    scores = {"sleep": sleep, "energy": energy, "recovery": recovery}
    weak = min(scores, key=scores.get)

    actions = {
        "sleep": "오늘 밤 취침 30분 전, 휴대폰을 내려놓고 조용한 음악이나 독서로 마무리해 보세요.",
        "energy": "10분 가볍게 걷거나 물 한 컵을 마시는 것처럼 작은 행동부터 시작해 보세요.",
        "recovery": "오늘 할 일 목록에서 한 가지를 지워보세요. 비우는 것도 회복입니다.",
    }

    return actions[weak]


# -----------------------------------------------------------------------------
# 체크인 요약 사이드바
# -----------------------------------------------------------------------------
def render_checkin_sidebar() -> None:
    render_checkin_styles()
    with st.sidebar:
        st.markdown(
            f"""
            <div class="soft-card">
              <div style="font-size:20px;font-weight:950;letter-spacing:-.04em;margin-bottom:4px;">
                🌱 마음온도 RAG
              </div>
              <div style="color:{MUTED};font-size:12px;line-height:1.55;">
                Check-in · Mind Temperature · Recovery
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(temp_ring(st.session_state.mind_temp), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        color, risk_text = risk_label(st.session_state.risk_level)
        risk_emoji = (
            "🌧️" if st.session_state.risk_level == RISK_HIGH
            else "🌤️" if st.session_state.risk_level == RISK_MID
            else "☀️"
        )
        st.markdown(
            f'<div class="soft-card">'
            f'<div style="color:{MUTED};font-size:12px;font-weight:800;margin-bottom:8px;">현재 상태</div>'
            f'{badge(risk_text, color, risk_emoji)}'
            f'</div>',
            unsafe_allow_html=True,
        )

        answers = st.session_state.get("answers", {})

        if answers:
            st.markdown(
                "<div class='soft-card'>"
                "<div style='font-size:13px;font-weight:900;margin-bottom:10px;'>체크인 요약</div>"
                + check_bar("수면", answers.get("sleep", 3), "🌙", BLUE)
                + check_bar("에너지", answers.get("energy", 3), "⚡", YELLOW)
                + check_bar("회복력", answers.get("recovery", 3), "🌿", PRIMARY)
                + check_bar("기분", answers.get("mood", 3), "😊", MINT)
                + check_bar("스트레스", 6 - answers.get("stress", 3), "🔥", CORAL)
                + check_bar("피로도", 6 - answers.get("fatigue", 3), "😴", CORAL)
                + "</div>",
                unsafe_allow_html=True,
            )

            st.markdown(recovery_tip(answers), unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 온보딩 — 이름 입력
# -----------------------------------------------------------------------------
def _render_name_input() -> None:
    st.markdown(
        '<div class="hero app-shell">'
        '<div class="hero-eyebrow">WELCOME</div>'
        '<div class="hero-title">어서오세요.</div>'
        '<p class="hero-sub">괜찮으시다면 제가 당신을 어떻게 부를지 알려주세요.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    with st.form("name_form", clear_on_submit=False, border=False):
        name = st.text_input(
            "닉네임",
            placeholder="이름 또는 닉네임을 입력해주세요",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("다음 →", type="primary", use_container_width=True)

    if submitted:
        if name.strip():
            st.session_state.user_name = name.strip()
            st.rerun()
        else:
            st.warning("이름을 입력해주세요.")

    st.caption("입력하신 이름은 이 세션 동안만 사용되며 저장되지 않습니다.")


# -----------------------------------------------------------------------------
# 온보딩 — 아이콘 선택
# -----------------------------------------------------------------------------
def _render_avatar_picker() -> None:
    safe_name = _html.escape(st.session_state.user_name)
    user_av = st.session_state.user_avatar
    asst_av = st.session_state.assistant_avatar

    st.markdown(
        f'<div class="hero app-shell">'
        f'<div class="hero-eyebrow">ICON SETUP</div>'
        f'<div class="hero-title">{safe_name}님,<br>아이콘을 골라주세요.</div>'
        f'<p class="hero-sub">채팅에서 사용할 나와 어시스턴트의 아이콘을 각각 선택해요.</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    for section, key, current in [
        ("내 아이콘", "user_avatar", user_av),
        ("어시스턴트 아이콘", "assistant_avatar", asst_av),
    ]:
        st.markdown(
            f'<div style="font-size:13px;font-weight:900;color:{TEXT};margin:16px 0 8px;">'
            f'{section} &nbsp;<span style="font-size:24px;">{current}</span></div>',
            unsafe_allow_html=True,
        )
        for row_start in range(0, len(AVATARS), 8):
            row = AVATARS[row_start:row_start + 8]
            cols = st.columns(len(row))
            for col, emoji in zip(cols, row):
                with col:
                    if st.button(emoji, key=f"{key}_{emoji}", use_container_width=True):
                        st.session_state[key] = emoji
                        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("체크인 시작하기 →", type="primary", use_container_width=True):
        st.session_state.avatar_selected = True
        st.rerun()


# -----------------------------------------------------------------------------
# 체크인 화면
# -----------------------------------------------------------------------------
def render_checkin(rag: MindCareRAGPipeline) -> None:
    render_checkin_sidebar()

    if not st.session_state.user_name:
        _render_name_input()
        return

    if not st.session_state.avatar_selected:
        _render_avatar_picker()
        return

    onboarding_hero()

    step = st.session_state.survey_step
    total = len(SURVEY)

    # 1) 설문 진행 화면
    if step < total:
        key, icon, question, _, scale = SURVEY[step]

        progress = step / total
        st.progress(progress, text=f"체크인 {step + 1} / {total}")

        st.markdown(
            f"""
            <div class="soft-card app-shell">
              <div style="font-size:44px;margin-bottom:8px;">{icon}</div>
              <div style="font-size:25px;font-weight:950;letter-spacing:-.045em;margin-bottom:8px;">
                {question}
              </div>
              <div style="color:{MUTED};font-size:13px;margin-bottom:18px;">
                가장 가까운 답을 하나 골라주세요.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        cols = st.columns([1, 1, 1, 1, 1])

        for idx, option in enumerate(scale):
            with cols[idx]:
                if st.button(
                    f"{idx + 1}\n\n{option}",
                    key=f"survey_{key}_{idx}",
                    use_container_width=True,
                ):
                    st.session_state.answers[key] = idx + 1
                    st.session_state.survey_step += 1
                    st.rerun()

        nav_l, nav_r = st.columns([1, 5])

        with nav_l:
            if step > 0 and st.button("← 이전", use_container_width=True):
                st.session_state.survey_step -= 1
                st.rerun()

        with nav_r:
            st.caption("응답은 현재 세션의 맞춤 답변을 위해 사용됩니다.")

        return

    # 2) 설문 완료 후 결과 계산
    filled = {
        key: st.session_state.answers.get(key, 3)
        for key, *_ in SURVEY
    }

    temp = _set_initial_temperature(
        rag,
        st.session_state.get("session_id", "default-user"),
        filled,
    )

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

    _, _, emoji = color_for_temp(temp)
    risk_color, risk_text = risk_label(risk)

    # 3) 결과 카드
    st.markdown(
        f"""
        <div class="hero app-shell" style="text-align:center;">
            <div class="hero-eyebrow">CHECK-IN RESULT</div>
            <div style="display:flex;justify-content:center;margin:12px 0 20px;">
                {temp_ring(temp)}
            </div>
            <div style="margin-bottom:12px;">
                {badge(risk_text, risk_color, emoji)}
            </div>
            <div style="font-size:25px;font-weight:950;letter-spacing:-.045em;margin-bottom:8px;">
                {result_title}
            </div>
            <p class="hero-sub" style="margin:0 auto;">
                {result_body}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 4) 채팅 진입
    if st.button("상담 시작하기 →", type="primary", use_container_width=True):
        st.session_state.mind_temp = temp
        st.session_state.risk_level = risk

        name = st.session_state.get("user_name", "")
        name_prefix = f"{name}님, " if name else ""
        greet = (
            f"{name_prefix}체크인 결과 마음 온도가 {temp}°로 나왔어요. "
            f"지금 많이 지쳐 계실 수 있으니, 가장 부담되는 부분부터 천천히 이야기해 주세요."
            if risk == RISK_HIGH
            else f"{name_prefix}마음 온도는 {temp}°예요. "
            f"조금 지쳐 있는 상태로 보여요. 오늘 가장 신경 쓰이는 것부터 가볍게 이야기해볼까요?"
            if risk == RISK_MID
            else f"{name_prefix}마음 온도는 {temp}°예요. "
            f"비교적 안정적인 상태네요. 번아웃 예방이나 회복에 대해 궁금한 점을 편하게 물어보세요."
        )

        st.session_state.messages = [{"role": "assistant", "content": greet}]
        st.session_state.stage = "chat"
        st.rerun()

    st.caption("본 서비스는 의료 행위가 아니며, 마음 온도는 진단이 아닌 참고용 지표입니다.")
