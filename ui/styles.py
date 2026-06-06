from __future__ import annotations

import streamlit as st

from ui.theme import get_theme


def inject_styles(theme: dict | None = None) -> None:
    if theme is None:
        theme = get_theme()

    css = f"""
        /* ── Reset & base ── */
        html, body, [class*="css"] {{
            font-family: 'Pretendard', 'Apple SD Gothic Neo', 'Noto Sans KR', sans-serif;
        }}

        /* ── App background ── */
        .stApp {{
            background: {theme['GRADIENT']};
            background-attachment: fixed;
        }}

        /* ── Main block ── */
        .block-container {{
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 780px;
        }}

        /* ── Sidebar ── */
        [data-testid="stSidebar"] {{
            background: {theme['CARD']} !important;
            border-right: 1px solid {theme['BORDER']};
        }}
        [data-testid="stSidebar"] > div:first-child {{
            padding: 1.5rem 1rem;
        }}

        /* ── Chat input ── */
        [data-testid="stChatInput"] textarea {{
            background: {theme['CARD']} !important;
            border: 1.5px solid {theme['BORDER']} !important;
            border-radius: 14px !important;
            color: {theme['TEXT']} !important;
            font-size: 0.95rem;
        }}
        [data-testid="stChatInput"] textarea:focus {{
            border-color: {theme['PRIMARY']} !important;
            box-shadow: 0 0 0 3px {theme['PRIMARY']}22 !important;
        }}

        /* ── Chat messages area ── */
        [data-testid="stChatMessageContainer"],
        section.main > div > div > div > div > div[data-testid="stVerticalBlock"] {{
            background: transparent;
        }}

        /* ── User chat bubble ── */
        [data-testid="stChatMessageContentUser"] {{
            background: linear-gradient(135deg, #FFFDF8 0%, #F5EFE2 100%) !important;
            border-radius: 18px 18px 4px 18px !important;
            padding: 0.75rem 1rem !important;
            border: 1px solid rgba(79,124,90,0.13) !important;
            box-shadow: 0 2px 12px rgba(79,124,90,0.08) !important;
        }}

        /* ── Assistant chat bubble ── */
        [data-testid="stChatMessageContentAssistant"] {{
            background: linear-gradient(135deg, #E4F2EA 0%, #D2EBD9 100%) !important;
            border-radius: 4px 18px 18px 18px !important;
            padding: 0.75rem 1rem !important;
            border: 1px solid rgba(79,124,90,0.15) !important;
            box-shadow: 0 2px 12px rgba(79,124,90,0.10) !important;
        }}

        /* ── Buttons ── */
        .stButton > button {{
            background: {theme['PRIMARY']} !important;
            color: #fff !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 0.45rem 1.2rem !important;
            font-weight: 600;
            transition: opacity 0.2s;
        }}
        .stButton > button:hover {{
            opacity: 0.88;
        }}

        /* ── Sliders ── */
        [data-testid="stSlider"] div[role="slider"] {{
            background: {theme['PRIMARY']} !important;
        }}

        /* ── Expander ── */
        [data-testid="stExpander"] {{
            background: {theme['CARD_ALT']} !important;
            border: 1px solid {theme['BORDER']} !important;
            border-radius: 12px !important;
        }}

        /* ── Divider ── */
        hr {{
            border-color: {theme['BORDER']} !important;
        }}

        /* ── Scrollbar ── */
        ::-webkit-scrollbar {{ width: 6px; }}
        ::-webkit-scrollbar-track {{ background: transparent; }}
        ::-webkit-scrollbar-thumb {{
            background: {theme['BORDER']};
            border-radius: 3px;
        }}

        /* ── 배경 부드러운 움직임 애니메이션 ── */
        @keyframes bgShift {{
            0%   {{ background-position: 0% 0%, 100% 100%, 0% 50%, 50% 0%, center; }}
            33%  {{ background-position: 30% 10%, 70% 90%, 20% 60%, 60% 20%, center; }}
            66%  {{ background-position: 10% 40%, 90% 60%, 40% 20%, 30% 70%, center; }}
            100% {{ background-position: 0% 0%, 100% 100%, 0% 50%, 50% 0%, center; }}
        }}

        .stApp {{
            animation: bgShift 20s ease-in-out infinite !important;
            background-size: 200% 200%, 180% 180%, 160% 160%, 150% 150%, cover !important;
        }}

        /* ── 채팅 메시지 영역에 유리 느낌 ── */
        [data-testid="stChatMessage"] {{
            backdrop-filter: blur(2px);
        }}

        /* ── 채팅 입력 영역 고정 시각 분리선 ── */
        [data-testid="stBottom"] {{
            background: linear-gradient(to top, rgba(246,241,232,0.95) 70%, transparent) !important;
            padding-top: 1rem !important;
        }}
    """

    st.markdown(
        f"<style>{css}</style>",
        unsafe_allow_html=True,
    )


def inject_sidebar_hidden_styles() -> None:
    """체크인 단계: 사이드바와 사이드바 토글 버튼을 완전히 숨김."""
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"],
        [data-testid="collapsedControl"] {
            display: none !important;
        }
        /* 사이드바 없을 때 본문 영역을 중앙 정렬로 확장 */
        .main > .block-container {
            margin-left: auto !important;
            margin-right: auto !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
