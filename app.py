from __future__ import annotations

import streamlit as st

from base import SelfRAGPipeline
from ui.checkin_insights import init_checkin_state, render_checkin, render_checkin_sidebar
from ui.components import render_safety_note
from ui.layout import render_answer, render_main_header
from ui.styles import inject_sidebar_hidden_styles, inject_styles
from ui.theme import get_theme

st.set_page_config(
    page_title="조용한 회복 공간",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

theme = get_theme()
inject_styles(theme)

if "rag" not in st.session_state:
    try:
        st.session_state.rag = SelfRAGPipeline()
        st.session_state.rag.build_vectorstore()
        st.session_state.rag.build_chain()
    except Exception as exc:
        st.error(
            f"초기화 오류: {exc}\n\n"
            "`data/.env` 파일에 `OPENAI_API_KEY`가 설정되어 있는지 확인하세요."
        )
        st.stop()

init_checkin_state()

SESSION_ID = "main"

# ── 스테이지 라우팅 ───────────────────────────────────────────────────────────
if st.session_state.stage != "chat":
    # 체크인 단계: 사이드바·채팅 입력·채팅 메시지 모두 숨김
    inject_sidebar_hidden_styles()
    render_checkin(st.session_state.rag)

else:
    # 체크인 완료 후: 사이드바 표시
    render_checkin_sidebar()

    # 채팅 히어로 표시 (체크인 완료 후에만)
    render_main_header(theme)

    # 채팅 히스토리 표시
    user_av = st.session_state.get("user_avatar", "🧑")
    asst_av = st.session_state.get("assistant_avatar", "🌿")

    for msg in st.session_state.messages:
        avatar = asst_av if msg["role"] == "assistant" else user_av
        with st.chat_message(msg["role"], avatar=avatar):
            if msg["role"] == "assistant":
                st.markdown(msg["content"])
                if msg.get("caveat"):
                    st.markdown(
                        f"<p style='margin:0.6rem 0 0; font-size:0.72rem;"
                        f" color:{theme['MUTED']}; line-height:1.5;"
                        f" font-style:italic;'>{msg['caveat']}</p>",
                        unsafe_allow_html=True,
                    )
                if msg.get("safety_note"):
                    render_safety_note(msg["safety_note"], theme)
            else:
                st.markdown(msg["content"])

    # 채팅 입력창 표시 (체크인 완료 후에만)
    user_input = st.chat_input("오늘 어떤 이야기를 나누고 싶으세요?")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar=user_av):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar=asst_av):
            with st.spinner("생각하는 중..."):
                checkin_answers = st.session_state.answers or None
                response = st.session_state.rag.ask(
                    question=user_input,
                    session_id=SESSION_ID,
                    checkin=checkin_answers,
                )

            render_answer(response, theme)

            st.session_state.mind_temp = response.get(
                "mind_temperature", st.session_state.mind_temp
            )
            st.session_state.risk_level = response.get("risk_level")
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response.get("core_answer") or response.get("answer", ""),
                    "caveat": response.get("caveat", ""),
                    "safety_note": response.get("safety_note", ""),
                }
            )

        st.rerun()
