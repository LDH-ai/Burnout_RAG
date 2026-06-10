from __future__ import annotations

import streamlit as st

from base import SelfRAGPipeline
from ui.checkin_insights import init_checkin_state, render_checkin, render_checkin_sidebar
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

# RAG ON/OFF 답변이 서로의 chat_history에 섞이지 않도록 세션을 분리한다.
RAG_SESSION_ID      = "main_rag"
LLM_ONLY_SESSION_ID = "main_llm_only"

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

    # 발표용 RAG ON/OFF 토글 — OFF면 retriever·context 없이 LLM 단독 답변
    rag_enabled = st.toggle(
        "RAG 사용",
        value=True,
        key="rag_enabled",
    )
    st.caption("RAG ON: 논문/문서 검색 기반 답변 · RAG OFF: LLM 단독 답변")

    # 채팅 히스토리 표시
    user_av = st.session_state.get("user_avatar", "🧑")
    asst_av = st.session_state.get("assistant_avatar", "🌿")

    for msg in st.session_state.messages:
        avatar = asst_av if msg["role"] == "assistant" else user_av
        with st.chat_message(msg["role"], avatar=avatar):
            if msg["role"] == "assistant":
                render_answer(msg, theme)
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
                if rag_enabled:
                    response = st.session_state.rag.ask(
                        question=user_input,
                        session_id=RAG_SESSION_ID,
                        checkin=checkin_answers,
                    )
                    response["answer_mode"] = "RAG ON"
                    response["rag_enabled"] = True
                else:
                    response = st.session_state.rag.ask_llm_only(
                        question=user_input,
                        session_id=LLM_ONLY_SESSION_ID,
                        checkin=checkin_answers,
                    )
                    response["answer_mode"] = "RAG OFF"
                    response["rag_enabled"] = False

            render_answer(response, theme)

            st.session_state.mind_temp = response.get(
                "mind_temperature", st.session_state.mind_temp
            )
            st.session_state.risk_level = response.get("risk_level")
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response.get("core_answer") or response.get("answer", ""),
                    "core_answer": response.get("core_answer", ""),
                    "answer_mode": response.get("answer_mode", "RAG ON" if rag_enabled else "RAG OFF"),
                    "rag_enabled": response.get("rag_enabled", rag_enabled),
                    "caveat": response.get("caveat", ""),
                    "safety_note": response.get("safety_note", ""),
                    "risk_level": response.get("risk_level"),
                    "mind_temperature": response.get("mind_temperature"),
                }
            )

        st.rerun()
