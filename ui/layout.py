from __future__ import annotations

import streamlit as st

from ui.components import render_page_header, render_safety_note
from ui.theme import get_theme


def render_main_header(theme: dict | None = None) -> None:
    if theme is None:
        theme = get_theme()
    render_page_header(theme)


def render_answer(response: dict, theme: dict | None = None) -> None:
    if theme is None:
        theme = get_theme()

    core_answer = (
        response.get("core_answer")
        or response.get("content")
        or response.get("answer")
        or ""
    )
    caveat      = response.get("caveat", "")
    safety_note = response.get("safety_note", "")

    # RAG ON/OFF 상태 뱃지 — answer_mode가 없는 과거 메시지는 뱃지를 생략한다.
    answer_mode = response.get("answer_mode")
    rag_enabled = response.get("rag_enabled")

    if answer_mode or rag_enabled is not None:
        if rag_enabled is False or answer_mode == "RAG OFF":
            badge_text = "RAG OFF · LLM 단독 답변"
        else:
            badge_text = "RAG ON · 문서 검색 기반 답변"

        badge_bg = theme.get("CARD_ALT") or theme.get("CARD", "transparent")
        st.markdown(
            f"<div style='display:inline-block;margin-bottom:0.45rem;"
            f"padding:0.22rem 0.55rem;border-radius:999px;"
            f"font-size:0.68rem;font-weight:800;"
            f"background:{badge_bg};"
            f"color:{theme['MUTED']};border:1px solid {theme['BORDER']};'>"
            f"{badge_text}</div>",
            unsafe_allow_html=True,
        )

    if core_answer:
        st.markdown(core_answer)

    if caveat:
        st.markdown(
            f"<p style='margin:0.6rem 0 0; font-size:0.72rem; color:{theme['MUTED']};"
            f"line-height:1.5; font-style:italic;'>{caveat}</p>",
            unsafe_allow_html=True,
        )

    if safety_note:
        render_safety_note(safety_note, theme)
