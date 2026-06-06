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

    core_answer = response.get("core_answer") or response.get("answer", "")
    caveat      = response.get("caveat", "")
    safety_note = response.get("safety_note", "")

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
