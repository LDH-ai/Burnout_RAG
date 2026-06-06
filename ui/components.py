from __future__ import annotations

import streamlit as st


def render_safety_note(note: str, theme: dict) -> None:
    if not note:
        return
    html = f"""
<div style="
    background:{theme['CORAL']}15;
    border-left:3px solid {theme['CORAL']};
    border-radius:0 10px 10px 0;
    padding:0.65rem 0.9rem;
    margin-top:0.5rem;
    font-size:0.83rem;
    color:{theme['TEXT']};
    line-height:1.55;
">
    {note}
</div>
"""
    st.markdown(html, unsafe_allow_html=True)


def render_page_header(theme: dict) -> None:
    html = f"""
<div style="margin-bottom:1.5rem;">
    <h1 style="
        margin:0 0 0.2rem;
        font-size:1.55rem;
        font-weight:800;
        color:{theme['TEXT']};
        letter-spacing:-0.02em;
    ">조용한 회복 공간 🌿</h1>
    <p style="
        margin:0;
        font-size:0.88rem;
        color:{theme['MUTED']};
        line-height:1.5;
    ">번아웃 회복을 위한 대화 공간입니다. 오늘의 상태를 체크인하고 대화를 시작해 보세요.</p>
</div>
"""
    st.markdown(html, unsafe_allow_html=True)
