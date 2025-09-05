# ui_styles.py
import streamlit as st

# 전역으로 안전한 스타일만 (폰트/배경 등) — 사이드바 숨김 X
BASE_STYLE = """
<style>
/* 필요하면 전역 스타일 넣기 (사이드바 관련 코드는 금지) */
</style>
"""

# 숨김 스타일 (필요한 페이지에서만 호출)
HIDE_SIDEBAR_STYLE = """
<style>
[data-testid="stSidebar"] { display: none !important; }
[data-testid="stSidebarCollapsedControl"] { display: none !important; }
</style>
"""

def apply_base_style():
    st.markdown(BASE_STYLE, unsafe_allow_html=True)

def hide_sidebar():
    st.markdown(HIDE_SIDEBAR_STYLE, unsafe_allow_html=True)
