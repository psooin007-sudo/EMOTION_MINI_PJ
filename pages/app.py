import streamlit as st
import streamlit.components.v1 as components
import os

# --- 페이지 설정 (가장 먼저 실행되어야 함) ---
# 이 페이지의 제목과 아이콘을 설정하고, 레이아웃을 'wide'로 하여 게임 화면을 넓게 씁니다.
st.set_page_config(page_title="분노의 암초 부수기", page_icon="⛵", layout="wide")

# --- HTML 게임을 불러와 화면에 띄우는 메인 함수 ---
def show_game():
    """HTML 파일을 읽어 Streamlit 앱에 표시하는 함수"""
    
    # 이 파이썬 파일의 현재 위치를 기준으로 'game.html' 파일의 경로를 정확히 찾습니다.
    # 이렇게 하면 어디서 실행하든 경로 문제가 발생하지 않습니다.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    html_file_path = os.path.join(current_dir, 'anger_management_game.html')

    try:
        # UTF-8 인코딩으로 HTML 파일을 엽니다.
        with open(html_file_path, 'r', encoding='utf-8') as f:
            html_string = f.read()
            
    except FileNotFoundError:
        st.error("게임 파일('anger_management_game.html')을 찾을 수 없습니다! 파이썬 파일과 같은 폴더에 있는지 확인해주세요.")
        return

    # st.components.v1.html을 사용하여 HTML 콘텐츠를 앱 본문에 삽입합니다.
    # height 값을 넉넉하게 주어 게임 화면 내부에 스크롤이 생기지 않도록 합니다.
    components.html(html_string, height=950, scrolling=False)

# --- 스크립트 실행 ---
# 다른 페이지 로직 없이, 오직 게임만 실행합니다.
show_game()

# --- 힐링센터 홈으로 돌아가기 버튼 ---
import streamlit as _st
if _st.button('🏠 홈으로 돌아가기', key='go_hc_home', use_container_width=True):
    # healing_center.py에서 사용하는 상태키로 직접 이동 (멀티페이지/exec 환경 모두 대응)
    _st.session_state['current_page'] = 'home'
    _st.rerun()
