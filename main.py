# === 스타일링 (기본 Streamlit 사이즈 + 사이드바 숨김) ===
APP1_STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;600;700&display=swap');

/* 사이드바 완전히 숨기기 */
[data-testid="stSidebar"],
[data-testid="stSidebarNav"],
[data-testid="stSidebarNavItems"], 
[data-testid="stSidebarNavSeparator"],
[data-testid="stSidebarUserContent"] {
    display: none !important;
}

/* 메인 영역을 전체 너비로 확장 */
.main .block-container {
    padding-left: 1rem !important;
    padding-right: 1rem !important;
    max-width: 100% !important;
}

/* 전체 배경 */
.stApp {
    background: #f8fafc;
    font-family: 'Noto Sans KR', system-ui, -apple-system, sans-serif;
}

html, body, [class*="css"] {
    color: #0f172a;
    font-weight: 400;
    line-height: 1.6;
}

/* 제목 */
.main-title {
    text-align: center;
    color: #020617;
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    letter-spacing: -0.025em;
}

.subtitle {
    text-align: center;
    color: #64748b;
    font-size: 1.125rem;
    margin-bottom: 3rem;
    font-weight: 400;
}

/* 버튼 스타일 */
.stButton > button {
    background: #ffffff;
    color: #334155;
    font-weight: 500;
    font-size: 0.95rem;
    padding: 1rem 1.5rem;
    border-radius: 0.75rem;
    border: 1px solid #e2e8f0;
    margin: 0.5rem 0;
    transition: all 0.2s ease-out;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    width: 100%;
    min-height: 3.5rem;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    font-family: inherit;
}

.stButton > button:hover {
    background: #f8fafc;
    border-color: #3b82f6;
    color: #1e40af;
    box-shadow: 0 4px 12px 0 rgba(59, 130, 246, 0.15);
    transform: translateY(-1px);
}

.stButton > button:active {
    transform: translateY(0);
}

/* Primary 버튼 */
.stButton > button[kind="primary"] {
    background: #3b82f6;
    color: white;
    border: 1px solid #3b82f6;
    font-weight: 600;
}

.stButton > button[kind="primary"]:hover {
    background: #2563eb;
    border-color: #2563eb;
    color: white;
    box-shadow: 0 8px 25px 0 rgba(59, 130, 246, 0.25);
}

/* Secondary 버튼 */
.stButton > button[kind="secondary"] {
    background: #f1f5f9;
    color: #475569;
    border: 1px solid #cbd5e1;
}

.stButton > button[kind="secondary"]:hover {
    background: #e2e8f0;
    border-color: #94a3b8;
}

/* 텍스트 영역 */
textarea {
    font-size: 0.95rem !important;
    padding: 1rem !important;
    border-radius: 0.75rem !important;
    border: 1px solid #d1d5db !important;
    background: #ffffff !important;
    transition: border-color 0.2s ease !important;
    line-height: 1.6 !important;
    resize: vertical !important;
    min-height: 8rem !important;
    font-family: inherit !important;
    color: #111827 !important;
}

textarea:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
    outline: none !important;
}

textarea::placeholder {
    color: #9ca3af !important;
}

/* 라디오 버튼 */
.stRadio > div {
    display: grid !important;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)) !important;
    gap: 1rem !important;
    margin: 1.5rem 0 !important;
}

.stRadio input[type="radio"] {
    display: none !important;
}

.stRadio label {
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    background: #ffffff !important;
    padding: 1.25rem !important;
    border-radius: 0.75rem !important;
    border: 1px solid #e5e7eb !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1) !important;
    color: #374151 !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    text-align: center !important;
    min-height: 5rem !important;
    gap: 0.5rem !important;
}

.stRadio label:hover {
    background: #f9fafb !important;
    border-color: #3b82f6 !important;
    box-shadow: 0 4px 12px 0 rgba(59, 130, 246, 0.1) !important;
}

.stRadio input[type="radio"]:checked + div label {
    background: #eff6ff !important;
    border-color: #3b82f6 !important;
    color: #1e40af !important;
    box-shadow: 0 4px 12px 0 rgba(59, 130, 246, 0.15) !important;
}

/* 섹션 스타일 */
.section {
    background: #ffffff;
    padding: 2rem;
    border-radius: 1rem;
    margin: 1.5rem 0;
    border: 1px solid #f1f5f9;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
}

.section-title {
    color: #0f172a;
    font-size: 1.25rem;
    font-weight: 600;
    margin-bottom: 1.5rem;
    text-align: center;
}

/* 상태 메시지 */
.element-container .stSuccess {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 0.75rem;
    padding: 1rem;
}

.element-container .stInfo {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 0.75rem;
    padding: 1rem;
}

.element-container .stWarning {
    background: #fffbeb;
    border: 1px solid #fed7aa;
    border-radius: 0.75rem;
    padding: 1rem;
}

.element-container .stError {
    background: #fef2f2;
    border: 1px solid #fecaca;
    border-radius: 0.75rem;
    padding: 1rem;
}

/* Progress bar */
.stProgress > div > div > div > div {
    background: #3b82f6;
    border-radius: 0.25rem;
}

/* Selectbox */
.stSelectbox > div > div > div {
    background: #ffffff;
    border: 1px solid #d1d5db;
    border-radius: 0.5rem;
}

/* Checkbox */
.stCheckbox > label {
    background: #ffffff;
    padding: 0.75rem 1rem;
    border-radius: 0.5rem;
    border: 1px solid #e5e7eb;
    transition: all 0.2s ease;
}

.stCheckbox > label:hover {
    background: #f9fafb;
    border-color: #3b82f6;
}

/* 반응형 */
@media (max-width: 768px) {
    .main-title { 
        font-size: 2rem; 
    }
    .stRadio > div { 
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)) !important;
    }
}

/* 스크롤바 */
::-webkit-scrollbar {
    width: 6px;
}

::-webkit-scrollbar-track {
    background: #f1f5f9;
}

::-webkit-scrollbar-thumb {
    background: #cbd5e1;
    border-radius: 3px;
}

::-webkit-scrollbar-thumb:hover {
    background: #94a3b8;
}
</style>
"""# integrated_main.py
import streamlit as st
import subprocess
import sys
import os
from datetime import datetime, timedelta
import json
import time
import base64
import gzip
import atexit
from transformers import pipeline

import streamlit as st
from ui_styles import apply_base_style, hide_sidebar

st.set_page_config(page_title="메인", initial_sidebar_state="collapsed")
apply_base_style()
hide_sidebar()   # ← 이 한 줄이 핵심



# 외부 라이브러리 자동 설치
import plotly.graph_objects as go
import pandas as pd
from plotly.subplots import make_subplots

from emotion_model import analyze_emotion_from_image, detect_face_and_analyze, get_latest_emotion, reset_emotion_state
import emotion_list

# === 한 런(run)에서 중복 버튼 렌더 방지용 가드 ===
_STOP_BTN_DRAWN = False

# 케어페이지 스크립트 경로 (파일명은 네가 실제로 실행할 파일명으로)
CARE_PAGE_PATH = os.path.join(os.path.dirname(__file__), "app1.py")

# 감정 데이터 (통합된 버전)
EMOTIONS = emotion_list.emotions

# app1.py의 감정 설정을 통합
EMOTION_CONFIG = {
    "슬픔": {"emoji": "😢", "desc": "마음이 아프고 우울할 때"},
    "화남": {"emoji": "😡", "desc": "분노와 짜증이 날 때"}, 
    "기쁨": {"emoji": "😊", "desc": "행복하고 즐거울 때"},
    "걱정": {"emoji": "😩", "desc": "걱정돼서 힘이들 때"},
    "불안/두려움": {"emoji": "😰", "desc": "불안하고 두려울 때"},
    "스트레스": {"emoji": "😵", "desc": "압박감과 피로를 느낄 때"},
    "복잡": {"emoji": "😵‍💫", "desc": "혼란스럽고 복잡할 때"},
    "모르겠음": {"emoji": "🤔", "desc": "감정이 애매하고 모호할 때"}
}

EMOTION_TO_PAGE = {
    "화남": "pages/anger_game.py",
    "슬픔": "pages/music.py",
    "기쁨": "pages/rr.py", 
    "걱정": "pages/breathing.py",
    "불안/두려움": "pages/anxiety_safety_check.py",
    "스트레스": "pages/game.py",
    "복잡": "pages/proverb_quiz_streamlit2.py",
    "모르겠음": "pages/app_streamlit.py"
}

# === 스타일링 (app1.py 기반) ===
APP1_STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;600;700&display=swap');

/* 전체 배경 - 밝고 깔끔하게 */
.stApp {
    background: linear-gradient(135deg, #f0f8ff 0%, #e6f3ff 100%);
    min-height: 100vh;
}

html, body, [class*="css"] {
    font-family: 'Noto Sans KR', sans-serif;
    color: #2c3e50;
}

/* 메인 컨테이너 */
.main-container {
    background: #ffffff;
    padding: 40px;
    border-radius: 20px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.05);
    margin: 20px auto;
    max-width: 900px;
    border: 1px solid #e1e8f0;
}

/* 제목 */
.main-title {
    text-align: center;
    color: #2c3e50;
    font-size: 2.8em;
    font-weight: 700;
    margin-bottom: 10px;
}

.subtitle {
    text-align: center;
    color: #64748b;
    font-size: 1.1em;
    margin-bottom: 40px;
    font-weight: 400;
}

/* 버튼 스타일 - 통일된 크기 */
.stButton > button {
    background: #ffffff;
    color: #2c3e50;
    font-weight: 600;
    font-size: 16px;
    padding: 20px 25px;
    border-radius: 12px;
    border: 2px solid #e1e8f0;
    margin: 10px 0;
    transition: all 0.2s ease;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    width: 100%;
    min-height: 65px;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    line-height: 1.4;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
    background: #f8fafc;
    border-color: #4facfe;
}

/* 힐링센터 버튼 */
.healing-btn {
    background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%) !important;
    color: #2c3e50 !important;
    border-color: #ff9a9e !important;
    font-weight: 700 !important;
    min-height: 70px !important;
    margin-top: 30px !important;
}

.healing-btn:hover {
    background: linear-gradient(135deg, #fecfef 0%, #ff9a9e 100%) !important;
    box-shadow: 0 10px 25px rgba(255, 154, 158, 0.3) !important;
}

/* 텍스트 영역 */
textarea {
    font-size: 15px !important;
    padding: 20px !important;
    border-radius: 12px !important;
    border: 2px solid #e1e8f0 !important;
    background: #ffffff !important;
    transition: all 0.2s ease !important;
    line-height: 1.5 !important;
    resize: none !important;
    min-height: 180px !important;
}

textarea:focus {
    border-color: #4facfe !important;
    box-shadow: 0 0 0 3px rgba(79, 172, 254, 0.1) !important;
    outline: none !important;
}

/* 라디오 버튼 - 4열 2행 배치 */
.stRadio > div {
    display: grid !important;
    grid-template-columns: repeat(4, 1fr) !important;
    grid-template-rows: repeat(2, 1fr) !important;
    gap: 15px !important;
    margin: 20px 0 !important;
    max-width: 800px !important;
    margin-left: auto !important;
    margin-right: auto !important;
}

.stRadio input[type="radio"] {
    display: none !important;
}

.stRadio label {
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    background: #ffffff !important;
    padding: 20px 10px !important;
    border-radius: 12px !important;
    border: 2px solid #e1e8f0 !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
    box-shadow: 0 3px 10px rgba(0, 0, 0, 0.05) !important;
    color: #2c3e50 !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    text-align: center !important;
    min-height: 85px !important;
    gap: 5px !important;
}

.stRadio label:hover {
    background: #f8fafc !important;
    border-color: #4facfe !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 15px rgba(0, 0, 0, 0.08) !important;
}

.stRadio input[type="radio"]:checked + div label {
    background: linear-gradient(135deg, #4facfe 0%, #00d2ff 100%) !important;
    color: white !important;
    border-color: #4facfe !important;
    box-shadow: 0 8px 20px rgba(79, 172, 254, 0.2) !important;
}

/* 섹션 스타일 */
.section {
    background: #f8fafc;
    padding: 30px;
    border-radius: 15px;
    margin: 25px 0;
    border: 1px solid #e1e8f0;
}

.section-title {
    color: #2c3e50;
    font-size: 1.4em;
    font-weight: 600;
    margin-bottom: 20px;
    text-align: center;
}

/* 상태 메시지 */
.element-container .stSuccess {
    background: rgba(34, 197, 94, 0.08);
    border: 1px solid rgba(34, 197, 94, 0.2);
    border-radius: 12px;
}

.element-container .stInfo {
    background: rgba(59, 130, 246, 0.08);
    border: 1px solid rgba(59, 130, 246, 0.2);
    border-radius: 12px;
}

.element-container .stWarning {
    background: rgba(245, 158, 11, 0.08);
    border: 1px solid rgba(245, 158, 11, 0.2);
    border-radius: 12px;
}

/* 반응형 */
@media (max-width: 768px) {
    .main-title { font-size: 2.2em; }
    .main-container { margin: 15px; padding: 25px; }
    .stRadio > div { 
        grid-template-columns: repeat(2, 1fr) !important;
        grid-template-rows: repeat(4, 1fr) !important;
        max-width: 400px !important;
    }
}
</style>
"""

# 페이지 설정
st.set_page_config(
    page_title="감정 분석 시스템",
    page_icon="🎭",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 스타일 적용
st.markdown(APP1_STYLE, unsafe_allow_html=True)

# === 세션 상태 초기화 ===
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'main'

if 'webcam_process' not in st.session_state:
    st.session_state.webcam_process = None

if 'emotion_history' not in st.session_state:
    st.session_state.emotion_history = []
    
if 'was_webcam_running' not in st.session_state:
    st.session_state.was_webcam_running = False

if 'page_mode' not in st.session_state:
    st.session_state.page_mode = 'main'

# === AI 모델 로드 (app1.py에서 가져옴) ===
@st.cache_resource(show_spinner="AI 모델 준비 중...")
def load_emotion_model():
    try:
        classifier = pipeline(
            "sentiment-analysis", 
            model="beomi/KcELECTRA-base-v2022",
            return_all_scores=True
        )
        return classifier
    except Exception as e:
        st.error(f"모델 로드 실패: {str(e)}")
        return None

# === 감정 분석 함수 (app1.py에서 가져옴) ===
def analyze_emotion(text):
    classifier = load_emotion_model()
    if not classifier:
        return "복잡", 75.0
    
    try:
        emotion_keywords = {
            "화남": ["화", "짜증", "분노", "열받", "빡치", "억울", "답답", "미치겠", "화나"],
            "슬픔": ["슬프", "우울", "눈물", "상처", "외로", "허전", "멘탈", "아파", "서러"],
            "기쁨": ["기쁜", "행복", "좋", "즐거", "웃", "신나", "만족", "뿌듯", "최고"],
            "스트레스": ["스트레스", "피곤", "힘들", "지쳐", "번아웃", "압박", "부담"],
            "걱정": ["걱정", "긴장"],
            "불안/두려움": ["불안", "두려", "무서", "겁나", "초조"],
            "복잡": ["복잡", "혼란", "갈등", "고민", "애매", "헷갈", "어렵"],
            "모르겠음": ["모르", "글쎄", "잘모르", "확실하지", "별로", "그냥", "음"]
        }
        
        text_lower = text.lower()
        scores = {}
        
        for emotion, keywords in emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                scores[emotion] = score
        
        if scores:
            emotion = max(scores, key=scores.get)
            confidence = min(65 + scores[emotion] * 10, 90)
        else:
            result = classifier(text)
            emotion = "기쁨" if result[0]['label'] == 'POSITIVE' else "복잡"
            confidence = max(result[0]['score'] * 80, 70)
        
        return emotion, confidence
        
    except:
        return "복잡", 75.0

# === 정리 함수들 (기존 main.py에서) ===
def cleanup_processes():
    """앱 종료 시 웹캠 프로세스 정리 및 JSON 파일 삭제"""
    # 웹캠 프로세스 종료
    if 'webcam_process' in st.session_state and st.session_state.webcam_process:
        try:
            st.session_state.webcam_process.terminate()
            print("✅ 웹캠 프로세스가 종료되었습니다.")
        except:
            pass
    
    # emotion_history.json 파일 삭제
    try:
        if os.path.exists('emotion_history.json'):
            os.remove('emotion_history.json')
            print("✅ emotion_history.json 파일이 삭제되었습니다.")
    except Exception as e:
        print(f"❌ JSON 파일 삭제 실패: {e}")
    
    # latest_emotion_result.json 파일 삭제
    try:
        if os.path.exists('latest_emotion_result.json'):
            os.remove('latest_emotion_result.json')
            print("✅ latest_emotion_result.json 파일이 삭제되었습니다.")
    except Exception as e:
        print(f"❌ latest_emotion_result.json 파일 삭제 실패: {e}")

def shutdown_app():
    """앱 수동 종료 함수"""
    st.success("🔄 프로그램을 종료합니다...")
    cleanup_processes()
    st.info("✅ 정리 작업이 완료되었습니다. 브라우저 탭을 닫아주세요.")
    st.stop()

# 자동 정리 등록
atexit.register(cleanup_processes)

# === 웹캠 프로세스 관리 함수들 (기존 main.py에서) ===
def start_webcam_process():
    """web.py를 별도 프로세스로 실행"""
    try:
        python_exe = sys.executable
        web_py_path = os.path.join(os.path.dirname(__file__), 'webcam.py')
        
        process = subprocess.Popen([python_exe, web_py_path])
        return process
    except Exception as e:
        st.error(f"웹캠 프로세스 시작 실패: {e}")
        return None

def stop_webcam_process():
    """웹캠 프로세스 종료"""
    if st.session_state.webcam_process:
        try:
            st.session_state.webcam_process.terminate()
            st.session_state.webcam_process.wait(timeout=3)
            st.session_state.webcam_process = None
            return True
        except:
            try:
                st.session_state.webcam_process.kill()
                st.session_state.webcam_process = None
                return True
            except:
                return False
    return True

def is_webcam_running():
    """웹캠 프로세스가 실행 중인지 확인"""
    if st.session_state.webcam_process:
        return st.session_state.webcam_process.poll() is None
    return False

# === 차트 생성 함수들 (웹캠 분석 결과용) ===
def create_emotion_gauge(score, color):
    """감정 신뢰도 게이지 차트"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "신뢰도 (%)"},
        delta={'reference': 80},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "gray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75, 
                'value': 90
            }
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig

# def create_enhanced_timeline_chart(history_data, minutes=30):
#     """향상된 감정 변화 추이 차트"""
#     if not history_data:
#         return None
        
#     # 최근 N분간 데이터 필터링
#     cutoff_time = datetime.now() - timedelta(minutes=minutes)
#     recent_data = [h for h in history_data if h['timestamp'] > cutoff_time]
    
#     if len(recent_data) < 1:
#         return None
    
#     # 데이터 준비
#     df = pd.DataFrame([
#         {
#             'time': entry['timestamp'].strftime('%H:%M:%S'),
#             'timestamp': entry['timestamp'],
#             'emotion': EMOTIONS.get(entry['emotion'], {'korean': entry['emotion']})['korean'],
#             'emotion_en': entry['emotion'],
#             'score': entry['score'] * 100,
#             'color': EMOTIONS.get(entry['emotion'], {'color': '#808080'})['color'],
#             'emoji': EMOTIONS.get(entry['emotion'], {'emoji': '🤔'})['emoji']
#         }
#         for entry in recent_data
#     ])
    
#     # 라인 차트 생성
#     fig = go.Figure()
    
#     # 전체 감정 변화 라인
#     fig.add_trace(go.Scatter(
#         x=df['timestamp'],
#         y=df['score'],
#         mode='lines+markers',
#         name='감정 변화',
#         line=dict(color='#1f77b4', width=3),
#         marker=dict(size=8),
#         hovertemplate="<b>%{text}</b><br>" +
#                      "시간: %{x|%H:%M:%S}<br>" +
#                      "신뢰도: %{y:.1f}%<extra></extra>",
#         text=[f"{row['emoji']} {row['emotion']}" for _, row in df.iterrows()]
#     ))
    
#     # 감정별로 색상이 다른 점들 추가
#     for emotion in df['emotion_en'].unique():
#         emotion_data = df[df['emotion_en'] == emotion]
#         if not emotion_data.empty:
#             emotion_info = EMOTIONS.get(emotion, {'korean': emotion, 'color': '#808080', 'emoji': '🤔'})
#             fig.add_trace(go.Scatter(
#                 x=emotion_data['timestamp'],
#                 y=emotion_data['score'],
#                 mode='markers',
#                 marker=dict(
#                     size=15,
#                     color=emotion_info['color'],
#                     symbol='circle',
#                     line=dict(width=2, color='white')
#                 ),
#                 name=f"{emotion_info['emoji']} {emotion_info['korean']}",
#                 hovertemplate=f"<b>{emotion_info['emoji']} {emotion_info['korean']}</b><br>" +
#                              "시간: %{x|%H:%M:%S}<br>" +
#                              "신뢰도: %{y:.1f}%<extra></extra>",
#                 showlegend=True
#             ))
    
#     fig.update_layout(
#         title=f"감정 변화 추이 (최근 {minutes}분)",
#         xaxis_title="시간",
#         yaxis_title="감정",
#         height=500,
#         margin=dict(l=20, r=20, t=60, b=20),
#         hovermode='x unified',
#         legend=dict(
#             orientation="h",
#             yanchor="bottom",
#             y=1.02,
#             xanchor="right",
#             x=1
#         ),
#         yaxis=dict(range=[0, 100])
#     )
    
#     return fig


def create_enhanced_timeline_chart(history_data, minutes=30):
    """향상된 감정 변화 추이 차트"""
    if not history_data:
        return None
     
    # 최근 N분간 데이터 필터링
    cutoff_time = datetime.now() - timedelta(minutes=minutes)
    recent_data = [h for h in history_data if h['timestamp'] > cutoff_time]
     
    if len(recent_data) < 1:
        return None
     
    # 감정별 y축 위치 매핑 (감정을 숫자로 매핑)
    emotion_positions = {}
    emotion_labels = []
    
    # 모든 감정들을 수집하고 정렬
    all_emotions = list(set(entry['emotion'] for entry in recent_data))
    all_emotions.sort()  # 알파벳 순으로 정렬
    
    for i, emotion in enumerate(all_emotions):
        emotion_positions[emotion] = i
        emotion_info = EMOTIONS.get(emotion, {'korean': emotion, 'emoji': '🤔'})
        emotion_labels.append(f"{emotion_info['emoji']} {emotion_info['korean']}")
     
    # 데이터 준비
    df = pd.DataFrame([
        {
            'time': entry['timestamp'].strftime('%H:%M:%S'),
            'timestamp': entry['timestamp'],
            'emotion': EMOTIONS.get(entry['emotion'], {'korean': entry['emotion']})['korean'],
            'emotion_en': entry['emotion'],
            'y_position': emotion_positions[entry['emotion']],  # 감정별 y축 위치
            'score': entry['score'] * 100,
            'color': EMOTIONS.get(entry['emotion'], {'color': '#808080'})['color'],
            'emoji': EMOTIONS.get(entry['emotion'], {'emoji': '🤔'})['emoji']
        }
        for entry in recent_data
    ])
     
    # 라인 차트 생성
    fig = go.Figure()
     
    # 전체 감정 변화 라인 (이제 y축이 감정 위치)
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['y_position'],
        mode='lines+markers',
        name='감정 변화',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8),
        hovertemplate="<b>%{text}</b><br>" +
                     "시간: %{x|%H:%M:%S}<br>" +
                     "신뢰도: %{customdata:.1f}%<extra></extra>",
        text=[f"{row['emoji']} {row['emotion']}" for _, row in df.iterrows()],
        customdata=df['score']  # 신뢰도 정보를 customdata로 전달
    ))
     
    # 감정별로 색상이 다른 점들 추가
    for emotion in df['emotion_en'].unique():
        emotion_data = df[df['emotion_en'] == emotion]
        if not emotion_data.empty:
            emotion_info = EMOTIONS.get(emotion, {'korean': emotion, 'color': '#808080', 'emoji': '🤔'})
            fig.add_trace(go.Scatter(
                x=emotion_data['timestamp'],
                y=emotion_data['y_position'],
                mode='markers',
                marker=dict(
                    size=15,
                    color=emotion_info['color'],
                    symbol='circle',
                    line=dict(width=2, color='white')
                ),
                name=f"{emotion_info['emoji']} {emotion_info['korean']}",
                hovertemplate=f"<b>{emotion_info['emoji']} {emotion_info['korean']}</b><br>" +
                             "시간: %{x|%H:%M:%S}<br>" +
                             "신뢰도: %{customdata:.1f}%<extra></extra>",
                customdata=emotion_data['score'],
                showlegend=True
            ))
     
    fig.update_layout(
        title=f"감정 변화 추이 (최근 {minutes}분)",
        xaxis_title="시간",
        yaxis_title="감정",
        height=500,
        margin=dict(l=20, r=20, t=60, b=20),
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(emotion_labels))),
            ticktext=emotion_labels,
            range=[-0.5, len(emotion_labels) - 0.5]  # 감정 라벨이 잘 보이도록 여백 추가
        )
    )
     
    return fig


def create_emotion_distribution_chart(history_data, minutes=30):
    """감정 분포 파이 차트"""
    if not history_data:
        return None
        
    cutoff_time = datetime.now() - timedelta(minutes=minutes)
    recent_data = [h for h in history_data if h['timestamp'] > cutoff_time]
    
    if not recent_data:
        return None
    
    # 감정별 카운트
    emotion_counts = {}
    for entry in recent_data:
        emotion = entry['emotion']
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    
    # 데이터 준비
    emotions = list(emotion_counts.keys())
    counts = list(emotion_counts.values())
    colors = [EMOTIONS.get(e, {'color': '#808080'})['color'] for e in emotions]
    labels = [f"{EMOTIONS.get(e, {'emoji': '🤔', 'korean': e})['emoji']} {EMOTIONS.get(e, {'korean': e})['korean']}" for e in emotions]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=counts,
        marker_colors=colors,
        textinfo='label+percent',
        textposition='auto',
        hovertemplate="<b>%{label}</b><br>" +
                     "횟수: %{value}<br>" +
                     "비율: %{percent}<extra></extra>"
    )])
    
    fig.update_layout(
        title=f"감정 분포 (최근 {minutes}분)",
        height=400,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

def create_emotion_stats_table(history_data, minutes=30):
    """감정 통계 테이블"""
    if not history_data:
        return None
        
    cutoff_time = datetime.now() - timedelta(minutes=minutes)
    recent_data = [h for h in history_data if h['timestamp'] > cutoff_time]
    
    if not recent_data:
        return None
    
    # 통계 계산
    emotion_stats = {}
    for entry in recent_data:
        emotion = entry['emotion']
        score = entry['score']
        
        if emotion not in emotion_stats:
            emotion_stats[emotion] = {
                'count': 0,
                'scores': [],
                'total_score': 0
            }
        
        emotion_stats[emotion]['count'] += 1
        emotion_stats[emotion]['scores'].append(score)
        emotion_stats[emotion]['total_score'] += score
    
    # 테이블 데이터 생성
    table_data = []
    for emotion, stats in emotion_stats.items():
        emotion_info = EMOTIONS.get(emotion, {'emoji': '🤔', 'korean': emotion})
        avg_score = stats['total_score'] / stats['count'] if stats['count'] > 0 else 0
        max_score = max(stats['scores']) if stats['scores'] else 0
        min_score = min(stats['scores']) if stats['scores'] else 0
        
        table_data.append({
            '감정': f"{emotion_info['emoji']} {emotion_info['korean']}",
            '횟수': stats['count'],
            '평균 신뢰도': f"{avg_score*100:.1f}%",
            '최고 신뢰도': f"{max_score*100:.1f}%",
            '최저 신뢰도': f"{min_score*100:.1f}%"
        })
    
    # 횟수순으로 정렬
    table_data.sort(key=lambda x: x['횟수'], reverse=True)
    
    df = pd.DataFrame(table_data)
    return df

# === 메인 페이지 함수 ===
def show_main_page():
    """메인 선택 페이지"""
    st.markdown('<h1 class="main-title">🐬 나의 감정 항해일지</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">하루하루를 "감정의 바다"를 항해하는 여정을 떠나고 있는<br>당신이 길을 잃지 않도록 돕는 "나침반"이자 "등대"가 되어드리겠습니다.</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # 선택 버튼들을 세 개의 컬럼으로 배치
    col1 = st.columns(1)[0]

    with col1:
        st.markdown("#### ⛵ 여정을 떠나기 전, 당신의 감정을 체크해 보겠습니다.")
        st.write("웹캠을 통해 실시간으로 감정을 분석합니다. 너무 부담갖지 않으셔도 됩니다.")
        
        if st.button("🎥 웹캠으로 분석하기", use_container_width=True):
            st.session_state.start_webcam_requested = True
            st.session_state.current_page = 'webcam'
            st.rerun()

def show_webcam_page():
    """웹캠 전용 페이지"""
    st.title("🔹 실시간 웹캠 분석 (전용 페이지)")
    st.markdown("---")

    # 뒤로가기
    cols = st.columns([1,1,1])
    with cols[0]:
        if st.button("🔙 메인으로", use_container_width=True, key="web_nav_main"):
            st.session_state.current_page = 'main'
            st.rerun()

def show_webcam_page():
    """웹캠 전용 페이지"""
    st.title("🎥 실시간 웹캠 감정 분석")
    st.markdown("---")

    # 뒤로가기 버튼
    col1 = st.columns(1)[0]
    with col1:
        if st.button("🔙 메인으로", use_container_width=True):
            st.session_state.current_page = 'main'
            st.rerun()

    # 웹캠 상태 확인 및 시작
    webcam_running = is_webcam_running()
    
    if not webcam_running:
        if st.session_state.get("start_webcam_requested", False):
            st.info("웹캠을 시작하고 있습니다...")
            st.session_state.webcam_process = start_webcam_process()
            st.session_state.start_webcam_requested = False
            if is_webcam_running():
                st.success("웹캠이 성공적으로 시작되었습니다!")
                st.rerun()
            else:
                st.error("웹캠 시작에 실패했습니다.")
        else:
            st.warning("웹캠이 실행되지 않았습니다.")
            if st.button("🎥 웹캠 시작하기", type="primary", use_container_width=True):
                st.session_state.webcam_process = start_webcam_process()
                if is_webcam_running():
                    st.success("웹캠이 시작되었습니다!")
                    st.rerun()
                else:
                    st.error("웹캠 시작 실패")
    else:
        st.success("🟢 웹캠이 실행 중입니다!")
        
        st.markdown("---")
        st.subheader("🧭 안내 / 상태")
        st.info(
            "• 이 페이지는 웹캠 실행 전용입니다.\n"
            "• 표정을 지어보세요. 분석 결과는 기록으로 저장되고, 웹캠 종료 시 대시보드에서 확인할 수 있어요.\n"
            "• 아래 '웹캠 종료' 버튼을 누르면 자동으로 대시보드로 이동합니다."
        )
        
        # 웹캠 종료 버튼
        if st.button("🛑 웹캠 종료하기", use_container_width=True, type="secondary"):
            if stop_webcam_process():
                st.success("웹캠이 종료되었습니다.")
                time.sleep(1)
                st.success("나의 감정 항해일지로 이동합니다!")
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.02)
                    progress_bar.progress(i + 1)
                st.switch_page("pages\healing_center.py")
            else:
                st.error("웹캠 종료 실패")
    
    # 자동으로 웹캠이 종료된 경우 감지
    if not is_webcam_running() and st.session_state.get("was_webcam_running", False):
        st.info("웹캠이 종료되었습니다. 대시보드로 이동합니다...")
        st.session_state.current_page = 'analytics'
        st.rerun()
    
    # 웹캠 상태를 세션에 기록
    st.session_state.was_webcam_running = is_webcam_running()
    
    # 10초마다 자동 새로고침
    time.sleep(10)
    st.rerun()

# === 유틸리티 함수들 (기존 main.py에서 필요한 것들) ===
def safe_get_query_param(param_name, default_value):
    """안전한 쿼리 파라미터 추출"""
    try:
        if hasattr(st, 'query_params'):
            if param_name in st.query_params:
                return st.query_params[param_name]
        elif hasattr(st, 'experimental_get_query_params'):
            params = st.experimental_get_query_params()
            if param_name in params:
                return params[param_name][0]
        return default_value
    except Exception as e:
        st.error(f"쿼리 파라미터 읽기 오류: {e}")
        return default_value

def load_url_history_data():
    """URL에서 압축된 히스토리 데이터 복원"""
    try:
        hist_param = safe_get_query_param('hist', None)
        if not hist_param:
            return []
        
        print(f"📊 URL에서 히스토리 데이터 복원 중... (길이: {len(hist_param)})")
        
        # base64 디코딩 → gzip 압축 해제 → JSON 파싱
        compressed_data = base64.b64decode(hist_param.encode('utf-8'))
        json_str = gzip.decompress(compressed_data).decode('utf-8')
        compact_data = json.loads(json_str)
        
        # 압축된 형식을 원래 형식으로 복원
        restored_history = []
        for item in compact_data:
            try:
                timestamp = datetime.fromtimestamp(item['t'])
                restored_history.append({
                    'emotion': item['e'],
                    'score': float(item['s']),
                    'timestamp': timestamp,
                    'datetime': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    'raw_emotion': item['e']
                })
            except (KeyError, ValueError, OSError) as e:
                print(f"⚠️ 데이터 복원 중 오류: {e}")
                continue
        
        print(f"✅ {len(restored_history)}개 히스토리 복원 완료")
        return restored_history
        
    except Exception as e:
        print(f"❌ URL 히스토리 복원 실패: {e}")
        return []
    

def load_local_emotion_history():
    """로컬 감정 히스토리 로드 (파일에서)"""
    try:
        if os.path.exists('emotion_history.json'):
            with open('emotion_history.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
                # timestamp 문자열을 datetime 객체로 변환
                for item in data:
                    if isinstance(item.get('timestamp'), (int, float)):
                        item['timestamp'] = datetime.fromtimestamp(item['timestamp'])
                    elif isinstance(item.get('timestamp'), str):
                        try:
                            item['timestamp'] = datetime.fromisoformat(item['timestamp'])
                        except:
                            item['timestamp'] = datetime.strptime(item['timestamp'], '%Y-%m-%d %H:%M:%S')
                return data
    except Exception as e:
        print(f"로컬 히스토리 로드 실패: {e}")
    return []

def load_all_emotion_data():
    """모든 소스에서 감정 데이터 로드"""
    all_history = []
    
    # 1. URL에서 압축된 히스토리 데이터 로드
    url_history = load_url_history_data()
    if url_history:
        all_history.extend(url_history)
    
    # 2. 로컬 파일에서 로드
    local_history = load_local_emotion_history()
    if local_history:
        all_history.extend(local_history)
    
    # 3. 세션 상태에서 로드
    session_history = st.session_state.get('emotion_history', [])
    if session_history:
        all_history.extend(session_history)
    
    # 4. 중복 제거 (타임스탬프 기준)
    seen_timestamps = set()
    unique_history = []
    
    for entry in sorted(all_history, key=lambda x: x['timestamp']):
        timestamp_key = entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        if timestamp_key not in seen_timestamps:
            seen_timestamps.add(timestamp_key)
            unique_history.append(entry)
    
    return unique_history

def show_analytics_page():
    """고급 분석 대시보드 페이지 - 웹캠 결과 포함"""
    st.title("📊 감정 분석 대시보드")
    st.markdown("---")
    
    # 뒤로가기 버튼
    if st.button("🔙 메인으로 돌아가기"):
        st.session_state.current_page = 'main'
        st.rerun()
    
    # 모든 감정 데이터 로드
    all_history = load_all_emotion_data()
    
    if not all_history:
        st.warning("🔭 분석할 감정 데이터가 없습니다.")
        st.info("먼저 웹캠 프로그램을 실행하거나 수동으로 감정을 선택해주세요!")
        return
    
    st.success(f"✅ 총 {len(all_history)}개의 감정 기록이 로드되었습니다.")
    
    # 사이드바 설정
    st.sidebar.header("📈 분석 설정")
    
    # 시간 범위 선택
    time_options = {
        "최근 10분": 10,
        "최근 30분": 30,
        "최근 1시간": 60,
        "최근 2시간": 120,
        "최근 6시간": 360,
        "최근 12시간": 720,
        "최근 24시간": 1440,
        "전체": 99999
    }
    
    selected_time = st.sidebar.selectbox(
        "분석 시간 범위",
        list(time_options.keys()),
        index=1
    )
    
    minutes = time_options[selected_time]
    
    # 실시간 새로고침 옵션
    auto_refresh = st.sidebar.checkbox("자동 새로고침 (30초)", value=False)
    if auto_refresh:
        time.sleep(30)
        st.rerun()
    
    # 수동 새로고침 버튼
    if st.sidebar.button("🔄 데이터 새로고침"):
        st.rerun()
    # 메인 대시보드 - 세로 배치로 변경
    st.subheader(f"📈 감정 변화 추이")
    timeline_chart = create_enhanced_timeline_chart(all_history, minutes)
    if timeline_chart:
        st.plotly_chart(timeline_chart, use_container_width=True)
    else:
        st.info("해당 시간 범위에 데이터가 없습니다.")

    st.subheader(f"🥧 감정 분포")
    distribution_chart = create_emotion_distribution_chart(all_history, minutes)
    if distribution_chart:
        st.plotly_chart(distribution_chart, use_container_width=True)
    else:
        st.info("해당 시간 범위에 데이터가 없습니다.")
    # 통계 테이블
    st.subheader(f"📊 감정 통계")
    stats_table = create_emotion_stats_table(all_history, minutes)
    if stats_table is not None:
        st.dataframe(stats_table, use_container_width=True)
    else:
        st.info("해당 시간 범위에 데이터가 없습니다.")
    
    # 원시 데이터 표시 (선택사항)
    show_raw_data = st.checkbox("📋 원시 데이터 보기")
    if show_raw_data:
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        recent_data = [h for h in all_history if h['timestamp'] > cutoff_time]
        
        if recent_data:
            df_raw = pd.DataFrame([
                {
                    '시간': entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                    '감정': f"{EMOTIONS.get(entry['emotion'], {'emoji': '🤔'})['emoji']} {entry['emotion']}",
                    '신뢰도': f"{entry['score']*100:.1f}%",
                    '원본 감정': entry.get('raw_emotion', entry['emotion'])
                }
                for entry in reversed(recent_data)  # 최신순으로 정렬
            ])
            st.dataframe(df_raw, use_container_width=True)
        else:
            st.info("해당 시간 범위에 데이터가 없습니다.")

    # --- 하단 CTA: 페이지 맨 아래, 가운데 정렬 ---
    st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)  # 약간의 여백

    left, center, right = st.columns([1, 2, 1])  # 가운데 열만 넓게
    with center:
        # 높이/폰트는 이미 전역 CSS(.stButton > button)로 잡혀있으니 여기선 width만 채움
        if st.button("🌸 힐링센터 입장\n\n바로 힐링 공간으로 이동해요",
                    key="healing_mode", use_container_width=True):
            st.success("힐링센터로 이동합니다!")
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.02)
                progress_bar.progress(i + 1)
            st.switch_page("pages\healing_center.py")


# === 메인 라우터 ===    
def main():
        # 현재 페이지 라우팅
    if st.session_state.current_page == 'main':
        show_main_page()
    elif st.session_state.current_page == 'webcam':
        show_webcam_page()
    else:
        st.session_state.current_page = 'main'
        st.rerun()


# 앱 실행
if __name__ == "__main__":
    main()