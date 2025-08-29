# main_with_graphs_integrated.py
import streamlit as st
import subprocess
import sys
import os
import threading
from datetime import datetime, timedelta
import json
import time
import base64
import gzip
import numpy as np
import atexit

# web_no_key_rere.py
# 외부 라이브러리 자동 설치

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from plotly.subplots import make_subplots

from emotion_model import analyze_emotion_from_image, detect_face_and_analyze, get_latest_emotion, reset_emotion_state
import emotion_list

# 페이지 설정
st.set_page_config(
    page_title="감정 분석 시스템",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 세션 상태 초기화
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'main'

if 'webcam_process' not in st.session_state:
    st.session_state.webcam_process = None

if 'emotion_history' not in st.session_state:
    st.session_state.emotion_history = []

# 감정 데이터 (통합된 버전)
EMOTIONS = emotion_list.emotions

# === 정리 함수들 ===

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

# === 유틸리티 함수들 ===

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

# === 웹캠 프로세스 관리 함수들 ===

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

# === 차트 생성 함수들 ===

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

def create_enhanced_timeline_chart(history_data, minutes=30):
    """향상된 감정 변화 추이 차트"""
    if not history_data:
        return None
        
    # 최근 N분간 데이터 필터링
    cutoff_time = datetime.now() - timedelta(minutes=minutes)
    recent_data = [h for h in history_data if h['timestamp'] > cutoff_time]
    
    if len(recent_data) < 1:
        return None
    
    # 데이터 준비
    df = pd.DataFrame([
        {
            'time': entry['timestamp'].strftime('%H:%M:%S'),
            'timestamp': entry['timestamp'],
            'emotion': EMOTIONS.get(entry['emotion'], {'korean': entry['emotion']})['korean'],
            'emotion_en': entry['emotion'],
            'score': entry['score'] * 100,
            'color': EMOTIONS.get(entry['emotion'], {'color': '#808080'})['color'],
            'emoji': EMOTIONS.get(entry['emotion'], {'emoji': '🤔'})['emoji']
        }
        for entry in recent_data
    ])
    
    # 라인 차트 생성
    fig = go.Figure()
    
    # 전체 감정 변화 라인
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['score'],
        mode='lines+markers',
        name='감정 변화',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8),
        hovertemplate="<b>%{text}</b><br>" +
                     "시간: %{x|%H:%M:%S}<br>" +
                     "신뢰도: %{y:.1f}%<extra></extra>",
        text=[f"{row['emoji']} {row['emotion']}" for _, row in df.iterrows()]
    ))
    
    # 감정별로 색상이 다른 점들 추가
    for emotion in df['emotion_en'].unique():
        emotion_data = df[df['emotion_en'] == emotion]
        if not emotion_data.empty:
            emotion_info = EMOTIONS.get(emotion, {'korean': emotion, 'color': '#808080', 'emoji': '🤔'})
            fig.add_trace(go.Scatter(
                x=emotion_data['timestamp'],
                y=emotion_data['score'],
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
                             "신뢰도: %{y:.1f}%<extra></extra>",
                showlegend=True
            ))
    
    fig.update_layout(
        title=f"감정 변화 추이 (최근 {minutes}분)",
        xaxis_title="시간",
        yaxis_title="신뢰도 (%)",
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
        yaxis=dict(range=[0, 100])
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

# === 페이지 함수들 ===

def show_main_page():
    """메인 선택 페이지"""
    st.title("😊 감정 분석 시스템")
    st.markdown("---")
    
    # 로컬 히스토리 미리보기
    local_history = load_local_emotion_history()
    if local_history:
        st.sidebar.success(f"📊 로컬 데이터: {len(local_history)}개 기록")
        
        # 최근 감정 미리보기 (사이드바)
        if len(local_history) > 0:
            st.sidebar.subheader("📈 최근 감정")
            recent = local_history[-3:]  # 최근 3개
            for i, emotion_data in enumerate(reversed(recent)):
                emotion_info = EMOTIONS.get(emotion_data['emotion'], {
                    'emoji': '🤔', 'korean': emotion_data['emotion']
                })
                st.sidebar.write(f"{emotion_info['emoji']} {emotion_info['korean']} ({emotion_data['score']*100:.1f}%)")
    
    # 설명
    st.markdown("""
    ### 🎭 어떤 방법으로 감정을 분석하고 싶으세요?
    
    세 가지 방법 중 하나를 선택해주세요:
    """)
    
    # 선택 버튼들을 세 개의 컬럼으로 배치
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📹 실시간 웹캠 분석")
        st.write("웹캠을 통해 실시간으로 감정을 분석합니다")
        
        if is_webcam_running():
            if st.button("🔄 웹캠 창 다시 열기", use_container_width=True):
                stop_webcam_process()
                st.session_state.webcam_process = start_webcam_process()
                if st.session_state.webcam_process:
                    st.success("✅ 웹캠이 실행되고 있습니다!")
        else:
            if st.button("🎥 웹캠으로 분석하기", use_container_width=True):
                st.session_state.webcam_process = start_webcam_process()
                if st.session_state.webcam_process:
                    st.success("✅ 웹캠이 실행되고 있습니다!")
                    st.info("💡 웹캠 창이 켜진 상태에서 표정을 지어주세요.\n\n🛑 웹캠을 종료하면 감정 분석 결과가 제공됩니다.")
    
    with col2:
        st.markdown("#### ✋ 수동으로 감정 선택")
        st.write("직접 감정을 선택해서 결과를 확인합니다")
        if st.button("🎯 직접 선택하기", use_container_width=True):
            st.session_state.current_page = 'manual'
            st.rerun()
    
    # 웹캠 제어 버튼들 (웹캠이 실행 중일 때만 표시)
    if is_webcam_running():
        st.markdown("---")
        st.markdown("#### 🎮 웹캠 제어")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🛑 웹캠 종료", use_container_width=True):
                if stop_webcam_process():
                    st.success("✅ 웹캠이 종료되었습니다.")
                    # 웹캠 종료 시 자동으로 대시보드로 이동
                    st.session_state.current_page = 'analytics'
                    st.rerun()
                
        with col2:
            if st.button("🔄 상태 새로고침", use_container_width=True):
                st.rerun()
        
        with col3:
            st.info("🎯 **웹캠 사용 안내**\n\n"
                    "• 얼굴을 카메라 정면에 위치시키세요\n"
                    "• 다양한 표정을 지어 감정 인식 정확도를 확인하세요\n"
                    "• 세션 종료 시 감정 분석 결과가 화면에 표시됩니다\n")
        
    
        
def show_manual_page():
    """수동 선택 페이지"""
    st.title("✋ 감정을 직접 선택해주세요")
    st.markdown("---")
    
    # 뒤로가기 버튼
    if st.button("🔙 메인으로 돌아가기"):
        st.session_state.current_page = 'main'
        st.rerun()
    
    st.markdown("### 🎭 어떤 감정을 선택하시겠어요?")
    
    # 감정 선택 버튼들을 3x2 그리드로 배치
    cols = st.columns(3)
    
    for i, (emotion_key, emotion_data) in enumerate(EMOTIONS.items()):
        col = cols[i % 3]
        with col:
            if st.button(
                f"{emotion_data['emoji']} {emotion_data['korean']}", 
                use_container_width=True,
                key=f"emotion_{emotion_key}"
            ):
                # 수동 선택 시에도 히스토리에 추가
                current_time = datetime.now()
                st.session_state.emotion_history.append({
                    'emotion': emotion_key,
                    'score': 0.9,  # 수동 선택이므로 높은 신뢰도
                    'timestamp': current_time,
                    'datetime': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'raw_emotion': emotion_key
                })
                
                st.session_state.current_page = 'result'
                st.session_state.selected_emotion = emotion_key
                st.session_state.manual_score = 0.9
                st.rerun()

def show_analytics_page():
    """고급 분석 대시보드 페이지"""
    st.title("📊 감정 분석 대시보드")
    st.markdown("---")
    
    # 뒤로가기 버튼
    if st.button("🔙 메인으로 돌아가기"):
        st.session_state.current_page = 'main'
        st.rerun()
    
    # 모든 감정 데이터 로드
    all_history = load_all_emotion_data()
    
    if not all_history:
        st.warning("📭 분석할 감정 데이터가 없습니다.")
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
    
    # 메인 대시보드
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"📈 감정 변화 추이 ({selected_time})")
        timeline_chart = create_enhanced_timeline_chart(all_history, minutes)
        if timeline_chart:
            st.plotly_chart(timeline_chart, use_container_width=True)
        else:
            st.info("해당 시간 범위에 데이터가 없습니다.")
    
    with col2:
        st.subheader(f"🥧 감정 분포 ({selected_time})")
        distribution_chart = create_emotion_distribution_chart(all_history, minutes)
        if distribution_chart:
            st.plotly_chart(distribution_chart, use_container_width=True)
        else:
            st.info("해당 시간 범위에 데이터가 없습니다.")
    
    # 통계 테이블
    st.subheader(f"📊 감정 통계 ({selected_time})")
    stats_table = create_emotion_stats_table(all_history, minutes)
    if stats_table is not None:
        st.dataframe(stats_table, use_container_width=True)
    else:
        st.info("해당 시간 범위에 데이터가 없습니다.")
        
    # 원시 데이터 표시 (선택사항)
    if st.expander("📋 원시 데이터 보기"):
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

def show_result_page():
    """감정 결과 페이지 (향상된 버전)"""
    import random
    
    # URL 파라미터에서 감정 정보 가져오기
    emotion_param = safe_get_query_param('emotion', None)
    score_param = safe_get_query_param('score', None)
    
    # URL에서 히스토리 데이터 확인 및 세션 상태 업데이트
    url_history = load_url_history_data()
    if url_history:
        # 중복 제거하면서 세션에 추가
        existing_times = {h['timestamp'].strftime('%Y-%m-%d %H:%M:%S') 
                         for h in st.session_state.emotion_history}
        
        new_entries = []
        for entry in url_history:
            time_key = entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            if time_key not in existing_times:
                new_entries.append(entry)
        
        if new_entries:
            st.session_state.emotion_history.extend(new_entries)
            st.info(f"📊 웹캠에서 {len(new_entries)}개의 새로운 기록이 추가되었습니다!")
    
    # 감정 정보 결정 우선순위: URL > 세션 > 기본값
    if emotion_param:
        emotion_key = emotion_param
        st.session_state.selected_emotion = emotion_param
    else:
        emotion_key = st.session_state.get('selected_emotion', 'neutral')
    
    # 점수 정보 결정
    if score_param:
        try:
            score = float(score_param)
        except:
            score = st.session_state.get('manual_score', 0.8)
    else:
        score = st.session_state.get('manual_score', 0.8)
    
    emotion = EMOTIONS.get(emotion_key, EMOTIONS['neutral'])
    
    # 현재 감정을 히스토리에 추가 (중복 방지)
    current_time = datetime.now()
    should_add = True
    
    if st.session_state.emotion_history:
        last_entry = st.session_state.emotion_history[-1]
        time_diff = (current_time - last_entry['timestamp']).total_seconds()
        if last_entry['emotion'] == emotion_key and time_diff < 10:
            should_add = False
    
    if should_add and emotion_param:  # URL에서 온 경우만 자동 추가
        st.session_state.emotion_history.append({
            'emotion': emotion_key,
            'score': score,
            'timestamp': current_time,
            'datetime': current_time.strftime('%Y-%m-%d %H:%M:%S'),
            'raw_emotion': emotion_key
        })
        
        # 히스토리 길이 제한
        if len(st.session_state.emotion_history) > 100:
            st.session_state.emotion_history = st.session_state.emotion_history[-100:]
    
    # 뒤로가기 버튼
    if st.button("🔙 메인으로 돌아가기"):
        st.session_state.current_page = 'main'
        if 'selected_emotion' in st.session_state:
            del st.session_state.selected_emotion
        if 'manual_score' in st.session_state:
            del st.session_state.manual_score
        st.query_params.clear()
        st.rerun()
    
    # 메인 헤더
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, {emotion['color']}20 0%, {emotion['color']}40 100%); border-radius: 15px; margin-bottom: 2rem;">
            <div style="font-size: 5rem; margin: 0;">{emotion['emoji']}</div>
            <h2 style="color: {emotion['color']}; margin: 1rem 0; font-size: 2.5rem;">
                {emotion['korean']}
            </h2>
            <h3 style="color: {emotion['color']}; margin: 0.5rem 0; font-size: 1.5rem; text-transform: uppercase;">
                {emotion_key}
            </h3>
            <p style="font-size: 1.3rem; color: #666; margin: 1rem 0; line-height: 1.6;">
                {emotion['description']}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # 신뢰도 게이지
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        gauge_chart = create_emotion_gauge(score, emotion['color'])
        st.plotly_chart(gauge_chart, use_container_width=True)
    
    # 실시간 미니 차트 (히스토리가 충분할 때)
    if len(st.session_state.emotion_history) > 1:
        st.subheader("📈 실시간 감정 변화")
        mini_timeline = create_enhanced_timeline_chart(st.session_state.emotion_history, 10)
        if mini_timeline:
            st.plotly_chart(mini_timeline, use_container_width=True)
    
    # 솔루션 및 조언
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💡 추천 솔루션")
        for solution in emotion['solutions']:
            st.markdown(f"• {solution}")
        
        st.subheader("💭 명언")
        selected_quotes = random.sample(emotion['quotes'], min(2, len(emotion['quotes'])))
        for quote in selected_quotes:
            st.markdown(f"> {quote}")
    
    with col2:
        st.subheader("🎯 조언")
        st.info(emotion['tips'])
        
        # 관련 감정들 표시
        if len(st.session_state.emotion_history) > 0:
            recent_emotions = list(set([h['emotion'] for h in st.session_state.emotion_history[-10:]]))
            if len(recent_emotions) > 1:
                st.subheader("🔄 최근 감정들")
                emotion_cols = st.columns(min(len(recent_emotions), 4))
                for i, emo in enumerate(recent_emotions[:4]):  # 최대 4개까지만
                    with emotion_cols[i]:
                        emo_data = EMOTIONS.get(emo, {'emoji': '🤔', 'korean': emo})
                        st.markdown(f"<div style='text-align: center; padding: 0.5rem; background: {emo_data.get('color', '#808080')}20; border-radius: 8px;'>{emo_data['emoji']}<br><small>{emo_data['korean']}</small></div>", unsafe_allow_html=True)
    
    # 액션 버튼들
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 상세 분석 보기", use_container_width=True, type="secondary"):
            st.session_state.current_page = 'analytics'
            st.rerun()
    
    with col2:
        if st.button("🔄 다시 분석하기", use_container_width=True, type="primary"):
            st.session_state.current_page = 'main'
            if 'selected_emotion' in st.session_state:
                del st.session_state.selected_emotion
            if 'manual_score' in st.session_state:
                del st.session_state.manual_score
            st.query_params.clear()
            st.rerun()
    
    with col3:
        if st.button("📊 다른 감정 보기", use_container_width=True):
            st.session_state.current_page = 'manual'
            st.rerun()

# === 메인 라우터 ===

def main():
    """메인 라우터 - 현재 페이지에 따라 적절한 함수 호출"""
    
    # 사이드바에 현재 상태 표시
    with st.sidebar:
        st.header("🔧 상태 정보")
        st.write(f"**현재 페이지**: `{st.session_state.current_page}`")
        
        if 'selected_emotion' in st.session_state:
            emotion = EMOTIONS[st.session_state.selected_emotion]
            st.write(f"**선택된 감정**: {emotion['emoji']} {emotion['korean']}")
        
        # 웹캠 상태 표시
        webcam_status = "🟢 실행중" if is_webcam_running() else "🔴 중지됨"
        st.write(f"**웹캠 상태**: {webcam_status}")
        
        # 히스토리 상태
        history_count = len(st.session_state.emotion_history)
        st.write(f"**세션 히스토리**: {history_count}개")
        
        # 로컬 데이터 상태
        local_count = len(load_local_emotion_history())
        st.write(f"**로컬 데이터**: {local_count}개")
        
        # URL 데이터 상태
        url_count = len(load_url_history_data())
        if url_count > 0:
            st.write(f"**URL 데이터**: {url_count}개")
        
        st.markdown("---")
        
        # 웹캠 강제 종료 버튼
        if st.button("🛑 웹캠 강제 종료", type="secondary"):
            if stop_webcam_process():
                st.success("✅ 웹캠이 종료되었습니다.")
                # 웹캠 종료 시 자동으로 대시보드로 이동
                st.session_state.current_page = 'analytics'
                st.rerun()
        
        st.markdown("---")

        # 대시보드 바로가기 버튼 추가
        if st.button("📊 분석 대시보드", use_container_width=True, type="primary"):
            st.session_state.current_page = 'analytics'
            st.rerun()

        st.markdown("---")

        # 히스토리 초기화 버튼
        if st.session_state.emotion_history:
            if st.button("🗑️ 세션 히스토리 초기화"):
                st.session_state.emotion_history = []
                st.success("✅ 세션 히스토리가 초기화되었습니다.")
                st.rerun()
        
        st.markdown("---")
    
    # 현재 페이지에 따라 적절한 함수 호출
    if st.session_state.current_page == 'main':
        show_main_page()
    elif st.session_state.current_page == 'manual':
        show_manual_page()
    elif st.session_state.current_page == 'analytics':
        show_analytics_page()
    elif st.session_state.current_page == 'result':
        show_result_page()
    else:
        # 예상치 못한 페이지면 메인으로
        st.session_state.current_page = 'main'
        st.rerun()
    
    # 푸터

    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.caption("🎭 Made with Streamlit | 감정 분석 시스템 🚀")
        # 프로그램 종료 버튼
        st.subheader("⚠️ 시스템 제어")
        if st.button("🛑 프로그램 완전 종료", use_container_width=True, type="secondary"):
            st.session_state.confirm_shutdown = True
            st.rerun()

        # 확인 대화상자 (여기서는 버튼만, 함수 호출은 하지 않음)
        if st.session_state.get('confirm_shutdown', False):
            st.warning("⚠️ 프로그램을 종료하시겠습니까?")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("✅ 예, 종료", key="confirm_yes"):
                    st.session_state.do_shutdown = True  # ← 플래그만 세움
                    st.rerun()
            with c2:
                if st.button("❌ 아니오", key="confirm_no"):
                    st.session_state.confirm_shutdown = False
                    st.rerun()

        # =========================
        # 👇 반드시 컬럼 블록 '밖'에 둬야 함 (main() 맨 아래쪽이면 OK)
        # 실제 종료 로직은 전역 레이아웃에서 호출 → 메시지 박스가 전체 폭
        # =========================
        if st.session_state.get('do_shutdown'):
            # 플래그 정리(선택)
            st.session_state.do_shutdown = False
            st.session_state.confirm_shutdown = False
            shutdown_app()  # ← 여기서 실행되면 success()/info()가 전체 너비로 렌더링됨


# 앱 실행
if __name__ == "__main__":
    main()