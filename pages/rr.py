import streamlit as st
import cv2
import mediapipe as mp
import random
import time
import numpy as np

# ---------- NAV HELPERS (공통) ----------
def go_home():
    candidates = ["app_streamlit.py", "app1.py", "app.py", "Home.py", "main.py"]
    for p in candidates:
        try:
            st.switch_page(p)
            return
        except Exception:
            continue
    st.warning("메인 페이지 파일을 찾지 못했어요. go_home() 후보 목록을 프로젝트에 맞게 수정해 주세요.")

def healing_center_block(*args, **kwargs):
    """disabled: no Healing Center callout"""
    return

# 사이드바 숨기기
st.markdown("""
    <style>
    [data-testid="stSidebar"] {display: none;}
    </style>
""", unsafe_allow_html=True)

# MediaPipe Hands 초기화
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1, 
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# 세션 상태 초기화
if "scores" not in st.session_state:
    st.session_state.scores = {"player": 0, "ai": 0, "draws": 0}
if "game_history" not in st.session_state:
    st.session_state.game_history = []
if "playing" not in st.session_state:
    st.session_state.playing = False
if "ai_difficulty" not in st.session_state:
    st.session_state.ai_difficulty = "normal"

choices = ["가위", "바위", "보"]
choice_emojis = {"가위": "✌️", "바위": "✊", "보": "✋"}

# 개선된 손 모양 판별 함수
def get_hand_sign(landmarks):
    # 손가락 끝점과 관절점 인덱스
    finger_tips = [4, 8, 12, 16, 20]  # 엄지, 검지, 중지, 약지, 새끼
    finger_pips = [3, 6, 10, 14, 18]  # 각 손가락의 중간 관절
    
    fingers_up = []
    
    # 엄지 (좌우 비교 - 엄지는 다른 방향)
    if landmarks[finger_tips[0]].x > landmarks[finger_pips[0]].x:
        fingers_up.append(1)
    else:
        fingers_up.append(0)
    
    # 나머지 손가락들 (상하 비교)
    for i in range(1, 5):
        if landmarks[finger_tips[i]].y < landmarks[finger_pips[i]].y:
            fingers_up.append(1)
        else:
            fingers_up.append(0)
    
    up_count = sum(fingers_up)
    
    # 제스처 판별 (더 정확한 로직)
    if up_count <= 1:  # 주먹 (0~1개 손가락)
        return "바위"
    elif up_count == 2 and fingers_up[1] == 1 and fingers_up[2] == 1:  # 검지+중지
        return "가위"
    elif up_count >= 4:  # 4개 이상 손가락
        return "보"
    else:
        return None

# AI 선택 로직 (난이도별)
def get_ai_choice():
    if st.session_state.ai_difficulty == "easy":
        return random.choice(choices)
    elif st.session_state.ai_difficulty == "hard" and len(st.session_state.game_history) >= 2:
        # 플레이어 패턴 분석
        recent_moves = [game["player_choice"] for game in st.session_state.game_history[-3:]]
        most_used = max(set(recent_moves), key=recent_moves.count)
        
        # 카운터 전략
        counter_moves = {"가위": "바위", "바위": "보", "보": "가위"}
        return counter_moves.get(most_used, random.choice(choices))
    else:
        # 보통 난이도 - 약간의 랜덤성
        return random.choice(choices)

def determine_winner(user, computer):
    if user == computer:
        return "무승부"
    elif (user == "가위" and computer == "보") or \
         (user == "바위" and computer == "가위") or \
         (user == "보" and computer == "바위"):
        return "플레이어 승리"
    else:
        return "AI 승리"

# 페이지 설정
st.set_page_config(
    page_title="🤖 AI 가위바위보", 
    page_icon="🤖", 
    layout="centered"
)

st.title("🤖 AI 가위바위보 게임")
st.markdown("### 손 인식 기술로 즐기는 가위바위보!")

# 점수판
col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
with col1:
    st.markdown("#### 🏆 점수판")
with col2:
    st.metric("플레이어", st.session_state.scores["player"], delta=None)
with col3:
    st.metric("무승부", st.session_state.scores["draws"], delta=None)
with col4:
    st.metric("AI", st.session_state.scores["ai"], delta=None)

# 점수 초기화 버튼
if st.button("🗑️ 점수 초기화"):
    st.session_state.scores = {"player": 0, "ai": 0, "draws": 0}
    st.session_state.game_history = []
    st.success("점수가 초기화되었습니다!")

# AI 난이도 설정
st.markdown("#### ⚙️ AI 난이도")
difficulty_options = {
    "easy": "🟢 쉬움 - 완전 랜덤",
    "normal": "🟡 보통 - 약간의 전략", 
    "hard": "🔴 어려움 - 패턴 분석"
}
st.session_state.ai_difficulty = st.selectbox(
    "난이도를 선택하세요:",
    options=list(difficulty_options.keys()),
    format_func=lambda x: difficulty_options[x],
    index=1
)

st.markdown("---")

# 게임 시작 버튼
col1, col2 = st.columns(2)
with col1:
    start_game = st.button("🎮 게임 시작", type="primary", use_container_width=True)
with col2:
    if st.button("🔄 다시 하기", use_container_width=True):
        st.session_state.playing = False

# 게임 진행
if start_game:
    st.session_state.playing = True

if st.session_state.playing:
    st.markdown("### 📷 카메라 준비")
    st.info("⏳ 3초 카운트다운 후 손 모양을 보여주세요!")
    
    # 카메라 캡처 영역
    camera_placeholder = st.empty()
    countdown_placeholder = st.empty()
    
    # 카메라 열기
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("카메라를 열 수 없습니다. 카메라 연결을 확인해주세요.")
        st.session_state.playing = False
    else:
        # 카운트다운
        for i in range(3, 0, -1):
            countdown_placeholder.markdown(f"## ⏱️ {i}")
            
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                camera_placeholder.image(frame_rgb, caption=f"카운트다운: {i}", use_container_width=True)
            time.sleep(1)
        
        countdown_placeholder.markdown("## ✋ 지금 보여주세요!")
        
        # 최종 캡처 및 분석
        user_choice = None
        detection_attempts = 0
        max_attempts = 5
        
        while user_choice is None and detection_attempts < max_attempts:
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 손 감지
                results = hands.process(frame_rgb)
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # 손 랜드마크 그리기
                        mp_draw.draw_landmarks(
                            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                        )
                        user_choice = get_hand_sign(hand_landmarks.landmark)
                        if user_choice:
                            break
                
                camera_placeholder.image(frame_rgb, caption="손 모양 분석 중...", use_container_width=True)
                detection_attempts += 1
                time.sleep(0.2)
        
        cap.release()
        countdown_placeholder.empty()
        
        # 게임 결과 처리
        if user_choice:
            ai_choice = get_ai_choice()
            game_result = determine_winner(user_choice, ai_choice)
            
            # 점수 업데이트
            if game_result == "플레이어 승리":
                st.session_state.scores["player"] += 1
            elif game_result == "AI 승리":
                st.session_state.scores["ai"] += 1
            else:
                st.session_state.scores["draws"] += 1
            
            # 게임 히스토리 추가
            st.session_state.game_history.append({
                "player_choice": user_choice,
                "ai_choice": ai_choice,
                "result": game_result
            })
            
            # 결과 표시
            st.markdown("---")
            st.markdown("### 🎯 게임 결과")
            
            result_col1, result_col2, result_col3 = st.columns(3)
            
            with result_col1:
                st.markdown(f"""
                <div style='text-align: center; padding: 20px; background-color: #f0f8ff; border-radius: 15px;'>
                    <h3>👤 플레이어</h3>
                    <div style='font-size: 4rem;'>{choice_emojis[user_choice]}</div>
                    <h4>{user_choice}</h4>
                </div>
                """, unsafe_allow_html=True)
            
            with result_col2:
                st.markdown(f"""
                <div style='text-align: center; padding: 20px;'>
                    <h2 style='color: #666;'>VS</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with result_col3:
                st.markdown(f"""
                <div style='text-align: center; padding: 20px; background-color: #f5f5f5; border-radius: 15px;'>
                    <h3>🤖 AI</h3>
                    <div style='font-size: 4rem;'>{choice_emojis[ai_choice]}</div>
                    <h4>{ai_choice}</h4>
                </div>
                """, unsafe_allow_html=True)
            
            # 승부 결과
            if game_result == "플레이어 승리":
                st.success(f"🎉 {game_result}")
            elif game_result == "AI 승리":
                st.error(f"🤖 {game_result}")
            else:
                st.warning(f"🤝 {game_result}")
        
        else:
            st.error("❌ 손 모양을 인식하지 못했습니다. 다시 시도해주세요!")
            st.markdown("""
            **손 인식 팁:**
            - 카메라에서 30cm 정도 떨어져 주세요
            - 손을 화면 중앙에 위치시켜 주세요  
            - 조명이 밝은 곳에서 해주세요
            - 손 모양을 명확하게 만들어 주세요
            """)
        
        st.session_state.playing = False

# 게임 히스토리 표시
if st.session_state.game_history:
    st.markdown("---")
    st.markdown("### 📊 최근 게임 기록")
    
    # 최근 10게임만 표시
    recent_games = st.session_state.game_history[-10:]
    
    history_cols = st.columns(min(len(recent_games), 10))
    for i, game in enumerate(reversed(recent_games)):
        with history_cols[i % len(history_cols)]:
            result_color = {
                "플레이어 승리": "🟢", 
                "AI 승리": "🔴", 
                "무승부": "🟡"
            }
            st.markdown(f"""
            <div style='text-align: center; padding: 10px; border-radius: 8px; background-color: #f8f9fa;'>
                <div>{result_color[game['result']]}</div>
                <small>{game['player_choice']} vs {game['ai_choice']}</small>
            </div>
            """, unsafe_allow_html=True)

# 통계 정보
if st.session_state.game_history:
    st.markdown("---")
    st.markdown("### 📈 게임 통계")
    
    total_games = len(st.session_state.game_history)
    win_rate = (st.session_state.scores["player"] / total_games * 100) if total_games > 0 else 0
    
    stat_col1, stat_col2, stat_col3 = st.columns(3)
    with stat_col1:
        st.metric("총 게임 수", total_games)
    with stat_col2:
        st.metric("승률", f"{win_rate:.1f}%")
    with stat_col3:
        # 가장 많이 사용한 패턴
        if st.session_state.game_history:
            player_choices = [game["player_choice"] for game in st.session_state.game_history]
            most_used = max(set(player_choices), key=player_choices.count)
            st.metric("자주 사용", f"{choice_emojis[most_used]} {most_used}")

# 게임 팁
with st.expander("🎯 게임 팁 & 손 인식 가이드"):
    st.markdown("""
    **손 인식을 위한 팁:**
    - 🤚 손바닥이 카메라를 향하도록 해주세요
    - 💡 밝은 조명 아래에서 플레이하세요
    - 📏 카메라에서 30-50cm 거리를 유지하세요
    - 🎯 손을 화면 중앙에 위치시켜 주세요
    
    **AI 난이도별 특징:**
    - 🟢 **쉬움**: 완전한 랜덤 선택
    - 🟡 **보통**: 기본적인 전략 사용
    - 🔴 **어려움**: 당신의 패턴을 분석해서 대응
    
    **손 모양 기준:**
    - ✊ **바위**: 주먹을 꽉 쥔 상태
    - ✌️ **가위**: 검지와 중지만 펴기
    - ✋ **보**: 모든 손가락 펴기
    """)

# AI 난이도 설정
st.markdown("---")
st.markdown("#### 🤖 AI 설정")
difficulty_map = {
    "easy": "🟢 쉬움 - 완전 랜덤",
    "normal": "🟡 보통 - 약간의 전략",
    "hard": "🔴 어려움 - 패턴 분석"
}

st.session_state.ai_difficulty = st.selectbox(
    "AI 난이도를 선택하세요:",
    options=["easy", "normal", "hard"],
    format_func=lambda x: difficulty_map[x],
    index=1,
    disabled=st.session_state.playing
)

# 추가 기능들
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    if st.button("📊 상세 통계 보기"):
        if st.session_state.game_history:
            st.markdown("#### 📈 상세 게임 분석")
            
            # 선택 빈도 분석
            player_choices = [game["player_choice"] for game in st.session_state.game_history]
            choice_counts = {choice: player_choices.count(choice) for choice in choices}
            
            for choice, count in choice_counts.items():
                percentage = (count / len(player_choices) * 100) if player_choices else 0
                st.write(f"{choice_emojis[choice]} {choice}: {count}회 ({percentage:.1f}%)")
        else:
            st.info("아직 게임 기록이 없습니다.")

with col2:
    if st.button("🎮 빠른 재시작"):
        st.session_state.playing = False
        st.experimental_rerun()

# 개발자 정보
with st.expander("ℹ️ 정보"):
    st.markdown("""
    **개선된 기능들:**
    - ✅ 향상된 손 인식 정확도
    - ✅ 연속 게임 플레이
    - ✅ 실시간 점수 추적
    - ✅ AI 난이도 조절
    - ✅ 패턴 분석 AI
    - ✅ 게임 통계 제공
    - ✅ 사용자 친화적 인터페이스
    
    **기술 스택:** OpenCV, MediaPipe, Streamlit
    """)
st.write("")
healing_center_block(text_variant="game", key="enter_hc_from_rr")

# --- 힐링센터 홈으로 돌아가기 버튼 ---
import streamlit as _st
if _st.button('🏠 홈으로 돌아가기', key='go_hc_home', use_container_width=True):
    # healing_center.py에서 사용하는 상태키로 직접 이동 (멀티페이지/exec 환경 모두 대응)
    _st.session_state['current_page'] = 'home'
    _st.rerun()
