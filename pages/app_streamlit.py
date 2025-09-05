# app_streamlit.py  # 파일 이름(사용자 제공)
# -------------------------------------------------------------  # 구분선/설명
# 🎭 AI 표정 매칭 게임 (Streamlit 버전) - 최종 수정본  # 앱 설명
# - 혼선이 있는 '거울 모드' 기능 전체 삭제  # 기능 변경 요약
# - 버그 수정: 라운드 수 부족 시 미션 중복 허용  # 게임 로직 수정 요약
# - UI 수정: 미지원 이모지(🪞) 교체, deprecated 파라미터 수정  # UI 변경 요약
# -------------------------------------------------------------

import io  # 바이트 버퍼 등 I/O 핸들링을 위해 사용(현재 직접 사용은 없음)
from datetime import datetime  # 게임 ID(타임스탬프) 생성을 위해 사용
import random  # 난수(여기서는 numpy를 주로 사용), 보조 용도
import numpy as np  # 확률 샘플링 및 수치 연산
import cv2  # OpenCV: 얼굴 검출 및 이미지 전처리
from PIL import Image  # 이미지 객체 변환 및 처리
import streamlit as st  # Streamlit UI 프레임워크

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
""", unsafe_allow_html=True)  # CSS를 주입해 사이드바를 숨김

# ====== 모델 로딩: Hugging Face Transformers ======
from transformers import pipeline  # 이미지 감정 분류 파이프라인 로딩 함수
try:
    import torch  # GPU/MPS 사용 가능 여부 체크를 위해 임포트
    DEVICE_ARG = "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else -1  # Mac MPS 지원 시 'mps', 아니면 CPU(-1)
except Exception:
    DEVICE_ARG = -1  # torch 임포트 실패 시 CPU 강제

@st.cache_resource(show_spinner="🤖 AI 모델을 불러오는 중입니다...")
def load_emotion_classifier():  # 감정 분류 모델을 로드하고 캐싱하는 함수
    clf = pipeline(
        "image-classification",  # 작업 유형: 이미지 분류
        model="trpakov/vit-face-expression",  # 공개된 얼굴 감정 분류 모델(ViT 기반)
        device=DEVICE_ARG  # 실행 디바이스 지정: 'mps' 또는 -1(CPU)
    )
    return clf  # 파이프라인 객체 반환(캐시됨)

# ====== 전역 상수/맵 ======
MODEL_EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']  # 모델이 반환 가능하다고 가정하는 라벨 목록
EMOTION_WEIGHTS = {
    'angry': 1.20, 'disgust': 1.15, 'fear': 1.15, 'happy': 1.00,
    'neutral': 0.85, 'sad': 1.10, 'surprise': 1.10,
}  # 라벨별 가중치(점수 계산 시 반영)
EMOTION_PICK_WEIGHTS = {
    'happy': 3.0, 'sad': 2.2, 'surprise': 2.0, 'angry': 1.0,
    'fear': 1.0, 'disgust': 0.8, 'neutral': 0.8,
}  # 미션 뽑기에서 각 감정이 등장할 확률 가중치
EMOTION_EMOJI_MAP = {
    'angry': {'emoji': '😠', 'name': '화남'}, 'disgust': {'emoji': '🤢', 'name': '역겨움'},
    'fear': {'emoji': '😨', 'name': '두려움'}, 'happy': {'emoji': '😄', 'name': '행복'},
    'neutral': {'emoji': '😐', 'name': '중립'}, 'sad': {'emoji': '😢', 'name': '슬픔'},
    'surprise': {'emoji': '😮', 'name': '놀람'}
}  # 라벨 ↔ 이모지/한글명 매핑
GAME_SITUATIONS = {
    'easy': [  # 쉬움 난이도 상황 목록
        {"id": 1, "situation": "드디어 에러를 해결했을 때!", "target_emotion": "happy", "emoji": "😄", "difficulty": 1},
        {"id": 2, "situation": "갑자기 뒤에서 누가 깜짝 놀래켰을 때!", "target_emotion": "surprise", "emoji": "😮", "difficulty": 1},
        {"id": 3, "situation": "지하철에 딱 한 자리 남아있어서 앉았을 때!", "target_emotion": "happy", "emoji": "😄", "difficulty": 1},
        {"id": 5, "situation": "세탁기에 휴지가 들어가서 빨래가 보풀투성이가 되었을 때...", "target_emotion": "sad", "emoji": "😢", "difficulty": 1},
        {"id": 6, "situation": "내일이 월요일이라는 사실을 깨달았을 때...", "target_emotion": "sad", "emoji": "😢", "difficulty": 1}
    ],
    'medium': [  # 보통 난이도 상황 목록
        {"id": 7, "situation": "코드가 홀라당! 날아가버렸을 때", "target_emotion": "angry", "emoji": "😠", "difficulty": 2},
        {"id": 8, "situation": "수인씨가 시끄럽게 할 때", "target_emotion": "angry", "emoji": "😠", "difficulty": 2},
        {"id": 9, "situation": "버스 두 대가 연속 만차라 그냥 지나가버렸을 때...", "target_emotion": "sad", "emoji": "😢", "difficulty": 2},
        {"id": 4, "situation": "출석 체크 시간 1분 넘겨서 지각 처리됐을 때...", "target_emotion": "sad", "emoji": "😢", "difficulty": 2},
        {"id": 10, "situation": "수업이 예상보다 일찍 끝났을 때!", "target_emotion": "happy", "emoji": "😄", "difficulty": 2},
        {"id": 11, "situation": "주머니에서 잊고 있던 5천 원이 불쑥 나왔을 때!", "target_emotion": "happy", "emoji": "😄", "difficulty": 2},
        {"id": 12, "situation": "프로젝트 발표 직전에 노트북 배터리가 1%인 걸 봤을 때", "target_emotion": "fear", "emoji": "😨", "difficulty": 2}
    ],
    'hard': [  # 어려움 난이도 상황 목록
        {"id": 13, "situation": "기옥님이 재미없는 농담을 했을 때", "target_emotion": "disgust", "emoji": "🤢", "difficulty": 3},
        {"id": 14, "situation": "강사님이 뒤에서 쳐다보고 계실 때", "target_emotion": "fear", "emoji": "😨", "difficulty": 3},
        {"id": 15, "situation": "점심 메뉴가 마음에 안 들 때", "target_emotion": "disgust", "emoji": "🤢", "difficulty": 3},
        {"id": 16, "situation": "월급이 들어왔을 때!", "target_emotion": "happy", "emoji": "😄", "difficulty": 3},
        {"id": 17, "situation": "웨이팅 30분이라더니 갑자기 자리 나서 바로 입장하게 됐을 때!", "target_emotion": "happy", "emoji": "😄", "difficulty": 3},
        {"id": 18, "situation": "게임 마지막 라운드에서 역전패했을 때", "target_emotion": "angry", "emoji": "😠", "difficulty": 3}
    ]
}  # 난이도별 미션 풀 정의

# ====== OpenCV 얼굴 검출기 및 이미지 처리 ======
@st.cache_resource
def get_face_cascade():  # Haar 캐스케이드 분류기를 캐싱하여 재사용
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  # 기본 얼굴 검출기 경로 사용

def crop_face_for_model(frame_bgr):  # 모델 입력용 얼굴 크롭 및 전처리 함수
    face_cascade = get_face_cascade()  # 캐시된 얼굴 검출기 가져오기
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)  # 얼굴 검출을 위해 그레이스케일 변환
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))  # 얼굴 후보 박스 탐지
    if len(faces) == 0:  # 얼굴이 검출되지 않은 경우
        h, w = frame_bgr.shape[:2]; side = min(h, w)  # 정사각형 크롭을 위한 한 변 계산
        x, y = (w - side) // 2, (h - side) // 2  # 중앙 정렬 좌표
        crop = frame_bgr[y:y+side, x:x+side]  # 중앙 사각형 영역 크롭
    else:
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])  # 가장 큰 얼굴(면적 최대) 선택
        mx, my = int(w * 0.2), int(h * 0.2)  # 주변 여백 20% 추가(문맥 포함)
        x0, y0 = max(0, x - mx), max(0, y - my)  # 좌상단 좌표 보정(0 이하 방지)
        x1, y1 = min(frame_bgr.shape[1], x + w + mx), min(frame_bgr.shape[0], y + h + my)  # 우하단 좌표 보정(경계 초과 방지)
        crop = frame_bgr[y0:y1, x0:x1]  # 얼굴 주변을 포함한 영역 크롭
    ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)  # 조명 보정을 위한 색공간 변환
    ych, cr, cb = cv2.split(ycrcb)  # 밝기(Y)와 색차(Cr/Cb) 분리
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # 대비 향상(국소 히스토그램 평활화)
    ych = clahe.apply(ych)  # 밝기 채널에 CLAHE 적용
    ycrcb = cv2.merge([ych, cr, cb])  # 다시 YCrCb 병합
    crop = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)  # BGR로 되돌리기
    return cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)  # 모델 입력 크기(224x224)로 리사이즈

def predict_emotion(image_pil, classifier):  # PIL 이미지와 분류기를 받아 예측 수행
    frame_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)  # PIL→NumPy→BGR 변환
    face_bgr = crop_face_for_model(frame_bgr)  # 얼굴 중심 크롭/전처리
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)  # 모델용 RGB로 변환
    predictions = classifier(Image.fromarray(face_rgb), top_k=None)  # HF 파이프라인으로 감정 예측(모든 결과 반환)
    return sorted([{'label': p['label'].lower(), 'score': float(p['score'])} for p in predictions], key=lambda x: x['score'], reverse=True)  # 스코어 내림차순 정렬 및 포맷팅

def calculate_score(target_emotion_label, predictions):  # 목표 감정과 예측 결과로 점수(1~10) 산출
    if not predictions: return 0  # 예측이 없으면 0점
    weighted = {p['label']: p['score'] * EMOTION_WEIGHTS.get(p['label'], 1.0) for p in predictions}  # 라벨별 가중치 곱
    if target_emotion_label == 'neutral':  # 목표가 중립인 경우 별도 로직
        top, nw = max(weighted, key=weighted.get), weighted.get('neutral', 0.0)  # 최고 확률 라벨, 중립 확률
        score = 8 + (nw * 2) if top == 'neutral' else nw * 7  # 중립이 1등이면 가점, 아니면 비율 점수
    else:
        non_neutral = {k: v for k, v in weighted.items() if k != 'neutral'}  # 중립 제외
        if not non_neutral: return 1  # 모두 중립만 있으면 최저점 처리
        top = max(non_neutral, key=non_neutral.get)  # 중립 제외 라벨 중 최고치
        if top == target_emotion_label:  # 목표와 일치할 때
            score = 8 + (non_neutral[top] * 2)  # 높은 기본점 + 가중 가점
        else:
            tv = non_neutral.get(target_emotion_label, 0.0)  # 목표 감정의 가중 확률
            score = 2 + (tv * 4) if tv > 0 else 1  # 일부라도 맞췄으면 보정, 아니면 거의 최저점
    return min(10, round(score))  # 10점 상한, 반올림

# ====== 게임 상태 관리 ======
def init_state():  # 세션 상태 초기화(최초 진입 시)
    if 'game_state' not in st.session_state:  # 키 존재 여부 확인
        st.session_state.game_id = None  # 게임 ID(타임스탬프)
        st.session_state.current_round = 0  # 현재 라운드 인덱스(0부터)
        st.session_state.total_score = 0  # 누적 점수
        st.session_state.selected_situations = []  # 선택된 미션 목록
        st.session_state.round_results = []  # 라운드별 결과 기록
        st.session_state.game_state = 'start'  # 초기 화면 상태

def start_game(difficulty, rounds):  # 난이도/라운드 수로 게임 시작
    init_state()  # 상태 보장
    st.session_state.game_id = datetime.now().strftime("%Y%m%d_%H%M%S")  # 고유 게임 ID 생성
    st.session_state.current_round = 0  # 첫 라운드로 설정
    st.session_state.total_score = 0  # 점수 초기화
    st.session_state.round_results = []  # 결과 초기화
    st.session_state.rounds_per_game = rounds  # 총 라운드 수 저장

    pool = []  # 미션 후보 풀
    if difficulty == 'mixed':  # 혼합 난이도
        for level in GAME_SITUATIONS.values(): pool.extend(level)  # easy/medium/hard 모두 합침
    else:
        pool = list(GAME_SITUATIONS.get(difficulty, []))  # 해당 난이도의 미션만 사용
        
    if not pool:
        st.warning("선택된 난이도에 해당하는 상황이 없습니다."); st.session_state.selected_situations = []; return  # 빈 풀일 경우 경고 후 종료
        
    allow_repeats = len(pool) < rounds  # 풀 크기가 라운드 수보다 작으면 중복 허용
    weights = np.array([EMOTION_PICK_WEIGHTS.get(s['target_emotion'], 1.0) for s in pool], dtype=float)  # 미션별 가중치 배열
    probs = None if weights.sum() == 0 else weights / weights.sum()  # 정규화된 확률 벡터(합이 1)
    
    chosen_idx = np.random.choice(len(pool), size=rounds, replace=allow_repeats, p=probs)  # 확률에 따라 인덱스 샘플링
    st.session_state.selected_situations = [pool[i] for i in chosen_idx]  # 선택된 미션 리스트 저장
    st.session_state.game_state = 'playing'  # 게임 진행 상태로 전환

def add_round_result(result):  # 라운드 결과를 세션에 추가
    st.session_state.round_results.append(result)  # 결과 목록에 추가
    st.session_state.total_score += result['score']  # 누적 점수 갱신
    st.session_state.current_round += 1  # 다음 라운드로 이동

def is_game_over_check():  # 게임 종료 여부 판단
    return st.session_state.current_round >= len(st.session_state.selected_situations)  # 모든 미션을 소화했는지 체크

def get_performance_level():  # 최종 등급을 문자열로 반환
    if not st.session_state.round_results: return "게임 미완료"  # 결과가 없으면 미완료 처리
    avg = st.session_state.total_score / len(st.session_state.round_results)  # 평균 점수 계산
    if avg >= 8: return "표정왕! 👑"  # 최고 등급
    if avg >= 6: return "표정 달인 🌟"  # 상위 등급
    if avg >= 4: return "표정 뚝딱이 🔨"  # 보통 등급
    return "표정 초보 🌱"  # 최하 등급

# ===================================================================
# ====== 🎨 UI 디자인 🎨 ======
# ===================================================================

st.set_page_config(page_title="AI 표정 매칭 게임", page_icon="🎭", layout="centered")  # 페이지 메타 및 레이아웃 설정

st.markdown("""<style>
.st-emotion-cache-1g6gooi {
    width: 100% !important;
}
</style>""", unsafe_allow_html=True)  # 특정 컨테이너의 폭을 강제로 100%로 설정하는 CSS 해킹

st.title("🎭 AI 표정 매칭 게임")  # 타이틀 출력
st.markdown("### AI와 함께 표정 연기를 즐겨봐요!")  # 부제 출력

init_state()  # 세션 상태 초기화/보장
classifier = load_emotion_classifier()  # 감정 분류 모델 로딩(캐시됨)

# --- 1. 게임 시작 전 (설정 화면) ---
if st.session_state.game_state == 'start':  # 초기 상태일 때
    with st.container(border=True):  # 테두리 있는 컨테이너 시작
        st.markdown("<h3 style='text-align: center;'>🚀 게임 설정을 선택해주세요</h3>", unsafe_allow_html=True)  # 안내 헤더
        
        col1, col2 = st.columns(2)  # 2열 레이아웃 생성
        with col1:
            difficulty = st.selectbox("난이도", ["easy", "medium", "hard", "mixed"], index=1)  # 난이도 선택(기본 'medium')
        with col2:
            rounds = st.selectbox("라운드 수", [3, 5], index=0)  # 라운드 수 선택(기본 3)
        
        st.markdown("---")  # 구분선
        if st.button("게임 시작", type="primary", use_container_width=True):  # 시작 버튼(가로폭 100%)
            start_game(difficulty, rounds)  # 게임 세팅
            st.rerun()  # 상태 변경 반영을 위해 즉시 리렌더

    st.image("images\image.jpg", use_container_width=True)  # 하단 이미지 표시(폴더 경로 기준)

# --- 2. 게임 진행 중 ---
elif st.session_state.game_state == 'playing':  # 플레이 중 상태
    if is_game_over_check():  # 모든 라운드를 마쳤는지 확인
        st.session_state.game_state = 'finished'  # 종료 상태로 전환
        st.rerun()  # 즉시 리렌더

    situation = st.session_state.selected_situations[st.session_state.current_round]  # 현재 라운드의 미션 객체
    
    st.markdown(f"### <div style='text-align:center;'>📝 미션: Round {st.session_state.current_round + 1}</div>", unsafe_allow_html=True)  # 미션 번호 표시
    st.markdown(f"## <div style='text-align:center; color: #1D4ED8;'>“ {situation['situation']} ”</div>", unsafe_allow_html=True)  # 상황 설명(파란색)
    st.markdown(f"### <div style='text-align:center;'>🎯 목표 표정: {EMOTION_EMOJI_MAP[situation['target_emotion']]['emoji']} **{EMOTION_EMOJI_MAP[situation['target_emotion']]['name']}**</div>", unsafe_allow_html=True)  # 목표 감정 안내
    st.write("")  # 여백

    img_file = st.camera_input("표정을 짓고 촬영하세요 📸", key=f"camera_{st.session_state.current_round}")  # 웹캠 캡처 입력 위젯(라운드별 키 고유)

    if img_file is not None:  # 사진이 업로드되면
        with st.spinner('📸 AI가 당신의 표정을 분석 중입니다...'):  # 분석 스피너 표기
            image_pil = Image.open(img_file).convert("RGB")  # 업로드 파일을 PIL RGB 이미지로 변환
            
            predictions = predict_emotion(image_pil, classifier)  # 감정 예측 실행
            user_emotion, user_conf = (predictions[0]['label'], predictions[0]['score']) if predictions else ('neutral', 0.0)  # 최상위 라벨과 점수
            score = calculate_score(situation["target_emotion"], predictions)  # 목표 감정 대비 점수 산출

            round_result = {
                "round": st.session_state.current_round + 1, "situation": situation["situation"],  # 라운드 번호, 상황 텍스트
                "target_emoji": situation["emoji"], "target_emotion_name": EMOTION_EMOJI_MAP[situation["target_emotion"]]['name'],  # 목표 이모지/이름
                "user_emotion_label": user_emotion,  # 예측된 감정 라벨(영문)
                "user_emotion_emoji": EMOTION_EMOJI_MAP.get(user_emotion, {'emoji': '❔'})['emoji'],  # 예측 감정 이모지(없으면 ❔)
                "user_emotion_name": EMOTION_EMOJI_MAP.get(user_emotion, {'name': '알 수 없음'})['name'],  # 예측 감정 한글명(없으면 대체)
                "user_confidence": round(user_conf * 100, 1), "score": score,  # 신뢰도(%)와 점수
                "predictions": {p['label']: round(p['score'], 3) for p in predictions[:7]}  # 상위 라벨별 확률(소수 3자리)
            }
            add_round_result(round_result)  # 결과를 세션에 반영
            
            if is_game_over_check():  # 게임 종료 시점 확인
                st.session_state.game_state = 'finished'  # 종료 상태 전환
            else:
                st.session_state.game_state = 'playing_next_round'  # 다음 라운드 준비 상태로 전환
            st.rerun()  # 즉시 리렌더로 상태 반영
    
    st.divider()  # 구분선
    cols = st.columns(2)  # 2열 레이아웃
    cols[0].metric("ዙ 라운드", f"{st.session_state.current_round + 1} / {st.session_state.rounds_per_game}")  # 진행 라운드 표시(왼쪽)
    cols[1].metric("🏆 총 점수", f"{st.session_state.total_score} 점")  # 총점 표시(오른쪽)

# --- 2-1. 다음 라운드 준비 화면 ---
elif st.session_state.game_state == 'playing_next_round':  # 라운드 완료 직후의 대기 상태
    last_score = st.session_state.round_results[-1]['score']  # 직전 라운드 점수
    st.success(f"✅ Round {st.session_state.current_round} 완료! (+{last_score}점 획득, 현재 총 {st.session_state.total_score}점)")  # 알림 배너
    if st.button("다음 라운드 시작 ▶️", type="primary", use_container_width=True):  # 다음 라운드 버튼
        st.session_state.game_state = 'playing'  # 다시 플레이 상태로 전환
        st.rerun()  # 즉시 리렌더

# --- 3. 게임 종료 ---
elif st.session_state.game_state == 'finished':  # 게임 종료 상태
    st.balloons()  # 풍선 이펙트
    st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>🎉 게임 종료! 수고하셨습니다! 🎉</h1>", unsafe_allow_html=True)  # 종료 헤드라인
    
    with st.container(border=True):  # 결과 요약 카드
        performance = get_performance_level()  # 등급 계산
        st.markdown(f"<h3 style='text-align: center;'>당신의 표정 연기 등급은...</h3>", unsafe_allow_html=True)  # 안내 텍스트
        st.markdown(f"<h1 style='text-align: center; color: #1E3A8A;'>{performance}</h1>", unsafe_allow_html=True)  # 등급 표시
        st.metric("🏆 최종 점수", f"{st.session_state.total_score} 점")  # 최종 점수 메트릭

    st.divider()  # 구분선

    st.subheader("📜 라운드별 상세 결과")  # 상세 결과 섹션 제목
    for r in st.session_state.round_results:  # 각 라운드 결과 루프
        with st.container(border=True):  # 각 라운드를 카드로 표시
            st.markdown(f"<h5><b>Round {r['round']}</b>: {r['situation']}</h5>", unsafe_allow_html=True)  # 라운드 제목/상황
            res_cols = st.columns([1, 1, 2])  # 3열 레이아웃(가중 폭)
            res_cols[0].metric("🎯 목표", f"{r['target_emoji']} {r['target_emotion_name']}")  # 목표 감정 표시
            res_cols[1].metric("☑️ 결과", f"{r['user_emotion_emoji']} {r['user_emotion_name']}")  # 예측 결과 표시
            res_cols[2].metric("📈 점수", f"{r['score']} 점", f"{r['user_confidence']}% 신뢰도")  # 점수 + 신뢰도
    st.divider()  # 구분선
    
    if st.button("🔁 다시 하기", type="secondary", use_container_width=True):  # 게임 재시작 버튼
        for k in list(st.session_state.keys()):  # 세션 상태의 모든 키 순회
            del st.session_state[k]  # 세션 상태 초기화를 위해 삭제
        st.rerun()  # 초기 상태로 리렌더
        
st.write("")
healing_center_block(text_variant="done", key="enter_hc_from_app_streamlit")

# if st.button("🏠 메인으로 돌아가기", key="go_home_from_app_streamlit"):
#     go_home()

# --- 힐링센터 홈으로 돌아가기 버튼 ---
import streamlit as _st
if _st.button('🏠 홈으로 돌아가기', key='go_hc_home', use_container_width=True):
    # healing_center.py에서 사용하는 상태키로 직접 이동 (멀티페이지/exec 환경 모두 대응)
    _st.session_state['current_page'] = 'home'
    _st.rerun()
