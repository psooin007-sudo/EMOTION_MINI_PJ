# app_streamlit.py
# 🎭 표정 매칭 게임 (Streamlit 버전)

import io
from datetime import datetime
import random
import numpy as np
import cv2
from PIL import Image
import streamlit as st

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

def healing_center_block(text_variant: str = "done", key: str = "enter_hc_from_app_streamlit"):
    msg = "✅ 완료가 되었다면, 아래 버튼으로 힐링센터에 입장하세요." if text_variant == "done"           else "✅ 게임이 끝났다면, 아래 버튼으로 힐링센터에 입장하세요."
    st.write("")
    st.markdown("""
        <style>
        .hc-card { border: 1px solid #eaeaea; background: #fafafa; padding: 14px 16px; border-radius: 12px; margin-top: 4px; }
        .hc-card p { margin: 0; font-size: 0.95rem; color: #444; }
        </style>
    """, unsafe_allow_html=True)
    st.markdown(f"<div class='hc-card'><p>{msg}</p></div>", unsafe_allow_html=True)
    st.write("")
    if st.button("🌸 힐링센터 입장하기", key=key):
        st.switch_page("pages/healing_center.py")

# ====== 게임 전역 설정 ======
EMOJIS = {
    "happy": "😄", "sad": "😢", "angry": "😠", "surprised": "😲", "neutral": "😐", "fear": "😨", "disgust": "🤢"
}
EMOTION_KR = {
    "happy": "행복", "sad": "슬픔", "angry": "분노", "surprised": "놀람", "neutral": "중립", "fear": "두려움", "disgust": "혐오"
}

from transformers import pipeline
try:
    import torch
    DEVICE_ARG = "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else -1
except Exception:
    DEVICE_ARG = -1

@st.cache_resource(show_spinner="🤖 AI 모델을 불러오는 중입니다...")
def load_emotion_classifier():
    clf = pipeline(
        "image-classification",
        model="trpakov/vit-face-expression",
        device=DEVICE_ARG
    )
    return clf

def analyze_emotion(clf, frame_bgr):
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)
    outputs = clf(pil_img)
    top = max(outputs, key=lambda x: x['score'])
    label = top['label'].lower()
    score = float(top['score'] * 100)
    emoji = EMOJIS.get(label, "😐")
    return label, score, emoji

def get_performance_level():
    ts = st.session_state.total_score
    if ts >= 90: return "S"
    if ts >= 75: return "A"
    if ts >= 60: return "B"
    if ts >= 45: return "C"
    return "D"

# ====== 초기화 ======
if 'game_state' not in st.session_state:
    st.session_state.game_state = 'start'
if 'total_score' not in st.session_state:
    st.session_state.total_score = 0
if 'current_round' not in st.session_state:
    st.session_state.current_round = 0
if 'round_results' not in st.session_state:
    st.session_state.round_results = []

# ====== UI: 시작 화면 ======
st.set_page_config(page_title="🎭 표정 매칭", page_icon="🎭", layout="centered")
st.title("🎭 표정 매칭 게임")
st.caption("주어진 이모지와 같은 표정을 지어보세요!")

if st.session_state.game_state == 'start':
    st.session_state.rounds = st.number_input("라운드 수 선택", min_value=3, max_value=10, value=5, step=1)
    if st.button("게임 시작 ▶️", type="primary", use_container_width=True):
        st.session_state.game_state = 'playing'
        st.session_state.current_round = 1
        st.session_state.total_score = 0
        st.session_state.round_results = []
        st.rerun()

elif st.session_state.game_state in ('playing', 'playing_next_round'):
    st.markdown("#### 현재 라운드: {}".format(st.session_state.current_round))
    target_emotion = random.choice(list(EMOJIS.keys()))
    target_emoji = EMOJIS[target_emotion]
    st.metric("이번 목표", f"{target_emoji} {EMOTION_KR[target_emotion]}")
    
    # 데모용 빈 프레임
    dummy = np.zeros((200, 200, 3), dtype=np.uint8)
    st.image(dummy, caption="카메라 프리뷰(데모)", use_container_width=True)

    clf = load_emotion_classifier()
    label, score, emoji = analyze_emotion(clf, dummy)
    st.write(f"예측: {emoji} {EMOTION_KR.get(label, label)} ({score:.1f}%)")

    if st.button("라운드 종료", type="primary", use_container_width=True):
        st.session_state.total_score += int(score // 20) * 10
        st.session_state.round_results.append({
            'round': st.session_state.current_round,
            'situation': f"{EMOTION_KR.get(label, label)} 표정", 
            'target_emoji': target_emoji,
            'target_emotion_name': EMOTION_KR[target_emotion],
            'user_emotion_emoji': emoji,
            'user_emotion_name': EMOTION_KR.get(label, label),
            'score': int(score // 20) * 10,
            'user_confidence': f"{score:.1f}"
        })
        if st.session_state.current_round >= st.session_state.rounds:
            st.session_state.game_state = 'finished'
        else:
            st.session_state.current_round += 1
        st.rerun()

elif st.session_state.game_state == 'finished':
    st.balloons()
    st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>🎉 게임 종료! 수고하셨습니다! 🎉</h1>", unsafe_allow_html=True)
    
    with st.container(border=True):
        performance = get_performance_level()
        st.markdown(f"<h3 style='text-align: center;'>당신의 표정 연기 등급은...</h3>", unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: center; color: #1E3A8A;'>{performance}</h1>", unsafe_allow_html=True)
        st.metric("🏆 최종 점수", f"{st.session_state.total_score} 점")

    st.divider()
    st.subheader("📜 라운드별 상세 결과")
    for r in st.session_state.round_results:
        with st.container(border=True):
            st.markdown(f"<h5><b>Round {r['round']}</b>: {r['situation']}</h5>", unsafe_allow_html=True)
            res_cols = st.columns([1, 1, 2])
            res_cols[0].metric("🎯 목표", f"{r['target_emoji']} {r['target_emotion_name']}")
            res_cols[1].metric("☑️ 결과", f"{r['user_emotion_emoji']} {r['user_emotion_name']}")
            res_cols[2].metric("📈 점수", f"{r['score']} 점", f"{r['user_confidence']}% 신뢰도")
    st.divider()
    
    if st.button("🔁 다시 하기", type="secondary", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

    st.write("")
    healing_center_block(text_variant="done", key="enter_hc_from_app_streamlit")
