import streamlit as st
import time
from datetime import datetime, timedelta
import random
import json
import streamlit.components.v1 as components
import os
from transformers import pipeline
import atexit
import pandas as pd
import emotion_list
from main import (
    load_all_emotion_data,
    create_enhanced_timeline_chart,
    create_emotion_distribution_chart,
    create_emotion_stats_table,
)
import openai
# [ADD] 재귀 스캔을 위한 glob, 시작 시각 기록
import glob
APP_START_TIME = time.time()  # 이 파일이 import된 시점(앱 시작 시점)

# ===============================
# OpenAI API 키 설정
# - 파일: C:\Users\Admin\AIX\emotion_mini_project\.streamlit\secrets.toml
# - 형식: OPENAI_API_KEY = "sk-..."
# ===============================
try:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
except KeyError:
    st.error("⚠️ OpenAI API 키를 찾을 수 없습니다! 프로젝트 폴더 내 .streamlit/secrets.toml 파일을 확인해 주세요.")
    st.info("API 키 없이도 나머지 기능은 사용 가능하지만, AI 기반 명상 이미지는 작동하지 않습니다.")
    openai.api_key = None  # API 키가 없을 경우 None으로 설정

# ===============================
# 바다 감정 경험 매핑 (이미지 프롬프트/사운드 프롬프트)
# ===============================
OCEAN_EXPERIENCE_MAPPING = {
    "화남": {
        "visual_prompt": "거친 폭풍우 속 높은 파도, 번개가 치는 어두운 바다를 추상적이고 예술적으로 묘사한 그림",
        "sound_prompt": "우르릉거리는 천둥과 강한 파도 소리. 화난 감정을 밖으로 표출하도록 유도하는 격렬한 소리."
    },
    "슬픔": {
        "visual_prompt": "어두운 밤의 바다, 보슬비가 내리고, 파도가 잔잔히 밀려오는 모습을 슬프고 감성적으로 묘사한 그림",
        "sound_prompt": "조용히 내리는 빗소리와 멀리서 들리는 차분한 파도 소리. 슬픔을 받아들이고 흘려보내도록 유도하는 잔잔한 소리."
    },
    "기쁨": {
        "visual_prompt": "눈부신 햇살이 비추는 맑고 투명한 바다, 돌고래 떼가 뛰노는 모습을 밝고 환하게 묘사한 그림",
        "sound_prompt": "경쾌하고 반짝이는 파도 소리와 돌고래의 울음소리. 기쁨의 에너지를 확장시키는 긍정적인 소리."
    },
    "스트레스": {
        "visual_prompt": "회오리가 치는 바다, 물속에서 거품이 소용돌이치는 모습을 긴장감 있게 묘사한 그림",
        "sound_prompt": "빠르게 몰아치는 파도 소리와 억눌린 감정이 밖으로 터져 나오도록 유도하는 거친 소리."
    },
    "걱정": {
        "visual_prompt": "안개가 자욱한 바다, 어디로 가야 할지 모르는 작은 배 한 척이 떠 있는 모습을 불안하게 묘사한 그림",
        "sound_prompt": "낮고 불안정한 파도 소리와 안개 속에서 들리는 흐릿한 소리. 걱정의 무게를 인식하고 가벼워지도록 유도하는 소리."
    },
    "불안/두려움": {
        # ✅ 무서운 이미지 대신 잔잔하고 편안한 느낌으로 변경
        "visual_prompt": "달빛이 비추는 잔잔한 밤바다, 부드러운 파도와 고요한 수평선을 따뜻하고 편안하게 묘사한 그림",
        "sound_prompt": "느리고 규칙적인 파도 소리. 불안을 차분하게 진정시키는 안정적인 소리."
    },
    "복잡": {
        "visual_prompt": "다양한 색깔의 물감들이 섞여 흐르는 바다, 예측할 수 없는 물결을 혼란스럽게 묘사한 그림",
        "sound_prompt": "여러 종류의 파도 소리가 섞여 불협화음을 내는 소리. 복잡한 마음을 해체하도록 유도하는 소리."
    },
    "평온": {
        "visual_prompt": "고요한 새벽 바다, 수평선 위로 해가 떠오르는 모습을 평화롭게 묘사한 그림",
        "sound_prompt": "아주 잔잔하고 규칙적인 파도 소리. 심리적 안정을 위한 명상음."
    },
}

# ===============================
# 오디오 소스 (명시적 플레이어용) — 다중 후보 & MIME 지정
# - 브라우저 자동재생 차단을 피하려면 사용자가 '재생'을 눌러야 함
# ===============================
AUDIO_SOURCES = [
    # 안정적인 공개 소스(직접 재생을 위해 플레이어로 표시)
    {"url": "https://www.soundjay.com/misc/sounds/ocean-wave-1.wav", "mime": "audio/wav"},
    {"url": "https://www.zapsplat.com/wp-content/uploads/2015/sound-effects-one/zapsplat_nature_sea_ocean_waves_gentle_lapping_rocks_stones_beach_calm_peaceful_001_22281.mp3", "mime": "audio/mp3"},
]
def get_ocean_audio_source():
    # 필요시 추가 후보를 넣을 수 있음. 첫 번째를 기본으로 사용.
    return AUDIO_SOURCES[0]


# ===============================
# (추가) 로컬 파도 소리 WAV 생성기 — 외부 링크 실패 대비 확실한 재생 보장
# ===============================
def synthesize_offline_ocean_wav(duration_sec=300, sample_rate=44100):
    """
    잔잔한 명상용 파도 소리 합성 (저역 중심, 포말 최소)
    - 아주 느린 스웰(바다 숨결) + 깊은 럼블 + 드문 포말
    - 고역 히스 최소화 → '매미 소리' 느낌 제거
    """
    import numpy as np, wave, uuid

    n = int(duration_sec * sample_rate)
    t = np.arange(n) / sample_rate
    rng = np.random.default_rng()

    # ===== 파라미터 (더 잔잔하게 듣고 싶으면 아래 값만 조절) =====
    swell_rate  = 0.06   # 스웰 주기(Hz). 더 느리게 = 0.04~0.06
    swell_depth = 0.25   # 스웰 깊이(0~1). 작게 유지
    deep_rumble_gain = 0.85  # 깊은 저역(바다의 '웅—')
    mid_rumble_gain  = 0.18  # 중역 물결(너무 크면 시끄러워짐)
    distant_wave_gain = 0.12 # 먼 파도층
    foam_level  = 0.015  # 포말(고역 히스). 매미 느낌 나면 0.00~0.02
    breaker_events_per_min = 1  # 큰 파도 이벤트(드물게)
    breaker_sigma = 0.55        # 큰 파도 꼬리 지속(초)
    # ===========================================================

    # 1pole 저역 필터(부드럽게)
    def smooth_filter(data, cutoff_hz, fs):
        RC = 1.0 / (2 * np.pi * cutoff_hz)
        dt = 1.0 / fs
        alpha = dt / (RC + dt)
        y = np.empty_like(data)
        y[0] = data[0]
        for i in range(1, data.size):
            y[i] = y[i-1] + alpha * (data[i] - y[i-1])
        return y

    # 자연스러운 스웰 엔벌로프(복합 저주파)
    def natural_envelope(t, f, depth):
        env = np.ones_like(t)
        env += depth * (0.50 * np.sin(2*np.pi*f*t)
                        + 0.25 * np.sin(2*np.pi*1.3*f*t + 0.6)
                        + 0.15 * np.sin(2*np.pi*0.7*f*t + 1.1))
        return np.clip(env, 0.25, 1.6)

    # 1) 깊은 바다 럼블(매우 저역)
    base = rng.normal(0, 1, n)
    deep = smooth_filter(base, 18, sample_rate)
    deep = smooth_filter(deep, 10, sample_rate)  # 더 느리게

    # 2) 중간 물결층(소량)
    midn = rng.normal(0, 1, n)
    mid  = smooth_filter(midn, 70, sample_rate)
    mid  = smooth_filter(mid, 35, sample_rate)

    # 3) 먼 파도층(핑크스러운 감쇠)
    dist = rng.normal(0, 1, n)
    for fcut in (300, 200, 120):
        dist = smooth_filter(dist, fcut, sample_rate)

    # 4) 포말(아주 미세, 고역 최소화)
    foamn = rng.normal(0, 1, n) * 0.25
    foam  = smooth_filter(foamn, 1800, sample_rate)
    foam  = smooth_filter(foam, 1000, sample_rate)

    # 5) 스웰 + 라핑(얕은 물결) 엔벌로프
    main_env = natural_envelope(t, swell_rate, swell_depth)
    lap_env  = 0.97 + 0.03 * np.sin(2*np.pi*0.32*t)  # 0.32Hz ≈ 3.1초 주기

    # 6) 드문 큰 파도(작게, 길게)
    wave_env = np.ones_like(t)
    num_events = max(1, int(breaker_events_per_min * duration_sec / 60))
    for _ in range(num_events):
        center = rng.uniform(duration_sec*0.2, duration_sec*0.8)
        strength = rng.uniform(0.08, 0.16)
        dur = rng.uniform(12, 20)  # 길고 부드럽게
        mask = np.abs(t - center) < dur
        shape = np.zeros_like(t)
        shape[mask] = strength * np.exp(-((t[mask]-center)/(dur*0.55))**2)
        tail_mask = (t > center) & (t < center + dur * 2.2)
        if np.any(tail_mask):
            shape[tail_mask] += strength * 0.28 * np.exp(-(t[tail_mask]-center)/(dur*0.9))
        wave_env += shape

    # 7) 합성(저역 중심, 고역 최소)
    audio = (
        deep * deep_rumble_gain * main_env +
        mid  * mid_rumble_gain  * main_env * 0.7 +
        dist * distant_wave_gain * wave_env * lap_env +
        foam * foam_level * (0.4*main_env + 0.6*wave_env) * lap_env
    )

    # 8) 최종 부드러움 확보(고역 억제 + DC 제거)
    audio = smooth_filter(audio, 1600, sample_rate)  # 1.6kHz 이상 살짝 감쇠
    audio = audio - np.mean(audio)

    # 9) 페이드 인/아웃(더 길게, 잔잔)
    fade_len = int(3.0 * sample_rate)
    if audio.size > 2*fade_len:
        fade_in  = np.sqrt(np.linspace(0, 1, fade_len))
        fade_out = np.sqrt(np.linspace(1, 0, fade_len))
        audio[:fade_len] *= fade_in
        audio[-fade_len:] *= fade_out

    # 10) RMS 기준 정규화(자연스러운 볼륨)
    rms = np.sqrt(np.mean(audio**2)) if np.any(audio) else 0.0
    if rms > 0:
        target_rms = 0.22   # 너무 크면 0.10, 너무 작으면 0.14
        audio *= (target_rms / rms)

    # 11) 아주 부드러운 리미팅(클리핑 방지)
    audio = np.tanh(audio * 1.8) * 0.98

    # 12) 저장(WAV, 16-bit PCM, 모노)
    pcm = (audio * 32767).astype(np.int16)
    path = f"natural_ocean_{uuid.uuid4().hex}.wav"
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return path


# ===============================
# 명상용 이미지 생성 (크기 확대: 720px)
# ===============================
def generate_and_display_ocean_image(user_emotion):
    """
    AI가 사용자의 감정 상태를 기반으로 바다 이미지를 생성해 표시합니다.
    - 표시 크기 width=720
    """
    if not openai.api_key:
        st.warning("API 키가 없어 AI 이미지 기능을 사용할 수 없습니다.")
        return False, None

    experience = OCEAN_EXPERIENCE_MAPPING.get(user_emotion, OCEAN_EXPERIENCE_MAPPING["복잡"])
    visual_prompt = experience["visual_prompt"]

    with st.spinner("🎨 바다 이미지를 그리는 중..."):
        try:
            # NOTE: 사용 환경에 따라 openai python SDK가 client 기반일 수도 있음.
            # 현재 프로젝트에서 사용하던 방식(openai.images.generate)을 유지.
            image_response = openai.images.generate(
                model="dall-e-3",
                prompt=visual_prompt,
                n=1,
                size="1024x1024",
                quality="hd"
            )
            image_url = image_response.data[0].url
            st.image(image_url, caption=f"AI가 그린 '{user_emotion}'의 바다", width=720)
            st.success("✅ 바다 이미지 생성 완료!")
            return True, image_url
        except Exception as e:
            st.error(f"이미지 생성에 실패했습니다: {e}")
            return False, None

# ===============================
# 명상 경험 준비 (이미지 + 오디오)
# ▶ 요구사항 반영:
#   - 버튼 아래 즉시 이미지 표시
#   - 바로 아래에 오디오 플레이어 항상 표시
#   - 이전 명상/로그 저장/복원 제거
#   - 외부 소스 실패 시 로컬 WAV로 확실하게 재생
# ===============================
def generate_and_display_ocean_experience(user_emotion):
    """
    명상 페이지에서 이미지와 오디오를 모두 생성하고, 성공 여부를 반환합니다.
    - 이미지는 기존 DALL·E 호출 그대로 사용
    - 오디오는 '오프라인 WAV'를 기본으로 재생 (외부 URL은 백업용)
    - 세션에 last_meditation 저장해서 페이지 이동 후에도 복원 가능
    """
    # 1) 이미지 생성
    ok, image_url = generate_and_display_ocean_image(user_emotion)
    if not ok:
        return False

    # 2) 오디오: 오프라인 WAV를 먼저 생성/재생 (브라우저/네트워크 영향 최소화)
    try:
        wav_path = synthesize_offline_ocean_wav(duration_sec=300)  
        st.audio(wav_path, format="audio/wav")
        st.caption("🔈 ▶ 버튼을 눌러 파도 소리를 들으세요. (오프라인 WAV)")

        # 세션/로그 업데이트 (페이지 나갔다 와도 복원됨)
        st.session_state.last_meditation = {
            "emotion": user_emotion,
            "image_url": image_url,
            "audio_url": wav_path,
            "audio_mime": "audio/wav",
            "ts": datetime.now().isoformat(timespec="seconds"),
        }
        st.session_state.show_audio_player = True
        st.session_state.audio_nonce = time.time()
        st.session_state.meditation_log.append(st.session_state.last_meditation.copy())
        save_data_to_json()
        return True

    except Exception:
        # 3) 만약 로컬 합성이 실패하면 외부 소스를 백업으로 사용
        audio_src = get_ocean_audio_source()
        st.audio(audio_src["url"], format=audio_src["mime"])
        st.caption("🔈 ▶ 버튼을 눌러 파도 소리를 들으세요. (외부 소스)")

        st.session_state.last_meditation = {
            "emotion": user_emotion,
            "image_url": image_url,
            "audio_url": audio_src["url"],
            "audio_mime": audio_src["mime"],
            "ts": datetime.now().isoformat(timespec="seconds"),
        }
        st.session_state.show_audio_player = True
        st.session_state.audio_nonce = time.time()
        st.session_state.meditation_log.append(st.session_state.last_meditation.copy())
        save_data_to_json()
        return True


# ===============================
# 자동 정리 (JSON 파일)
# ===============================
def auto_cleanup():
    """앱 종료 시 임시 JSON 파일 정리"""
    json_files = [
        "ocean_healing_data.json",
        "emotion_history.json",
        "latest_emotion_result.json"
    ]

    # [ADD] 서브폴더 포함 모든 *.wav 수집 (중복 제거)
    try:
        wav_candidates = glob.glob("*.wav") + glob.glob("**/*.wav", recursive=True)
        wav_files = []
        cutoff = APP_START_TIME - 60  # [ADD] 60초 버퍼 포함, 시작 이후 생성/수정 파일만 정리
        for p in set(os.path.abspath(x) for x in wav_candidates if os.path.isfile(x)):
            try:
                if os.path.getmtime(p) >= cutoff:
                    wav_files.append(p)
            except Exception:
                # mtime 접근 실패 시 그냥 건너뜀
                pass
    except Exception:
        wav_files = []

    # [NOTE] 정말로 모든 .wav(시작 전 생성 포함)를 지우고 싶다면 위의 mtime 필터링을 제거하세요.
    #   -> wav_files = list(set(os.path.abspath(x) for x in wav_candidates if os.path.isfile(x)))

    # [ADD] JSON + WAV 통합 정리 루프
    targets = json_files + wav_files

    for target in targets:
        max_attempts = 4
        for attempt in range(max_attempts):
            try:
                if os.path.exists(target):
                    os.chmod(target, 0o777)
                    os.remove(target)
                    print(f"🧹 자동 정리: {target} 삭제됨")
                    break
            except Exception as e:
                if attempt < max_attempts - 1:
                    time.sleep(0.5)
                else:
                    print(f"🧹 자동 정리 실패: {target} ({e})")

    
atexit.register(auto_cleanup)

EMOTIONS = emotion_list.emotions

# ===============================
# 페이지 설정
# ===============================
st.set_page_config(
    page_title="🌊 바다 힐링센터",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===============================
# AI 분류 모델 로드 (한국어 감정 키워드 기반 보조)
# ===============================
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

def analyze_emotion(text):
    """간단 키워드 매칭 기반 감정 추론 (모델 실패 대비용)"""
    classifier = load_emotion_model()
    if not classifier:
        return "복잡", 75.0

    try:
        emotion_keywords = {
            "화남": ["화", "짜증", "분노", "열받", "빡치", "억울", "답답", "미치겠", "화나", "안 좋", "안좋", "불합격", "킹받", "좆", "짜증나", "열받아", "분통", "개빡", "억까", "빡대가리", "X발", "짜증남", "젠장", "ㅈ같", "ㅈ발", "개같"],
            "슬픔": ["슬프", "우울", "눈물", "상처", "외로", "허전", "멘탈", "아파", "서러", "불합격", "떨어", "서럽", "허망", "상실", "패배", "좌절", "망했", "멘붕", "공허", "처참", "멘탈붕괴", "멘붕", "현타"],
            "기쁨": ["기쁜", "행복", "좋", "즐거", "웃", "신나", "만족", "뿌듯", "최고", "합격", "신남", "즐겜", "개꿀", "개이득", "쩐다", "갓", "인생템", "사랑", "감동", "ㅎㅇㅌ", "ㅋㅋ", "ㅎㅎ", "ㄱㅇㅇ", "다행"],
            "스트레스": ["스트레스", "피곤", "힘들", "지쳐", "번아웃", "압박", "부담", "씨발", "ㅆㅂ", "번거", "미치겠", "벅차", "귀찮", "짜증폭발", "죽겠", "멘탈나감", "노답", "헬조선", "빡겜"],
            "걱정": ["걱정", "긴장", "좋을까", "조마조마", "노심초사", "근심", "잘될까", "걱정됨", "불안정"],
            "불안/두려움": ["불안", "두려", "무서", "겁나", "초조", "떨려", "식은땀", "겁", "초조해", "덜덜", "무섭"],
            "평온": ["평온", "차분", "고요", "안정", "편안", "평화", "온화", "느긋", "잔잔"],
            "복잡": ["복잡", "혼란", "갈등", "고민", "애매", "헷갈", "어렵", "뒤죽박죽", "정신없", "혼동", "모르겠", "답없", "골치", "정리안됨"],
        }

        text_lower = text.lower()
        scores = {e: sum(1 for k in kws if k in text_lower) for e, kws in emotion_keywords.items()}
        if not any(scores.values()):
            return "복잡", 75.0

        top_emotion = max(scores, key=scores.get)
        confidence = min(scores[top_emotion] * 25 + 50, 100)
        return top_emotion, confidence
    except Exception:
        return "복잡", 75.0

# ===============================
# 데이터 저장/로드 (+ last_meditation & meditation_log 추가)
# ===============================
def save_data_to_json():
    data = {
        "mood_history": st.session_state.get("mood_history", []),
        "breathing_sessions": st.session_state.get("breathing_sessions", 0),
        "meditation_time": st.session_state.get("meditation_time", 0),
        "gratitude_list": st.session_state.get("gratitude_list", []),
        "journal_entries": st.session_state.get("journal_entries", []),
        "sleep_records": st.session_state.get("sleep_records", []),
        "habit_tracker": {k: list(v["done_dates"]) for k, v in st.session_state.get("habit_tracker", {}).items()},
        "emotion_history": st.session_state.get("emotion_history", []),
        "daily_ocean_message": st.session_state.get("daily_ocean_message", None),
        "last_meditation": st.session_state.get("last_meditation", None),
        "meditation_log": st.session_state.get("meditation_log", []),
    }
    try:
        with open("ocean_healing_data.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def load_data_from_json():
    try:
        with open("ocean_healing_data.json", "r", encoding="utf-8") as f:
            data = json.load(f)

        st.session_state.mood_history = data.get("mood_history", [])
        st.session_state.breathing_sessions = data.get("breathing_sessions", 0)
        st.session_state.meditation_time = data.get("meditation_time", 0)
        st.session_state.gratitude_list = data.get("gratitude_list", [])
        st.session_state.journal_entries = data.get("journal_entries", [])
        st.session_state.sleep_records = data.get("sleep_records", [])
        habit_data = data.get("habit_tracker", {})
        st.session_state.habit_tracker = {k: {"done_dates": set(v)} for k, v in habit_data.items()}
        st.session_state.emotion_history = data.get("emotion_history", [])
        st.session_state.daily_ocean_message = data.get("daily_ocean_message", None)
        st.session_state.last_meditation = data.get("last_meditation", None)
        st.session_state.meditation_log = data.get("meditation_log", [])
    except FileNotFoundError:
        pass

# ===============================
# 스타일 / 배경음 (루프)
# ===============================
OCEAN_THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;600;700&display=swap');
.ocean-audio { position: fixed; top: -100px; left: -100px; width: 1px; height: 1px; opacity: 0; pointer-events: none; }
.stApp {
    background: linear-gradient(135deg, #87CEEB 0%, #ADD8E6 25%, #B0E0E6 50%, #E0F6FF 75%, #F0F8FF 100%);
    background-attachment: fixed; font-family: 'Noto Sans KR', sans-serif; color: #2c3e50;
}
/* 🔒 오디오용 st.components iframe을 화면/레이아웃에서 완전히 숨김 */
[data-testid="stIFrame"]{
  width:0 !important;
  height:0 !important;
  min-height:0 !important;
  border:0 !important;
  position:absolute !important;
  left:-9999px !important;
  pointer-events:none !important;
}
[data-testid="stSidebar"] { min-width: 380px !important; max-width: 380px !important;
    background: linear-gradient(180deg, #4A90E2 0%, #5DADE2 30%, #7FB3D3 100%) !important;
    border-right: 2px solid rgba(255,255,255,0.4) !important; box-shadow: 0 0 30px rgba(74, 144, 226, 0.2) !important; }
[data-testid="stSidebar"] > div { background: transparent !important; display: flex !important; flex-direction: column !important; height: 100% !important; padding-top: 10px !important; padding-bottom: 20px !important; }
[data-testid="stSidebar"] h1 { color: white !important; text-align: center !important; text-shadow: 2px 2px 4px rgba(0,0,0,0.3) !important; margin: 10px 0 20px 0 !important; font-size: 1.8rem !important; animation: wave 3s ease-in-out infinite !important; }
@keyframes wave { 0%, 100% { transform: translateY(0px); } 50% { transform: translateY(-3px); } }
[data-testid="stSidebar"] .stButton > button { background: rgba(255,255,255,0.15) !important; color: white !important; font-weight: 500 !important; font-size: 0.95rem !important; padding: 0.8rem 1rem !important; border-radius: 12px !important; border: 1px solid rgba(255,255,255,0.25) !important; margin: 0.3rem 8px !important; transition: all 0.3s ease !important; width: calc(100% - 16px) !important; text-align: left !important; backdrop-filter: blur(5px) !important; box-shadow: 0 2px 10px rgba(0,0,0,0.1) !important; }
[data-testid="stSidebar"] .stButton > button:hover { background: rgba(255,255,255,0.25) !important; border-color: rgba(255,255,255,0.4) !important; transform: translateX(5px) scale(1.02) !important; box-shadow: 0 4px 20px rgba(0,0,0,0.15) !important; }
.exit-section { margin-top: auto !important; padding: 20px 8px !important; border-top: 1px solid rgba(255,255,255,0.2) !important; }
.exit-section .stButton > button { background: rgba(220,20,60,0.2) !important; border-color: rgba(255,255,255,0.3) !important; color: white !important; font-weight: 600 !important; }
.exit-section .stButton > button:hover { background: rgba(220,20,60,0.4) !important; transform: scale(1.02) !important; }
.main .block-container { padding: 2rem !important; max-width: none !important; }
.ocean-page-title { text-align: center; font-size: 2.8rem; font-weight: 700; margin: 1rem 0 2rem 0; color: #4A90E2; text-shadow: 2px 2px 4px rgba(0,0,0,0.1); background: linear-gradient(45deg, #4A90E2, #7FB3D3); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
.wave-decoration { width: 100%; height: 60px; background: url("data:image/svg+xml,%3Csvg viewBox='0 0 1200 120' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M0,60 Q300,120 600,60 T1200,60 V120 H0 Z' fill='%237FB3D3' opacity='0.3'/%3E%3C/svg%3E"); background-size: cover; margin: 20px 0; animation: waveMove 6s ease-in-out infinite; }
@keyframes waveMove { 0%, 100% { background-position-x: 0px; } 50% { background-position-x: -200px; } }
.main .stButton > button { background: linear-gradient(45deg, #4A90E2, #7FB3D3); color: white; font-weight: 500; font-size: 1rem; padding: 1.2rem 2rem; border-radius: 15px; border: none; margin: 0.5rem 0; transition: all 0.3s ease; box-shadow: 0 4px 15px rgba(74,144,226,0.3); width: 100%; min-height: 3.5rem; }
.main .stButton > button:hover { background: linear-gradient(45deg, #7FB3D3, #4A90E2); transform: translateY(-2px); box-shadow: 0 6px 25px rgba(74,144,226,0.4); }
.stTextInput > div > div > input, .stTextArea > div > div > textarea, .stSelectbox > div > div > div { background: rgba(255,255,255,0.9) !important; border: 2px solid rgba(127, 179, 211, 0.3) !important; border-radius: 12px !important; color: #2c3e50 !important; }
.stTextInput > div > div > input:focus, .stTextArea > div > div > textarea:focus { border-color: #7FB3D3 !important; box-shadow: 0 0 15px rgba(127, 179, 211, 0.3) !important; }
[data-testid="metric-container"] { background: rgba(255,255,255,0.9); border-radius: 15px; padding: 1rem; border: 1px solid rgba(127, 179, 211, 0.3); box-shadow: 0 4px 15px rgba(74, 144, 226, 0.1); }
.stProgress > div > div > div { background: linear-gradient(90deg, #7FB3D3, #4A90E2) !important; border-radius: 10px !important; }
.stSuccess { background: rgba(46, 204, 113, 0.1) !important; border: 1px solid rgba(46, 204, 113, 0.3) !important; border-radius: 15px !important; }
.stInfo { background: rgba(127, 179, 211, 0.1) !important; border: 1px solid rgba(127, 179, 211, 0.3) !important; border-radius: 15px !important; }
.ocean-quote { background: linear-gradient(135deg, rgba(127, 179, 211, 0.1), rgba(176, 224, 230, 0.1)); border-radius: 20px; padding: 25px; margin: 25px 0; text-align: center; font-style: italic; font-size: 1.1rem; color: #4A90E2; border: 2px solid rgba(127, 179, 211, 0.3); box-shadow: 0 4px 20px rgba(127, 179, 211, 0.1); }
@media (max-width: 768px) { .ocean-page-title { font-size: 2rem; } [data-testid="stSidebar"] { min-width: 300px !important; max-width: 300px !important; } }
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: rgba(176, 224, 230, 0.1); border-radius: 10px; }
::-webkit-scrollbar-thumb { background: linear-gradient(45deg, #7FB3D3, #4A90E2); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: linear-gradient(45deg, #4A90E2, #7FB3D3); }
[data-testid="stSidebarNav"], [data-testid="stSidebarNavItems"], [data-testid="stSidebarNavSeparator"] { display: none !important; }
</style>
"""
# --- 최상단 여백 확 줄이기: 가장 마지막에 넣기 ---
st.markdown("""
<style>
/* 1) 상단 헤더를 완전히 제거해서 헤더 높이(여백) 자체를 없앱니다. */
[data-testid="stHeader"] { display: none; height: 0; }

/* 2) 메인 컨테이너의 위쪽 패딩을 최소화합니다. (원하면 0으로) */
.main .block-container { padding-top: 0.25rem !important; }

/* 3) 페이지 타이틀(호흡과 명상) 위 마진을 없앱니다. */
.ocean-page-title, h1 { margin-top: 0 !important; }
</style>
""", unsafe_allow_html=True)


# 배경 루프 오디오 (분위기용)
OCEAN_AUDIO_HTML = """
<audio autoplay loop class="ocean-audio" id="oceanAudio">
    <source src="https://www.soundjay.com/misc/sounds/ocean-wave-1.wav" type="audio/wav">
    <source src="https://www.zapsplat.com/wp-content/uploads/2015/sound-effects-one/zapsplat_nature_sea_ocean_waves_gentle_lapping_rocks_stones_beach_calm_peaceful_001_22281.mp3" type="audio/mpeg">
</audio>
<script>
// 사용자 상호작용 이후 자동 재생 보장 시도 (브라우저 정책 회피)
document.addEventListener('click', function() {
    const audio = document.getElementById('oceanAudio');
    if (audio && audio.paused) { audio.play().catch(()=>{}); }
}, { once: true });
window.onload = function() {
    const audio = document.getElementById('oceanAudio');
    if (audio) { audio.volume = 0.25; }  // 기본 볼륨
}
</script>
"""

# 스타일/오디오 적용
st.markdown(OCEAN_THEME_CSS, unsafe_allow_html=True)
st.components.v1.html(OCEAN_AUDIO_HTML, height=0)

# ===============================
# 세션 상태 초기화 (last_meditation/meditation_log 추가)
# ===============================
DEFAULTS = {
    "current_page": "home",
    "mood_history": [],
    "breathing_sessions": 0,   # 명상 세션 카운터
    "meditation_time": 0,
    "gratitude_list": [],
    "journal_entries": [],
    "sleep_records": [],
    "habit_tracker": {},
    "emotion_history": [],
    "webcam_process": None,
    "was_webcam_running": False,
    "page_mode": "main",
    "daily_ocean_message": None,
    "last_meditation": None,       # {emotion, image_url, audio_url, audio_mime, ts}
    "meditation_log": [],          # 누적 로그
    "show_audio_player": False,    # 명시적 오디오 플레이어 표시 여부
    "audio_nonce": 0,              # 오디오 재표시 트리거
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

VALID_PAGES = {
    "home", "breathing_meditation", "emotion_journal", "sleep", "habits", "gratitude",
    "rps_game", "anger_game", "proverb_game", "emotion_game", "compass_card"
}
if st.session_state.get("current_page") not in VALID_PAGES:
    st.session_state.current_page = "home"

if "boot_routed" not in st.session_state:
    st.session_state.boot_routed = True
    st.session_state.current_page = "home"
    st.rerun()

# 초기 데이터 로드
if "data_loaded" not in st.session_state:
    load_data_from_json()
    st.session_state.data_loaded = True

# 바다 명언
OCEAN_MESSAGES = [
    "🌊 바다처럼 깊고 넓은 마음으로 하루를 시작해보세요",
    "🐚 조개가 진주를 만들 듯, 어려움도 소중한 경험이 됩니다",
    "🌅 새벽 바다의 고요함처럼 평온한 하루 되세요",
    "⚓ 항해에는 목적지가 있듯, 당신의 여정도 의미가 있습니다",
    "🏖️ 파도가 모래사장을 부드럽게 만들듯, 시간이 상처를 치유합니다",
    "🐠 물고기가 바다에서 자유롭듯, 당신도 자유로운 마음을 가지세요",
    "🌊 거센 파도도 결국 잔잔해지듯, 모든 감정은 흘러갑니다",
    "🗨️ 바다의 소리에 귀 기울이듯, 내 마음의 목소리를 들어보세요"
]
if not st.session_state.daily_ocean_message:
    st.session_state.daily_ocean_message = random.choice(OCEAN_MESSAGES)

def run_game_file(file_path):
    """외부 게임 스크립트 실행 (필요 시 사용)"""
    try:
        if os.path.exists(file_path):
            exec(open(file_path, encoding='utf-8').read())
        else:
            st.error(f"게임 파일을 찾을 수 없습니다: {file_path}")
    except Exception as e:
        st.error(f"게임 실행 중 오류: {str(e)}")

# ===============================
# 타로 (인라인 렌더)
# ===============================
def render_tarot_inline():
    import random as _random
    import streamlit as _st

    CARD_BACK = r"images\tarot.JPG.jpg"
    TAROT_MAJOR_ARCANA = {
        "0. The Fool (광대)": {"img": "https://upload.wikimedia.org/wikipedia/commons/9/90/RWS_Tarot_00_Fool.jpg", "msg": "새로운 시작을 두려워하지 마세요. 가벼운 마음으로 내딛는 첫걸음이 멋진 여정의 시작이 될 거예요."},
        "1. The Magician (마법사)": {"img": "https://upload.wikimedia.org/wikipedia/commons/d/de/RWS_Tarot_01_Magician.jpg", "msg": "당신 안에는 무한한 잠재력과 창의력이 있어요. 지금이 바로 그 힘을 믿고 발휘할 때입니다."},
        "2. The High Priestess (여사제)": {"img": "https://upload.wikimedia.org/wikipedia/commons/8/88/RWS_Tarot_02_High_Priestess.jpg", "msg": "때로는 시끄러운 세상의 소리보다 내면의 목소리에 귀 기울여 보세요. 답은 이미 당신 안에 있을지 몰라요."},
        "3. The Empress (여제)": {"img": "https://upload.wikimedia.org/wikipedia/commons/d/d2/RWS_Tarot_03_Empress.jpg", "msg": "풍요로움은 결과만이 아닌 과정 속에 있어요. 당신의 오늘 하루를 소중히 여기고 스스로를 아껴주세요."},
        "4. The Emperor (황제)": {"img": "https://upload.wikimedia.org/wikipedia/commons/c/c3/RWS_Tarot_04_Emperor.jpg", "msg": "안정감과 자신감을 가지세요. 당신은 스스로의 삶을 이끌어갈 충분한 힘과 책임감을 가지고 있습니다."},
        "5. The Hierophant (교황)": {"img": "https://upload.wikimedia.org/wikipedia/commons/8/8d/RWS_Tarot_05_Hierophant.jpg", "msg": "때로는 전통과 원칙 속에서 지혜를 얻을 수 있어요. 믿음직한 사람의 조언이 힘이 될 수 있습니다."},
        "6. The Lovers (연인)": {"img": "https://upload.wikimedia.org/wikipedia/commons/d/db/RWS_Tarot_06_Lovers.jpg", "msg": "당신의 마음이 이끄는 선택을 존중하세요. 조화로운 관계와 소통이 당신에게 기쁨을 가져다줄 거예요."},
        "7. The Chariot (전차)": {"img": "https://upload.wikimedia.org/wikipedia/commons/9/9b/RWS_Tarot_07_Chariot.jpg", "msg": "목표를 향해 힘차게 나아가세요. 강한 의지와 추진력이 있다면 어떤 어려움도 극복할 수 있습니다."},
        "8. Strength (힘)": {"img": "https://upload.wikimedia.org/wikipedia/commons/f/f5/RWS_Tarot_08_Strength.jpg", "msg": "진정한 힘은 억지로 누르는 것이 아니라 부드러움으로 다스리는 내면의 용기에서 나옵니다."},
        "9. The Hermit (은둔자)": {"img": "https://upload.wikimedia.org/wikipedia/commons/4/4d/RWS_Tarot_09_Hermit.jpg", "msg": "잠시 멈춰서 조용히 자신을 돌아볼 시간이 필요해요. 성찰의 시간은 당신의 길을 더 밝게 비춰줄 등불이 될 거예요."},
        "10. Wheel of Fortune (운명의 수레바퀴)": {"img": "https://upload.wikimedia.org/wikipedia/commons/3/3c/RWS_Tarot_10_Wheel_of_Fortune.jpg", "msg": "삶은 돌고 도는 것. 지금의 어려움은 곧 지나가고 새로운 기회가 찾아올 거예요. 변화의 흐름을 받아들이세요."},
        "11. Justice (정의)": {"img": "https://upload.wikimedia.org/wikipedia/commons/e/e0/RWS_Tarot_11_Justice.jpg", "msg": "균형과 조화를 찾아보세요. 당신이 내린 현명하고 공정한 판단이 좋은 결과를 가져올 것입니다."},
        "12. The Hanged Man (매달린 남자)": {"img": "https://upload.wikimedia.org/wikipedia/commons/2/2b/RWS_Tarot_12_Hanged_Man.jpg", "msg": "상황을 다른 관점에서 바라볼 필요가 있어요. 잠시 멈춰서 생각을 전환하면 새로운 해답이 보일 수 있습니다."},
        "13. Death (죽음)": {"img": "https://upload.wikimedia.org/wikipedia/commons/d/d7/RWS_Tarot_13_Death.jpg", "msg": "끝은 새로운 시작을 의미해요. 과거의 것을 떠나보낼 때, 더 나은 미래를 위한 공간이 생깁니다."},
        "14. Temperance (절제)": {"img": "https://upload.wikimedia.org/wikipedia/commons/f/f8/RWS_Tarot_14_Temperance.jpg", "msg": "마음의 평온과 조화를 유지하는 것이 중요해요. 차분하게 서로 다른 것들을 융화시킬 때 더 큰 시너지가 납니다."},
        "15. The Devil (악마)": {"img": "https://upload.wikimedia.org/wikipedia/commons/5/55/RWS_Tarot_15_Devil.jpg", "msg": "당신을 얽매는 부정적인 생각이나 습관이 있다면, 그것을 직시하고 벗어날 용기를 내보세요."},
        "16. The Tower (탑)": {"img": "https://upload.wikimedia.org/wikipedia/commons/5/53/RWS_Tarot_16_Tower.jpg", "msg": "예상치 못한 변화가 찾아올 수 있지만, 무너진 자리에 더 견고하고 새로운 것을 세울 기회가 될 수 있습니다."},
        "17. The Star (별)": {"img": "https://upload.wikimedia.org/wikipedia/commons/d/db/RWS_Tarot_17_Star.jpg", "msg": "희망을 잃지 마세요. 어두운 밤하늘에서도 별은 빛나듯, 당신의 꿈과 희망이 길을 밝혀줄 거예요."},
        "18. The Moon (달)": {"img": "https://upload.wikimedia.org/wikipedia/commons/7/7f/RWS_Tarot_18_Moon.jpg", "msg": "마음이 불안하고 미래가 흐릿하게 느껴질 수 있어요. 하지만 새벽이 오기 전이 가장 어두운 법입니다. 당신의 직감을 믿으세요."},
        "19. The Sun (태양)": {"img": "https://upload.wikimedia.org/wikipedia/commons/1/17/RWS_Tarot_19_Sun.jpg", "msg": "긍정적인 에너지와 성공이 당신과 함께합니다. 자신감을 갖고 밝은 미래를 마음껏 즐기세요."},
        "20. Judgement (심판)": {"img": "https://upload.wikimedia.org/wikipedia/commons/d/dd/RWS_Tarot_20_Judgement.jpg", "msg": "과거의 경험을 발판 삼아 한 단계 더 성장할 시간입니다. 스스로를 용서하고 새로운 부름에 응답하세요."},
        "21. The World (세계)": {"img": "https://upload.wikimedia.org/wikipedia/commons/f/ff/RWS_Tarot_21_World.jpg", "msg": "하나의 여정이 성공적으로 마무리되었어요. 당신이 이룬 성과를 축하하고, 또 다른 완성을 향해 나아가세요."}
    }

    IMG_WIDTH = 330
    _st.markdown('<div style="padding:8px 0"></div>', unsafe_allow_html=True)

    if "tarot_card_drawn" not in _st.session_state:
        _st.session_state.tarot_card_drawn = None

    if _st.session_state.tarot_card_drawn:
        name, img, msg = _st.session_state.tarot_card_drawn
        _st.markdown("#### 🃏 오늘의 나침반 카드")
        col1, col2 = _st.columns([1, 1.2])
        with col1:
            _st.image(img, width=IMG_WIDTH)
        with col2:
            _st.markdown(f"**{name}**")
            _st.markdown(
                f'<div style="background:#0b1220;color:#dbeafe;border:1px solid rgba(255,255,255,.08);'
                f'border-radius:12px;padding:14px 16px;line-height:1.7;max-width:720px">{msg}</div>',
                unsafe_allow_html=True
            )
        c1, c2 = _st.columns(2)
        with c1:
            if _st.button("🔁 다시 뽑기", use_container_width=True, key="tarot_redraw"):
                _st.session_state.tarot_card_drawn = None
                _st.rerun()
        with c2:
            if _st.button("🏠 항해 일지 홈", use_container_width=True, key="tarot_home"):
                _st.session_state.current_page = "home"
                _st.rerun()
    else:
        _st.markdown("#### 🎴 카드를 뒤집어 오늘의 메시지를 확인하세요")
        _st.image(CARD_BACK, width=IMG_WIDTH)
        if _st.button("🎴 카드 뽑기", use_container_width=True, key="tarot_draw"):
            name = _random.choice(list(TAROT_MAJOR_ARCANA.keys()))
            entry = TAROT_MAJOR_ARCANA[name]
            _st.session_state.tarot_card_drawn = (name, entry["img"], entry["msg"])
            _st.rerun()

def handle_exit_button():
    """종료 버튼 처리"""
    if 'confirm_exit' not in st.session_state:
        st.session_state.confirm_exit = False
    if not st.session_state.confirm_exit:
        if st.button("🚪 프로그램 종료", key="exit_app", use_container_width=True):
            st.session_state.confirm_exit = True
            st.rerun()
    else:
        st.warning("⚠️ 나의 감정 항해 일지를 떠나시겠습니까?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ 예", key="confirm_exit_yes", use_container_width=True):
                progress_placeholder = st.empty()
                with progress_placeholder.container():
                    with st.spinner("JSON 파일들을 정리하는 중..."):
                        st.write("🧹 세션 데이터 정리 중...")
                        time.sleep(1)
                        st.write("🗑️ 임시 파일 삭제 중...")
                        cleanup_success = cleanup_and_exit()
                        time.sleep(1)
                        if cleanup_success:
                            st.write("✅ 정리 완료!")
                        else:
                            st.write("⚠️ 일부 파일 정리 실패")
                st.rerun()
        with col2:
            if st.button("❌ 아니오", key="confirm_exit_no", use_container_width=True):
                st.session_state.confirm_exit = False
                st.rerun()

def show_analytics_page():
    all_history = load_all_emotion_data()
    if not all_history:
        st.warning("🔭 분석할 감정 데이터가 없습니다.")
        st.info("먼저 웹캠 프로그램을 실행하거나 수동으로 감정을 선택해주세요!")
        return

    st.success(f"✅ 총 {len(all_history)}개의 감정 기록이 로드되었습니다.")
    st.sidebar.header("📈 분석 설정")

    time_options = {
        "최근 10분": 10, "최근 30분": 30, "최근 1시간": 60, "최근 2시간": 120,
        "최근 6시간": 360, "최근 12시간": 720, "최근 24시간": 1440, "전체": 99999
    }
    selected_time = st.sidebar.selectbox("분석 시간 범위", list(time_options.keys()), index=1)
    minutes = time_options[selected_time]

    auto_refresh = st.sidebar.checkbox("자동 새로고침 (30초)", value=False)
    if auto_refresh:
        time.sleep(30)
        st.rerun()
    if st.sidebar.button("🔄 데이터 새로고침"):
        st.rerun()

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

    st.subheader(f"📊 감정 통계")
    stats_table = create_emotion_stats_table(all_history, minutes)
    if stats_table is not None:
        st.dataframe(stats_table, use_container_width=True)
    else:
        st.info("해당 시간 범위에 데이터가 없습니다.")

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
                } for entry in reversed(recent_data)
            ])
            st.dataframe(df_raw, use_container_width=True)
        else:
            st.info("해당 시간 범위에 데이터가 없습니다.")

def cleanup_and_exit():
    """프로그램 종료 시 JSON 파일 삭제"""
    json_files = [
        "ocean_healing_data.json",
        "emotion_history.json",
        "latest_emotion_result.json"
    ]
    deleted_files, failed_files = [], []

    cleanup_keys = [
        "mood_history", "breathing_sessions", "meditation_time",
        "gratitude_list", "journal_entries", "sleep_records",
        "habit_tracker", "emotion_history", "daily_ocean_message",
        "data_loaded", "webcam_process", "was_webcam_running",
        "last_meditation", "meditation_log", "show_audio_player"
    ]
    for key in cleanup_keys:
        if key in st.session_state:
            try: del st.session_state[key]
            except: pass

    time.sleep(0.5)

    for json_file in json_files:
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                if os.path.exists(json_file):
                    os.chmod(json_file, 0o777)
                    os.remove(json_file)
                    deleted_files.append(json_file)
                    print(f"✅ {json_file} 파일 삭제 (시도 {attempt + 1})")
                    break
                else:
                    print(f"ℹ️ {json_file} 파일이 존재하지 않습니다.")
                    break
            except PermissionError:
                if attempt < max_attempts - 1:
                    print(f"⏳ 권한 오류, 재시도... ({attempt + 1}/{max_attempts})")
                    time.sleep(1)
                else:
                    print(f"❌ 권한 없음: {json_file}")
                    failed_files.append(json_file)
            except OSError as e:
                if attempt < max_attempts - 1:
                    print(f"⏳ OSError, 재시도... ({attempt + 1}/{max_attempts}): {e}")
                    time.sleep(1)
                else:
                    print(f"❌ 삭제 실패: {json_file} - {e}")
                    failed_files.append(json_file)
            except Exception as e:
                if attempt < max_attempts - 1:
                    print(f"⏳ 예외, 재시도... ({attempt + 1}/{max_attempts}): {e}")
                    time.sleep(1)
                else:
                    print(f"❌ 예외로 삭제 실패: {json_file} - {e}")
                    failed_files.append(json_file)

    if deleted_files:
        st.success(f"🗑️ 삭제된 파일: {', '.join(deleted_files)}")
    if failed_files:
        st.error(f"❌ 삭제 실패 파일: {', '.join(failed_files)}")
        st.warning("일부 파일이 사용 중일 수 있습니다. 앱 종료 후 수동으로 삭제해주세요.")

    st.session_state.exit_in_progress = True
    return len(failed_files) == 0

def show_exit_screen():
    st.markdown("""
    <div style="text-align:center; padding: 40px; background: linear-gradient(135deg,#E0F6FF,#B0E0E6); border-radius: 20px; margin: 20px 0;">
        <div style="font-size: 4em; margin-bottom: 20px;">🌊</div>
        <h1 style="color:#4A90E2; margin: 0 0 15px 0;">바다로 돌아가는 중...</h1>
        <p style="color:#5DADE2; font-size:1.1em; margin: 0 0 20px 0;">소중한 항해 기록들을 안전하게 보관하고 있어요</p>
        <div style="background:#ffffff; border-radius:15px; padding:20px; margin: 20px auto; max-width: 600px; box-shadow:0 4px 20px rgba(74,144,226,.2);">
            <div style="color:#4A90E2; font-size:1.1em;">🐚 추억들을 조심스럽게 정리하고 있어요...</div>
        </div>
        <div style="color:#5DADE2; font-style:italic; margin-top: 15px; padding:15px; background: rgba(127,179,211,0.15); border-radius:12px; display:inline-block;">
            오늘도 마음의 항해를 함께해주셔서 감사했어요 🌊💙
        </div>
    </div>
    """, unsafe_allow_html=True)

    progress_bar = st.progress(0)
    messages = [
        "🌊 항해 일지를 정리하고 있어요",
        "⚓ 감정의 닻을 안전하게 내리고 있어요",
        "🐚 소중한 기억들을 보관하고 있어요",
        "🌅 오늘의 항해를 마무리하고 있어요"
    ]
    status_placeholder = st.empty()
    for i, message in enumerate(messages):
        status_placeholder.markdown(f"<div style='text-align: center; color: #4A90E2; font-size: 1.2em;'>{message}</div>", unsafe_allow_html=True)
        progress_bar.progress((i + 1) / len(messages))
        time.sleep(1.5)

    save_data_to_json()
    st.success("🌊 정리가 완료되었습니다. 브라우저 탭을 닫으시면 됩니다.")
    st.stop()

if st.session_state.get('exit_in_progress', False):
    show_exit_screen()

# 사이드바 네비게이션
# ─────────────────────────────────────────
# 사이드바
# ─────────────────────────────────────────


st.markdown("""
<style>
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    # 헤더
    st.markdown('<h1>🌊 마음 항해 연구소</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center; color:rgba(255,255,255,0.9); margin-bottom:18px;">마음의 항해를 시작해요</p>', unsafe_allow_html=True)

    # 항해일지 (메인)
    st.markdown("### 🧭 항해일지")
    if st.button("🧭 항해일지", key="nav_home", use_container_width=True):
        st.session_state.current_page = "home"
        save_data_to_json()
        st.rerun()

    nav_categories = [
        ("🌬️ AI 파도 명상", "breathing_meditation"),
        ("💤 수면관리", "sleep"),
        ("🎯 습관트래커", "habits"),
        ("📓 감정 저널", "emotion_journal"),
        ("🙏 감사 일기", "gratitude"),
    ]
    for name, key in nav_categories:
        if st.button(name, key=f"nav_{key}", use_container_width=True):
            st.session_state.current_page = key
            save_data_to_json()
            st.rerun()

    # 파도놀이터: 게임 카테고리
    st.markdown("### 🌊 파도놀이터")
    game_categories = [
        ("✂️ 가위바위보 게임", "rps_game", "pages/rr.py"),
        ("⚡ 암초 깨기 게임", "anger_game", "pages/app.py"),
        ("📚 속담 게임", "proverb_game", "pages/seafaring_proverb_quiz.py"),
        ("🎭 표정 연기 게임", "emotion_game", "pages/app_streamlit.py"),
        ("🧭 오늘의 나침반 카드", "compass_card", "healing_center_tarot.py"),
    ]
    for name, key, file_path in game_categories:
        if st.button(name, key=f"nav_{key}", use_container_width=True):
            st.session_state.current_page = key
            st.session_state.game_file_path = file_path
            save_data_to_json()
            st.rerun()

    # 종료 섹션
    st.markdown('<div class="exit-section">', unsafe_allow_html=True)
    handle_exit_button()
    st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# 메인 컨텐츠
# ===============================
if st.session_state.current_page == "home":
    st.markdown('<div class="ocean-page-title">🧭 나의 감정 항해 일지</div>', unsafe_allow_html=True)

    st.markdown(f'<div class="ocean-quote">"{st.session_state.daily_ocean_message}"</div>', unsafe_allow_html=True)

    if st.button('🌊 새로운 바다 메시지', key='btn_new_ocean_msg', use_container_width=False):
        try:
            new_msg = random.choice([m for m in OCEAN_MESSAGES if m != st.session_state.daily_ocean_message])
        except Exception:
            new_msg = st.session_state.daily_ocean_message
        st.session_state.daily_ocean_message = new_msg
        save_data_to_json()
        st.rerun()

    st.markdown('<div class="wave-decoration"></div>', unsafe_allow_html=True)
    show_analytics_page()

    st.markdown('<div class="ocean-container">', unsafe_allow_html=True)
    st.markdown("### 🌊 오늘의 항해 현황")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🧘 명상 세션", f"{st.session_state.breathing_sessions}회")
    with col2:
        st.metric("🧘 명상 시간", f"{st.session_state.meditation_time}분")
    with col3:
        st.metric("📓 감정 기록", f"{len(st.session_state.emotion_history)}개")
    with col4:
        st.metric("🙏 감사 표현", f"{len(st.session_state.gratitude_list)}개")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="ocean-container">', unsafe_allow_html=True)
    st.markdown("### ⚡ 빠른 항해")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🧘 바로 명상하기", use_container_width=True):
            st.session_state.current_page = "breathing_meditation"
            save_data_to_json()
            st.rerun()
    with col2:
        if st.button("📓 감정 기록하기", use_container_width=True):
            st.session_state.current_page = "emotion_journal"
            save_data_to_json()
            st.rerun()
    with col3:
        if st.button("🧭 나침반 뽑기", use_container_width=True):
            st.session_state.current_page = "compass_card"
            save_data_to_json()
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.current_page == "breathing_meditation":
    # 제목은 기존 스타일 유지
    st.markdown('<div class="ocean-page-title">🌬️ AI 파도 명상</div>', unsafe_allow_html=True)
    st.markdown('<div class="ocean-container">', unsafe_allow_html=True)

    # ✅ 명상 전용 UI (요청: 이전 명상/기록 UI 제거, 버튼 아래 이미지+오디오 표시)
    st.subheader("💡 AI 기반 맞춤 명상")
    st.info("AI가 당신의 현재 감정에 맞는 바다 명상 경험을 준비해 드릴게요.")

    # 이전 명상/로그 UI **제거** (요청 반영)

    # 2) 새 명상 생성 영역  ⛳️ (텍스트 입력 → 선택지)
    EMOTION_CHOICES = list(OCEAN_EXPERIENCE_MAPPING.keys())

    # 기본값을 '평온'으로(없으면 첫 항목)
    default_idx = EMOTION_CHOICES.index("평온") if "평온" in EMOTION_CHOICES else 0
    selected_emotion = st.selectbox(
        "🧠 AI가 그릴 당신의 지금 감정을 고르세요",
        EMOTION_CHOICES,
        index=default_idx,
        key="meditation_emotion_select",
    )
    st.caption("위 목록에서 감정을 선택하면, 이미지와 파도 소리를 준비합니다.")

    # ▼ 여기 key를 'btn_generate_select'로 변경
    if st.button("🤖 AI 바다 경험 생성", use_container_width=True, key="btn_generate_select"):
        # 선택된 감정으로 바로 생성 (analyze_emotion 호출 안 함)
        if not openai.api_key:
            st.warning("API 키가 없어 AI 이미지를 생성할 수 없습니다. (오디오는 재생 가능합니다)")
        if generate_and_display_ocean_experience(selected_emotion):
            st.session_state.breathing_sessions += 1      # 명상 세션 카운트
            st.session_state.meditation_time += 5         # 명상 시간(분)
            save_data_to_json()
            st.success("🧘 명상을 준비했어요. 위쪽 '이전 명상 이어서 보기'에서 ▶ 버튼을 눌러 파도 소리를 들으세요.")


    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.current_page == "emotion_journal":
    st.markdown('<div class="ocean-page-title">📓 감정 저널</div>', unsafe_allow_html=True)
    st.markdown('<div class="ocean-container">', unsafe_allow_html=True)

    st.markdown("### 🌊 감정의 바다 일기")
    col1, col2 = st.columns([2, 1])
    with col1:
        mood_slider = st.select_slider(
            "현재 기분을 파도로 표현한다면?",
            options=["🌊 큰 파도 (매우 안 좋음)", "🌀 거친 파도 (안 좋음)", "〰️ 잔잔한 물결 (보통)", "🌊 순한 파도 (좋음)", "✨ 고요한 바다 (매우 좋음)"],
            value="〰️ 잔잔한 물결 (보통)"
        )
        emotion_text = st.text_area(
            "💭 지금의 감정과 상황을 자세히 적어보세요",
            placeholder="오늘 있었던 일이나 지금 느끼는 감정을 솔직하게 표현해보세요. AI가 당신의 감정을 분석해드립니다.",
            height=150
        )
        emotion_tags = st.multiselect("추가 감정 태그 (선택)",
                                      ["😊행복", "😢슬픔", "😰불안", "😡분노", "🥱피로", "🤩열정", "😌평온", "😔우울", "🤗감사", "😤짜증"])
    with col2:
        st.markdown("""
        <div style='text-align:center; padding:20px; background:rgba(127,179,211,0.1); border-radius:15px; margin-bottom:20px;'>
            <div style='font-size:3em; margin-bottom:10px;'>🌊</div>
            <div style='color:#4A90E2; font-weight:bold;'>감정 분석 AI</div>
            <div style='color:#666; font-size:0.9em; margin-top:5px;'>텍스트를 분석하여<br>감정을 파악합니다</div>
        </div>
        """, unsafe_allow_html=True)
        is_private = st.checkbox("🔒 개인 일기 (비공개)")

    if st.button("🌊 감정을 바다에 기록하기", type="primary", use_container_width=True) and emotion_text:
        emotion, confidence = analyze_emotion(emotion_text)
        emotion_record = {
            "mood_slider": mood_slider,
            "content": emotion_text,
            "ai_emotion": emotion,
            "confidence": confidence,
            "emotion_tags": emotion_tags,
            "is_private": is_private,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "time": datetime.now().strftime("%H:%M"),
            "word_count": len(emotion_text.split())
        }
        st.session_state.journal_entries.append(emotion_record)
        save_data_to_json()
        st.success("✅ 감정이 바다에 기록되었습니다!")
        col_ai1, col_ai2 = st.columns(2)
        with col_ai1:
            st.info(f"🤖 **AI 감정 분석**\n감정: {emotion}")
        with col_ai2:
            st.info(f"📊 **분석 신뢰도**\n{confidence:.1f}%")

    if st.session_state.emotion_history:
        st.markdown("#### 🌊 최근 감정의 파도들")
        show_private = st.checkbox("🔒 비공개 일기도 보기")
        display_records = [r for r in st.session_state.emotion_history if (show_private or not r.get("is_private", False))]
        for record in reversed(display_records[-5:]):
            with st.expander(f"{record.get('mood_slider', '').split('(')[0] if record.get('mood_slider') else '감정기록'} - {record['date']} {record['time']} {'🔒' if record.get('is_private') else ''}"):
                content = record.get('content', record.get('note', ''))
                preview = content[:150] + "..." if len(content) > 150 else content
                st.write(f"**내용:** {preview}")
                if record.get('ai_emotion'):
                    st.write(f"**AI 분석:** {record['ai_emotion']} ({record.get('confidence', 0):.1f}%)")
                if record.get('emotion_tags'):
                    st.write(f"**태그:** {', '.join(record['emotion_tags'])}")
                if record.get('word_count'):
                    st.write(f"**글자 수:** {record['word_count']}단어")

    if len(st.session_state.emotion_history) > 0:
        st.markdown("#### 📊 나의 감정 통계")
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        with col_stat1:
            st.metric("📝 총 기록", f"{len(st.session_state.emotion_history)}개")
        with col_stat2:
            recent = [r.get('ai_emotion', '알 수 없음') for r in st.session_state.emotion_history[-7:]]
            most_common = max(set(recent), recent.count) if recent else "없음"
            st.metric("🎭 주요 감정", most_common)
        with col_stat3:
            total_words = sum(r.get('word_count', 0) for r in st.session_state.emotion_history)
            st.metric("✍️ 총 단어", f"{total_words}개")

    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.current_page == "sleep":
    st.markdown('<div class="ocean-page-title">💤 수면관리</div>', unsafe_allow_html=True)
    st.markdown('<div class="ocean-container">', unsafe_allow_html=True)

    st.markdown("### 🌙 밤바다 수면 일지")
    col1, col2 = st.columns(2)
    with col1:
        sleep_quality = st.slider("수면의 질 (1~10)", 1, 10, 7)
        sleep_hours = st.number_input("수면 시간(시간)", 0.0, 14.0, 7.5, 0.5)
    with col2:
        bedtime = st.time_input("잠든 시간", value=datetime.strptime("23:00", "%H:%M").time())
        wake_time = st.time_input("일어난 시간", value=datetime.strptime("07:00", "%H:%M").time())

    sleep_issues = st.multiselect("수면 방해 요소",
                                  ["스트레스", "카페인", "스마트폰", "소음", "온도", "몸 불편", "걱정", "파도 소리가 그리워서", "없음"])
    dream_note = st.text_area("기억나는 꿈 (선택사항)", height=80, placeholder="바다나 항해 관련 꿈을 꾸셨나요?")

    if st.button("🌙 수면 일지 저장", use_container_width=True):
        sleep_record = {
            "quality": sleep_quality,
            "hours": sleep_hours,
            "bedtime": bedtime.strftime("%H:%M"),
            "wake_time": wake_time.strftime("%H:%M"),
            "issues": sleep_issues,
            "dream": dream_note or None,
            "date": datetime.now().strftime("%Y-%m-%d")
        }
        st.session_state.sleep_records.append(sleep_record)
        save_data_to_json()
        st.success("💤 수면 일지가 저장되었습니다! 좋은 꿈 꾸세요 🌊")

    if st.session_state.sleep_records:
        with st.expander("🌙 최근 수면 기록 보기"):
            for record in reversed(st.session_state.sleep_records[-7:]):
                issues_text = ', '.join(record['issues']) if record['issues'] else '없음'
                st.write(f"**{record['date']}** — {record['hours']}시간, 질 {record['quality']}/10, 잠:{record['bedtime']}, 기상:{record['wake_time']} / 방해요소: {issues_text}")
                if record.get('dream'):
                    st.write(f"꿈: {record['dream']}")

    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.current_page == "habits":
    st.markdown('<div class="ocean-page-title">🎯 습관트래커</div>', unsafe_allow_html=True)
    st.markdown('<div class="ocean-container">', unsafe_allow_html=True)

    st.markdown("### ⚓ 매일의 항해 습관")
    new_habit = st.text_input("새로운 항해 습관 추가", placeholder="예: 바다 소리 5분 듣기, 깊은 호흡 10회")
    if st.button("➕ 습관 추가") and new_habit:
        if new_habit not in st.session_state.habit_tracker:
            st.session_state.habit_tracker[new_habit] = {"done_dates": set()}
            save_data_to_json()
        st.success(f"⚓ '{new_habit}' 습관이 추가되었습니다!")

    today = datetime.now().strftime("%Y-%m-%d")
    if st.session_state.habit_tracker:
        st.markdown("#### 🌊 오늘의 습관 체크")
        for habit, data in st.session_state.habit_tracker.items():
            checked = today in data["done_dates"]
            new_checked = st.checkbox(f"{habit}", value=checked, key=f"habit_{habit}")
            if new_checked and not checked:
                data["done_dates"].add(today); save_data_to_json(); st.success(f"🎉 '{habit}' 완료!")
            elif not new_checked and checked:
                data["done_dates"].discard(today); save_data_to_json()

        with st.expander("📊 항해 습관 통계 보기"):
            for habit, data in st.session_state.habit_tracker.items():
                completion_count = len(data['done_dates'])
                st.write(f"⚓ **{habit}**: {completion_count}일 완료")
                dates = sorted(data['done_dates'], reverse=True)
                streak = 0
                for i, date_str in enumerate(dates):
                    expected_date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
                    if date_str == expected_date: streak += 1
                    else: break
                if streak > 0:
                    st.write(f"  🔥 연속 {streak}일 달성!")
    else:
        st.info("🌊 아직 설정된 항해 습관이 없습니다. 위에서 새로운 습관을 추가해보세요!")

    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.current_page == "gratitude":
    st.markdown('<div class="ocean-page-title">🙏 감사 일기</div>', unsafe_allow_html=True)
    st.markdown('<div class="ocean-container">', unsafe_allow_html=True)

    st.markdown("### 🌊 바다처럼 넓은 감사의 마음")
    gratitude_categories = {
        "🌊 바다": "바다, 파도, 해변, 바닷바람",
        "👥 사람": "가족, 친구, 동료, 지인",
        "🌟 경험": "특별한 순간, 배움, 성취",
        "🏠 일상": "평범한 하루, 작은 행복",
        "🌍 자연": "날씨, 계절, 주변 환경",
        "💪 건강": "몸과 마음의 건강",
        "🎁 물질": "소유하고 있는 것들"
    }
    category = st.selectbox("감사 카테고리:", list(gratitude_categories.keys()))
    st.info(f"💡 {gratitude_categories[category]}")
    gratitude_item = st.text_input("구체적으로 감사한 것:", placeholder="예: 오늘 본 아름다운 석양")
    gratitude_reason = st.text_area("왜 감사한가요?", height=100, placeholder="그 순간이 마음에 평화를 가져다주었기 때문에...")

    if st.button("🙏 감사 기록 추가", use_container_width=True) and gratitude_item:
        gratitude_record = {
            "item": gratitude_item,
            "reason": gratitude_reason,
            "category": category,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M")
        }
        st.session_state.gratitude_list.append(gratitude_record)
        save_data_to_json()
        st.success("🌟 감사 기록이 바다에 새겨졌습니다!")
        st.balloons()

    if st.session_state.gratitude_list:
        st.markdown("#### 🌊 최근 감사의 파도들")
        for entry in reversed(st.session_state.gratitude_list[-5:]):
            with st.expander(f"{entry['category']} - {entry['item']} ({entry['date']})"):
                if entry["reason"]:
                    st.write(f"이유: {entry['reason']}")

    daily_challenges = [
        "오늘 만난 사람 중 한 명에게 고마움을 표현해보세요",
        "바다처럼 넓은 마음으로 평범한 순간에 감사해보세요",
        "어려웠던 상황에서도 배운 점을 찾아보세요",
        "내 몸이 해주는 일들에 감사해보세요",
        "오늘의 작은 성취를 인정해보세요"
    ]
    if "daily_gratitude_challenge" not in st.session_state:
        st.session_state.daily_gratitude_challenge = random.choice(daily_challenges)

    st.info(f"🌊 오늘의 감사 챌린지: {st.session_state.daily_gratitude_challenge}")
    if st.button("🔄 새로운 챌린지"):
        st.session_state.daily_gratitude_challenge = random.choice(daily_challenges)
        save_data_to_json()
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.current_page in ["rps_game", "anger_game", "proverb_game", "emotion_game", "compass_card"]:
    game_titles = {
        "rps_game": "✂️ 가위바위보 게임",
        "anger_game": "⚡ 암초 깨기 게임",
        "proverb_game": "📚 속담 게임",
        "emotion_game": "🎭 표정 연기 게임",
        "compass_card": "🧭 오늘의 나침반 카드"
    }
    st.markdown(f'<div class="ocean-page-title">{game_titles[st.session_state.current_page]}</div>', unsafe_allow_html=True)
    st.markdown('<div class="ocean-container">', unsafe_allow_html=True)

    if st.session_state.current_page == 'compass_card':
        render_tarot_inline()
    elif hasattr(st.session_state, 'game_file_path'):
        file_path = st.session_state.game_file_path
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    game_code = file.read()
                    import streamlit as _st
                _orig_spc = getattr(_st, 'set_page_config', None)
                if os.path.basename(file_path) == 'healing_center_tarot.py' and callable(_orig_spc):
                    try:
                        _st.set_page_config = lambda *a, **k: None
                        exec(game_code, globals())
                    finally:
                        _st.set_page_config = _orig_spc
                else:
                    exec(game_code, globals())
            except Exception as e:
                st.error(f"게임 실행 중 오류: {str(e)}")
                st.info("게임 파일을 확인해주세요.")
        else:
            st.error(f"게임 파일을 찾을 수 없습니다: {file_path}")
            st.info("게임 파일이 올바른 위치에 있는지 확인해주세요.")
    else:
        st.info(f"🚧 {game_titles[st.session_state.current_page]} 로딩 중입니다...")

    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown('<div class="ocean-page-title">🌊 알 수 없는 항해</div>', unsafe_allow_html=True)
    st.markdown('<div class="ocean-container">', unsafe_allow_html=True)
    st.warning("🧭 잘못된 항로입니다. 메뉴에서 올바른 목적지를 선택해주세요.")
    st.markdown('</div>', unsafe_allow_html=True)

# 자동 저장
save_data_to_json()

# 하단 정보
st.markdown('<div class="ocean-container">', unsafe_allow_html=True)
st.markdown("### 🆘 긴급 지원 & 도움말")
st.info("""
**자살예방상담전화**: 109 (24시간) 	
**생명의전화**: 1588-9191 	
**청소년전화**: 1388 	
**정신건강위기상담전화**: 1577-0199
""")
st.warning("""
다음과 같은 경우 전문가의 도움을 받으세요: 	
• 2주 이상 지속되는 우울감 	
• 자해나 자살 생각 	
• 일상생활이 어려울 정도의 불안 	
• 수면/식사 패턴의 심각한 변화
""")
st.markdown('</div>', unsafe_allow_html=True)
