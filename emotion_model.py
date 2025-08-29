# emotion_model.py

"""
감정 분석 모델 로드 및 추론 모듈
"""

import cv2
import numpy as np
from PIL import Image
import streamlit as st
import threading
import time

# 전역 변수
emotion_pipeline = None
latest_emotion = None
latest_confidence = 0.0
emotion_lock = threading.Lock()

# 감정 매핑 (모델 출력 -> 우리 앱 형식)
EMOTION_MAPPING = {
    'angry': 'angry',
    'sad': 'sad', 
    'happy': 'happy',
    'fear': 'fear',
    'surprise': 'surprise',
    'neutral': 'neutral',
    'disgust': 'neutral'  # disgust는 neutral로 매핑
}

@st.cache_resource
def load_emotion_model():
    """
    감정 분석 모델을 로드합니다 (캐시됨)
    Returns:
        pipeline: Hugging Face 감정 분석 파이프라인
    """
    try:
        from transformers import pipeline
        
        # Hugging Face 모델 로드
        pipe = pipeline(
            'image-classification', 
            model="dima806/facial_emotions_image_detection",
            device=-1  # CPU 사용 (-1), GPU 사용시 0
        )
        return pipe
    except ImportError:
        st.error("transformers 라이브러리가 설치되지 않았습니다. requirements.txt를 확인해주세요.")
        return None
    except Exception as e:
        st.error(f"모델 로드 실패: {str(e)}")
        return None

def analyze_emotion_from_image(image):
    """
    PIL 이미지에서 감정을 분석합니다
    Args:
        image: PIL Image 객체
    Returns:
        tuple: (emotion_key, confidence)
    """
    global emotion_pipeline
    
    if emotion_pipeline is None:
        emotion_pipeline = load_emotion_model()
    
    if emotion_pipeline is None:
        return 'neutral', 0.0
    
    try:
        # 감정 분석 실행
        results = emotion_pipeline(image)
        
        if results and len(results) > 0:
            # 가장 높은 확률의 감정
            top_result = results[0]
            emotion = top_result['label']
            confidence = top_result['score']
            
            # 우리 앱의 감정 키로 매핑
            mapped_emotion = EMOTION_MAPPING.get(emotion, 'neutral')
            
            return mapped_emotion, confidence
        else:
            return 'neutral', 0.0
            
    except Exception as e:
        print(f"감정 분석 오류: {e}")
        return 'neutral', 0.0

def detect_face_and_analyze(image_array):
    """
    이미지에서 얼굴을 감지하고 감정을 분석합니다
    Args:
        image_array: OpenCV 형식의 이미지 배열 (BGR)
    Returns:
        tuple: (emotion_key, confidence, face_coordinates)
    """
    try:
        # 얼굴 감지기 초기화
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # 그레이스케일 변환
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        
        # 얼굴 감지
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            # 가장 큰 얼굴 선택
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = largest_face
            
            # 얼굴 영역 추출
            face_img = image_array[y:y+h, x:x+w]
            
            # BGR -> RGB 변환 후 PIL 이미지로 변환
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_pil = Image.fromarray(face_rgb)
            
            # 감정 분석
            emotion, confidence = analyze_emotion_from_image(face_pil)
            
            return emotion, confidence, (x, y, w, h)
        else:
            return 'neutral', 0.0, None
            
    except Exception as e:
        print(f"얼굴 감지 및 분석 오류: {e}")
        return 'neutral', 0.0, None

def update_latest_emotion(emotion, confidence):
    """
    최신 감정 상태를 업데이트합니다 (스레드 안전)
    """
    global latest_emotion, latest_confidence
    
    with emotion_lock:
        latest_emotion = emotion
        latest_confidence = confidence

def get_latest_emotion():
    """
    최신 감정 상태를 가져옵니다 (스레드 안전)
    Returns:
        tuple: (emotion_key, confidence)
    """
    global latest_emotion, latest_confidence
    
    with emotion_lock:
        return latest_emotion, latest_confidence

def reset_emotion_state():
    """
    감정 상태를 초기화합니다
    """
    global latest_emotion, latest_confidence
    
    with emotion_lock:
        latest_emotion = None
        latest_confidence = 0.0