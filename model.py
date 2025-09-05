"""
감정 분석 모델 로드 및 추론 모듈 - 통합 최적화 버전
web_with_graphs.py와 호환되는 통합 모델 매니저
"""

import cv2
import numpy as np
from PIL import Image
import streamlit as st
import threading
import time
import logging
from typing import Tuple, Optional, Union
from functools import lru_cache
import warnings

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaceTracker:
    """안정적인 단일 얼굴 추적을 위한 클래스 - web_with_graphs.py와 동일"""
    
    def __init__(self, max_missing_frames=10, min_overlap_ratio=0.3):
        self.tracked_face = None
        self.missing_frames = 0
        self.max_missing_frames = max_missing_frames
        self.min_overlap_ratio = min_overlap_ratio
        self.face_history = []
        self.max_history = 5
        
    def calculate_overlap_ratio(self, rect1, rect2):
        """두 사각형의 겹치는 비율 계산"""
        x1_1, y1_1, w1, h1 = rect1
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        
        x1_2, y1_2, w2, h2 = rect2
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2
        
        # 겹치는 영역 계산
        overlap_x1 = max(x1_1, x1_2)
        overlap_y1 = max(y1_1, y1_2)
        overlap_x2 = min(x2_1, x2_2)
        overlap_y2 = min(y2_1, y2_2)
        
        if overlap_x1 >= overlap_x2 or overlap_y1 >= overlap_y2:
            return 0.0
        
        overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
        rect1_area = w1 * h1
        
        return overlap_area / rect1_area if rect1_area > 0 else 0.0
    
    def find_best_face(self, faces):
        """가장 적합한 얼굴 선택"""
        if len(faces) == 0:
            return None
        
        if self.tracked_face is None:
            return max(faces, key=lambda x: x[2] * x[3])
        
        # 기존 추적 얼굴과 겹치는 얼굴들 찾기
        matching_faces = []
        for face in faces:
            overlap = self.calculate_overlap_ratio(self.tracked_face, face)
            if overlap >= self.min_overlap_ratio:
                matching_faces.append((face, overlap))
        
        if matching_faces:
            return max(matching_faces, key=lambda x: x[1])[0]
        else:
            return max(faces, key=lambda x: x[2] * x[3])
    
    def update_tracking(self, faces, scale=1.0):
        """얼굴 추적 업데이트"""
        if len(faces) == 0:
            self.missing_frames += 1
            if self.missing_frames > self.max_missing_frames:
                self.tracked_face = None
                self.face_history.clear()
            return self.tracked_face
        
        # 스케일 조정
        scaled_faces = []
        for x, y, w, h in faces:
            if scale != 1.0:
                x, y, w, h = int(x/scale), int(y/scale), int(w/scale), int(h/scale)
            scaled_faces.append((x, y, w, h))
        
        best_face = self.find_best_face(scaled_faces)
        
        if best_face is not None:
            self.tracked_face = best_face
            self.missing_frames = 0
            
            self.face_history.append(best_face)
            if len(self.face_history) > self.max_history:
                self.face_history.pop(0)
        
        return self.tracked_face
    
    def get_stable_face(self):
        """안정화된 얼굴 위치 반환"""
        if not self.face_history:
            return self.tracked_face
        
        # 최근 얼굴 위치들의 평균으로 안정화
        avg_x = sum(face[0] for face in self.face_history) // len(self.face_history)
        avg_y = sum(face[1] for face in self.face_history) // len(self.face_history)
        avg_w = sum(face[2] for face in self.face_history) // len(self.face_history)
        avg_h = sum(face[3] for face in self.face_history) // len(self.face_history)
        
        return (avg_x, avg_y, avg_w, avg_h)
    
    def reset(self):
        """트래커 리셋"""
        self.tracked_face = None
        self.missing_frames = 0
        self.face_history.clear()


class UnifiedModelManager:
    """통합 모델 관리 싱글톤 클래스 - 중복 로딩 방지"""
    _instance = None
    _lock = threading.RLock()  # 재진입 가능한 락
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        self._emotion_pipeline = None
        self._face_cascade = None
        self._model_id = "dima806/facial_emotions_image_detection"
        self._device = -1  # CPU 사용
        self._initialized = True
        
        logger.info("🏗️ UnifiedModelManager 초기화")
    
    @property
    def emotion_pipeline(self):
        """지연 로딩으로 감정 분석 파이프라인 반환"""
        if self._emotion_pipeline is None:
            self._emotion_pipeline = self._load_emotion_model()
        return self._emotion_pipeline
    
    @property 
    def face_cascade(self):
        """지연 로딩으로 얼굴 감지기 반환"""
        if self._face_cascade is None:
            self._face_cascade = self._load_face_cascade()
        return self._face_cascade
    
    def _load_emotion_model(self):
        """감정 분석 모델 로드"""
        try:
            logger.info("🔄 감정 분석 모델 로드 중...")
            
            # 경고 메시지 억제
            warnings.filterwarnings("ignore", category=UserWarning)
            
            from transformers import pipeline, AutoImageProcessor
            
            # Streamlit 환경에서는 캐시 사용
            if 'st' in globals() and hasattr(st, 'cache_resource'):
                @st.cache_resource
                def create_pipeline():
                    processor = AutoImageProcessor.from_pretrained(self._model_id, use_fast=True)
                    return pipeline(
                        'image-classification', 
                        model=self._model_id,
                        image_processor=processor,
                        device=self._device,
                        return_all_scores=False  # 성능 향상
                    )
                pipe = create_pipeline()
            else:
                # 일반 환경에서는 직접 생성
                processor = AutoImageProcessor.from_pretrained(self._model_id, use_fast=True)
                pipe = pipeline(
                    'image-classification', 
                    model=self._model_id,
                    image_processor=processor,
                    device=self._device,
                    return_all_scores=False
                )
            
            # 모델 테스트
            test_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            test_result = pipe(test_img)
            
            logger.info(f"✅ 감정 분석 모델 로드 완료! 테스트: {test_result[0]['label'] if test_result else 'No result'}")
            return pipe
            
        except ImportError as e:
            error_msg = "transformers 라이브러리가 설치되지 않았습니다."
            logger.error(f"{error_msg}: {e}")
            if 'st' in globals():
                st.error(error_msg)
            return None
        except Exception as e:
            error_msg = f"감정 분석 모델 로드 실패: {str(e)}"
            logger.error(error_msg)
            if 'st' in globals():
                st.error(error_msg)
            return None
    
    def _load_face_cascade(self):
        """얼굴 감지기 로드"""
        try:
            logger.info("🔄 얼굴 감지기 로드 중...")
            
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            face_cascade = cv2.CascadeClassifier(cascade_path)
            
            if face_cascade.empty():
                raise ValueError("얼굴 감지기 로드 실패")
            
            logger.info("✅ 얼굴 감지기 로드 완료!")
            return face_cascade
            
        except Exception as e:
            logger.error(f"얼굴 감지기 로드 오류: {e}")
            return None
    
    def get_emotion_pipeline(self):
        """외부 호출용 - web_with_graphs.py 호환성"""
        return self.emotion_pipeline
    
    def get_face_cascade(self):
        """외부 호출용 - web_with_graphs.py 호환성"""
        return self.face_cascade


class EmotionAnalyzer:
    """통합 감정 분석 클래스"""
    
    # 감정 매핑 (모델 출력 -> 앱 형식)
    EMOTION_MAPPING = {
        'angry': 'angry',
        'anger': 'angry',
        'sad': 'sad',
        'sadness': 'sad',
        'happy': 'happy',
        'happiness': 'happy',
        'fear': 'fear',
        'fearful': 'fear',
        'surprise': 'surprise',
        'surprised': 'surprise',
        'neutral': 'neutral',
        'disgust': 'disgust',
        'disgusted': 'disgust'
    }
    
    def __init__(self):
        self._model_manager = UnifiedModelManager()
        self._latest_emotion = None
        self._latest_confidence = 0.0
        self._lock = threading.RLock()
        self._face_tracker = FaceTracker()
        
        logger.info("🎭 EmotionAnalyzer 초기화 완료")
    
    @property
    def pipeline(self):
        """감정 분석 파이프라인 반환"""
        return self._model_manager.emotion_pipeline
    
    @property 
    def face_cascade(self):
        """얼굴 감지기 반환"""
        return self._model_manager.face_cascade
    
    def _preprocess_image(self, image: Image.Image, target_size: Tuple[int, int] = (224, 224)) -> Image.Image:
        """이미지 전처리 최적화"""
        try:
            # 이미지 크기 조정
            if image.size != target_size:
                image = image.resize(target_size, Image.Resampling.LANCZOS)
            
            # RGB 변환 확인
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            return image
        except Exception as e:
            logger.error(f"이미지 전처리 오류: {e}")
            return image
    
    def analyze_emotion(self, image: Union[Image.Image, np.ndarray]) -> Tuple[str, float]:
        """
        이미지에서 감정 분석
        Args:
            image: PIL Image 또는 numpy array
        Returns:
            tuple: (emotion_key, confidence)
        """
        if self.pipeline is None:
            return 'neutral', 0.0
        
        try:
            # numpy array인 경우 PIL Image로 변환
            if isinstance(image, np.ndarray):
                if len(image.shape) == 3 and image.shape[2] == 3:
                    # BGR -> RGB 변환
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
            
            # 이미지 전처리
            processed_image = self._preprocess_image(image)
            
            # 감정 분석 실행
            results = self.pipeline(processed_image)
            
            if results and len(results) > 0:
                top_result = results[0]
                emotion = top_result['label'].lower()
                confidence = float(top_result['score'])
                
                # 감정 매핑
                mapped_emotion = self.EMOTION_MAPPING.get(emotion, 'neutral')
                
                logger.debug(f"감정 분석: {emotion} -> {mapped_emotion} ({confidence:.2f})")
                return mapped_emotion, confidence
            
            return 'neutral', 0.0
            
        except Exception as e:
            logger.error(f"감정 분석 오류: {e}")
            return 'neutral', 0.0
    
    def detect_face_and_analyze(self, image_array: np.ndarray, 
                               min_face_size: Tuple[int, int] = (50, 50),
                               use_tracking: bool = True) -> Tuple[str, float, Optional[Tuple[int, int, int, int]]]:
        """
        단일 얼굴 감지 및 감정 분석
        Args:
            image_array: OpenCV BGR 이미지
            min_face_size: 최소 얼굴 크기
            use_tracking: 얼굴 추적 사용 여부
        Returns:
            tuple: (emotion_key, confidence, face_coordinates)
        """
        if self.face_cascade is None:
            logger.warning("얼굴 감지기가 로드되지 않았습니다.")
            return 'neutral', 0.0, None
        
        try:
            # 성능 최적화를 위한 이미지 크기 조정
            height, width = image_array.shape[:2]
            max_width = 640
            
            if width > max_width:
                scale = max_width / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                resized_img = cv2.resize(image_array, (new_width, new_height))
            else:
                scale = 1.0
                resized_img = image_array
            
            # 그레이스케일 변환
            gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
            
            # 얼굴 감지
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.2, 
                minNeighbors=5,
                minSize=min_face_size,
                maxSize=(min(width//2, 400), min(height//2, 400)),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # 얼굴 추적 또는 선택
            if use_tracking:
                tracked_face = self._face_tracker.update_tracking(faces, scale)
                target_face = tracked_face
            else:
                # 추적 없이 가장 큰 얼굴 선택
                target_face = max(faces, key=lambda x: x[2] * x[3]) if len(faces) > 0 else None
            
            if target_face is not None:
                x, y, w, h = target_face
                
                # 얼굴 영역 추출 (여백 추가)
                margin = max(w, h) // 10
                x1, y1 = max(0, x - margin), max(0, y - margin)
                x2, y2 = min(image_array.shape[1], x + w + margin), min(image_array.shape[0], y + h + margin)
                
                face_img = image_array[y1:y2, x1:x2]
                
                # 얼굴이 너무 작으면 스킵
                if face_img.shape[0] < 50 or face_img.shape[1] < 50:
                    return 'neutral', 0.0, None
                
                # 감정 분석
                emotion, confidence = self.analyze_emotion(face_img)
                
                return emotion, confidence, target_face
            
            return 'neutral', 0.0, None
            
        except Exception as e:
            logger.error(f"얼굴 감지 및 분석 오류: {e}")
            return 'neutral', 0.0, None
    
    def update_emotion_state(self, emotion: str, confidence: float) -> None:
        """최신 감정 상태 업데이트 (스레드 안전)"""
        with self._lock:
            self._latest_emotion = emotion
            self._latest_confidence = float(confidence)
    
    def get_emotion_state(self) -> Tuple[Optional[str], float]:
        """최신 감정 상태 반환 (스레드 안전)"""
        with self._lock:
            return self._latest_emotion, self._latest_confidence
    
    def reset_emotion_state(self) -> None:
        """감정 상태 초기화"""
        with self._lock:
            self._latest_emotion = None
            self._latest_confidence = 0.0
            self._face_tracker.reset()
    
    def reset_face_tracking(self) -> None:
        """얼굴 추적만 리셋"""
        self._face_tracker.reset()


# 전역 인스턴스 (싱글톤 패턴)
_emotion_analyzer = None
_model_manager = None

def get_emotion_analyzer() -> EmotionAnalyzer:
    """EmotionAnalyzer 싱글톤 인스턴스 반환"""
    global _emotion_analyzer
    if _emotion_analyzer is None:
        _emotion_analyzer = EmotionAnalyzer()
    return _emotion_analyzer

def get_model_manager() -> UnifiedModelManager:
    """ModelManager 싱글톤 인스턴스 반환"""
    global _model_manager
    if _model_manager is None:
        _model_manager = UnifiedModelManager()
    return _model_manager


# === 하위 호환성을 위한 래퍼 함수들 ===

@lru_cache(maxsize=1)
def load_emotion_model():
    """하위 호환성을 위한 래퍼 - Streamlit 캐시 지원"""
    if 'st' in globals():
        # Streamlit 환경
        return get_model_manager().emotion_pipeline
    else:
        # 일반 환경
        return get_emotion_analyzer().pipeline

def analyze_emotion_from_image(image: Image.Image) -> Tuple[str, float]:
    """하위 호환성을 위한 래퍼"""
    analyzer = get_emotion_analyzer()
    return analyzer.analyze_emotion(image)

def detect_face_and_analyze(image_array: np.ndarray) -> Tuple[str, float, Optional[Tuple[int, int, int, int]]]:
    """하위 호환성을 위한 래퍼"""
    analyzer = get_emotion_analyzer()
    return analyzer.detect_face_and_analyze(image_array)

def update_latest_emotion(emotion: str, confidence: float) -> None:
    """하위 호환성을 위한 래퍼"""
    analyzer = get_emotion_analyzer()
    analyzer.update_emotion_state(emotion, confidence)

def get_latest_emotion() -> Tuple[Optional[str], float]:
    """하위 호환성을 위한 래퍼"""
    analyzer = get_emotion_analyzer()
    return analyzer.get_emotion_state()

def reset_emotion_state() -> None:
    """하위 호환성을 위한 래퍼"""
    analyzer = get_emotion_analyzer()
    analyzer.reset_emotion_state()

def reset_face_tracking() -> None:
    """얼굴 추적 리셋"""
    analyzer = get_emotion_analyzer()
    analyzer.reset_face_tracking()


# === 새로운 통합 함수들 ===

def create_unified_analyzer(use_tracking: bool = True) -> EmotionAnalyzer:
    """통합 분석기 생성"""
    analyzer = get_emotion_analyzer()
    if not use_tracking:
        analyzer.reset_face_tracking()
    return analyzer

def batch_analyze_emotions(images: list) -> list:
    """배치 감정 분석"""
    analyzer = get_emotion_analyzer()
    results = []
    
    for img in images:
        emotion, confidence = analyzer.analyze_emotion(img)
        results.append((emotion, confidence))
    
    return results


# === 디버깅 및 상태 확인 함수들 ===

def get_model_status() -> dict:
    """모델 로드 상태 확인"""
    manager = get_model_manager()
    return {
        'emotion_model_loaded': manager.emotion_pipeline is not None,
        'face_cascade_loaded': manager.face_cascade is not None,
        'model_id': manager._model_id,
        'device': manager._device
    }

def print_model_info():
    """모델 정보 출력"""
    status = get_model_status()
    print("=" * 50)
    print("🔍 모델 상태 정보")
    print("=" * 50)
    print(f"감정 분석 모델: {'✅ 로드됨' if status['emotion_model_loaded'] else '❌ 로드 안됨'}")
    print(f"얼굴 감지기: {'✅ 로드됨' if status['face_cascade_loaded'] else '❌ 로드 안됨'}")
    print(f"모델 ID: {status['model_id']}")

    # 🔧 여기 고침: 중첩 f-string 제거
    device_str = "CPU" if status["device"] == -1 else f"GPU {status['device']}"
    print(f"디바이스: {device_str}")

    print("=" * 50)



if __name__ == "__main__":
    # 테스트 실행
    print("모델 테스트 실행")
    print_model_info()
    
    try:
        # 테스트 이미지로 분석 테스트
        test_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        emotion, confidence = analyze_emotion_from_image(test_img)
        print(f"테스트 결과: {emotion} ({confidence:.2f})")
    except Exception as e:
        print(f"테스트 실패: {e}")