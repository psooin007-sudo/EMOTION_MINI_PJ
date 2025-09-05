<<<<<<< HEAD
# web_with_graphs.py - 개선된 버전
=======
# web_with_graphs.py
>>>>>>> f81054bf3e5cea299f28fc141e2bd08a1635d760
import cv2
import numpy as np
from PIL import Image
from transformers import pipeline, AutoImageProcessor
import webbrowser
import urllib.parse
import time
import json
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from collections import defaultdict, Counter
<<<<<<< HEAD
from functools import lru_cache
import threading
=======
>>>>>>> f81054bf3e5cea299f28fc141e2bd08a1635d760

# 설정
MODEL_ID = "dima806/facial_emotions_image_detection"
DEVICE = -1
CAM_INDEX = 0
WIDTH, HEIGHT = 640, 480

# 배포된 Streamlit 앱 URL
DEPLOYED_APP_URL = "https://emotiondetector0827.streamlit.app/"

<<<<<<< HEAD
# 수정된 감정 매핑
=======
# 수정된 감정 매핑 (disgust를 다시 활성화)
>>>>>>> f81054bf3e5cea299f28fc141e2bd08a1635d760
EMOTION_MAPPING = {
    "sadness": "sad", 
    "happiness": "happy", 
    "anger": "angry",
    "fearful": "fear", 
    "surprised": "surprise", 
<<<<<<< HEAD
    "disgusted": "disgust",
=======
    "disgusted": "disgust",  # 다시 disgust로 매핑
>>>>>>> f81054bf3e5cea299f28fc141e2bd08a1635d760
    "neutral": "neutral"
}

# 감정별 색상 (BGR for OpenCV, RGB for matplotlib)
EMOTION_COLORS = {
    "angry": (0, 0, 255),
    "sad": (255, 0, 0), 
    "happy": (0, 255, 0),
    "fear": (0, 165, 255),
    "surprise": (255, 255, 0),
<<<<<<< HEAD
    "disgust": (128, 0, 128),
=======
    "disgust": (128, 0, 128),  # disgust 색상 추가
>>>>>>> f81054bf3e5cea299f28fc141e2bd08a1635d760
    "neutral": (128, 128, 128)
}

# matplotlib용 색상 (RGB 정규화)
EMOTION_COLORS_MPL = {
    "angry": (1.0, 0.0, 0.0),
    "sad": (0.0, 0.0, 1.0), 
    "happy": (0.0, 1.0, 0.0),
    "fear": (1.0, 0.65, 0.0),
    "surprise": (1.0, 1.0, 0.0),
    "disgust": (0.5, 0.0, 0.5),
    "neutral": (0.5, 0.5, 0.5)
}

<<<<<<< HEAD

class FaceTracker:
    """안정적인 단일 얼굴 추적을 위한 클래스"""
    
    def __init__(self, max_missing_frames=10, min_overlap_ratio=0.3):
        self.tracked_face = None
        self.missing_frames = 0
        self.max_missing_frames = max_missing_frames
        self.min_overlap_ratio = min_overlap_ratio
        self.face_history = []  # 최근 얼굴 위치 기록
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
        """가장 적합한 얼굴 선택 (크기와 추적 일관성 고려)"""
        if len(faces) == 0:
            return None
        
        # 추적 중인 얼굴이 없으면 가장 큰 얼굴 선택
        if self.tracked_face is None:
            return max(faces, key=lambda x: x[2] * x[3])
        
        # 기존 추적 얼굴과 겹치는 얼굴들 찾기
        matching_faces = []
        for face in faces:
            overlap = self.calculate_overlap_ratio(self.tracked_face, face)
            if overlap >= self.min_overlap_ratio:
                matching_faces.append((face, overlap))
        
        if matching_faces:
            # 겹치는 비율이 가장 높은 얼굴 선택
            return max(matching_faces, key=lambda x: x[1])[0]
        else:
            # 겹치는 얼굴이 없으면 가장 큰 얼굴 선택
            return max(faces, key=lambda x: x[2] * x[3])
    
    def update_tracking(self, faces):
        """얼굴 추적 업데이트 - 가장 적합한 단일 얼굴만 반환"""
        if len(faces) == 0:
            self.missing_frames += 1
            if self.missing_frames > self.max_missing_frames:
                self.tracked_face = None
                self.face_history.clear()
            return self.tracked_face
        
        # 가장 적합한 얼굴 선택
        best_face = self.find_best_face(faces)
        
        if best_face is not None:
            self.tracked_face = best_face
            self.missing_frames = 0
            
            # 얼굴 히스토리 업데이트
            self.face_history.append(best_face)
            if len(self.face_history) > self.max_history:
                self.face_history.pop(0)
        
        return self.tracked_face
    
    def get_stable_face(self):
        """안정화된 얼굴 위치 반환 (최근 히스토리의 평균)"""
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


class ModelManager:
    """모델 관리를 위한 싱글톤 클래스 - 중복 로딩 방지"""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.emotion_pipeline = None
        self.face_cascade = None
        self._initialized = True
    
    @lru_cache(maxsize=1)
    def get_emotion_pipeline(self):
        """감정 분석 모델을 캐시하여 한 번만 로드"""
        if self.emotion_pipeline is None:
            try:
                print("🔄 감정 분석 모델 로드 중...")
                processor = AutoImageProcessor.from_pretrained(MODEL_ID, use_fast=True)
                self.emotion_pipeline = pipeline(
                    "image-classification", 
                    model=MODEL_ID,
                    image_processor=processor, 
                    device=DEVICE
                )
                print("✅ 감정 분석 모델 로드 완료!")
                
                # 모델 테스트
                test_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
                results = self.emotion_pipeline(test_img)
                print(f"🧪 모델 테스트 성공: {results[0]['label'] if results else 'No result'}")
                
            except Exception as e:
                print(f"❌ 감정 분석 모델 로드 실패: {e}")
                
        return self.emotion_pipeline
    
    @lru_cache(maxsize=1)
    def get_face_cascade(self):
        """얼굴 감지기를 캐시하여 한 번만 로드"""
        if self.face_cascade is None:
            try:
                print("🔄 얼굴 감지기 로드 중...")
                self.face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
                
                if self.face_cascade.empty():
                    raise ValueError("얼굴 감지기 로드 실패")
                
                print("✅ 얼굴 감지기 로드 완료!")
                
            except Exception as e:
                print(f"❌ 얼굴 감지기 로드 실패: {e}")
                
        return self.face_cascade


=======
>>>>>>> f81054bf3e5cea299f28fc141e2bd08a1635d760
class EmotionHistory:
    """감정 히스토리 관리 클래스 (그래프 기능 포함)"""
    def __init__(self, history_file="emotion_history.json"):
        self.history_file = history_file
        self.history = []  # [{timestamp, emotion, score, raw_emotion}]
        self.load_history()
    
    def load_history(self):
        """저장된 히스토리 로드"""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    self.history = json.load(f)
                print(f"📚 기존 히스토리 로드: {len(self.history)}개 기록")
        except Exception as e:
            print(f"⚠️ 히스토리 로드 실패: {e}")
            self.history = []
    
    def add_record(self, emotion, score, raw_emotion=None):
        """새 감정 기록 추가"""
        record = {
            "timestamp": time.time(),
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "emotion": emotion,
            "score": float(score),
            "raw_emotion": raw_emotion or emotion
        }
        self.history.append(record)
        self.save_history()
        print(f"📝 히스토리 추가: {emotion} ({score*100:.1f}%)")
    
    def save_history(self):
        """히스토리를 파일에 저장"""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"❌ 히스토리 저장 실패: {e}")
    
    def get_recent_records(self, minutes=10):
        """최근 N분간의 기록 반환"""
        cutoff_time = time.time() - (minutes * 60)
        return [r for r in self.history if r["timestamp"] > cutoff_time]
    
    def get_emotion_stats(self, minutes=None):
        """감정별 통계 (전체 또는 최근 N분)"""
        records = self.get_recent_records(minutes) if minutes else self.history
        stats = defaultdict(list)
        
        for record in records:
            stats[record["emotion"]].append(record["score"])
        
        # 평균 스코어와 횟수 계산
        result = {}
        for emotion, scores in stats.items():
            result[emotion] = {
                "count": len(scores),
                "avg_score": np.mean(scores) if scores else 0,
                "max_score": max(scores) if scores else 0,
                "min_score": min(scores) if scores else 0
            }
        
        return result
    
<<<<<<< HEAD
=======
    def get_timeline_data(self, minutes=60):
        """시간대별 감정 변화 데이터"""
        recent_records = self.get_recent_records(minutes)
        return [(r["datetime"], r["emotion"], r["score"]) for r in recent_records]
    
>>>>>>> f81054bf3e5cea299f28fc141e2bd08a1635d760
    def print_summary(self, minutes=10):
        """히스토리 요약 출력"""
        print(f"\n📊 감정 히스토리 요약 (최근 {minutes}분)")
        print("=" * 50)
        
        recent = self.get_recent_records(minutes)
        if not recent:
            print("📭 기록이 없습니다.")
            return
        
        stats = self.get_emotion_stats(minutes)
        
        # 감정별 통계 출력 (빈도순 정렬)
        sorted_emotions = sorted(stats.items(), key=lambda x: x[1]["count"], reverse=True)
        
        print("🎭 감정별 통계:")
        for emotion, data in sorted_emotions:
            print(f"  {emotion.upper():8} | 횟수: {data['count']:2d} | "
                  f"평균: {data['avg_score']*100:5.1f}% | "
                  f"최고: {data['max_score']*100:5.1f}%")
        
        # 최근 5개 기록
        print(f"\n⏱️  최근 감정 변화:")
        for record in recent[-5:]:
            print(f"  {record['datetime'][:19]} | {record['emotion'].upper():8} | {record['score']*100:5.1f}%")
<<<<<<< HEAD
=======
    
    def export_csv(self, filename="emotion_history.csv"):
        """CSV로 내보내기"""
        try:
            import pandas as pd
            df = pd.DataFrame(self.history)
            df.to_csv(filename, index=False, encoding='utf-8')
            print(f"📄 히스토리를 {filename}으로 내보냄 ({len(self.history)}개 기록)")
        except ImportError:
            # pandas 없을 때 수동으로 CSV 생성
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write("timestamp,datetime,emotion,score,raw_emotion\n")
                    for record in self.history:
                        f.write(f"{record['timestamp']},{record['datetime']},{record['emotion']},{record['score']},{record['raw_emotion']}\n")
                print(f"📄 히스토리를 {filename}으로 내보냄 ({len(self.history)}개 기록)")
            except Exception as e:
                print(f"❌ CSV 내보내기 실패: {e}")
        except Exception as e:
            print(f"❌ CSV 내보내기 실패: {e}")

    def plot_emotion_timeline(self, minutes=60, save_file=None):
        """감정 변화 타임라인 그래프"""
        recent_records = self.get_recent_records(minutes)
        
        if not recent_records:
            print("📭 그래프를 그릴 데이터가 없습니다.")
            return
        
        # 데이터 준비
        timestamps = [datetime.fromtimestamp(r["timestamp"]) for r in recent_records]
        emotions = [r["emotion"] for r in recent_records]
        scores = [r["score"] for r in recent_records]
        
        # 그래프 생성
        plt.figure(figsize=(12, 8))
        
        # 서브플롯 1: 감정별 색상으로 시간 변화
        plt.subplot(2, 1, 1)
        
        for emotion in set(emotions):
            emotion_times = [timestamps[i] for i, e in enumerate(emotions) if e == emotion]
            emotion_scores = [scores[i] for i, e in enumerate(emotions) if e == emotion]
            
            plt.scatter(emotion_times, emotion_scores, 
                       c=[EMOTION_COLORS_MPL[emotion]], 
                       label=emotion.upper(), 
                       alpha=0.7, s=60)
        
        plt.title(f'감정 변화 타임라인 (최근 {minutes}분)', fontsize=16, fontweight='bold')
        plt.xlabel('시간')
        plt.ylabel('감정 신뢰도')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # x축 시간 포맷
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.gca().xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
        plt.xticks(rotation=45)
        
        # 서브플롯 2: 감정별 빈도 막대 그래프
        plt.subplot(2, 1, 2)
        
        emotion_counts = Counter(emotions)
        emotions_list = list(emotion_counts.keys())
        counts = list(emotion_counts.values())
        colors = [EMOTION_COLORS_MPL[e] for e in emotions_list]
        
        bars = plt.bar(emotions_list, counts, color=colors, alpha=0.7)
        plt.title('감정별 빈도', fontsize=14, fontweight='bold')
        plt.xlabel('감정')
        plt.ylabel('횟수')
        
        # 막대 위에 숫자 표시
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        
        # 파일로 저장
        if save_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_file = f"emotion_timeline_{timestamp}.png"
        
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"📈 그래프를 {save_file}에 저장했습니다")
        
        plt.show()
    
    def plot_emotion_heatmap(self, save_file=None):
        """시간대별 감정 히트맵"""
        if not self.history:
            print("📭 그래프를 그릴 데이터가 없습니다.")
            return
        
        # 시간대별 데이터 준비
        hours_data = defaultdict(lambda: defaultdict(int))
        
        for record in self.history:
            dt = datetime.fromtimestamp(record["timestamp"])
            hour = dt.hour
            emotion = record["emotion"]
            hours_data[hour][emotion] += 1
        
        # 모든 감정 목록
        all_emotions = list(EMOTION_MAPPING.values())
        all_hours = range(24)
        
        # 히트맵 데이터 생성
        heatmap_data = []
        for emotion in all_emotions:
            row = [hours_data[hour][emotion] for hour in all_hours]
            heatmap_data.append(row)
        
        # 그래프 생성
        plt.figure(figsize=(15, 8))
        
        im = plt.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
        
        # 축 설정
        plt.yticks(range(len(all_emotions)), [e.upper() for e in all_emotions])
        plt.xticks(range(24), [f'{h:02d}:00' for h in range(24)])
        
        # 제목과 라벨
        plt.title('시간대별 감정 분포 히트맵', fontsize=16, fontweight='bold')
        plt.xlabel('시간')
        plt.ylabel('감정')
        
        # 컬러바
        cbar = plt.colorbar(im)
        cbar.set_label('빈도', rotation=270, labelpad=20)
        
        # 값 표시
        for i in range(len(all_emotions)):
            for j in range(24):
                text = plt.text(j, i, heatmap_data[i][j], 
                               ha="center", va="center", color="black", fontsize=8)
        
        plt.tight_layout()
        
        # 파일로 저장
        if save_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_file = f"emotion_heatmap_{timestamp}.png"
        
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"🔥 히트맵을 {save_file}에 저장했습니다")
        
        plt.show()
    
    def plot_emotion_pie_chart(self, minutes=None, save_file=None):
        """감정별 비율 원형 그래프"""
        records = self.get_recent_records(minutes) if minutes else self.history
        
        if not records:
            print("📭 그래프를 그릴 데이터가 없습니다.")
            return
        
        # 감정별 카운트
        emotions = [r["emotion"] for r in records]
        emotion_counts = Counter(emotions)
        
        # 그래프 데이터
        labels = list(emotion_counts.keys())
        sizes = list(emotion_counts.values())
        colors = [EMOTION_COLORS_MPL[emotion] for emotion in labels]
        
        # 그래프 생성
        plt.figure(figsize=(10, 8))
        
        wedges, texts, autotexts = plt.pie(sizes, labels=[l.upper() for l in labels], 
                                          colors=colors, autopct='%1.1f%%',
                                          startangle=90, textprops={'fontsize': 12})
        
        # 제목
        period_text = f"최근 {minutes}분" if minutes else "전체 기간"
        plt.title(f'감정 분포 ({period_text})', fontsize=16, fontweight='bold')
        
        # 범례
        plt.legend(wedges, [f"{l.upper()} ({c}회)" for l, c in zip(labels, sizes)],
                  title="감정별 횟수", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        
        plt.axis('equal')
        
        # 파일로 저장
        if save_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            period_suffix = f"_{minutes}min" if minutes else "_total"
            save_file = f"emotion_pie{period_suffix}_{timestamp}.png"
        
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"🥧 원형 그래프를 {save_file}에 저장했습니다")
        
        plt.show()
    
    def plot_emotion_score_distribution(self, save_file=None):
        """감정별 신뢰도 분포 상자 그래프"""
        if not self.history:
            print("📭 그래프를 그릴 데이터가 없습니다.")
            return
        
        # 감정별 스코어 데이터 준비
        emotion_scores = defaultdict(list)
        for record in self.history:
            emotion_scores[record["emotion"]].append(record["score"])
        
        # 데이터가 있는 감정만 필터링
        filtered_emotions = {k: v for k, v in emotion_scores.items() if v}
        
        if not filtered_emotions:
            print("📭 그래프를 그릴 스코어 데이터가 없습니다.")
            return
        
        # 그래프 생성
        plt.figure(figsize=(12, 8))
        
        emotions = list(filtered_emotions.keys())
        scores_data = list(filtered_emotions.values())
        
        # 상자 그래프
        box_plot = plt.boxplot(scores_data, labels=[e.upper() for e in emotions], patch_artist=True)
        
        # 색상 설정
        for patch, emotion in zip(box_plot['boxes'], emotions):
            patch.set_facecolor(EMOTION_COLORS_MPL[emotion])
            patch.set_alpha(0.7)
        
        plt.title('감정별 신뢰도 분포', fontsize=16, fontweight='bold')
        plt.xlabel('감정')
        plt.ylabel('신뢰도')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # 통계 정보 추가
        stats_text = []
        for emotion in emotions:
            scores = filtered_emotions[emotion]
            avg_score = np.mean(scores)
            stats_text.append(f"{emotion.upper()}: 평균 {avg_score:.2f}")
        
        plt.figtext(0.02, 0.02, " | ".join(stats_text), fontsize=10, alpha=0.7)
        
        plt.tight_layout()
        
        # 파일로 저장
        if save_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_file = f"emotion_score_dist_{timestamp}.png"
        
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"📦 상자 그래프를 {save_file}에 저장했습니다")
        
        plt.show()
>>>>>>> f81054bf3e5cea299f28fc141e2bd08a1635d760


class WebcamEmotionAnalyzer:
    def __init__(self):
<<<<<<< HEAD
        # 싱글톤 모델 매니저 사용 - 중복 로딩 방지
        self.model_manager = ModelManager()
=======
        self.emotion_pipeline = None
        self.face_cascade = None
>>>>>>> f81054bf3e5cea299f28fc141e2bd08a1635d760
        
        # 히스토리 관리 객체
        self.emotion_history = EmotionHistory()
        
<<<<<<< HEAD
        # 얼굴 트래커 - 단일 얼굴 안정적 추적
        self.face_tracker = FaceTracker()
        
        # 실시간 통계 (기존 방식 유지)
        self.emotion_stats = {}
        
        print("📡 모델 초기화...")
        self.load_models()
        
    def load_models(self):
        """모델 로드 - 이제 중복 로딩 방지"""
        # 모델 매니저를 통해 한 번만 로드
        self.emotion_pipeline = self.model_manager.get_emotion_pipeline()
        self.face_cascade = self.model_manager.get_face_cascade()
        
        if self.emotion_pipeline and self.face_cascade:
            print("✅ 모든 모델 로드 완료!")
        else:
            print("⚠️ 일부 모델 로드에 실패했습니다.")
            
    def detect_faces(self, frame):
        """얼굴 검출 - 향상된 파라미터"""
        if self.face_cascade is None:
            return []
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 더 안정적인 얼굴 검출 설정
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1,      # 더 정밀한 스케일 팩터
            minNeighbors=5,       # 더 엄격한 필터링
            minSize=(80, 80),     # 더 큰 최소 크기
            maxSize=(min(frame.shape[1]//2, 300), min(frame.shape[0]//2, 300)),  # 최대 크기 제한
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # 얼굴 트래커를 통해 단일 얼굴만 반환
        tracked_face = self.face_tracker.update_tracking(faces)
        
        if tracked_face is not None:
            return [tracked_face]  # 리스트로 반환하여 기존 코드와 호환성 유지
        else:
            return []
=======
        # 실시간 통계 (기존 방식 유지)
        self.emotion_stats = {}
        
        self.load_models()
        
    def load_models(self):
        """모델 로드"""
        print("Loading models...")
        
        # 얼굴 검출기
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # 감정 분석 모델
        try:
            print("Loading emotion classification model...")
            processor = AutoImageProcessor.from_pretrained(MODEL_ID, use_fast=True)
            self.emotion_pipeline = pipeline(
                "image-classification", 
                model=MODEL_ID,
                image_processor=processor, 
                device=DEVICE
            )
            print("✅ Models loaded successfully!")
            
            # 모델 테스트
            self.test_model()
            
        except Exception as e:
            print(f"❌ Model loading error: {e}")
            
    def test_model(self):
        """모델 동작 테스트"""
        try:
            # 테스트용 더미 이미지 생성
            test_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            results = self.emotion_pipeline(test_img)
            print(f"🧪 모델 테스트 결과: {results}")
        except Exception as e:
            print(f"⚠️ 모델 테스트 실패: {e}")
            
    def detect_faces(self, frame):
        """얼굴 검출 (더 민감하게 조정)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 더 민감한 얼굴 검출 설정
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.05,     # 더 작은 스케일 팩터
            minNeighbors=3,       # 더 적은 이웃 수
            minSize=(50, 50),     # 더 큰 최소 크기
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        return faces
>>>>>>> f81054bf3e5cea299f28fc141e2bd08a1635d760
    
    def crop_and_preprocess_face(self, frame, faces):
        """얼굴 크롭 및 전처리 개선"""
        if len(faces) == 0:
            return None
<<<<<<< HEAD
        
        # 안정화된 얼굴 위치 사용
        stable_face = self.face_tracker.get_stable_face()
        if stable_face is None:
            return None
            
        x, y, w, h = stable_face
        
        # 더 적절한 마진으로 얼굴 영역 확장
=======
            
        largest_face = max(faces, key=lambda face: face[2] * face[3])
        x, y, w, h = largest_face
        
        # 더 넓은 마진으로 얼굴 영역 확장
>>>>>>> f81054bf3e5cea299f28fc141e2bd08a1635d760
        margin_x = int(w * 0.2)  # 가로 20% 마진
        margin_y = int(h * 0.3)  # 세로 30% 마진 (머리 포함)
        
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(frame.shape[1], x + w + margin_x)
        y2 = min(frame.shape[0], y + h + margin_y)
        
        face_img = frame[y1:y2, x1:x2]
        
        # 이미지 품질 개선
<<<<<<< HEAD
        if face_img.shape[0] > 50 and face_img.shape[1] > 50:
            # 크기 정규화 (224x224로 리사이즈)
            face_img = cv2.resize(face_img, (224, 224))
            
            # 대비 및 밝기 개선
            face_img = cv2.convertScaleAbs(face_img, alpha=1.1, beta=5)
=======
        if face_img.shape[0] > 0 and face_img.shape[1] > 0:
            # 크기 정규화 (224x224로 리사이즈)
            face_img = cv2.resize(face_img, (224, 224))
            
            # 대비 개선
            face_img = cv2.convertScaleAbs(face_img, alpha=1.2, beta=10)
>>>>>>> f81054bf3e5cea299f28fc141e2bd08a1635d760
            
        return face_img
    
    def analyze_emotion(self, face_img, save_to_history=True):
        """감정 분석 (히스토리 저장 옵션 추가)"""
        if face_img is None or self.emotion_pipeline is None:
            return None, None
            
        try:
            # OpenCV BGR을 RGB로 변환
            rgb_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_img)
            
<<<<<<< HEAD
            # 감정 분석 실행
=======
            # 감정 분석 결과를 모두 확인
>>>>>>> f81054bf3e5cea299f28fc141e2bd08a1635d760
            results = self.emotion_pipeline(pil_img)
            
            if results and len(results) > 0:
                # 모든 결과 출력 (디버깅용)
                print("🎭 전체 감정 분석 결과:")
<<<<<<< HEAD
                for i, result in enumerate(results[:3]):  # 상위 3개만 출력
=======
                for i, result in enumerate(results[:5]):  # 상위 5개만 출력
>>>>>>> f81054bf3e5cea299f28fc141e2bd08a1635d760
                    raw_label = result["label"]
                    mapped_label = EMOTION_MAPPING.get(raw_label.lower(), raw_label.lower())
                    confidence = result["score"]
                    print(f"  {i+1}. {raw_label} -> {mapped_label}: {confidence*100:.2f}%")
                
                # 가장 높은 신뢰도의 결과 선택
                best_result = results[0]
                raw_emotion = best_result["label"]
                emotion = EMOTION_MAPPING.get(raw_emotion.lower(), raw_emotion.lower())
                score = float(best_result["score"])
                
                # 기존 통계 업데이트
                if emotion in self.emotion_stats:
                    self.emotion_stats[emotion] += 1
                else:
                    self.emotion_stats[emotion] = 1
                
                # 히스토리에 저장
                if save_to_history:
                    self.emotion_history.add_record(emotion, score, raw_emotion)
                
                print(f"🎯 선택된 감정: {raw_emotion} -> {emotion} ({score*100:.1f}%)")
                print(f"📊 실시간 통계: {self.emotion_stats}")
                
                return emotion, score
                
        except Exception as e:
            print(f"❌ 감정 분석 오류: {e}")
            import traceback
            traceback.print_exc()
            
        return None, None
    
    def draw_face_info(self, frame, faces, emotion=None, score=None):
<<<<<<< HEAD
        """얼굴 박스와 감정 정보 그리기 - 단일 얼굴용"""
        if len(faces) == 0:
            return frame
        
        # 단일 얼굴 처리
        face = faces[0]  # 이미 트래커를 통해 선택된 얼굴
        x, y, w, h = face
        
        color = EMOTION_COLORS.get(emotion, (0, 255, 0))
        
        # 얼굴 박스 (더 두껍게)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
        
        # 감정 라벨
        if emotion and score is not None:
            # 신뢰도에 따른 색상 조정
            confidence_color = color if score > 0.6 else (128, 128, 128)
            
            label_text = f"{emotion.upper()} ({score*100:.1f}%)"
            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            
            # 라벨 배경
            cv2.rectangle(frame, (x, y - 35), (x + label_size[0] + 15, y), confidence_color, -1)
            
            # 라벨 텍스트
            cv2.putText(frame, label_text, (x + 7, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 트래킹 상태 표시
            status_text = f"Tracking: {self.face_tracker.missing_frames}/{self.face_tracker.max_missing_frames}"
            cv2.putText(frame, status_text, (x, y + h + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
=======
        """얼굴 박스와 감정 정보 그리기"""
        for (x, y, w, h) in faces:
            color = EMOTION_COLORS.get(emotion, (0, 255, 0))
            
            # 얼굴 박스 (더 두껍게)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
            
            # 감정 라벨
            if emotion and score is not None:
                # 신뢰도에 따른 색상 조정
                confidence_color = color if score > 0.5 else (128, 128, 128)
                
                label_text = f"{emotion.upper()} ({score*100:.1f}%)"
                label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                
                # 라벨 배경
                cv2.rectangle(frame, (x, y - 35), (x + label_size[0] + 15, y), confidence_color, -1)
                
                # 라벨 텍스트
                cv2.putText(frame, label_text, (x + 7, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
>>>>>>> f81054bf3e5cea299f28fc141e2bd08a1635d760
        return frame
    
    def draw_history_info(self, frame, show_recent=True):
        """화면에 히스토리 정보 표시"""
        if show_recent:
            recent_records = self.emotion_history.get_recent_records(5)  # 최근 5분
            if recent_records:
                # 최근 감정 변화 표시
                y_start = 120
                cv2.putText(frame, "Recent Emotions:", (WIDTH - 200, y_start),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                for i, record in enumerate(recent_records[-3:]):  # 최근 3개만
                    emotion = record["emotion"]
                    score = record["score"]
                    time_str = record["datetime"][-8:-3]  # HH:MM만
                    
                    color = EMOTION_COLORS.get(emotion, (255, 255, 255))
                    text = f"{time_str} {emotion[:4].upper()} {score*100:.0f}%"
                    
                    cv2.putText(frame, text, (WIDTH - 200, y_start + 20 + i*15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return frame
    
    def save_result_locally(self, emotion, score):
        """결과를 로컬 파일로 저장 (히스토리 포함)"""
        result = {
            "emotion": emotion,
            "score": score,
            "timestamp": time.time(),
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "emotion_stats": self.emotion_stats.copy(),
            "recent_history": self.emotion_history.get_recent_records(10),  # 최근 10분
            "total_records": len(self.emotion_history.history)
        }
        
        with open("latest_emotion_result.json", "w", encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    
    def run(self):
        """메인 루프 실행"""
        cap = cv2.VideoCapture(CAM_INDEX)
        if not cap.isOpened():
            raise RuntimeError("웹캠을 열 수 없습니다")
            
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        
        # 카메라 설정 개선
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)
        cap.set(cv2.CAP_PROP_CONTRAST, 0.5)
        
<<<<<<< HEAD
        window_name = "Real-time Emotion Analysis (Single Face Tracking)"
=======
        window_name = "Real-time Emotion Analysis"
>>>>>>> f81054bf3e5cea299f28fc141e2bd08a1635d760
        cv2.namedWindow(window_name)
        
        last_emotion, last_score = None, None
        frame_count = 0
        show_history = True
        
<<<<<<< HEAD
        print("🎥 웹캠 시작! (단일 얼굴 추적 모드)")
        print(f"🌐 결과 페이지: {DEPLOYED_APP_URL}")
        print("💡 팁: 카메라 앞에서 다양한 표정을 지어보세요!")
        print("⌨️  조작법:")
        print("  ESC: 종료")
        print("  R: 얼굴 트래킹 리셋")
=======
        print("웹캠 시작!")
        print(f"결과 페이지: {DEPLOYED_APP_URL}")
        print("팁: 다양한 표정을 지어보세요 (웃음, 화남, 슬픔 등)")
        print("조작법:")
        print("  ESC: 종료")
>>>>>>> f81054bf3e5cea299f28fc141e2bd08a1635d760
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_count += 1
<<<<<<< HEAD
                faces = self.detect_faces(frame)  # 이미 단일 얼굴만 반환
                
                # 감정 분석 (15프레임마다, 히스토리는 덜 자주)
                if frame_count % 15 == 0 and len(faces) > 0:
                    face_img = self.crop_and_preprocess_face(frame, faces)
                    # 히스토리 저장은 45프레임마다만 (너무 많은 데이터 방지)
                    save_history = frame_count % 45 == 0
=======
                faces = self.detect_faces(frame)
                
                # 더 자주 감정 분석 (10프레임마다, 히스토리는 덜 자주)
                if frame_count % 10 == 0 and len(faces) > 0:
                    face_img = self.crop_and_preprocess_face(frame, faces)
                    # 히스토리 저장은 30프레임마다만 (너무 많은 데이터 방지)
                    save_history = frame_count % 30 == 0
>>>>>>> f81054bf3e5cea299f28fc141e2bd08a1635d760
                    emotion, score = self.analyze_emotion(face_img, save_to_history=save_history)
                    if emotion:
                        last_emotion = emotion
                        last_score = score
                        # 결과를 로컬에 저장
                        self.save_result_locally(emotion, score)
                
                # 얼굴 정보 표시
                frame = self.draw_face_info(frame, faces, last_emotion, last_score)
                
                # 히스토리 정보 표시
                if show_history:
                    frame = self.draw_history_info(frame)
                
                # 상태 정보 표시
<<<<<<< HEAD
                face_status = f"Face: {'✓' if len(faces) > 0 else '✗'}"
                if len(faces) > 0:
                    missing = self.face_tracker.missing_frames
                    face_status += f" (Missing: {missing})"
                
                emotion_status = ""
                if last_emotion:
                    emotion_status = f"Emotion: {last_emotion.upper()} ({(last_score or 0)*100:.1f}%)"
                else:
                    emotion_status = "Detecting emotions..."
                
                status = f"{face_status} | {emotion_status} | Records: {len(self.emotion_history.history)}"
                
                cv2.putText(frame, status, (10, HEIGHT - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # 사용법 안내
                help_text = "Single Face Tracking | ESC: quit | R: reset tracking"
                cv2.putText(frame, help_text, (10, HEIGHT - 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                
                cv2.imshow(window_name, frame)
                
                # 키보드 입력 처리
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC 키
                    print("👋 종료합니다...")
                    break
                elif key == ord('r') or key == ord('R'):  # R 키로 트래킹 리셋
                    print("🔄 얼굴 트래킹 리셋")
                    self.face_tracker.reset()
                    last_emotion, last_score = None, None
=======
                status = f"Faces: {len(faces)} | "
                if last_emotion:
                    status += f"Emotion: {last_emotion.upper()} ({(last_score or 0)*100:.1f}%)"
                else:
                    status += "Detecting emotions..."
                    
                status += f" | Records: {len(self.emotion_history.history)}"
                
                cv2.putText(frame, status, (10, HEIGHT - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # 사용법 안내
                help_text = "Real-time emotion detection | ESC to quit"
                cv2.putText(frame, help_text, (10, HEIGHT - 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                cv2.imshow(window_name, frame)
                
                # ESC 키로 종료
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC 키
                    break
>>>>>>> f81054bf3e5cea299f28fc141e2bd08a1635d760
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # 최종 통계 출력
<<<<<<< HEAD
            print("\n🏁 세션 완료!")
            print(f"📊 총 기록된 감정: {len(self.emotion_history.history)}개")
            print(f"🎭 실시간 통계: {self.emotion_stats}")
=======
            print("\n세션 완료!")
            print(f"총 기록된 감정: {len(self.emotion_history.history)}개")
>>>>>>> f81054bf3e5cea299f28fc141e2bd08a1635d760
            self.emotion_history.print_summary(60)  # 최근 1시간


def main():
<<<<<<< HEAD
    """메인 함수"""
    print("=" * 60)
    print("🎭 실시간 감정 분석기 (개선된 버전)")
    print("=" * 60)
    print("✨ 주요 개선사항:")
    print("  - 모델 중복 로딩 방지 (메모리 절약)")
    print("  - 단일 얼굴 안정적 추적")
    print("  - 향상된 얼굴 인식 정확도")
    print("  - 트래킹 상태 시각화")
    print()
    print(f"🌐 결과 페이지: {DEPLOYED_APP_URL}")
    print("💡 다양한 감정 표현을 시도해보세요!")
    print("=" * 60)
    
    try:
        analyzer = WebcamEmotionAnalyzer()
        analyzer.run()
    except Exception as e:
        print(f"❌ 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("🔚 프로그램 종료")
=======
    print("실시간 감정 분석기 시작...")
    print(f"결과 페이지: {DEPLOYED_APP_URL}")
    print("다양한 감정 표현을 시도해보세요!")
    
    analyzer = WebcamEmotionAnalyzer()
    analyzer.run()
>>>>>>> f81054bf3e5cea299f28fc141e2bd08a1635d760


if __name__ == "__main__":
    main()