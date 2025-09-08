# ⛵ 나의 감정 항해 일지 (My Emotional Voyage Log)

**An AI-Powered Compass for Your Emotional Ocean**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-streamlit-app-url.com) 매일 예측 불가능한 감정의 바다를 항해하는 20대들을 위한 AI 기반 디지털 정신 건강 케어 솔루션입니다. 사용자가 자신의 감정 항해 경로를 기록하고 돌아볼 수 있는 '항해 일지'가 되어, 감정의 폭풍우 속에서 길을 잃지 않도록 돕는 '나침반'이자 '등대'가 되는 것을 목표로 합니다.

<br>

## ✨ 핵심 기능 (Core Features)

| 기능 | 설명 | 주요 기술 |
| :--- | :--- | :--- |
| **1. 실시간 표정 분석 & 대시보드** | 웹캠으로 사용자의 표정을 실시간 분석하고, 감정 변화 추이를 시각화된 대시보드에 자동 기록하여 객관적인 자기 이해를 돕습니다. | `OpenCV`, `Hugging Face` |
| **2. AI 생성형 파도 명상** | 사용자가 선택한 현재 감정을 기반으로, OpenAI의 DALL-E 3 모델이 세상에 하나뿐인 '감정의 바다' 이미지를 생성하여 맞춤형 명상 경험을 제공합니다. | `OpenAI (DALL-E 3)` |
| **3. AI 표정 연기 게임** | 주어진 상황에 맞는 표정을 연기하면 AI가 정확도를 판단하는 게임으로, 즐겁게 감정 표현을 연습하고 자신을 이해하도록 돕습니다. | `Hugging Face` |
| **4. 인터랙티브 미니게임** | '파도놀이터' 메뉴에서 가위바위보, 암초 부수기 등 사용자가 직접 선택하여 즐기는 게임 콘텐츠를 통해 스트레스를 해소하고 참여를 유도합니다. | `Streamlit` |

<br>

## 🏛️ 프로젝트 철학 (Our Philosophy)

저희는 기술이 사용자의 여정을 방해하지 않고, 필요할 때만 정확한 길잡이가 되어야 한다고 믿습니다.

* **Invisible Support:** AI는 배경에서 조용히 작동합니다.
* **Seamless Experience:** 사용자는 불필요한 개입 없이 자신의 여정에만 집중합니다.
* **Guiding Light:** 필요할 때만 정확한 길잡이가 되어줍니다.
* **Human-Centered:** 기술이 아닌 사용자의 경험이 중심입니다.

<br>

## 🛠️ 개발 환경 (Development Environment)

* **Language:** `Python`
* **Framework:** `Streamlit`
* **AI & ML:** `transformers`, `openai`, `torch`
* **Image Processing:** `opencv-python`, `Pillow`
* **Data Visualization:** `pandas`, `plotly`
* **AI Models:**
    * `Hugging Face` (dima806/facial_emotions_image_detection, trpakov/vit-face-expression, beomi/KcELECTRA-base-v2022)
    * `OpenAI` (DALL-E 3)

<br>

## 🚀 시작하기 (Getting Started)

프로젝트를 실행하기 위해서는 `streamlit`, `opencv-python`, `pandas`, `plotly`, `transformers`, `torch`, `openai`, `Pillow` 등의 라이브러리가 필요합니다.

### OpenAI API 키 설정

프로젝트 폴더 내에 `.streamlit` 폴더를 만들고, 그 안에 `secrets.toml` 파일을 생성하여 아래와 같이 OpenAI API 키를 입력해주세요.

```toml
# .streamlit/secrets.toml
OPENAI_API_KEY = "sk-..."
앱 실행 (Run)
Bash

streamlit run main.py
