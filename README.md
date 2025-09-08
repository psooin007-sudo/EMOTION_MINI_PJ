⛵ 나의 감정 항해 일지 (My Emotional Voyage Log)
An AI-Powered Compass for Your Emotional Ocean

 매일 예측 불가능한 감정의 바다를 항해하는 20대들을 위한 AI 기반 디지털 정신 건강 케어 솔루션입니다. 사용자가 자신의 감정 항해 경로를 기록하고 돌아볼 수 있는 '항해 일지'가 되어, 감정의 폭풍우 속에서 길을 잃지 않도록 돕는 '나침반'이자 '등대'가 되는 것을 목표로 합니다.

<br>

✨ 핵심 기능 (Core Features)
기능	설명	주요 기술
1. 실시간 표정 분석 & 대시보드	웹캠으로 사용자의 표정을 실시간 분석하고, 감정 변화 추이를 시각화된 대시보드에 자동 기록하여 객관적인 자기 이해를 돕습니다.	OpenCV, Hugging Face
2. AI 생성형 파도 명상	사용자가 선택한 현재 감정을 기반으로, OpenAI의 DALL-E 3 모델이 세상에 하나뿐인 '감정의 바다' 이미지를 생성하여 맞춤형 명상 경험을 제공합니다.	OpenAI (DALL-E 3)
3. AI 표정 연기 게임	주어진 상황에 맞는 표정을 연기하면 AI가 정확도를 판단하는 게임으로, 즐겁게 감정 표현을 연습하고 자신을 이해하도록 돕습니다.	Hugging Face
4. 인터랙티브 미니게임	'파도놀이터' 메뉴에서 가위바위보, 암초 부수기 등 사용자가 직접 선택하여 즐기는 게임 콘텐츠를 통해 스트레스를 해소하고 참여를 유도합니다.	Streamlit
<br>

🏛️ 프로젝트 철학 (Our Philosophy)
저희는 기술이 사용자의 여정을 방해하지 않고, 필요할 때만 정확한 길잡이가 되어야 한다고 믿습니다.

Invisible Support: AI는 배경에서 조용히 작동합니다.

Seamless Experience: 사용자는 불필요한 개입 없이 자신의 여정에만 집중합니다.

Guiding Light: 필요할 때만 정확한 길잡이가 되어줍니다.

Human-Centered: 기술이 아닌 사용자의 경험이 중심입니다.

<br>

🛠️ 개발 환경 (Development Environment)
Language: Python

Framework: Streamlit

AI & ML: transformers, openai, torch

Image Processing: opencv-python, Pillow

Data Visualization: pandas, plotly

<br>

🚀 시작하기 (Getting Started)
1. 필요 라이브러리 설치 (Installation)
Bash

pip install -r requirements.txt
2. OpenAI API 키 설정
프로젝트 폴더 내에 .streamlit 폴더를 만들고, 그 안에 secrets.toml 파일을 생성하여 아래와 같이 OpenAI API 키를 입력해주세요.

Ini, TOML

# .streamlit/secrets.toml
OPENAI_API_KEY = "sk-..."
3. 앱 실행 (Run)
Bash

streamlit run main.py
<br>

🧭 앱 사용법 (How to Use)
첫 항해 시작 (실시간 감정 분석)

streamlit run main.py 실행 시 나타나는 첫 화면에서 [🎥 웹캠으로 분석하기] 버튼을 클릭하세요.

별도의 웹캠 창이 열리고, 실시간으로 당신의 표정을 분석하여 감정 데이터를 기록합니다.

웹캠 창을 닫으면(ESC 키) 자동으로 분석 결과가 저장되고 '나의 감정 항해 일지' 메인 페이지로 이동합니다.

항해 일지 둘러보기 (메인 대시보드)

메인 페이지에서는 방금 웹캠으로 분석된 당신의 감정 변화 추이를 타임라인과 원 그래프로 한눈에 확인할 수 있습니다.

화면 왼쪽의 사이드바가 이 앱의 모든 기능을 탐험할 수 있는 항해 지도입니다.

다양한 섬 탐험하기 (핵심 기능 사용)

🌬️ AI 파도 명상: 사이드바에서 'AI 파도 명상'을 선택하세요. 현재 감정을 고르면, AI가 당신만을 위한 바다 이미지를 그려줍니다.

📓 감정 저널: 오늘 있었던 일을 글로 기록해보세요. AI가 당신의 글에 숨겨진 감정을 분석해 알려줍니다.

🌊 파도놀이터: 사이드바의 '파도놀이터' 메뉴에서 다양한 게임을 즐길 수 있습니다.

🎭 표정 연기 게임: AI와 함께 표정 연기 실력을 테스트해보세요.

⚡ 암초 깨기 게임: 스트레스가 쌓였다면 '분노의 암초'를 부수며 해소하세요.

🧭 오늘의 나침반 카드: 타로 카드를 뽑으며 하루의 방향성을 점쳐보세요.


