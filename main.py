import streamlit as st
import emotion_list

# 페이지 설정
st.set_page_config(
    page_title="😊 감정 분석 데모", 
    page_icon="🎭", 
    layout="wide"
)

# 세션 상태 초기화 (처음 실행시에만)
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'main'  # 시작 페이지

# 감정 리스트 정의 (새로운 상세 구조)
EMOTIONS = emotion_list.emotions

# === 페이지 함수들 ===

def show_main_page():
    """메인 선택 페이지"""
    st.title("😊 감정 분석 프로젝트")
    st.markdown("---")
    
    # 설명
    st.markdown("""
    ### 🎭 어떤 방법으로 감정을 분석하고 싶으세요?
    
    두 가지 방법 중 하나를 선택해주세요:
    """)
    
    # 선택 버튼들을 두 개의 컬럼으로 배치
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📹 실시간 웹캠 분석")
        st.write("웹캠을 통해 실시간으로 감정을 분석합니다")
        if st.button("🎥 웹캠으로 분석하기", use_container_width=True):
            st.session_state.current_page = 'webcam'
            st.rerun()  # 페이지 새로고침
    
    with col2:
        st.markdown("#### ✋ 수동으로 감정 선택")
        st.write("직접 감정을 선택해서 결과를 확인합니다")
        if st.button("🎯 직접 선택하기", use_container_width=True):
            st.session_state.current_page = 'manual'
            st.rerun()


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
            # 각 감정별 스타일링된 버튼
            if st.button(
                f"{emotion_data['emoji']} {emotion_data['korean']}", 
                use_container_width=True,
                key=f"emotion_{emotion_key}"
            ):
                st.session_state.current_page = 'result'
                st.session_state.selected_emotion = emotion_key
                st.rerun()


def show_webcam_page():
    """웹캠 페이지 (아직 기본 구조만)"""
    st.title("📹 웹캠 감정 분석")
    st.markdown("---")
    
    # 뒤로가기 버튼
    if st.button("🔙 메인으로 돌아가기"):
        st.session_state.current_page = 'main'
        st.rerun()
    
    st.info("🚧 웹캠 기능은 다음 단계에서 구현할 예정입니다!")
    
    # 임시로 테스트용 버튼들
    st.markdown("### 테스트용 감정 결과")
    cols = st.columns(3)
    
    test_emotions = ['happy', 'sad', 'angry']
    for i, emotion in enumerate(test_emotions):
        with cols[i]:
            if st.button(f"테스트: {EMOTIONS[emotion]['korean']}", key=f"test_{emotion}"):
                st.session_state.current_page = 'result'
                st.session_state.selected_emotion = emotion
                st.rerun()


def show_result_page():
    """감정 결과 페이지"""
    import random  # 랜덤 선택을 위해 추가
    
    emotion_key = st.session_state.get('selected_emotion', 'neutral')
    emotion = EMOTIONS[emotion_key]
    
    st.title(f"{emotion['emoji']} {emotion['korean']} 감정 결과")
    st.markdown("---")
    
    # 기본 설명
    st.markdown(f"""
    ### 🎯 분석 결과: **{emotion['korean']}**
    
    {emotion['emoji']} {emotion['description']}
    """)
    
    # 랜덤으로 솔루션 하나 선택해서 표시
    random_solution = random.choice(emotion['solutions'])
    st.success(f"💡 **추천 해결책**: {random_solution}")
    
    # 랜덤으로 명언 하나 선택해서 표시
    random_quote = random.choice(emotion['quotes'])
    st.info(f"📝 **오늘의 명언**: {random_quote}")
    
    # 팁 표시
    st.markdown(f"### 💭 **도움이 되는 팁**")
    st.write(f"✨ {emotion['tips']}")
    
    # 더 많은 정보 보기 (접을 수 있는 형태)
    with st.expander("📋 더 많은 해결책 보기"):
        st.write("**모든 해결책:**")
        for solution in emotion['solutions']:
            st.write(f"• {solution}")
    
    with st.expander("💬 더 많은 명언 보기"):
        st.write("**관련 명언들:**")
        for quote in emotion['quotes']:
            st.write(f"• {quote}")
    
    # 액션 버튼들
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 다시 분석하기", use_container_width=True):
            st.session_state.current_page = 'main'
            # 이전 결과 초기화
            if 'selected_emotion' in st.session_state:
                del st.session_state.selected_emotion
            st.rerun()
    
    with col2:
        if st.button("📊 다른 감정 보기", use_container_width=True):
            st.session_state.current_page = 'manual'
            st.rerun()


# === 메인 라우터 ===

def main():
    """메인 라우터 - 현재 페이지에 따라 적절한 함수 호출"""
    
    # 사이드바에 현재 상태 표시 (디버깅용)
    with st.sidebar:
        st.header("🔧 상태 정보")
        st.write(f"현재 페이지: `{st.session_state.current_page}`")
        if 'selected_emotion' in st.session_state:
            emotion = EMOTIONS[st.session_state.selected_emotion]
            st.write(f"선택된 감정: {emotion['emoji']} {emotion['korean']}")
    
    # 현재 페이지에 따라 적절한 함수 호출
    if st.session_state.current_page == 'main':
        show_main_page()
    elif st.session_state.current_page == 'manual':
        show_manual_page()
    elif st.session_state.current_page == 'webcam':
        show_webcam_page()
    elif st.session_state.current_page == 'result':
        show_result_page()
    else:
        # 예상치 못한 페이지면 메인으로
        st.session_state.current_page = 'main'
        st.rerun()
    
    # 푸터
    st.markdown("---")
    st.caption("Made with Streamlit 🚀")


# 앱 실행
if __name__ == "__main__":
    main()