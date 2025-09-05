# -*- coding: utf-8 -*-
"""
속담 항해 퀴즈 (바다 테마 · 클릭형 빈칸)
- '항로 1/2' 표기를 '빈칸 ①/②'로 변경
- 빈칸 번호가 문장 순서대로 ① → ②가 되도록 정렬
- 선택지에서 '(선택)' 제거
- 등대 힌트는 커스텀 박스로 길게 표시 (HTML 태그 노출 방지)
Python 3.8+ / requirements: streamlit
"""

import streamlit as st
import random
import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# -----------------------------
# 페이지 설정
# -----------------------------
st.set_page_config(page_title="속담 항해 퀴즈", page_icon="⛵", layout="centered")

# -----------------------------
# 바다 테마 CSS
# -----------------------------
st.markdown(
    """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;600;700&display=swap');
  [data-testid="stSidebar"] { display:none; }
  html, body, [class*="css"] { font-family:'Noto Sans KR', sans-serif; color:#0b2545; }
  .stApp { background: radial-gradient(ellipse at top, #b3e5ff 0%, #7cc6f7 35%, #4e80ee 100%); min-height: 100vh; }
  .main-container { background: rgba(255,255,255,0.92); padding: 28px; border-radius: 20px; backdrop-filter: blur(10px);
    box-shadow: 0 15px 40px rgba(0,0,0,0.18); max-width: 860px; margin: 24px auto; }
  .quiz-title { text-align:center; font-size:2.2rem; font-weight:800; color:#08395e; text-shadow:0 2px 6px rgba(0,0,0,0.15); margin: 6px 0 14px; }
  .subtitle { text-align:center; color:#0b2545; opacity:.8; margin-top:-2px; }

  .score-display{ text-align:center; background:#ffffffcc; border:2px solid #e6f0ff; border-radius:14px; padding:14px;
    box-shadow:0 6px 18px rgba(0,0,0,0.08); margin: 10px 0 4px; }
  .badge { display:inline-block; padding:6px 10px; border-radius:999px; font-weight:700; font-size:.92rem; margin:0 6px; }
  .badge-blue{ background:#e8f1ff; color:#114b8a; border:1px solid #cfe0ff; }
  .badge-green{ background:#e9fbef; color:#0a6d2d; border:1px solid #c9f1d8; }
  .badge-storm{ background:#f7e8ff; color:#5a189a; border:1px solid #e9d2ff; }

  .question-card { background: linear-gradient(135deg,#eef7ff 0%,#e6f0ff 100%); padding: 22px; border-radius: 16px;
    border-left: 6px solid #1f6feb; box-shadow: 0 8px 22px rgba(31,111,235,0.12); margin-top: 8px; }
  .masked-proverb { font-size:1.7rem; font-weight:800; text-align:center; background:#fff; border-radius:12px; padding:14px; letter-spacing:2px; margin-top: 8px; }

  .stButton > button { background: linear-gradient(135deg,#667eea 0%,#764ba2 100%); color:white; font-weight:700; font-size:16px;
    padding:10px 24px; border-radius:26px; border:none; transition:all .18s ease; width:100%; box-shadow:0 4px 14px rgba(102,126,234,.3); }
  .stButton > button:hover { transform: translateY(-1px); box-shadow:0 8px 20px rgba(102,126,234,.4); }
  .stButton > button:disabled { opacity:.6; }

  /* 라디오를 칩 버튼처럼 */
  div[role="radiogroup"] > label {
    display:inline-block; margin:6px 8px 6px 0; padding:10px 16px; border-radius:999px;
    border:1px solid #cfe0ff; background:#f8fbff; cursor:pointer; transition:all .15s ease;
  }
  div[role="radiogroup"] > label:hover { transform: translateY(-1px); box-shadow:0 2px 8px rgba(31,111,235,.18); }

  .note-card { background:#ffffffcc; border:1px solid #eaeaea; border-radius:12px; padding:12px 14px; }

  /* 힌트 박스 (넓고 길게) */
  .hint-box{
    background: linear-gradient(135deg, #cfe9ff 0%, #c0ddff 100%);
    border: 1px solid #7aa2ff;
    border-radius: 14px;
    padding: 18px 20px;
    margin: 10px 0 16px;
    min-height: 140px;
    box-shadow: 0 6px 18px rgba(31,111,235,.12);
    display: flex;
    align-items: center;
  }
  .hint-title{
    font-weight: 800;
    margin-right: 10px;
    color: #0b3c82;
    white-space: nowrap;
  }
  .hint-body{
    font-size: 1rem;
    line-height: 1.8;
    color: #07305f;
  }
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# 데이터
# -----------------------------
PROVERBS: List[Dict[str, str]] = [
    {"proverb": "가는 말이 고와야 오는 말이 곱다", "meaning": "말을 좋게 해야 남도 좋게 한다"},
    {"proverb": "호랑이 굴에 가야 호랑이 새끼를 잡는다", "meaning": "위험을 무릅써야 큰 성과를 얻는다"},
    {"proverb": "백지장도 맞들면 낫다", "meaning": "쉬운 일도 협력하면 더 쉽다"},
    {"proverb": "우물을 파도 한 우물을 파라", "meaning": "한 가지 일을 끝까지 해야 성공한다"},
    {"proverb": "고래 싸움에 새우 등 터진다", "meaning": "강자 싸움에 약자가 피해를 본다"},
    {"proverb": "발 없는 말이 천 리 간다", "meaning": "말은 퍼지기 쉽다"},
    {"proverb": "등잔 밑이 어둡다", "meaning": "가까운 일을 오히려 모른다"},
    {"proverb": "서당 개 삼 년이면 풍월을 읊는다", "meaning": "오래 접하면 알게 된다"},
    {"proverb": "소 잃고 외양간 고친다", "meaning": "이미 일이 잘못된 뒤에 대책을 세운다"},
    {"proverb": "원숭이도 나무에서 떨어진다", "meaning": "아무리 잘하는 사람도 실수할 수 있다"},
    {"proverb": "티끌 모아 태산", "meaning": "작은 것도 모이면 큰 것이 된다"},
    {"proverb": "하늘이 무너져도 솟아날 구멍이 있다", "meaning": "아무리 어려워도 해결책은 있다"},
    {"proverb": "가는 날이 장날이다", "meaning": "뜻밖의 일을 우연히 만난다"},
    {"proverb": "벼는 익을수록 고개를 숙인다", "meaning": "겸손의 미덕을 말한다"},
    {"proverb": "급할수록 돌아가라", "meaning": "급할수록 침착해야 한다"},
    {"proverb": "꿩 먹고 알 먹는다", "meaning": "한 번에 두 가지 이익을 얻는다"},
    {"proverb": "닭 쫓던 개 지붕 쳐다본다", "meaning": "헛수고만 하고 허탈해진다"},
    {"proverb": "뛰는 놈 위에 나는 놈 있다", "meaning": "더 뛰어난 사람이 있다"},
    {"proverb": "바늘 도둑이 소 도둑 된다", "meaning": "작은 잘못이 큰 범죄로 이어진다"},
    {"proverb": "낮말은 새가 듣고 밤말은 쥐가 듣는다", "meaning": "말은 언제나 누군가 들을 수 있다"},
    {"proverb": "세 살 버릇 여든 간다", "meaning": "어린 시절 습관은 평생 간다"},
    {"proverb": "하룻강아지 범 무서운 줄 모른다", "meaning": "철없는 사람이 무모한 행동을 한다"},
    {"proverb": "윗물이 맑아야 아랫물이 맑다", "meaning": "윗사람이 바르면 아랫사람도 바르다"},
    {"proverb": "불난 집에 부채질한다", "meaning": "힘든 상황을 더 어렵게 만든다"},
    {"proverb": "아니 땐 굴뚝에 연기 나랴", "meaning": "원인 없는 결과는 없다"},
    {"proverb": "가는 떡이 커야 오는 떡도 크다", "meaning": "주는 만큼 받는다"},
    {"proverb": "금강산도 식후경", "meaning": "무슨 일도 먹은 뒤에 해야 한다"},
    {"proverb": "돌다리도 두들겨 보고 건너라", "meaning": "확실해 보여도 확인은 필요하다"},
    {"proverb": "공든 탑이 무너지랴", "meaning": "정성 들인 일은 무너지지 않는다"},
]

SAVE_FILE = "wrong_notes.json"
BLANK_TERM = "빈칸"  # 라벨 텍스트 (예: "해도", "좌표", "표식" 등)
CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩"  # circled numbers

# -----------------------------
# 오답노트 유틸
# -----------------------------
def load_wrong_notes() -> list:
    try:
        with open(SAVE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_wrong_notes(notes: list):
    with open(SAVE_FILE, "w", encoding="utf-8") as f:
        json.dump(notes, f, ensure_ascii=False, indent=2)

# -----------------------------
# 클릭 빈칸 문제 생성 (정렬/레이블 반영)
# -----------------------------
def build_click_blanks_question(
    proverb: str,
    all_proverbs: List[Dict[str, str]],
    max_choices: int = 6,
    force_blanks_cnt: Optional[int] = None,
) -> Dict[str, Any]:
    tokens = proverb.split()
    cand_idx = [i for i, w in enumerate(tokens) if len(w) >= 2] or [0]

    # 빈칸 개수 결정 및 "문장 순서대로" 정렬
    blanks_cnt = force_blanks_cnt if force_blanks_cnt else (2 if len(cand_idx) >= 2 and random.random() < 0.5 else 1)
    blanks = sorted(random.sample(cand_idx, blanks_cnt))  # 정렬하여 ①,② 순서 보장

    # 오답 풀 수집
    pool = set()
    for p in all_proverbs:
        for w in p["proverb"].split():
            if len(w) >= 2:
                pool.add(w)
    pool = list(pool)

    choices, correct = [], []
    for b in blanks:
        ans = tokens[b]
        distract = [w for w in pool if w != ans and abs(len(w) - len(ans)) <= 1]
        random.shuffle(distract)
        # '(선택)' 제거 버전: 정답 + 오답 후보만
        opts = [ans] + distract[:max(0, max_choices - 1)]
        random.shuffle(opts)
        choices.append(opts)
        correct.append(ans)

    # 마스킹 텍스트 (해도 ①/②)
    idx2no = {b: i + 1 for i, b in enumerate(blanks)}
    def label_for(i: int) -> str:
        no = idx2no[i]
        circ = CIRCLED[no-1] if 1 <= no <= len(CIRCLED) else str(no)
        return f"[{BLANK_TERM} {circ}]"
    masked_tokens = [label_for(i) if i in idx2no else t for i, t in enumerate(tokens)]
    masked_text = " ".join(masked_tokens)

    return {
        "tokens": tokens,
        "blanks": blanks,
        "choices": choices,
        "correct": correct,
        "masked_text": masked_text,
    }

# -----------------------------
# 데이터클래스
# -----------------------------
@dataclass
class QuizItem:
    proverb: str
    meaning: str
    @classmethod
    def from_dict(cls, d: Dict[str, str]): 
        return cls(d["proverb"], d["meaning"])

# -----------------------------
# 세션 상태
# -----------------------------
if "rounds" not in st.session_state: st.session_state.rounds = 10
if "current" not in st.session_state: st.session_state.current = 0
if "score" not in st.session_state: st.session_state.score = 0
if "pool" not in st.session_state: st.session_state.pool = []
if "mode" not in st.session_state: st.session_state.mode = "menu"  # menu | quiz | note
if "finished" not in st.session_state: st.session_state.finished = False
if "start_time" not in st.session_state: st.session_state.start_time = None
if "qbuild" not in st.session_state: st.session_state.qbuild = {}
if "storm" not in st.session_state: st.session_state.storm = False
if "show_hint" not in st.session_state: st.session_state.show_hint = False  # 힌트 토글

# -----------------------------
# UI 컨테이너
# -----------------------------
st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.markdown('<h1 class="quiz-title">⛵ 속담 항해 퀴즈</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">빈칸을 클릭해 <b>해도</b>를 완성하세요!</p>', unsafe_allow_html=True)

# -----------------------------
# 메뉴 화면
# -----------------------------
if st.session_state.mode == "menu":
    st.markdown(
        """
    <div class="note-card">
      <b>설명</b><br>
      • 각 문제에는 <b>빈칸 ①/②</b>가 뚫려 있어요. 후보 중에서 골라 채우세요.<br>
      • 가끔 <b>🌩️ 폭풍경보</b>가 떠서 빈칸이 2개가 되거나 오답이 까다로워질 수 있어요.
    </div>
    """,
        unsafe_allow_html=True,
    )
    st.write("")
    st.session_state.rounds = st.slider("🏝️ 문제 수 선택", 5, 30, st.session_state.rounds)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🚀 항해 시작"):
            st.session_state.pool = random.sample(PROVERBS, min(st.session_state.rounds, len(PROVERBS)))
            st.session_state.current = 0
            st.session_state.score = 0
            st.session_state.finished = False
            st.session_state.mode = "quiz"
            st.session_state.start_time = time.time()
            st.rerun()
    with c2:
        if st.button("📓 오답노트 보기"):
            st.session_state.mode = "note"
            st.rerun()

# -----------------------------
# 오답노트 화면
# -----------------------------
elif st.session_state.mode == "note":
    st.markdown("### 📓 오답 노트")
    notes = load_wrong_notes()
    if not notes:
        st.info("아직 오답이 없습니다. 항해를 시작해 보세요!")
    else:
        for i, n in enumerate(notes, 1):
            st.markdown(
                f"""
            <div class="note-card" style="margin:8px 0;">
              <b>{i}. {n['proverb']}</b><br>
              뜻: {n['meaning']}<br>
              당신의 답: <i>{n['wrong_answer']}</i>
            </div>
            """,
                unsafe_allow_html=True,
            )
    st.write("")
    if st.button("🔙 돌아가기"):
        st.session_state.mode = "menu"
        st.rerun()

# -----------------------------
# 퀴즈 화면
# -----------------------------
elif st.session_state.mode == "quiz" and st.session_state.pool:
    if st.session_state.finished:
        st.balloons()
        accuracy = (st.session_state.score / st.session_state.rounds) * 100
        duration = time.time() - (st.session_state.start_time or time.time())
        st.markdown(
            f"""
        <div class="score-display">
          🎉 항해 완료! 🎉<br>
          최종 점수: <b>{st.session_state.score} / {st.session_state.rounds}</b> ({accuracy:.1f}%)<br>
          소요 시간: {duration:.1f}초
        </div>
        """,
            unsafe_allow_html=True,
        )
        if accuracy >= 90:
            st.success("🏆 훌륭합니다! 잔잔한 바다 위의 항해였어요.")
        elif accuracy >= 70:
            st.success("👏 좋아요! 파도도 잘 헤쳐 나갔네요.")
        elif accuracy >= 50:
            st.info("📚 조금 더 연습하면 금방 대항해가 될 거예요!")
        else:
            st.warning("💪 파도가 거셌네요. 다시 도전!")
        st.write("")
        if st.button("🔄 다시 시작"):
            for k in ["current", "score", "pool", "finished", "start_time", "qbuild", "storm", "show_hint"]:
                if k in st.session_state:
                    del st.session_state[k]
            st.session_state.mode = "menu"
            st.rerun()
    else:
        progress = (st.session_state.current / st.session_state.rounds)
        st.progress(progress)
        st.markdown(
            f"""
        <div class="score-display">
          <span class="badge badge-blue">⛵ 항해 {st.session_state.current + 1}/{st.session_state.rounds}</span>
          <span class="badge badge-green">🏝️ 정복 섬 {st.session_state.score}</span>
        </div>
        """,
            unsafe_allow_html=True,
        )

        item = QuizItem.from_dict(st.session_state.pool[st.session_state.current])

        # 폭풍 이벤트 (25% 확률)
        if ("idx" not in st.session_state.qbuild) or (st.session_state.qbuild.get("idx") != st.session_state.current):
            storm_flag = random.random() < 0.25
            st.session_state.storm = storm_flag
            st.session_state.qbuild = {
                "idx": st.session_state.current,
                **build_click_blanks_question(
                    item.proverb,
                    PROVERBS,
                    max_choices=7 if storm_flag else 6,
                    force_blanks_cnt=2 if storm_flag else None,
                ),
            }
            # 힌트는 새 문제에서 닫기
            st.session_state.show_hint = False
        qb = st.session_state.qbuild

        storm_badge = '<span class="badge badge-storm">🌩️ 폭풍경보</span>' if st.session_state.storm else ""
        st.markdown(
            f"""
        <div class="question-card">
          <h3>다음 속담의 빈칸을 완성하세요 {storm_badge}</h3>
          <div class="masked-proverb">{qb['masked_text']}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # 등대 힌트 토글 + 박스 (HTML 태그 노출 방지)
        col_hint, _ = st.columns([1, 3])
        with col_hint:
            btn_label = "🗼 등대 힌트 켜기" if not st.session_state.show_hint else "🗼 힌트 접기"
            if st.button(btn_label):
                st.session_state.show_hint = not st.session_state.show_hint
                st.rerun()

        if st.session_state.show_hint:
            st.markdown(
                f"""
            <div class="hint-box">
              <div class="hint-title">🗼 등대 힌트</div>
              <div class="hint-body">{item.meaning}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        # 빈칸별 후보 (라디오) — '(선택)' 제거, 기본 0번 선택
        selections: List[str] = []
        for i, opts in enumerate(qb["choices"]):
            circ = CIRCLED[i] if i < len(CIRCLED) else str(i + 1)
            sel = st.radio(
                f"🏝️ {BLANK_TERM} {circ} 선택",
                options=opts,
                index=0,  # 첫 항목 기본 선택
                horizontal=True,
                key=f"sel_{st.session_state.current}_{i}",
            )
            selections.append(sel)

        # 제출
        if st.button("⚓ 정박하고 채점하기"):
            # 사용자가 완성한 문장 만들어 저장/채점
            user_tokens = qb["tokens"][:]
            for b_idx, choice in zip(qb["blanks"], selections):
                user_tokens[b_idx] = choice
            user_proverb = " ".join(user_tokens)

            is_all_correct = all(s == ans for s, ans in zip(selections, qb["correct"]))

            if is_all_correct:
                st.success("🌊 정답! 잔잔한 항해 계속 갑니다.")
                st.session_state.score += 1
            else:
                st.error(f"🌪️ 오답! 파고가 높았습니다. 정답은 👉 <b>{item.proverb}</b>", icon="🚨")
                notes = load_wrong_notes()
                notes.append(
                    {"proverb": item.proverb, "meaning": item.meaning, "wrong_answer": user_proverb}
                )
                save_wrong_notes(notes)

            st.session_state.current += 1
            if st.session_state.current >= st.session_state.rounds:
                st.session_state.finished = True
            time.sleep(0.9)
            st.rerun()

# 닫기
st.markdown('</div>', unsafe_allow_html=True)

# --- 힐링센터 홈으로 돌아가기 버튼 ---
import streamlit as _st
if _st.button('🏠 홈으로 돌아가기', key='go_hc_home', use_container_width=True):
    # healing_center.py에서 사용하는 상태키로 직접 이동 (멀티페이지/exec 환경 모두 대응)
    _st.session_state['current_page'] = 'home'
    _st.rerun()
