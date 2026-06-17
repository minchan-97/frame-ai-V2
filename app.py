"""
GasCore — 쓸수록 나를 닮아가는 AI
교사용 인터페이스 (사용자 친화적 리디자인)
"""
import streamlit as st
import pickle, io, os, time
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(
    page_title="GasCore — 나만의 AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── 디자인 ─────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&family=Space+Mono:wght@400;700&display=swap');

:root {
    --bg:    #0f1117;
    --sf:    #1a1d27;
    --sf2:   #22263a;
    --bd:    #2a2f45;
    --ac:    #7c6ef5;
    --ac2:   #a594f9;
    --gn:    #4ade80;
    --yw:    #fbbf24;
    --rd:    #f87171;
    --tx:    #e8eaf6;
    --mt:    #7986a8;
    --pass:  #4ade80;
    --warn:  #fbbf24;
    --fatal: #f87171;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--tx) !important;
    font-family: 'Noto Sans KR', sans-serif !important;
}

[data-testid="stSidebar"] {
    background: var(--sf) !important;
    border-right: 1px solid var(--bd) !important;
}

/* 카드 */
.gc-card {
    background: var(--sf);
    border: 1px solid var(--bd);
    border-radius: 12px;
    padding: 1.2rem;
    margin-bottom: 0.8rem;
    transition: border-color 0.2s;
}
.gc-card:hover { border-color: var(--ac); }

/* 히어로 */
.gc-hero {
    background: linear-gradient(135deg, #1a1040 0%, #0f1117 60%);
    border: 1px solid #3a2f6a;
    border-radius: 16px;
    padding: 2.5rem 2rem;
    margin-bottom: 1.5rem;
    text-align: center;
}
.gc-hero h1 {
    font-size: 2.2rem;
    font-weight: 700;
    color: var(--ac2);
    margin-bottom: 0.4rem;
}
.gc-hero p {
    color: var(--mt);
    font-size: 1rem;
    margin: 0;
}

/* 성장 바 */
.gc-grow {
    background: var(--sf2);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.8rem;
}

/* 뱃지 */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.5px;
}
.badge-pass  { background: #1a3a1a; color: var(--pass); border: 1px solid var(--pass); }
.badge-warn  { background: #3a2e0a; color: var(--warn); border: 1px solid var(--warn); }
.badge-fatal { background: #3a0f0f; color: var(--fatal); border: 1px solid var(--fatal); }

/* 채팅 */
.chat-user {
    text-align: right;
    margin: 8px 0;
}
.chat-user span {
    background: #2d2060;
    color: var(--tx);
    padding: 8px 14px;
    border-radius: 16px 16px 4px 16px;
    display: inline-block;
    max-width: 82%;
    font-size: 0.9rem;
    line-height: 1.5;
}
.chat-ai {
    text-align: left;
    margin: 8px 0 14px 0;
}
.chat-ai span {
    background: var(--sf2);
    color: var(--tx);
    padding: 8px 14px;
    border-radius: 16px 16px 16px 4px;
    display: inline-block;
    max-width: 82%;
    font-size: 0.9rem;
    line-height: 1.6;
}
.chat-meta {
    font-size: 0.65rem;
    color: var(--mt);
    margin-top: 3px;
}

/* 토큰 XAI */
.tok-ok  { background:#1a3a1a; color:var(--pass);  padding:2px 5px; border-radius:3px; font-family:monospace; font-size:0.78rem; margin:1px; }
.tok-w   { background:#2a200a; color:var(--warn);  padding:2px 5px; border-radius:3px; font-family:monospace; font-size:0.78rem; margin:1px; }
.tok-bad { background:#2a0f0f; color:var(--fatal); padding:2px 5px; border-radius:3px; font-family:monospace; font-size:0.78rem; margin:1px; }

/* 버튼 */
[data-testid="stButton"] button {
    font-weight: 600 !important;
    border-radius: 8px !important;
    border: none !important;
    transition: opacity 0.15s !important;
}
[data-testid="stButton"] button:hover { opacity: 0.85 !important; }

hr { border-color: var(--bd) !important; }
[data-testid="stExpander"] { border-color: var(--bd) !important; }

/* 탭 */
[data-testid="stTabs"] button {
    font-size: 0.85rem !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)


# ── 엔진 로드 ──────────────────────────────────────────────
try:
    from gascore_engine import GasCoreFramework
    ENGINE_OK = True
except Exception as e:
    ENGINE_OK = False
    st.error(f"엔진을 불러오지 못했어요: {e}")
    st.stop()

# ── 세션 초기화 ──────────────────────────────────────────────
_defaults = {
    "fw":           None,
    "initialized":  False,
    "qa_offline":   [],
    "qa_history":   [],
    "api_key":      os.getenv("OPENAI_API_KEY", ""),
    "model":        "gpt-4o-mini",
    "corpus_bytes": None,
    "corpus_name":  "",
    "pkl_bytes":    None,
    "show_adv":     False,
    "logp_thr":     -11.5,
    "ev_thr":       -13.0,
    "max_retry":    3,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

def get_fw(): return st.session_state.fw
def set_fw(fw): st.session_state.fw = fw


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 사이드바
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with st.sidebar:
    st.markdown("## 🌱 GasCore")
    st.caption("쓸수록 나를 닮아가는 AI")
    st.markdown("---")

    # API Key
    st.session_state.api_key = st.text_input(
        "🔑 OpenAI API Key",
        value=st.session_state.api_key,
        type="password",
        help="GPT 답변 생성에 필요해요. 검증(가드레일)만 쓸 경우 없어도 됩니다."
    )

    st.markdown("---")

    # ── 문서 업로드 ──────────────────────────────────────
    st.markdown("### 📄 내 문서 불러오기")
    st.caption("교육계획서, 교과서, 업무매뉴얼 등 PDF/Word/텍스트 파일")

    uploaded = st.file_uploader(
        "파일 선택",
        type=["txt", "pdf", "docx"],
        key="corpus_up",
        label_visibility="collapsed"
    )
    if uploaded:
        st.session_state.corpus_bytes = uploaded.read()
        st.session_state.corpus_name  = uploaded.name
        st.success(f"📄 {uploaded.name}")

    if st.session_state.corpus_bytes:
        if st.button("🚀 AI 학습 시작", use_container_width=True,
                     type="primary"):
            with st.spinner("문서를 학습하는 중이에요..."):
                try:
                    raw  = st.session_state.corpus_bytes
                    name = st.session_state.corpus_name.lower()
                    if name.endswith(".pdf"):
                        import pypdf
                        text = "\n".join(
                            p.extract_text() or ""
                            for p in pypdf.PdfReader(io.BytesIO(raw)).pages)
                    elif name.endswith(".docx"):
                        import docx
                        text = "\n".join(
                            p.text for p in docx.Document(io.BytesIO(raw)).paragraphs
                            if p.text.strip())
                    else:
                        text = raw.decode("utf-8", errors="ignore")

                    fw = GasCoreFramework()
                    prog = st.progress(0)
                    fw.initialize(text, epochs=10, som_grid=6,
                                  on_progress=lambda p, m: prog.progress(p))
                    set_fw(fw)
                    st.session_state.initialized = True
                    st.session_state.qa_offline  = []
                    st.session_state.qa_history  = []
                    st.rerun()
                except Exception as e:
                    st.error(f"학습 실패: {e}")
                    import traceback; st.code(traceback.format_exc())

    # ── 저장된 학습 불러오기 ──────────────────────────────
    st.markdown("---")
    st.markdown("### 💾 이전 학습 불러오기")
    st.caption("저장해둔 .pkl 파일을 선택하세요")

    pkl_up = st.file_uploader(
        "학습 파일(.pkl)",
        type=None,
        key="pkl_up",
        label_visibility="collapsed"
    )
    if pkl_up:
        if pkl_up.name.endswith('.pkl'):
            st.session_state.pkl_bytes = pkl_up.read()
            st.success(f"💾 {pkl_up.name}")
        else:
            st.warning(".pkl 파일만 지원해요.")

    if st.session_state.pkl_bytes and not st.session_state.initialized:
        if st.button("📂 불러오기", use_container_width=True):
            try:
                with st.spinner("불러오는 중..."):
                    data = pickle.loads(st.session_state.pkl_bytes)
                    fw   = GasCoreFramework.from_dict(data)
                    set_fw(fw)
                    st.session_state.initialized = True
                    st.session_state.qa_offline  = []
                    st.session_state.qa_history  = []
                st.rerun()
            except Exception as e:
                st.error(f"불러오기 실패: {e}")

    # ── 현재 상태 ────────────────────────────────────────
    if st.session_state.initialized:
        fw = get_fw()
        s  = fw.summary()
        st.markdown("---")

        approved = s.get("ev_승인된 생성", 0)
        cycle    = s["cycle"]
        vocab    = s["nm_vocab"]

        st.markdown("### 📊 학습 현황")

        # 성장 게이지
        pct = min(approved / 50, 1.0)
        st.markdown(f"""
<div style="background:#1a1d27;border-radius:8px;padding:0.8rem;">
    <div style="font-size:0.75rem;color:#7986a8;margin-bottom:0.4rem;">
        AI 개인화 진행도
    </div>
    <div style="background:#2a2f45;border-radius:4px;height:8px;overflow:hidden;">
        <div style="background:linear-gradient(90deg,#7c6ef5,#a594f9);
                    width:{pct*100:.0f}%;height:100%;border-radius:4px;
                    transition:width 0.5s;"></div>
    </div>
    <div style="font-size:0.72rem;color:#7986a8;margin-top:0.3rem;">
        {approved}개 승인됨 · {cycle}회 학습 · 어휘 {vocab}개
    </div>
</div>
""", unsafe_allow_html=True)

        st.download_button(
            "💾 학습 저장",
            data=pickle.dumps(fw.to_dict()),
            file_name=f"gascore_c{fw.cycle}.pkl",
            mime="application/octet-stream",
            use_container_width=True,
        )

        # 고급 설정 토글
        st.markdown("---")
        if st.checkbox("⚙️ 고급 설정", value=st.session_state.show_adv):
            st.session_state.show_adv = True
            st.session_state.logp_thr = st.slider(
                "도메인 민감도", -15.0, -5.0, st.session_state.logp_thr, 0.5,
                help="낮을수록 도메인 외 답변을 더 엄격하게 차단해요")
            st.session_state.max_retry = st.slider(
                "재생성 최대 횟수", 1, 5, st.session_state.max_retry)
            st.session_state.model = st.selectbox(
                "GPT 모델", ["gpt-4o-mini", "gpt-4o"],
                index=["gpt-4o-mini","gpt-4o"].index(st.session_state.model))
        else:
            st.session_state.show_adv = False

    st.markdown("---")
    st.caption("⚡ GasCore | GPU 없이 작동 | 내 문서만 사용")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 메인
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ── 미초기화 랜딩 ───────────────────────────────────────────
if not st.session_state.initialized:
    st.markdown("""
<div class="gc-hero">
    <h1>🌱 GasCore</h1>
    <p style="font-size:1.2rem;color:#a594f9;font-weight:600;margin:0.4rem 0;">
        쓸수록 나를 닮아가는 AI
    </p>
    <p>내 문서를 올리면, 그 문서 안에서만 대답해요.<br>
    내가 승인할수록, 내 방식을 배워가요.</p>
</div>
""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    features = [
        ("📄", "내 문서 기반",
         "교육계획서, 교과서, 업무매뉴얼...\n어떤 문서든 올리면 그게 AI의 기준이 돼요."),
        ("🌱", "쓸수록 성장",
         "승인 버튼을 누를수록\nAI가 내 스타일과 기준을 기억해요."),
        ("🔒", "도메인 이탈 차단",
         "문서 밖의 내용은 자동으로 걸러져요.\n환각 없는 안전한 답변만 제공해요."),
    ]
    for col, (icon, title, desc) in zip([c1,c2,c3], features):
        with col:
            st.markdown(f"""
<div class="gc-card" style="text-align:center;min-height:150px;">
    <div style="font-size:2rem;margin-bottom:0.5rem;">{icon}</div>
    <div style="font-weight:700;margin-bottom:0.4rem;color:#a594f9;">{title}</div>
    <div style="font-size:0.82rem;color:#7986a8;white-space:pre-line;">{desc}</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
<div style="background:#1a1d27;border:1px solid #2a2f45;border-radius:12px;
            padding:1.2rem;text-align:center;color:#7986a8;font-size:0.85rem;">
    👈 왼쪽 사이드바에서 <strong style="color:#a594f9">파일을 올리고 학습을 시작</strong>하거나,
    이전에 저장한 학습 파일(.pkl)을 불러오세요.
</div>
""", unsafe_allow_html=True)

    # UnivAI와의 차이 강조
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### GasCore는 다른 AI 도구와 다르게 작동해요")
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("""
<div class="gc-card" style="border-color:#3a2f6a;">
    <div style="color:#7986a8;font-size:0.75rem;margin-bottom:0.5rem;">
        📱 일반 AI 도구 (GPT 래퍼)
    </div>
    <ul style="color:#7986a8;font-size:0.85rem;padding-left:1.2rem;margin:0;">
        <li>매번 새로 분석 — 기억 없음</li>
        <li>문서 밖으로 나갈 수 있음</li>
        <li>누가 써도 똑같은 답변</li>
        <li>10번 써도 처음과 같음</li>
    </ul>
</div>
""", unsafe_allow_html=True)
    with d2:
        st.markdown("""
<div class="gc-card" style="border-color:#7c6ef5;">
    <div style="color:#a594f9;font-size:0.75rem;margin-bottom:0.5rem;">
        🌱 GasCore
    </div>
    <ul style="font-size:0.85rem;padding-left:1.2rem;margin:0;">
        <li>승인할수록 내 판단을 기억</li>
        <li>내 문서 안에서만 대답</li>
        <li>나만의 AI로 점점 변해감</li>
        <li>100번 쓸수록 나를 닮아감</li>
    </ul>
</div>
""", unsafe_allow_html=True)
    st.stop()


# ── 초기화 후 메인 ──────────────────────────────────────────
fw = get_fw()
s  = fw.summary()
approved = s.get("ev_승인된 생성", 0)
cycle    = s["cycle"]

# 상단 상태 바
grow_msg = (
    "🌱 막 시작됐어요 — 승인할수록 AI가 나를 닮아가요" if approved == 0 else
    f"🌿 {approved}번 승인됨 — AI가 당신의 스타일을 배우는 중이에요" if approved < 20 else
    f"🌳 {approved}번 승인됨 — AI가 당신을 꽤 많이 알고 있어요"
)
st.markdown(f"""
<div style="background:linear-gradient(135deg,#1a1040,#1a1d27);
            border:1px solid #3a2f6a;border-radius:10px;
            padding:0.7rem 1.2rem;margin-bottom:1rem;
            display:flex;align-items:center;gap:0.8rem;">
    <div style="flex:1;font-size:0.85rem;color:#a594f9;">{grow_msg}</div>
    <div style="font-size:0.72rem;color:#7986a8;white-space:nowrap;">
        {cycle}회 학습 완료
    </div>
</div>
""", unsafe_allow_html=True)


# ── 탭 ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "💬 내 문서와 대화하기",
    "🧬 AI 발전시키기",
    "🔍 텍스트 직접 검증",
    "🔄 학습 진행하기",
])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 탭 1: LLM + 가드레일 + XAI (메인 대화)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab1:
    st.markdown("##### 내 문서 기반으로 GPT가 답해요")
    st.caption("문서 밖으로 나간 답변은 자동으로 다시 생성 · XAI로 이유 설명")

    if not st.session_state.api_key:
        st.warning("💡 사이드바에서 OpenAI API Key를 입력해주세요.")
    elif not fw.is_initialized:
        st.info("사이드바에서 문서를 먼저 학습해주세요.")
    else:
        # 대화 이력 표시
        for item in st.session_state.qa_history[-8:]:
            r    = item["result"]
            icon = {"PASS":"🟢","WARNING":"🟡","FATAL":"🔴"}.get(r.status,"⬜")
            label = {"PASS":"문서 내 정보","WARNING":"일부 확인 필요","FATAL":"문서 범위 초과"}.get(r.status,"")
            v_color = {"PASS":"#4ade80","WARNING":"#fbbf24","FATAL":"#f87171"}.get(r.status,"#7986a8")

            # 질문
            st.markdown(
                f'<div class="chat-user"><span>{item["question"]}</span></div>',
                unsafe_allow_html=True)

            # 답변 — 카드형으로 전체 표시
            st.markdown(f"""
<div style="background:#1a1d27;border:1px solid {v_color}40;border-left:3px solid {v_color};
            border-radius:0 12px 12px 12px;padding:1rem 1.2rem;margin:4px 0 4px 0;">
    <div style="white-space:pre-wrap;line-height:1.7;font-size:0.92rem;color:#e8eaf6;">
{r.answer}
    </div>
    <div style="margin-top:0.6rem;font-size:0.68rem;color:#7986a8;border-top:1px solid #2a2f45;padding-top:0.4rem;">
        {icon} {label} · {r.attempts}회 검증 · {r.total_ms:.0f}ms
    </div>
</div>
""", unsafe_allow_html=True)

            # XAI 경고 (문서 밖 단어 있을 때만)
            if r.xai and r.xai.outlier_tokens:
                st.markdown(
                    f'<div style="margin:2px 0 12px 0;font-size:0.72rem;color:#f87171;padding-left:4px;">'
                    f'⚠️ 문서 밖 단어 감지: {", ".join(r.xai.outlier_tokens)}</div>',
                    unsafe_allow_html=True)
            else:
                st.markdown("<div style='margin-bottom:12px;'></div>", unsafe_allow_html=True)

        # 입력
        q = st.text_area(
            "질문",
            height=70,
            placeholder="예: 이 교육계획서에서 안전교육 시수는 몇 시간인가요?",
            label_visibility="collapsed",
            key="qa_main_input"
        )
        c_send, c_clear = st.columns([5, 1])
        with c_send:
            send = st.button("📤 물어보기", use_container_width=True,
                             disabled=not q.strip(), type="primary")
        with c_clear:
            if st.button("🗑️", help="대화 기록 지우기"):
                st.session_state.qa_history = []
                st.rerun()

        if send and q.strip():
            def llm_fn_main(prompt):
                from openai import OpenAI
                client = OpenAI(api_key=st.session_state.api_key)
                msgs = []
                hint = get_fw().guideline_hint
                if hint:
                    msgs.append({"role":"system","content":
                        f"""당신은 전문적인 AI 어시스턴트입니다.
아래 문서를 주요 참고자료로 활용하되, 일반 지식으로도 충분히 보완하여 답하세요.
답변은 구체적이고 실용적으로, 충분히 상세하게 작성하세요.

[참고 문서]
{hint[:2000]}

답변 시 주의사항:
- 문서 내용을 바탕으로 하되 창의적으로 확장하세요
- 구체적인 활동, 방법, 예시를 포함하세요
- 충분한 길이로 상세하게 답변하세요"""})
                msgs.append({"role":"user","content":prompt})
                resp = client.chat.completions.create(
                    model=st.session_state.model,
                    messages=msgs,
                    max_tokens=1500)
                return resp.choices[0].message.content.strip()

            with st.spinner("문서를 참고하여 답변 중이에요..."):
                try:
                    result = fw.run_guardrail(
                        question=q.strip(),
                        llm_fn=llm_fn_main,
                        max_attempts=st.session_state.max_retry,
                        logp_thr=st.session_state.logp_thr,
                    )
                    st.session_state.qa_history.insert(0, {
                        "question": q.strip(),
                        "result":   result,
                    })
                    set_fw(fw)
                    st.rerun()
                except Exception as e:
                    st.error(f"오류: {e}")
                    import traceback; st.code(traceback.format_exc())

        # XAI 상세 (마지막 답변)
        if st.session_state.qa_history:
            last = st.session_state.qa_history[0]
            if last["result"].xai:
                with st.expander("🔍 마지막 답변 XAI 분석"):
                    xai = last["result"].xai
                    st.caption(xai.explanation)
                    if xai.token_scores:
                        html = ""
                        for tok, lp, _ in xai.token_scores:
                            cls = "tok-ok" if lp >= -10 else "tok-w" if lp >= -14 else "tok-bad"
                            html += f'<span class="{cls}">{tok}</span> '
                        st.markdown(html, unsafe_allow_html=True)
                        st.caption("🟢 문서 내 단어  🟡 경계  🔴 문서 밖 단어")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 탭 2: AI 발전시키기 (진화 인덱스)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab2:
    st.markdown("##### AI가 아이디어를 제안해요 — 승인할수록 AI가 나를 닮아가요")
    st.caption("승인하면 AI가 학습 — 거부하면 AI가 수정 — 쓸수록 내 스타일로")

    if not st.session_state.api_key:
        st.warning("💡 GPT 답변 생성을 위해 사이드바에서 API Key를 입력해주세요.")

    cl, cr = st.columns([1, 2])
    with cl:
        st.markdown("**제안 수**")
        max_cands = st.slider("", 3, 10, 5, label_visibility="collapsed",
                              key="l1_max")
        if st.button("✨ 아이디어 생성", use_container_width=True,
                     type="primary",
                     disabled=not st.session_state.api_key):
            with st.spinner("아이디어를 구상하는 중이에요..."):
                try:
                    if fw.evolving:
                        fw.evolving.pending = []
                    cands = fw.generate_candidates(
                        max_candidates=max_cands,
                        logp_thr=st.session_state.ev_thr,
                        api_key=st.session_state.api_key,
                        model=st.session_state.model,
                    )
                    if cands:
                        st.success(f"✓ {len(cands)}개 아이디어가 생성됐어요")
                    else:
                        st.info("새 아이디어를 생성하지 못했어요. 더 많은 문서 내용이 필요할 수 있어요.")
                    set_fw(fw)
                except Exception as e:
                    st.error(f"오류: {e}")
                    import traceback; st.code(traceback.format_exc())

        st.markdown("---")
        # 통계 (친화적)
        if fw.evolving:
            ev_s = fw.evolving.summary()
            approved_n = ev_s.get("승인된 생성", 0)
            total_n    = ev_s.get("총 문장", 0)
            st.markdown(f"""
<div style="font-size:0.8rem;color:#7986a8;line-height:2;">
    📝 전체 학습 문장: <strong>{total_n}</strong><br>
    ✅ 내가 승인한 것: <strong>{approved_n}</strong><br>
    🧠 AI 순환 횟수: <strong>{cycle}</strong>
</div>
""", unsafe_allow_html=True)

    with cr:
        pending = fw.evolving.pending if fw.evolving else []
        reasons = ["내 스타일이 아님","사실과 다름","표현이 어색함","이미 있는 내용","기타"]

        if not pending:
            st.markdown("""
<div class="gc-card" style="text-align:center;padding:2rem;color:#7986a8;">
    ← 아이디어 생성 버튼을 눌러보세요<br>
    <small>AI가 내 문서 기반으로 수업활동을 제안해요</small>
</div>
""", unsafe_allow_html=True)
        else:
            ca2, cb2 = st.columns(2)
            with ca2:
                if st.button("✅ 전부 좋아요", use_container_width=True, key="ap_all"):
                    for c in list(fw.evolving.pending):
                        fw.approve_candidate(c)
                    set_fw(fw); st.rerun()
            with cb2:
                if st.button("❌ 전부 거부", use_container_width=True, key="rj_all"):
                    for c in list(fw.evolving.pending):
                        fw.reject_candidate(c, "일괄")
                    set_fw(fw); st.rerun()

            st.caption(f"📋 {len(pending)}개의 아이디어가 기다리고 있어요")
            st.markdown("---")

            for i, cand in enumerate(pending):
                status = cand.get("coreai_status","?")
                badge_cls = {"PASS":"badge-pass","WARNING":"badge-warn"}.get(status,"")
                badge_txt = {"PASS":"문서 내 내용","WARNING":"확인 필요"}.get(status, status)
                tag = "🤖 AI 생성" if cand.get("by_llm") else "📐 조합"

                st.markdown(f"""
<div class="gc-card">
    <div style="font-size:0.7rem;color:#7986a8;margin-bottom:0.5rem;">
        {tag} &nbsp;
        <span class="badge {badge_cls}">{badge_txt}</span>
    </div>
    <div style="font-size:1rem;font-weight:600;margin:0.5rem 0;line-height:1.5;">
        {cand['sentence']}
    </div>
    <div style="font-size:0.72rem;color:#7986a8;">
        💡 {cand['from_a'][:40]} &middot; {cand['from_b'][:40]}
    </div>
</div>
""", unsafe_allow_html=True)

                key_base = f"{i}_{cand['sentence'][:6]}"
                ca, cb = st.columns([1, 1])
                with ca:
                    if st.button("✅ 승인 — 학습에 추가", key=f"ap_{key_base}",
                                 use_container_width=True, type="primary"):
                        fw.approve_candidate(cand)
                        set_fw(fw); st.rerun()
                with cb:
                    r = st.selectbox("거부 이유", reasons,
                                     key=f"rs_{key_base}",
                                     label_visibility="collapsed")
                    if st.button("❌ 거부", key=f"rj_{key_base}",
                                 use_container_width=True):
                        fw.reject_candidate(cand, r)
                        set_fw(fw); st.rerun()
                st.markdown("")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 탭 3: 가드레일 (답변 검증)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 탭 3: 텍스트 직접 검증 (XAI)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab3:
    st.markdown("##### 텍스트가 내 문서 범위 안인지 바로 확인해요")
    st.caption("어떤 텍스트든 붙여넣으면 XAI가 어느 단어가 문제인지 설명해줘요")

    if not fw.is_initialized:
        st.info("사이드바에서 문서를 먼저 학습해주세요.")
    else:
        xai_text = st.text_area(
            "확인할 텍스트",
            height=120,
            placeholder="예: 안전 교육은 연간 5시간 실시한다 (숫자 오류 테스트)\n예: 학교폭력 예방 교육은 학기별로 실시한다 (정상 테스트)",
            label_visibility="collapsed",
            key="l3_text"
        )
        run_xai = st.button("🔍 검증하기", use_container_width=True,
                             disabled=not xai_text.strip(), type="primary")

        if run_xai and xai_text.strip():
            try:
                xai = fw.explain(xai_text.strip(),
                                 logp_thr=st.session_state.logp_thr)
                if xai:
                    v = xai.verdict
                    v_color = {"PASS":"#4ade80","WARNING":"#fbbf24","FATAL":"#f87171"}.get(v,"#7986a8")
                    v_label = {
                        "PASS":    "✅ 문서 범위 안이에요",
                        "WARNING": "⚠️ 일부 단어를 확인해보세요",
                        "FATAL":   "❌ 문서 범위를 벗어났어요"
                    }.get(v, v)

                    c1, c2, c3 = st.columns(3)
                    c1.metric("판정", v_label)
                    c2.metric("처리 시간", f"{xai.ms:.1f}ms")
                    c3.metric("평균 점수", f"{xai.avg_logp:+.2f}")

                    st.markdown("---")
                    st.markdown("**단어별 분석**")
                    if xai.token_scores:
                        html = ""
                        for tok, lp, _ in xai.token_scores:
                            cls = "tok-ok" if lp >= -10 else "tok-w" if lp >= -14 else "tok-bad"
                            html += f'<span class="{cls}">{tok}</span> '
                        st.markdown(html, unsafe_allow_html=True)
                        st.caption("🟢 문서 내 단어  🟡 경계 단어  🔴 문서 밖 단어")

                    st.markdown("---")
                    st.info(f"📋 {xai.explanation}")

                    if xai.outlier_tokens:
                        st.error(f"문서 밖 단어: {', '.join(xai.outlier_tokens)}")
                    if xai.cluster_hint:
                        st.caption(f"💡 관련 문서 개념: {xai.cluster_hint}")
            except Exception as e:
                st.error(f"오류: {e}")
                import traceback; st.code(traceback.format_exc())


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 탭 4: 순환 관리 (학습 진행)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab4:
    st.markdown("##### 승인한 내용을 AI에 반영해요")
    st.caption("승인 내용이 쌓이면 '학습 반영하기'를 눌러주세요 — AI가 더 나를 닮아가요")

    s = fw.summary()
    c1, c2, c3 = st.columns(3)
    c1.metric("학습 횟수", f"{s['cycle']}회")
    c2.metric("승인한 아이디어", f"{s.get('ev_승인된 생성', 0)}개")
    c3.metric("AI가 아는 어휘", f"{s['nm_vocab']}개")

    st.markdown("---")

    has_gen = bool(fw.evolving and fw.evolving.generated)

    if not has_gen:
        st.markdown("""
<div class="gc-card" style="text-align:center;padding:2rem;color:#7986a8;">
    '수업활동 만들기' 탭에서 아이디어를 승인하면<br>
    여기서 AI에 반영할 수 있어요 🌱
</div>
""", unsafe_allow_html=True)
    else:
        gen_count = len(fw.evolving.generated)
        st.success(f"✅ {gen_count}개의 승인된 아이디어가 학습 대기 중이에요")

        if st.button(f"🔄 {gen_count}개 학습 반영하기",
                     use_container_width=True,
                     type="primary"):
            with st.spinner("AI가 학습 중이에요..."):
                fw.complete_cycle(epochs=5)
                set_fw(fw)
            st.success(f"✓ {fw.cycle}번째 학습 완료! AI가 조금 더 나를 닮아졌어요 🌿")
            st.balloons()
            st.rerun()

        st.markdown("---")
        st.markdown("**이번에 승인한 아이디어들**")
        for g in fw.evolving.generated:
            icon = "🟢" if g["coreai_status"] == "PASS" else "🟡"
            gen  = g.get("generation", 0)
            st.markdown(f"""
<div class="gc-card" style="padding:0.6rem 1rem;">
    {icon} {g['sentence']}
    <span style="font-size:0.68rem;color:#7986a8;float:right;">{gen}세대</span>
</div>
""", unsafe_allow_html=True)

