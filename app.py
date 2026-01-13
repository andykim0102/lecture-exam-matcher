# app.py
import time
import re
import streamlit as st
import google.generativeai as genai
import fitz  # PyMuPDF
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 0. Page config
# ==========================================
st.set_page_config(page_title="Med-Study OS", layout="wide", page_icon="🩺")
st.caption("📌 흐름: (1) 과목별 족보 업로드→DB 구축  (2) 강의본/전사텍스트 → 조교가 '족보 나온 포인트'만 요약")

# ==========================================
# 1. Session state
# ==========================================
if "db" not in st.session_state:
    # item: {"subject": str, "page": int, "text": str, "source": str, "embedding": list[float]}
    st.session_state.db = []

if "api_key" not in st.session_state:
    st.session_state.api_key = None

if "api_key_ok" not in st.session_state:
    st.session_state.api_key_ok = False

if "text_models" not in st.session_state:
    st.session_state.text_models = []

if "best_text_model" not in st.session_state:
    st.session_state.best_text_model = None

if "lecture_doc" not in st.session_state:
    st.session_state.lecture_doc = None

if "lecture_filename" not in st.session_state:
    st.session_state.lecture_filename = None

if "current_page" not in st.session_state:
    st.session_state.current_page = 0

# caches
if "last_page_sig" not in st.session_state:
    st.session_state.last_page_sig = None

if "last_ai_sig" not in st.session_state:
    st.session_state.last_ai_sig = None

if "last_ai_text" not in st.session_state:
    st.session_state.last_ai_text = ""

if "last_related" not in st.session_state:
    st.session_state.last_related = []

# ==========================================
# 2. Settings
# ==========================================
JOKBO_THRESHOLD = 0.72  # 추천 0.70~0.75


def has_jokbo_evidence(related: list[dict]) -> bool:
    return bool(related) and related[0]["score"] >= JOKBO_THRESHOLD


# ==========================================
# 3. Utils
# ==========================================
def ensure_configured():
    if st.session_state.get("api_key"):
        genai.configure(api_key=st.session_state["api_key"])


def extract_text_from_pdf(uploaded_file):
    """PDF -> pages [{page, text, source}]"""
    data = uploaded_file.getvalue()  # ✅ UploadedFile read() 이슈 방지
    doc = fitz.open(stream=data, filetype="pdf")
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text() or ""
        if text.strip():
            pages.append({"page": i + 1, "text": text, "source": uploaded_file.name})
    return pages


def get_embedding(text: str):
    text = (text or "").strip()
    if not text:
        return []
    text = text[:12000]  # 안정성 컷
    ensure_configured()

    try:
        return genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_document",
        )["embedding"]
    except Exception:
        try:
            return genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_document",
            )["embedding"]
        except Exception:
            return []


def filter_db_by_subject(subject: str, db: list[dict]):
    """subject가 '전체'면 전체 반환, 아니면 해당 과목만"""
    if not db:
        return []
    subject = (subject or "").strip()
    if subject in ["전체", "ALL", ""]:
        return db
    return [x for x in db if x.get("subject") == subject]


def find_relevant_jokbo(query_text: str, db: list[dict], top_k: int = 5):
    """유사도 검색"""
    if not db:
        return []

    query_emb = get_embedding(query_text)
    if not query_emb:
        return []

    valid_items = [item for item in db if item.get("embedding")]
    if not valid_items:
        return []

    db_embs = [item["embedding"] for item in valid_items]
    sims = cosine_similarity([query_emb], db_embs)[0]
    top_idxs = np.argsort(sims)[::-1][:top_k]
    return [{"score": float(sims[i]), "content": valid_items[i]} for i in top_idxs]


# ==========================================
# 4. AI (조교 설명)
# ==========================================
@st.cache_data(show_spinner=False)
def list_text_models(api_key: str):
    genai.configure(api_key=api_key)
    models = genai.list_models()
    out = []
    for m in models:
        methods = getattr(m, "supported_generation_methods", []) or []
        if "generateContent" in methods:
            out.append(m.name)
    return out


def pick_best_text_model(model_names: list[str]):
    if not model_names:
        return None
    flash = [m for m in model_names if "flash" in m.lower()]
    return flash[0] if flash else model_names[0]


def generate_with_fallback(prompt: str, model_names: list[str]):
    ensure_configured()
    last_err = None
    for name in model_names:
        if not name:
            continue
        try:
            model = genai.GenerativeModel(name)
            res = model.generate_content(prompt)
            text = getattr(res, "text", None)
            if text:
                return text, name
            return str(res), name
        except Exception as e:
            last_err = e
    raise last_err


def build_ta_prompt(lecture_text: str, related: list[dict], subject: str):
    """
    ✅ UI에는 '조교 설명'만 출력
    ✅ 근거(족보 발췌)는 프롬프트에만 넣고 화면에서는 숨김
    """
    ctx_lines = []
    for r in related[:3]:
        c = r["content"]
        src = c.get("source", "")
        pg = c.get("page", "?")
        txt = (c.get("text") or "")[:450]
        ctx_lines.append(f'- [{src} p{pg} | sim={r["score"]:.3f}] {txt}')

    jokbo_ctx = "\n".join(ctx_lines)

    return f"""
너는 의대 조교다. 학생이 강의를 듣는 중이며, 지금 텍스트가 족보에서 어떤 식으로 출제되었는지 빠르게 잡아줘야 한다.
과목: {subject}

중요 규칙:
- 아래 [관련 족보 발췌]에 근거해서만 말해라. (추측/상식 설명/창작 금지)
- 학생이 "수업 중" 바로 체크할 수 있게 짧고 명확하게.
- 강의 텍스트를 길게 다시 말하지 마라. 출제 포인트만.
- 근거 인용은 2개 이상 포함하되, 화면에는 족보 원문을 길게 붙이지 말고 "짧은 구절"로만.

출력 형식(반드시 지켜라):
[조교 한줄 코멘트]
- (한 문장)

[족보에서 나온 포인트 TOP3]
- (짧게 3개)

[족보 출제 방식]
- (객관식/단답/서술/비교/정의/기전 등) + 한 줄

[근거(짧은 인용 2개 이상)]
- "..." (출처: 파일명 p페이지)
- "..." (출처: 파일명 p페이지)

[학생 액션]
- 지금 외울 키워드 5개: 키워드1, 키워드2, ...

[입력 텍스트]
{lecture_text}

[관련 족보 발췌]
{jokbo_ctx}
""".strip()


def build_transcript_prompt(chunks: list[str], related_packs: list[list[dict]], subject: str):
    """
    전사 텍스트(여러 chunk)에서 '족보 관련 내용만' 뽑아서 조교가 정리
    """
    # 근거는 chunk별 top2 정도만
    lines = []
    for idx, (chunk, rel) in enumerate(zip(chunks, related_packs), start=1):
        if not has_jokbo_evidence(rel):
            continue
        ctx = []
        for r in rel[:2]:
            c = r["content"]
            ctx.append(f'- [{c.get("source","")} p{c.get("page","?")} sim={r["score"]:.3f}] {(c.get("text","")[:250])}')
        lines.append(f"""
(구간 {idx})
[강의 전사 일부]
{chunk}

[관련 족보 발췌]
{chr(10).join(ctx)}
""".strip())

    packed = "\n\n".join(lines)
    if not packed.strip():
        packed = "(족보 근거가 있는 구간이 없습니다.)"

    return f"""
너는 의대 조교다. 아래는 강의 전사 텍스트 일부 구간들이다.
목표: '족보에 실제로 나왔던 내용'에 해당하는 구간만 골라, 학생이 복습/수업 중 포인트를 잡게 요약해라.
과목: {subject}

중요 규칙:
- 각 구간은 반드시 [관련 족보 발췌] 근거가 있을 때만 포함해라.
- 추측 금지. 족보 발췌 기반으로만.
- 결과는 "족보 포인트 노트" 형태로 간결하게.

출력 형식:
[족보 포인트 노트]
1) (포인트 제목) - 한 줄 설명
   - 근거: "..." (출처)
   - 학생 액션 키워드: ...

2) ...

입력:
{packed}
""".strip()


# ==========================================
# 5. Transcript chunking
# ==========================================
def chunk_transcript(text: str, max_chars: int = 900):
    """
    전사 텍스트를 너무 길지 않게 분할.
    - 빈 줄/문장 기준으로 자르고
    - 길면 max_chars 기준으로 추가 분할
    """
    text = (text or "").strip()
    if not text:
        return []

    # 1차: 빈 줄 기준
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    # 2차: 너무 긴 덩어리 분할
    chunks = []
    for p in parts:
        if len(p) <= max_chars:
            chunks.append(p)
        else:
            start = 0
            while start < len(p):
                chunks.append(p[start:start + max_chars])
                start += max_chars
    return chunks


# ==========================================
# 6. Sidebar
# ==========================================
with st.sidebar:
    st.title("🩺 Med-Study")

    api_key = st.text_input("Gemini API Key", type="password", key="api_key_input")
    if api_key:
        try:
            st.session_state.api_key = api_key
            genai.configure(api_key=api_key)
            available_models = list_text_models(api_key)
            if not available_models:
                st.session_state.api_key_ok = False
                st.error("generateContent 가능한 모델이 없습니다.")
            else:
                st.session_state.api_key_ok = True
                st.session_state.text_models = available_models
                st.session_state.best_text_model = pick_best_text_model(available_models)
                st.success("AI 연결 완료")
                st.caption(f"텍스트 모델(자동): {st.session_state.best_text_model}")
        except Exception as e:
            st.session_state.api_key_ok = False
            st.error(f"모델 목록 조회 실패: {e}")

    st.divider()

    # 과목 목록(현재 DB 기반)
    subjects_in_db = sorted({x.get("subject", "") for x in st.session_state.db if x.get("subject")})
    st.caption(f"📚 학습된 족보 페이지 수: **{len(st.session_state.db)}**")
    st.caption(f"📚 학습된 과목: **{', '.join(subjects_in_db) if subjects_in_db else '(없음)'}**")

    if st.button("족보 DB 초기화", key="reset_db_btn"):
        st.session_state.db = []
        st.session_state.last_page_sig = None
        st.session_state.last_ai_sig = None
        st.session_state.last_ai_text = ""
        st.session_state.last_related = []
        st.rerun()


# ==========================================
# 7. Tabs
# ==========================================
tab1, tab2, tab3 = st.tabs(
    ["📂 1) 과목별 족보 업로드/학습", "📖 2) 강의본(PDF) → 조교 설명", "🎙️ 3) 강의 전사 텍스트 → 족보 포인트"]
)

# ==================================================
# TAB 1 — Subject-separated Jokbo DB build
# ==================================================
with tab1:
    st.header("📂 1) 과목별 족보 업로드/학습")
    st.info("업로드 시 과목을 지정하면, 이후 분석은 해당 과목 DB에서만 검색합니다.")

    c1, c2 = st.columns([1, 2])
    with c1:
        subject_for_upload = st.selectbox(
            "과목(족보 업로드용)",
            ["해부학", "생리학", "약리학", "기타(직접입력)"],
            index=1,
            key="subject_upload_select",
        )
    with c2:
        subject_custom = st.text_input(
            "기타 과목명",
            disabled=(subject_for_upload != "기타(직접입력)"),
            key="subject_upload_custom",
        )

    subject_final = subject_custom.strip() if subject_for_upload == "기타(직접입력)" else subject_for_upload
    subject_final = subject_final if subject_final else "기타(미입력)"

    st.caption(f"현재 업로드 과목: **{subject_final}**")

    files = st.file_uploader(
        "족보 PDF 업로드(여러 개 가능)",
        type="pdf",
        accept_multiple_files=True,
        key="jokbo_pdf_uploader",
    )

    col_a, col_b = st.columns([1, 2])
    with col_a:
        max_pages = st.number_input(
            "파일당 최대 학습 페이지(데모용)",
            min_value=1,
            max_value=500,
            value=60,
            step=1,
            key="max_pages_input",
        )
    with col_b:
        st.caption("많이 학습할수록 비용/시간이 늘어요. (데모는 30~80 추천)")

    if st.button("📚 족보 DB 구축 시작", key="build_db_btn"):
        if not st.session_state.api_key_ok or not st.session_state.get("api_key"):
            st.error("사이드바에서 유효한 API Key를 먼저 설정하세요.")
            st.stop()
        if not files:
            st.warning("족보 PDF를 업로드하세요.")
            st.stop()

        bar = st.progress(0)
        status = st.empty()
        new_db = []
        total_files = len(files)

        for i, f in enumerate(files):
            status.text(f"📖 파일 처리: {f.name}")
            pages = extract_text_from_pdf(f)[: int(max_pages)]
            if not pages:
                status.text(f"⚠️ 텍스트 없음: {f.name} (스캔본이면 OCR 필요)")
                bar.progress((i + 1) / total_files)
                continue

            for j, p in enumerate(pages):
                status.text(f"🧠 임베딩: {f.name} ({j+1}/{len(pages)}p)")
                emb = get_embedding(p["text"])
                if emb:
                    p["embedding"] = emb
                    p["subject"] = subject_final
                    new_db.append(p)
                time.sleep(0.7)  # 429 완화(구축 시에만)

            bar.progress((i + 1) / total_files)

        st.session_state.db.extend(new_db)
        status.text("✅ 학습 완료")
        st.success(f"과목 [{subject_final}]로 총 {len(new_db)} 페이지 학습 완료")
        st.info("👉 2번/3번 탭에서 과목을 선택하고 분석하세요.")


# ==================================================
# TAB 2 — Lecture PDF -> TA explanation only (no raw jokbo UI)
# ==================================================
with tab2:
    st.header("📖 2) 강의본(PDF) → 조교 설명")
    st.info("강의 페이지를 넘기면, 오른쪽에 조교가 '족보에서 나온 포인트'만 설명해줍니다. (원문 카드 출력 없음)")

    if not st.session_state.db:
        st.warning("먼저 1번 탭에서 **족보 DB를 구축**하세요.")

    # subject selection for analysis
    subjects_in_db = sorted({x.get("subject", "") for x in st.session_state.db if x.get("subject")})
    subject_options = ["전체"] + (subjects_in_db if subjects_in_db else ["(DB 없음)"])
    subject_pick = st.selectbox("분석 과목(이 과목 DB에서만 검색)", subject_options, key="tab2_subject_pick")

    lec_file = st.file_uploader("강의본 PDF 업로드", type="pdf", key="lec_pdf_uploader")

    # (optional) debug: show evidence snippets
    debug_show = st.toggle("디버그: 근거(짧은 발췌) 보기", value=False, key="debug_evidence_tab2")

    if lec_file:
        if st.session_state.lecture_doc is None or st.session_state.lecture_filename != lec_file.name:
            data = lec_file.getvalue()
            st.session_state.lecture_doc = fitz.open(stream=data, filetype="pdf")
            st.session_state.lecture_filename = lec_file.name
            st.session_state.current_page = 0
            st.session_state.last_page_sig = None
            st.session_state.last_ai_sig = None
            st.session_state.last_ai_text = ""
            st.session_state.last_related = []

        doc = st.session_state.lecture_doc
        col_view, col_right = st.columns([6, 4])

        # LEFT: PDF viewer
        with col_view:
            nav1, nav2, nav3 = st.columns([1, 2, 1])

            if nav1.button("◀", key="prev_page_btn"):
                if st.session_state.current_page > 0:
                    st.session_state.current_page -= 1

            nav2.markdown(
                f"<center><b>{st.session_state.current_page+1} / {len(doc)}</b></center>",
                unsafe_allow_html=True,
            )

            if nav3.button("▶", key="next_page_btn"):
                if st.session_state.current_page < len(doc) - 1:
                    st.session_state.current_page += 1

            page = doc.load_page(st.session_state.current_page)
            pix = page.get_pixmap(dpi=150)
            st.image(
                Image.frombytes("RGB", [pix.width, pix.height], pix.samples),
                use_container_width=True,
            )

            page_text = (page.get_text() or "").strip()
            if not page_text:
                st.warning("이 페이지에는 텍스트가 없습니다. (스캔 PDF면 OCR이 필요할 수 있어요)")

        # RIGHT: TA explain
        with col_right:
            st.markdown("### 🧑‍🏫 조교 설명")

            if not st.session_state.db:
                st.info("족보 DB가 없어서 비교할 수 없습니다.")
                st.stop()

            db_sub = filter_db_by_subject(subject_pick, st.session_state.db)

            if not page_text:
                st.info("텍스트가 없어 분석할 수 없습니다.")
                st.stop()

            # search only when page changes
            page_sig = hash(page_text)
            if page_sig != st.session_state.last_page_sig:
                st.session_state.last_page_sig = page_sig
                st.session_state.last_related = find_relevant_jokbo(page_text, db_sub, top_k=5)
                st.session_state.last_ai_sig = None  # force regen for new page

            related = st.session_state.last_related

            if not has_jokbo_evidence(related):
                st.warning("이 페이지는 족보 근거가 뚜렷하지 않아서(임계값 미만) 조교 설명을 생략했습니다.")
                st.caption(f"임계값: {JOKBO_THRESHOLD:.2f} / 최고 유사도: {related[0]['score']:.3f}" if related else "")
                st.stop()

            if not st.session_state.api_key_ok:
                st.warning("조교 설명을 쓰려면 사이드바에 Gemini API Key를 입력해야 합니다.")
                st.stop()

            # AI caching
            ai_sig = (page_sig, subject_pick, related[0]["content"].get("source"), related[0]["content"].get("page"))
            if ai_sig != st.session_state.last_ai_sig:
                prompt = build_ta_prompt(page_text, related, subject_pick)
                models = st.session_state.text_models or []
                with st.spinner("조교가 족보 근거로 설명 중..."):
                    result, used = generate_with_fallback(prompt, models)
                st.session_state.last_ai_sig = ai_sig
                st.session_state.last_ai_text = result
                st.caption(f"사용 모델: {used}")

            st.write(st.session_state.last_ai_text)

            if debug_show:
                with st.expander("디버그: 매칭 근거(상위 3개, 짧게)", expanded=False):
                    for i, r in enumerate(related[:3], start=1):
                        c = r["content"]
                        st.markdown(f"**#{i} sim={r['score']:.3f} · {c.get('source','')} p{c.get('page','?')} · 과목={c.get('subject','')}**")
                        st.write((c.get("text") or "")[:500] + "…")
    else:
        st.caption("강의본 PDF를 올리면, 오른쪽에 조교 설명이 자동으로 표시됩니다.")


# ==================================================
# TAB 3 — Transcript text -> pick only jokbo-related points
# ==================================================
with tab3:
    st.header("🎙️ 3) 강의 전사 텍스트 → 족보 포인트")
    st.info("교수님 강의를 녹음한 뒤 전사된 텍스트를 넣으면, '족보에 나온 내용'만 골라 조교가 정리합니다. (원문 카드 출력 없음)")

    if not st.session_state.db:
        st.warning("먼저 1번 탭에서 **족보 DB를 구축**하세요.")

    subjects_in_db = sorted({x.get("subject", "") for x in st.session_state.db if x.get("subject")})
    subject_options = ["전체"] + (subjects_in_db if subjects_in_db else ["(DB 없음)"])
    subject_pick = st.selectbox("분석 과목(이 과목 DB에서만 검색)", subject_options, key="tab3_subject_pick")

    up_txt = st.file_uploader("전사 텍스트(.txt) 업로드(선택)", type=["txt"], key="transcript_txt_uploader")
    transcript_text = ""
    if up_txt is not None:
        try:
            transcript_text = up_txt.getvalue().decode("utf-8", errors="ignore")
        except Exception:
            transcript_text = ""

    transcript_text = st.text_area(
        "전사 텍스트 붙여넣기(업로드 대신 가능)",
        value=transcript_text,
        height=240,
        key="transcript_text_area",
        placeholder="예) 오늘은 신경계의 ... (전사된 텍스트를 그대로 붙여넣기)",
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        max_chunks = st.number_input("최대 구간 수(데모)", min_value=1, max_value=40, value=12, step=1, key="max_chunks")
    with col2:
        st.caption("전사 텍스트가 길면 비용/시간이 늘어요. 데모는 8~15 추천")

    debug_show = st.toggle("디버그: 구간별 매칭 점수 보기", value=False, key="debug_evidence_tab3")

    if st.button("🧠 전사 텍스트에서 족보 포인트 뽑기", key="run_transcript_btn"):
        if not transcript_text.strip():
            st.error("전사 텍스트를 입력(또는 txt 업로드)하세요.")
            st.stop()
        if not st.session_state.api_key_ok:
            st.error("사이드바에 Gemini API Key를 입력하세요.")
            st.stop()

        db_sub = filter_db_by_subject(subject_pick, st.session_state.db)

        # chunking
        chunks = chunk_transcript(transcript_text, max_chars=900)[: int(max_chunks)]
        if not chunks:
            st.error("텍스트를 구간으로 나누지 못했습니다.")
            st.stop()

        # retrieve per chunk
        related_packs = []
        prog = st.progress(0)
        for i, ch in enumerate(chunks, start=1):
            rel = find_relevant_jokbo(ch, db_sub, top_k=3)
            related_packs.append(rel)
            prog.progress(i / len(chunks))

        # build + run AI summarizer (only evidence chunks)
        prompt = build_transcript_prompt(chunks, related_packs, subject_pick)
        models = st.session_state.text_models or []

        with st.spinner("족보 근거가 있는 구간만 모아 조교가 정리 중..."):
            result, used = generate_with_fallback(prompt, models)

        st.markdown("### 🧑‍🏫 족보 포인트 노트")
        st.caption(f"사용 모델: {used}")
        st.write(result)

        if debug_show:
            with st.expander("디버그: 구간별 최고 유사도", expanded=False):
                for idx, rel in enumerate(related_packs, start=1):
                    best = rel[0]["score"] if rel else 0.0
                    mark = "✅" if (rel and best >= JOKBO_THRESHOLD) else "—"
                    st.write(f"{mark} 구간 {idx}: best_sim={best:.3f}")
