# app.py
import time
import streamlit as st
import google.generativeai as genai
import fitz  # PyMuPDF
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 0. 페이지 설정
# ==========================================
st.set_page_config(page_title="Med-Study OS", layout="wide", page_icon="🩺")
st.caption("📌 흐름: (1) 족보 업로드→DB 구축  (2) 강의본 업로드→페이지 넘기며 옆에서 '족보가 어떻게 나왔는지' 조교 설명")

# ==========================================
# 1. 세션 상태 초기화
# ==========================================
if "db" not in st.session_state:
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

# 페이지별 캐시(중복 호출 방지)
if "last_page_sig" not in st.session_state:
    st.session_state.last_page_sig = None
if "last_related" not in st.session_state:
    st.session_state.last_related = []
if "last_ai_sig" not in st.session_state:
    st.session_state.last_ai_sig = None
if "last_ai_text" not in st.session_state:
    st.session_state.last_ai_text = ""

# ==========================================
# 2. 설정값
# ==========================================
JOKBO_THRESHOLD = 0.72  # 추천 0.70~0.75


def has_jokbo_evidence(related: list[dict]) -> bool:
    return bool(related) and related[0]["score"] >= JOKBO_THRESHOLD


# ==========================================
# 3. 유틸
# ==========================================
def ensure_configured():
    if st.session_state.get("api_key"):
        genai.configure(api_key=st.session_state["api_key"])


def extract_text_from_pdf(uploaded_file):
    data = uploaded_file.getvalue()
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
    text = text[:12000]
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


def find_relevant_jokbo(query_text: str, db: list[dict], top_k: int = 5):
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


def build_ta_explain_prompt(lecture_text: str, related: list[dict]) -> str:
    """
    목표: '조교가 옆에서 설명' 형태로
    - 강의에서 지금 뭐가 핵심인지
    - 족보에서 어떤 방식으로 나왔는지
    - 근거(족보 문장) 2개 이상
    """
    # 족보 근거 상위 3개만 넣어 속도/품질 균형
    ctx_lines = []
    for r in related[:3]:
        c = r["content"]
        src = c.get("source", "")
        pg = c.get("page", "?")
        txt = (c.get("text") or "")[:450]
        ctx_lines.append(f"- [{src} p{pg} | sim={r['score']:.3f}] {txt}")

    jokbo_ctx = "\n".join(ctx_lines)

    return f"""
너는 의대 조교다. 학생이 강의를 듣는 중이며, 지금 보고 있는 강의 페이지가 '족보에 나왔던 내용인지' 빠르게 잡아주면 된다.

중요 규칙:
- 아래 [관련 족보 발췌]를 근거로만 말해라. (추측/창작/상식 설명 금지)
- 강의 텍스트 자체를 길게 재진술하지 말고, 시험 포인트 중심으로만.
- 근거는 반드시 '족보 문장/구절'로 2개 이상 짧게 인용해라.

출력 형식(반드시 지켜라):
[조교 한줄 코멘트]
- (한 문장)

[족보에서 나온 포인트]
- 포인트 3개 (짧게)

[족보에서 실제로 나온 방식]
- 어떤 형태(단답/객관식/서술/비교/기전/정의 등)인지 1~2줄

[근거 인용]
- "..." (출처: 파일명 p페이지)
- "..." (출처: 파일명 p페이지)

[학생 액션]
- 지금 외워야 할 키워드 5개 (콤마로)

[강의 페이지 텍스트]
{lecture_text}

[관련 족보 발췌]
{jokbo_ctx}
""".strip()


# ==========================================
# 5. UI 렌더
# ==========================================
def render_jokbo_cards(related: list[dict]):
    st.markdown("### 📌 이 페이지와 유사한 족보 근거")
    if not related:
        st.info("족보 DB가 비어있거나, 텍스트 임베딩 생성에 실패했습니다.")
        return

    if not has_jokbo_evidence(related):
        st.warning("관련 족보 근거를 찾지 못했습니다.")
        return

    for i, r in enumerate(related):
        c = r["content"]
        score = r["score"]
        title = f"#{i+1}  유사도 {score:.3f} · {c.get('source','(unknown)')} · p{c.get('page','?')}"
        with st.container(border=True):
            st.markdown(f"**{title}**")
            snippet = (c.get("text") or "").strip()
            snippet = snippet[:900]
            st.write(snippet + ("…" if len((c.get("text") or "")) > 900 else ""))


def render_ta_panel(page_text: str, related: list[dict], auto_ai: bool):
    st.markdown("### 🧑‍🏫 조교 설명")

    if not has_jokbo_evidence(related):
        st.info("이 페이지는 족보 근거가 뚜렷하지 않아서 조교 설명을 생략했어요.")
        return

    if not auto_ai:
        st.caption("자동 조교 설명이 꺼져 있어요. 토글을 켜면 페이지 넘길 때마다 자동으로 설명해줘요.")
        return

    if not st.session_state.api_key_ok or not st.session_state.get("api_key"):
        st.warning("조교 설명(AI)을 쓰려면 사이드바에 Gemini API Key를 입력해야 해요.")
        return

    # 페이지 텍스트 + top1 유사도 기반 시그니처로 중복 생성 방지
    sig = (hash(page_text), related[0]["content"].get("source"), related[0]["content"].get("page"))
    if sig != st.session_state.last_ai_sig:
        prompt = build_ta_explain_prompt(page_text, related)
        models = st.session_state.text_models or []
        with st.spinner("조교가 족보 근거를 바탕으로 설명 중..."):
            try:
                result, used = generate_with_fallback(prompt, models)
                st.session_state.last_ai_sig = sig
                st.session_state.last_ai_text = result
                st.caption(f"사용 모델: {used}")
            except Exception as e:
                st.error(f"AI 설명 생성 실패: {e}")
                return
    else:
        st.caption("사용 모델: (캐시)")

    st.write(st.session_state.last_ai_text)


# ==========================================
# 6. 사이드바
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
    st.caption(f"📚 학습된 족보 페이지 수: **{len(st.session_state.db)}**")

    if st.button("족보 DB 초기화", key="reset_db_btn"):
        st.session_state.db = []
        st.session_state.last_page_sig = None
        st.session_state.last_related = []
        st.session_state.last_ai_sig = None
        st.session_state.last_ai_text = ""
        st.rerun()

# ==========================================
# 7. 메인 탭
# ==========================================
tab1, tab2 = st.tabs(["📂 1) 족보 업로드/학습", "📖 2) 강의본 보며 '조교 설명 + 족보 근거'"])

# ==================================================
# TAB 1 — 족보 업로드/학습
# ==================================================
with tab1:
    st.header("📂 1) 족보 업로드/학습")
    st.info("족보 PDF를 여러 개 올려서 페이지 단위로 임베딩 DB를 만들어둡니다.")

    files = st.file_uploader(
        "족보 PDF 업로드",
        type="pdf",
        accept_multiple_files=True,
        key="jokbo_pdf_uploader",
    )

    col_a, col_b = st.columns([1, 2])
    with col_a:
        max_pages = st.number_input(
            "파일당 최대 학습 페이지(데모용)",
            min_value=1,
            max_value=400,
            value=60,
            step=1,
            key="max_pages_input",
        )
    with col_b:
        st.caption("너무 많이 학습하면 임베딩 호출이 많아져 느릴 수 있어요. (데모는 30~80 추천)")

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
            status.text(f"📖 파일 처리 중: {f.name}")
            pages = extract_text_from_pdf(f)[: int(max_pages)]
            if not pages:
                status.text(f"⚠️ 텍스트 추출 실패/빈 PDF: {f.name} (스캔본이면 OCR 필요)")
                bar.progress((i + 1) / total_files)
                continue

            for j, p in enumerate(pages):
                status.text(f"🧠 DB 구축: {f.name} ({j+1}/{len(pages)}p)")
                emb = get_embedding(p["text"])
                if emb:
                    p["embedding"] = emb
                    new_db.append(p)
                time.sleep(0.7)  # 429 완화(족보 구축 때만)

            bar.progress((i + 1) / total_files)

        st.session_state.db.extend(new_db)
        status.text("✅ 학습 완료")
        st.success(f"총 {len(new_db)} 페이지(텍스트 있는 페이지만) 학습 완료")
        st.info("👉 다음 탭에서 강의본을 올리고, 페이지 넘기면서 오른쪽에서 조교 설명 + 족보 근거를 확인하세요.")

# ==================================================
# TAB 2 — 강의본 보며 조교 설명 + 족보 근거 (핵심)
# ==================================================
with tab2:
    st.header("📖 2) 강의본 보며 '조교 설명 + 족보 근거'")
    st.info("강의본을 페이지 넘기면, 오른쪽에 조교 설명이 먼저 뜨고 그 아래에 족보 근거가 표시됩니다.")

    if not st.session_state.db:
        st.warning("먼저 1번 탭에서 **족보 DB를 구축**하세요.")

    lec_file = st.file_uploader("강의본 PDF 업로드", type="pdf", key="lec_pdf_uploader")

    # ✅ 여기서 AI가 '메인' (기본 ON)
    auto_ai = st.toggle("자동 조교 설명(페이지 넘길 때마다 갱신)", value=True, key="auto_ai_toggle")

    if lec_file:
        if st.session_state.lecture_doc is None or st.session_state.lecture_filename != lec_file.name:
            data = lec_file.getvalue()
            st.session_state.lecture_doc = fitz.open(stream=data, filetype="pdf")
            st.session_state.lecture_filename = lec_file.name
            st.session_state.current_page = 0
            st.session_state.last_page_sig = None
            st.session_state.last_related = []
            st.session_state.last_ai_sig = None
            st.session_state.last_ai_text = ""

        doc = st.session_state.lecture_doc
        col_view, col_right = st.columns([6, 4])

        # ---------- LEFT ----------
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

        # ---------- RIGHT ----------
        with col_right:
            if not st.session_state.db:
                st.info("족보 DB가 없어서 비교할 수 없습니다.")
                st.stop()

            # 페이지가 바뀔 때만 검색
            page_sig = hash(page_text) if page_text else None
            if page_text and page_sig != st.session_state.last_page_sig:
                st.session_state.last_page_sig = page_sig
                st.session_state.last_related = find_relevant_jokbo(page_text, st.session_state.db, top_k=5)

            related = st.session_state.last_related

            # 1) 조교 설명(메인)
            render_ta_panel(page_text, related, auto_ai)

            st.divider()

            # 2) 족보 근거(서브)
            render_jokbo_cards(related)
    else:
        st.caption("강의본 PDF를 올리면, 오른쪽에 조교 설명(자동) + 족보 근거가 표시됩니다.")
