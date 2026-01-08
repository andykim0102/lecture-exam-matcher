import streamlit as st
import google.generativeai as genai
import fitz  # PyMuPDF
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time

# ==========================================
# 1. 설정 및 초기화
# ==========================================
st.set_page_config(page_title="Med-Study OS Final", layout="wide", page_icon="🩺")

if 'db' not in st.session_state: st.session_state.db = []
if 'lecture_doc' not in st.session_state: st.session_state.lecture_doc = None
if 'current_page' not in st.session_state: st.session_state.current_page = 0

# ==========================================
# 2. 핵심 함수 (Logic)
# ==========================================
def extract_text_from_pdf(file):
    """PDF를 텍스트로 변환 (fitz 사용)"""
    doc = fitz.open(stream=file.read(), filetype="pdf")
    pages_content = []
    for page_num, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            pages_content.append({"page": page_num + 1, "text": text, "source": file.name})
    return pages_content

def get_embedding(text):
    """임베딩 (Embedding-004 사용)"""
    try:
        return genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_document"
        )['embedding']
    except Exception:
        try:
            return genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_document"
            )['embedding']
        except:
            return []

def find_relevant_jokbo(query_text, db, top_k=3):
    """유사도 검색"""
    if not db: return []
    query_emb = get_embedding(query_text)
    if not query_emb: return []
    
    db_embs = [item['embedding'] for item in db]
    sims = cosine_similarity([query_emb], db_embs)[0]
    top_idxs = np.argsort(sims)[::-1][:top_k]
    
    return [{"score": sims[i], "content": db[i]} for i in top_idxs]

# ==========================================
# 3. 사이드바
# ==========================================
with st.sidebar:
    st.title("⚙️ 설정")
    api_key = st.text_input("Gemini API Key", type="password")
    if api_key:
        genai.configure(api_key=api_key)
        st.success("API Key 입력됨")
            
    st.divider()
    st.write(f"📚 학습된 족보: {len(st.session_state.db)} 페이지")
    if st.button("초기화"):
        st.session_state.db = []
        st.rerun()

# ==========================================
# 4. 메인 UI
# ==========================================
tab1, tab2 = st.tabs(["📂 족보 학습", "📖 강의 공부"])

# --- TAB 1: 족보 학습 ---
with tab1:
    st.header("1. 족보 업로드")
    files = st.file_uploader("족보 PDF", accept_multiple_files=True, type="pdf")
    
    if st.button("학습 시작 🚀") and files:
        if not api_key:
            st.error("API Key를 입력하세요.")
        else:
            bar = st.progress(0)
            status = st.empty()
            new_db = []
            total_files = len(files)
            
            for i, f in enumerate(files):
                status.text(f"📖 파일 읽는 중: {f.name}...")
                pages = extract_text_from_pdf(f)
                
                for j, p in enumerate(pages):
                    status.text(f"🧠 학습 중: {f.name} ({j+1}/{len(pages)} 페이지)...")
                    emb = get_embedding(p['text'])
                    if emb:
                        p['embedding'] = emb
                        new_db.append(p)
                    # [중요] 속도 제한 방지 대기
                    time.sleep(1.0) 
                
                bar.progress((i + 1) / total_files)
            
            st.session_state.db.extend(new_db)
            status.text("✅ 학습 완료!")
            st.success(f"{len(new_db)} 페이지 학습 완료!")

# --- TAB 2: 강의 분석 ---
with tab2:
    st.header("2. 강의 뷰어 & AI")
    lec_file = st.file_uploader("강의록 PDF", type="pdf", key="lec")
    
    if lec_file:
        if st.session_state.lecture_doc is None or st.session_state.lecture_doc.name != lec_file.name:
            st.session_state.lecture_doc = fitz.open(stream=lec_file.read(), filetype="pdf")
            st.session_state.current_page = 0
            
        doc = st.session_state.lecture_doc
        col_view, col_ai = st.columns([6, 4])
        
        with col_view:
            c1, c2, c3 = st.columns([1, 2, 1])
            if c1.button("◀"): 
                if st.session_state.current_page > 0: st.session_state.current_page -= 1
            c2.markdown(f"<center>{st.session_state.current_page + 1} / {len(doc)}</center>", unsafe_allow_html=True)
            if c3.button("▶"): 
                if st.session_state.current_page < len(doc) - 1: st.session_state.current_page += 1
            
            page = doc.load_page(st.session_state.current_page)
            pix = page.get_pixmap(dpi=150)
            st.image(Image.frombytes("RGB", [pix.width, pix.height], pix.samples), use_container_width=True)
            curr_text = page.get_text()

        with col_ai:
            if st.button("분석하기 ⚡"):
                if not api_key or not st.session_state.db:
                    st.error("API Key 또는 족보 데이터가 없습니다.")
                else:
                    if not curr_text.strip():
                        st.warning("텍스트가 없는 페이지입니다.")
                    else:
                        with st.spinner("AI가 분석 중입니다..."):
                            try:
                                # 1. 관련 족보 찾기
                                related = find_relevant_jokbo(curr_text, st.session_state.db)
                                ctx_str = "\n".join([f"- {i['content']['text'][:100]}" for i in related])
                                
                                prompt = f"강의: {curr_text}\n족보: {ctx_str}\n\n연관성, 키워드, 문제 생성해줘."

                                # [핵심] 무료 한도가 넉넉한 1.5-flash 모델 강제 사용
                                model = genai.GenerativeModel("gemini-1.5-flash")
                                
                                response = model.generate_content(prompt)
                                st.markdown(response.text)
                                    
                            except Exception as e:
                                if "429" in str(e):
                                    st.error("⚠️ 사용량이 많습니다. 30초 뒤에 다시 시도해주세요.")
                                else:
                                    st.error(f"에러 발생: {e}")
