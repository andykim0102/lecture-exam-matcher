import streamlit as st
import google.generativeai as genai
import fitz  # PyMuPDF
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import io

# ==========================================
# 1. 설정 및 초기화 (Configuration)
# ==========================================
st.set_page_config(page_title="Med-Study OS Base", layout="wide", page_icon="🧬")

# 세션 상태 초기화
if 'db' not in st.session_state: st.session_state.db = []  # 족보 데이터 저장소
if 'lecture_doc' not in st.session_state: st.session_state.lecture_doc = None
if 'current_page' not in st.session_state: st.session_state.current_page = 0

# ==========================================
# 2. 핵심 함수 (Core Logic)
# ==========================================

def extract_text_from_pdf(file):
    """PDF 파일을 페이지별 텍스트로 분리하여 리스트로 반환"""
    doc = fitz.open(stream=file.read(), filetype="pdf")
    pages_content = []
    for page_num, page in enumerate(doc):
        text = page.get_text()
        if text.strip():  # 빈 페이지 제외
            pages_content.append({
                "page": page_num + 1,
                "text": text,
                "source": file.name
            })
    return pages_content

def get_embedding(text):
    """Gemini API를 사용하여 텍스트를 벡터로 변환"""
    try:
        # 최신 임베딩 모델 사용 (text-embedding-004 권장)
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_document"
        )
        return result['embedding']
    except Exception as e:
        st.error(f"임베딩 실패: {e}")
        return []

def find_relevant_jokbo(query_text, db, top_k=3):
    """강의 내용(Query)과 족보(DB) 간의 코사인 유사도 계산"""
    if not db: return []
    
    # 1. 쿼리 임베딩
    query_embedding = get_embedding(query_text)
    if not query_embedding: return []

    # 2. DB 임베딩 매트릭스 생성
    db_embeddings = [item['embedding'] for item in db]
    
    # 3. 코사인 유사도 계산
    similarities = cosine_similarity([query_embedding], db_embeddings)[0]
    
    # 4. 상위 K개 추출
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        results.append({
            "score": similarities[idx],
            "content": db[idx]
        })
    return results

# ==========================================
# 3. 사이드바 (API 설정)
# ==========================================
with st.sidebar:
    st.title("⚙️ 시스템 설정")
    api_key = st.text_input("Gemini API Key", type="password")
    
    if api_key:
        genai.configure(api_key=api_key)
        st.success("API 연결됨")
    else:
        st.warning("API Key를 입력해주세요.")
        
    st.divider()
    st.write(f"📊 학습된 족보 데이터: {len(st.session_state.db)} 청크")

# ==========================================
# 4. 메인 UI (Tabs)
# ==========================================
tab1, tab2 = st.tabs(["📂 1. 족보 학습 (Knowledge Base)", "📖 2. 강의 학습 (Study Mode)"])

# --- TAB 1: 족보 데이터 구축 ---
with tab1:
    st.subheader("과거 기출문제(족보) 업로드")
    st.info("이곳에 업로드된 PDF는 AI가 검색할 수 있는 '지식 베이스'가 됩니다.")
    
    uploaded_jokbo = st.file_uploader("족보 PDF 파일들", accept_multiple_files=True, type="pdf")
    
    if st.button("데이터베이스 구축 시작 🚀"):
        if not api_key:
            st.error("먼저 API 키를 입력하세요.")
        elif not uploaded_jokbo:
            st.error("파일을 업로드하세요.")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            new_db = []
            total_files = len(uploaded_jokbo)
            
            for idx, file in enumerate(uploaded_jokbo):
                status_text.text(f"처리 중: {file.name}...")
                
                # 1. 텍스트 추출
                pages = extract_text_from_pdf(file)
                
                # 2. 임베딩 (페이지별로 벡터화)
                for p in pages:
                    emb = get_embedding(p['text'])
                    if emb:
                        p['embedding'] = emb
                        new_db.append(p)
                
                progress_bar.progress((idx + 1) / total_files)
            
            # 세션에 저장
            st.session_state.db.extend(new_db)
            status_text.text("완료!")
            st.success(f"총 {len(new_db)}개의 페이지가 학습되었습니다.")

# --- TAB 2: 강의 뷰어 및 분석 ---
with tab2:
    st.subheader("강의록 뷰어 & AI 분석")
    
    lecture_file = st.file_uploader("오늘 공부할 강의록 PDF", type="pdf", key="lecture")
    
    if lecture_file:
        # 파일을 PyMuPDF 객체로 로드
        if st.session_state.lecture_doc is None or st.session_state.lecture_doc.name != lecture_file.name:
            st.session_state.lecture_doc = fitz.open(stream=lecture_file.read(), filetype="pdf")
            st.session_state.current_page = 0 # 페이지 초기화
            
        doc = st.session_state.lecture_doc
        total_pages = len(doc)
        
        # 2-Column 레이아웃 (좌: PDF 뷰어, 우: AI 분석)
        col_view, col_ai = st.columns([1, 1])
        
        with col_view:
            # 페이지 컨트롤러
            c1, c2, c3 = st.columns([1, 2, 1])
            if c1.button("◀ 이전"):
                if st.session_state.current_page > 0: st.session_state.current_page -= 1
            c2.markdown(f"<center>{st.session_state.current_page + 1} / {total_pages} 페이지</center>", unsafe_allow_html=True)
            if c3.button("다음 ▶"):
                if st.session_state.current_page < total_pages - 1: st.session_state.current_page += 1
            
            # 현재 페이지 이미지 렌더링
            page = doc.load_page(st.session_state.current_page)
            pix = page.get_pixmap(dpi=150)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            st.image(img, use_container_width=True)
            
            # 현재 페이지 텍스트 추출 (분석용)
            current_text = page.get_text()

        with col_ai:
            st.markdown("### 🤖 AI 연관 분석")
            
            if st.button("이 페이지 분석하기 ⚡"):
                if len(st.session_state.db) == 0:
                    st.warning("먼저 '족보 학습' 탭에서 데이터를 구축해주세요.")
                elif not current_text.strip():
                    st.warning("이 페이지에는 텍스트가 거의 없습니다. (이미지 위주)")
                else:
                    with st.spinner("족보와 연결고리를 찾는 중..."):
                        # 1. 유사한 족보 검색
                        related_items = find_relevant_jokbo(current_text, st.session_state.db)
                        
                        # 2. 프롬프트 구성
                        context_str = ""
                        for item in related_items:
                            context_str += f"- [출처: {item['content']['source']} {item['content']['page']}p] (유사도: {item['score']:.2f})\n내용: {item['content']['text'][:200]}...\n\n"
                        
                        prompt = f"""
                        당신은 의대생의 공부를 도와주는 AI 튜터입니다.
                        
                        [현재 강의 내용]:
                        {current_text}
                        
                        [관련된 족보(기출) 내용]:
                        {context_str}
                        
                        명령:
                        1. 현재 강의 내용이 과거 족보의 어떤 부분과 연결되는지 설명하세요.
                        2. 시험에 나올만한 핵심 키워드(Key Concept)를 3개 추출하세요.
                        3. 출제 경향을 바탕으로 간단한 OX 퀴즈를 하나 만드세요.
                        """
                        
                        # 3. 답변 생성
                        model = genai.GenerativeModel('gemini-1.5-flash')
                        response = model.generate_content(prompt)
                        
                        st.markdown(response.text)
                        
                        with st.expander("참고한 족보 원문 보기"):
                            st.text(context_str)
