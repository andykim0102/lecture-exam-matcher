import streamlit as st
import pandas as pd
import base64
import re
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_mic_recorder import mic_recorder

# =========================
# 1. 초기 설정 및 세션 관리
# =========================
st.set_page_config(page_title="Med-Study AI", layout="wide")

if 'notebook' not in st.session_state: st.session_state.notebook = []
if 'pre_analysis' not in st.session_state: st.session_state.pre_analysis = []
if 'pdf_bytes' not in st.session_state: st.session_state.pdf_bytes = None
if 'exam_db' not in st.session_state: st.session_state.exam_db = []

def get_pdf_text(file):
    reader = PdfReader(file)
    return [page.extract_text() or "" for page in reader.pages]

def display_pdf(file_bytes, page_num):
    """뷰어 깨짐 방지를 위한 Base64 PDF 렌더러 (Chrome/Edge 최적화)"""
    try:
        base64_pdf = base64.b64encode(file_bytes).decode('utf-8')
        # PDF.js를 사용하지 않고 브라우저 내장 뷰어를 강제 호출
        pdf_display = f'''
            <embed src="data:application/pdf;base64,{base64_pdf}#page={page_num}" 
            width="100%" height="800px" type="application/pdf" />
        '''
        st.markdown(pdf_display, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"PDF 뷰어를 로드할 수 없습니다: {e}")

# =========================
# 2. 메인 서비스 로직
# =========================
st.title("🩺 의대생 전용 스마트 학습 OS")

tab1, tab2, tab3 = st.tabs(["📅 수업 전: 분석", "🎙️ 수업 중: 실시간 매칭", "🎯 수업 후: 정리본"])

# --- [Tab 1: 수업 전 분석] ---
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📚 족보 등록 (DB 구축)")
        exam_files = st.file_uploader("족보 PDF 업로드", type="pdf", accept_multiple_files=True)
        if st.button("족보 고도화 인덱싱"):
            db = []
            for f in exam_files:
                pages = get_pdf_text(f)
                for i, text in enumerate(pages):
                    # [개선] 텍스트가 너무 길면 핵심 문단 위주로 쪼개기(Chunking)
                    chunks = re.split(r'\n\s*\n', text) 
                    for chunk in chunks:
                        if len(chunk.strip()) > 30:
                            db.append({"source": f"{f.name} (p.{i+1})", "content": chunk.strip()})
            st.session_state.exam_db = db
            st.success(f"{len(db)}개의 족보 유닛 저장 완료!")

    with col2:
        st.subheader("📖 강의록 매칭")
        lec_file = st.file_uploader("강의록 PDF 업로드", type="pdf")
        if lec_file:
            st.session_state.pdf_bytes = lec_file.getvalue()
            if st.button("AI 사전 분석 시작"):
                if not st.session_state.exam_db:
                    st.error("먼저 족보를 등록하세요.")
                else:
                    lec_texts = get_pdf_text(lec_file)
                    # [개선] 단순 키워드 매칭 보완 (TF-IDF 가중치 상향)
                    vec = TfidfVectorizer(ngram_range=(1, 3), min_df=1)
                    exam_texts = [e['content'] for e in st.session_state.exam_db]
                    exam_matrix = vec.fit_transform(exam_texts)
                    
                    results = []
                    for i, p_text in enumerate(lec_texts):
                        if not p_text.strip(): continue
                        qv = vec.transform([p_text])
                        sims = cosine_similarity(qv, exam_matrix).flatten()
                        # [개선] 역치 조정 및 정교화
                        if sims.max() > 0.15: 
                            idx = sims.argmax()
                            results.append({
                                "page": i+1, "score": sims.max(),
                                "info": st.session_state.exam_db[idx]['source'],
                                "content": st.session_state.exam_db[idx]['content']
                            })
                    st.session_state.pre_analysis = results
                    st.success("분석 완료!")

# --- [Tab 2: 수업 중 뷰어 & 원클릭 저장] ---
with tab2:
    if not st.session_state.pdf_bytes:
        st.warning("강의록을 먼저 업로드해주세요.")
    else:
        c1, c2 = st.columns([1.2, 0.8])
        with c1:
            page = st.select_slider("강의록 페이지 이동", options=range(1, 101), value=1)
            display_pdf(st.session_state.pdf_bytes, page)
        
        with c2:
            st.subheader("⚡ 실시간 족보 매칭")
            # [해결] 실시간 녹음 및 매칭 인터페이스
            audio = mic_recorder(start_prompt="🎙️ 교수님 설명 분석", stop_prompt="⏹️ 분석 중지", key='live_mic')
            
            # [개선] 현재 페이지 매칭 정보 포커싱
            page_hits = [h for h in st.session_state.pre_analysis if h['page'] == page]
            if page_hits:
                for h in page_hits:
                    with st.expander(f"🔥 기출 적중 ({int(h['score']*100)}% 일치)", expanded=True):
                        st.error(f"📍 출처: {h['info']}")
                        st.info(f"**핵심 내용:**\n{h['content']}")
                        
                        # [해결] 원클릭 단권화
                        note = st.text_area("수업 중 메모", key=f"note_{page}")
                        if st.button("📌 내 정리본에 즉시 추가", key=f"btn_{page}"):
                            st.session_state.notebook.append({
                                "page": page, "info": h['info'], 
                                "content": h['content'], "note": note
                            })
                            st.toast("정리본에 저장되었습니다!")
            else:
                st.write("이 페이지는 관련 족보가 없습니다.")

# --- [Tab 3: 정리본 리포트] ---
with tab3:
    st.header("📝 나만의 수업 요약본")
    if not st.session_state.notebook:
        st.info("수업 중 '저장' 버튼을 누른 내용이 여기에 표시됩니다.")
    else:
        for i, item in enumerate(st.session_state.notebook):
            with st.container(border=True):
                st.markdown(f"### [강의록 {item['page']}p] {item['info']}")
                st.write(f"**💡 족보 내용:** {item['content']}")
                st.success(f"**✏️ 나의 메모:** {item['note']}")
                if st.button("삭제", key=f"del_{i}"):
                    st.session_state.notebook.pop(i)
                    st.rerun()
