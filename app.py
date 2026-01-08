import streamlit as st
import pandas as pd
import re
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_mic_recorder import mic_recorder
from streamlit_pdf_viewer import pdf_viewer # 전용 뷰어 라이브러리

# =========================
# 1. 초기 설정 및 세션 관리
# =========================
st.set_page_config(page_title="Med-Study AI", layout="wide")

# 세션 초기화 (코드 실행 중 데이터 유실 방지)
if 'notebook' not in st.session_state: st.session_state.notebook = []
if 'pre_analysis' not in st.session_state: st.session_state.pre_analysis = []
if 'pdf_bytes' not in st.session_state: st.session_state.pdf_bytes = None
if 'exam_db' not in st.session_state: st.session_state.exam_db = []

def get_pdf_text(file):
    reader = PdfReader(file)
    return [page.extract_text() or "" for page in reader.pages]

# =========================
# 2. 메인 UI 구성
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
                    # 문단 단위로 쪼개어 가독성 및 매칭률 향상
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
            st.session_state.pdf_bytes = lec_file.getvalue() # 바이너리 데이터 저장
            if st.button("AI 사전 분석 시작"):
                if not st.session_state.exam_db:
                    st.error("먼저 족보를 등록하세요.")
                else:
                    lec_texts = get_pdf_text(lec_file)
                    # 의미론적 매칭을 위한 TF-IDF 설정 강화
                    vec = TfidfVectorizer(ngram_range=(1, 3), min_df=1)
                    exam_texts = [e['content'] for e in st.session_db] if 'session_db' in globals() else [e['content'] for e in st.session_state.exam_db]
                    exam_matrix = vec.fit_transform(exam_texts)
                    
                    results = []
                    for i, p_text in enumerate(lec_texts):
                        if not p_text.strip(): continue
                        qv = vec.transform([p_text])
                        sims = cosine_similarity(qv, exam_matrix).flatten()
                        if sims.max() > 0.15: # 적중 역치
                            idx = sims.argmax()
                            results.append({
                                "page": i+1, "score": sims.max(),
                                "info": st.session_state.exam_db[idx]['source'],
                                "content": st.session_state.exam_db[idx]['content']
                            })
                    st.session_state.pre_analysis = results
                    st.success(f"분석 완료! {len(results)}개 페이지 적중.")

# --- [Tab 2: 수업 중 뷰어 & 원클릭 저장] ---
with tab2:
    if st.session_state.pdf_bytes is None:
        st.warning("강의록 PDF를 먼저 업로드하고 분석해주세요.")
    else:
        c1, c2 = st.columns([1.2, 0.8])
        
        with c1:
            st.subheader("📄 강의록 실시간 뷰어")
            # PDF 페이지 슬라이더
            page_num = st.number_input("페이지 선택", min_value=1, max_value=200, value=1)
            
            # [해결] 까만 화면 방지를 위한 전용 라이브러리 호출
            pdf_viewer(st.session_state.pdf_bytes, width=700, pages_to_render=[page_num])
        
        with c2:
            st.subheader("⚡ 실시간 족보 매칭")
            # 실시간 녹음 기능
            audio = mic_recorder(start_prompt="🎙️ 교수님 설명 분석", stop_prompt="⏹️ 분석 중지", key='live_mic')
            
            # 현재 페이지 기반 족보 알림
            page_hits = [h for h in st.session_state.pre_analysis if h['page'] == page_num]
            if page_hits:
                for h in page_hits:
                    with st.container(border=True):
                        st.error(f"🔥 기출 적중 ({int(h['score']*100)}% 일치)")
                        st.markdown(f"**📍 출처:** {h['info']}")
                        st.info(f"**📚 관련 원문:**\n{h['content']}")
                        
                        # 사용자 메모 및 저장
                        user_note = st.text_input("수업 내용 메모", key=f"note_{page_num}")
                        if st.button("📌 나만의 정리본에 추가", key=f"btn_{page_num}"):
                            st.session_state.notebook.append({
                                "page": page_num, "info": h['info'], 
                                "content": h['content'], "note": user_note
                            })
                            st.toast("정리본 탭에 저장되었습니다!")
            else:
                st.info("이 페이지는 관련 족보가 없습니다.")

# --- [Tab 3: 정리본 리포트] ---
with tab3:
    st.header("📝 나만의 수업 요약본")
    if not st.session_state.notebook:
        st.info("수업 중 저장한 내용이 없습니다.")
    else:
        for i, item in enumerate(st.session_state.notebook):
            with st.expander(f"📔 [강의록 {item['page']}p] {item['info']}", expanded=True):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**📂 족보 원문**")
                    st.write(item['content'])
                with col_b:
                    st.markdown("**✏️ 수업 메모**")
                    st.success(item['note'] if item['note'] else "추가 메모 없음")
                
                if st.button("삭제", key=f"del_{i}"):
                    st.session_state.notebook.pop(i)
                    st.rerun()
