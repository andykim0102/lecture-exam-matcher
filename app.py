import streamlit as st
import pandas as pd
import base64
import os
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_mic_recorder import mic_recorder

# =========================
# 1. 초기 설정 및 세션 관리 (에러 방지 및 데이터 유지)
# =========================
st.set_page_config(page_title="Med-Study OS Alpha", layout="wide")

# 세션 상태 초기화
if 'pre_analysis' not in st.session_state: st.session_state.pre_analysis = []
if 'exam_db' not in st.session_state: st.session_state.exam_db = []
if 'vectorizer' not in st.session_state: st.session_state.vectorizer = None
if 'matrix' not in st.session_state: st.session_state.matrix = None
if 'pdf_bytes' not in st.session_state: st.session_state.pdf_bytes = None
if 'notebook' not in st.session_state: st.session_state.notebook = [] # 단권화 바구니

def get_pdf_text_by_page(file):
    reader = PdfReader(file)
    return [page.extract_text() or "" for page in reader.pages]

def display_pdf(file_bytes, page_num):
    base64_pdf = base64.b64encode(file_bytes).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}#page={page_num}" width="100%" height="800" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def get_match_label(score):
    """소수점 점수를 직관적인 지표로 변환"""
    percent = int(score * 100)
    if score > 0.4: return f"🔥 {percent}% (족보 적중 확실)"
    if score > 0.25: return f"✅ {percent}% (연관성 높음)"
    return f"⚠️ {percent}% (확인 권장)"

# =========================
# 2. 메인 서비스 레이아웃
# =========================
st.title("🩺 Med-Study OS: 통합 단권화 솔루션")

tab1, tab2, tab3 = st.tabs(["📂 1. 수업 전 (사전 분석)", "🎙️ 2. 수업 중 (뷰어 & 실시간)", "🎯 3. 수업 후 (나만의 정리본)"])

# --- [Tab 1: 수업 전 사전 분석] ---
with tab1:
    st.header("강의 전 데이터 준비")
    col_ex, col_lec = st.columns(2)
    
    with col_ex:
        st.subheader("📚 족보 아카이브 등록")
        exam_files = st.file_uploader("족보 PDF들을 업로드하세요", type="pdf", accept_multiple_files=True)
        if st.button("족보 데이터 고도화 인덱싱"):
            all_exams = []
            for f in exam_files:
                pages = get_pdf_text_by_page(f)
                # 텍스트를 문단 단위로 쪼개어(Chunking) 정확도 향상 시도 가능
                for i, text in enumerate(pages):
                    if len(text.strip()) > 20: # 의미 있는 텍스트만 추출
                        all_exams.append({"source": f"{f.name} (p.{i+1})", "content": text})
            
            if all_exams:
                st.session_state.exam_db = all_exams
                vec = TfidfVectorizer(ngram_range=(1, 2), stop_words=None) # 의학 용어 보존을 위해 stop_words 미사용
                st.session_state.matrix = vec.fit_transform([e['content'] for e in all_exams])
                st.session_state.vectorizer = vec
                st.success("족보 데이터베이스 최적화 완료!")

    with col_lec:
        st.subheader("📖 오늘 강의록 분석")
        lec_file = st.file_uploader("오늘 수업용 강의록 PDF", type="pdf")
        if lec_file:
            st.session_state.pdf_bytes = lec_file.getvalue()
            if st.button("AI 사전 매칭 실행"):
                if st.session_state.vectorizer:
                    lec_pages = get_pdf_text_by_page(lec_file)
                    results = []
                    for i, p_text in enumerate(lec_pages):
                        if not p_text.strip(): continue
                        qv = st.session_state.vectorizer.transform([p_text])
                        sims = cosine_similarity(qv, st.session_state.matrix).flatten()
                        if sims.max() > 0.22: # 역치(Threshold) 조정 가능
                            best_idx = sims.argmax()
                            results.append({
                                "page": i+1, 
                                "score": sims.max(), 
                                "exam_info": st.session_state.exam_db[best_idx]['source'],
                                "exam_text": st.session_state.exam_db[best_idx]['content']
                            })
                    st.session_state.pre_analysis = results
                    st.success(f"분석 완료! 총 {len(results)}개의 핵심 기출 페이지를 찾았습니다.")
                else:
                    st.error("먼저 족보 데이터를 등록해주세요.")

# --- [Tab 2: 수업 중 시각적 뷰어 & 실시간 단권화] ---
with tab2:
    if st.session_state.pdf_bytes is None:
        st.warning("먼저 강의록을 업로드하고 분석해주세요.")
    else:
        # 좌측: PDF 뷰어 / 우측: 실시간 알림 및 간편 노트
        col_pdf, col_tool = st.columns([1.2, 0.8])
        
        with col_pdf:
            st.subheader("📄 강의록 실시간 뷰어")
            current_page = st.select_slider("페이지 이동", options=range(1, 101), value=1)
            display_pdf(st.session_state.pdf_bytes, current_page)

        with col_tool:
            st.subheader("⚡ 실시간 어시스턴트")
            
            # 실시간 녹음 및 분석부
            audio = mic_recorder(start_prompt="🎤 교수님 설명 분석 시작", stop_prompt="⏹️ 중지 및 매칭", key='live_mic')
            if audio:
                st.audio(audio['bytes'])
                # 데모 시나리오용 시뮬레이션 발언
                speech_text = "이 질환의 진단 기준은 작년 국시에도 나왔고 아주 핵심적인 내용입니다."
                st.info(f"🗣️ 인식된 강의 내용: \"{speech_text}\"")
                
                # 실시간 매칭 알림 (전체 DB 대상)
                if st.session_state.vectorizer:
                    qv_live = st.session_state.vectorizer.transform([speech_text])
                    sims_live = cosine_similarity(qv_live, st.session_state.matrix).flatten()
                    if sims_live.max() > 0.18:
                        hit = sims_live.argmax()
                        st.toast("🚨 족보 매칭 발견!", icon="🔥")
                        with st.status("🔥 실시간 족보 매칭 성공!", expanded=True):
                            st.write(f"**관련 족보:** {st.session_state.exam_db[hit]['source']}")
                            st.write(f"**기출 지문:** {st.session_state.exam_db[hit]['content'][:300]}...")
            
            st.divider()
            
            # 현재 페이지 기반 사전 정보 표시 및 간편 추가
            st.subheader(f"📍 {current_page}p 기출 포인트")
            page_hits = [h for h in st.session_state.pre_analysis if h['page'] == current_page]
            
            if page_hits:
                for h in page_hits:
                    with st.container(border=True):
                        st.markdown(f"**적중률:** {get_match_label(h['score'])}")
                        st.markdown(f"**출처:** {h['exam_info']}")
                        # 핵심: 족보 원문 포커싱 (문단 추출은 향후 정규표현식으로 고도화 가능)
                        st.caption(f"내용 요약: {h['exam_text'][:350]}...")
                        
                        # [해결 4] 사용자 입력 추가 기능
                        user_note = st.text_input("메모 추가", placeholder="교수님이 이 부분에서 강조하신 말씀은?", key=f"note_{h['page']}")
                        
                        if st.button("📌 이 페이지 단권화 저장", key=f"btn_{h['page']}"):
                            st.session_state.notebook.append({
                                "page": h['page'],
                                "exam_info": h['exam_info'],
                                "exam_text": h['exam_text'],
                                "user_note": user_note
                            })
                            st.toast("나만의 정리본에 추가되었습니다!")
            else:
                st.info("이 페이지와 관련된 기출 내역이 없습니다.")

# --- [Tab 3: 수업 후 나만의 정리본] ---
with tab3:
    st.header("📝 나만의 스마트 정리본 (단권화 완료)")
    
    if st.session_state.notebook:
        st.write(f"총 {len(st.session_state.notebook)}개의 핵심 포인트가 정리되었습니다.")
        
        for i, item in enumerate(st.session_state.notebook):
            with st.expander(f"📔 [강의록 {item['page']}p] {item['exam_info']} 관련 정리", expanded=True):
                col_res1, col_res2 = st.columns(2)
                with col_res1:
                    st.markdown("**📂 관련 족보 원문**")
                    st.info(item['exam_text'])
                with col_res2:
                    st.markdown("**✏️ 수업 중 나의 메모**")
                    st.success(item['user_note'] if item['user_note'] else "추가 메모 없음")
                
                if st.button("삭제", key=f"del_{i}"):
                    st.session_state.notebook.pop(i)
                    st.rerun()
        
        # 파일 내보내기
        final_df = pd.DataFrame(st.session_state.notebook)
        csv_data = final_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 최종 정리본 CSV 다운로드 (Anki 호환 가능)", csv_data, "med_summary.csv", "text/csv")
    else:
        st.info("수업 중에 '단권화 저장' 버튼을 누른 내용들이 여기에 모입니다.")
