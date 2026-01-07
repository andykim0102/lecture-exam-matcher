import streamlit as st
import pandas as pd
import os
import time
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_mic_recorder import mic_recorder

# =========================
# 1. 초기 설정 및 에러 방지 (세션 관리)
# =========================
st.set_page_config(page_title="의대생 실시간 족보 비서", layout="wide")

# NameError 방지를 위한 세션 상태 초기화
if 'pre_analysis' not in st.session_state: st.session_state.pre_analysis = []
if 'exam_db' not in st.session_state: st.session_state.exam_db = []
if 'vectorizer' not in st.session_state: st.session_state.vectorizer = None
if 'matrix' not in st.session_state: st.session_state.matrix = None

def get_pdf_text(file):
    reader = PdfReader(file)
    return [page.extract_text() or "" for page in reader.pages]

# =========================
# 2. 메인 UI 및 기능
# =========================
st.title("🩺 Med-Study OS: 실시간 족보 매칭 & 녹음")

tab1, tab2, tab3 = st.tabs(["📅 수업 전: 자동 정리", "🎙️ 수업 중: 실시간 녹음/알림", "🎯 수업 후: 단권화 리포트"])

# --- [Tab 1: 수업 전 사전 분석] ---
with tab1:
    st.header("강의실 가기 전: 족보 포인트 미리보기")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. 족보 PDF 등록")
        exam_files = st.file_uploader("과거 족보 파일들을 올려주세요", type="pdf", accept_multiple_files=True)
        if st.button("족보 데이터 분석 시작"):
            all_exams = []
            for f in exam_files:
                pages = get_pdf_text(f)
                for i, text in enumerate(pages):
                    if text.strip():
                        all_exams.append({"info": f"{f.name} (p.{i+1})", "text": text})
            
            if all_exams:
                st.session_state.exam_db = all_exams
                vec = TfidfVectorizer(ngram_range=(1, 2))
                st.session_state.matrix = vec.fit_transform([e['text'] for e in all_exams])
                st.session_state.vectorizer = vec
                st.success(f"{len(all_exams)}개의 족보 페이지 인덱싱 완료!")

    with col2:
        st.subheader("2. 오늘 강의록 매칭")
        lec_file = st.file_uploader("오늘 수업용 강의록 PDF", type="pdf")
        if lec_file and st.button("수업 전 자동 단권화 분석"):
            if st.session_state.vectorizer:
                lec_pages = get_pdf_text(lec_file)
                results = []
                for i, p_text in enumerate(lec_pages):
                    if not p_text.strip(): continue
                    qv = st.session_state.vectorizer.transform([p_text])
                    sims = cosine_similarity(qv, st.session_state.matrix).flatten()
                    if sims.max() > 0.2:
                        best_idx = sims.argmax()
                        results.append({
                            "page": i+1, 
                            "score": sims.max(), 
                            "exam_info": st.session_state.exam_db[best_idx]['info'],
                            "exam_text": st.session_state.exam_db[best_idx]['text']
                        })
                st.session_state.pre_analysis = results
                st.success(f"분석 완료! {len(results)}개 페이지에서 족보 적중 예상.")
            else:
                st.error("먼저 족보 데이터를 등록해주세요.")

# --- [Tab 2: 수업 중 실시간 매칭 & 녹음] ---
with tab2:
    st.header("🎧 실시간 강의 트래킹")
    if not st.session_state.pre_analysis:
        st.warning("수업 전 분석을 먼저 완료해야 실시간 매칭이 가능합니다.")
    else:
        st.info("교수님 설명을 녹음하면, 분석된 족보 데이터를 실시간으로 대조하여 알려줍니다.")
        
        # 실제 음성 녹음 도구 (이미지 에러 해결책)
        audio = mic_recorder(start_prompt="🔴 강의 녹음 시작", stop_prompt="⏹️ 중지 및 실시간 분석", key='recorder')
        
        if audio:
            st.audio(audio['bytes'])
            st.success("강의 녹음 완료 및 텍스트 변환 중... (Whisper 시뮬레이션)")
            
            # 데모용 시뮬레이션: 실제로는 녹음된 audio['bytes']를 Whisper API로 전송
            # 예시: 교수님이 심근경색(MI) 관련 족보 내용을 설명했다고 가정
            simulated_speech = "심근경색 환자가 응급실에 오면 가장 먼저 ST분절 상승 여부를 확인해야 합니다."
            st.subheader(f"🗣️ 교수님 발언 인식: \"{simulated_speech}\"")
            
            # 실시간 매칭 로직 (사전 분석 데이터 기반)
            hits = [item for item in st.session_state.pre_analysis if any(word in item['exam_text'] for word in simulated_speech.split()[:4])]
            
            if hits:
                for hit in hits:
                    st.toast(f"🚨 족보 적중 알림! 강의록 {hit['page']}p 관련", icon="🔥")
                    with st.warning():
                        st.markdown(f"### 🚩 실시간 기출 적중 (강의록 {hit['page']}페이지)")
                        st.write(f"**과거 출제 정보:** {hit['exam_info']}")
                        st.write(f"**과거 지문 내용:** {hit['exam_text'][:250]}...")
                        st.caption("💡 교수님이 방금 설명하신 내용은 과거 기출 지문의 핵심 키워드와 일치합니다.")
            else:
                st.info("현재 발언 중에는 일치하는 과거 기출 데이터가 없습니다.")

# --- [Tab 3: 수업 후 복습 리포트] ---
with tab3:
    st.header("🎯 오늘의 스마트 단권화 리포트")
    if st.session_state.pre_analysis:
        df = pd.DataFrame(st.session_state.pre_analysis)
        st.subheader("매칭 결과 요약")
        st.dataframe(df[['page', 'exam_info', 'score']])
        
        # Anki 카드 생성 기능
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 오늘 기출 기반 Anki 카드 다운로드", csv, "anki_cards.csv", "text/csv")
    else:
        st.write("표시할 분석 리포트가 없습니다.")
