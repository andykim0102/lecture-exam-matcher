import streamlit as st
import pandas as pd
import re
import google.generativeai as genai
from openai import OpenAI
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_mic_recorder import mic_recorder
from streamlit_pdf_viewer import pdf_viewer

# =========================
# 1. 초기 설정 및 UI 최적화
# =========================
st.set_page_config(page_title="Med-Study AI Alpha", layout="wide")

# [해결] 뷰어 하단 잘림 방지 CSS
st.markdown("""
    <style>
    .stMainBlockContainer { padding-top: 2rem; }
    iframe { min-height: 850px !important; border-radius: 12px; }
    .stAlert { border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# 세션 상태 초기화
for key in ['notebook', 'pre_analysis', 'pdf_bytes', 'exam_db']:
    if key not in st.session_state:
        st.session_state[key] = [] if key != 'pdf_bytes' else None

# =========================
# 2. AI 요약 엔진 (에러 복구 로직 포함)
# =========================
def get_ai_summary(text, api_key, provider="Gemini"):
    if not api_key: 
        return "🔑 사이드바에 API 키를 입력하면 AI 요약이 활성화됩니다."
    
    prompt = f"당신은 의대 전문 튜터입니다. 다음 기출 지문을 핵심 기전이나 암기 포인트 위주로 3줄 요약하세요:\n\n{text}"
    
    try:
        if provider == "Gemini":
            genai.configure(api_key=api_key)
            # [해결] 404 에러 방지용 모델명 자동 순회 로직
            for model_name in ['gemini-1.5-flash', 'gemini-1.5-flash-latest', 'gemini-pro']:
                try:
                    model = genai.GenerativeModel(model_name)
                    response = model.generate_content(prompt)
                    return response.text
                except:
                    continue
            return "❌ 모델을 찾을 수 없습니다(404). 새 API 키를 발급받아보세요."
            
        elif provider == "ChatGPT":
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini", 
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
            
    except Exception as e:
        if "429" in str(e):
            return "⚠️ 할당량 초과(429). 1분 뒤 다시 시도하거나 다른 API 키를 써보세요."
        return f"🚨 요약 실패: {str(e)}"

# =========================
# 3. 사이드바 및 레이아웃
# =========================
with st.sidebar:
    st.header("⚙️ AI 모델 설정")
    ai_provider = st.selectbox("LLM 엔진 선택", ["Gemini", "ChatGPT"])
    api_key = st.text_input(f"{ai_provider} API Key", type="password")
    st.caption("Gemini 사용 시 'Create API key in new project'로 발급된 키를 추천합니다.")

st.title("🩺 의대생 전용 스마트 학습 OS")
tab1, tab2, tab3 = st.tabs(["📅 1. 사전 분석", "🎙️ 2. 실시간 매칭", "🎯 3. 나만의 정리본"])

# --- [Tab 1: 사전 분석 및 DB화] ---
with tab1:
    col_ex, col_lec = st.columns(2)
    with col_ex:
        st.subheader("📚 족보 아카이브 등록")
        exam_files = st.file_uploader("족보 PDF들을 업로드하세요", type="pdf", accept_multiple_files=True)
        if st.button("족보 데이터 고도화 인덱싱"):
            db = []
            for f in exam_files:
                reader = PdfReader(f)
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text:
                        db.append({"source": f"{f.name} (p.{i+1})", "content": text.strip()})
            st.session_state.exam_db = db
            st.success(f"{len(db)}개의 족보 유닛이 등록되었습니다.")

    with col_lec:
        st.subheader("📖 오늘 강의록 분석")
        lec_file = st.file_uploader("강의록 PDF 업로드", type="pdf")
        if lec_file:
            st.session_state.pdf_bytes = lec_file.getvalue()
            if st.button("AI 매칭 가동"):
                if not st.session_state.exam_db:
                    st.error("먼저 족보 데이터를 등록해주세요.")
                else:
                    reader = PdfReader(lec_file)
                    lec_texts = [p.extract_text() for p in reader.pages]
                    vec = TfidfVectorizer(ngram_range=(1, 2))
                    exam_matrix = vec.fit_transform([e['content'] for e in st.session_state.exam_db])
                    results = []
                    for i, p_text in enumerate(lec_texts):
                        if not p_text: continue
                        qv = vec.transform([p_text])
                        sims = cosine_similarity(qv, exam_matrix).flatten()
                        if sims.max() > 0.18:
                            idx = sims.argmax()
                            results.append({
                                "page": i+1, "score": sims.max(),
                                "info": st.session_state.exam_db[idx]['source'],
                                "content": st.session_state.exam_db[idx]['content']
                            })
                    st.session_state.pre_analysis = results
                    st.success(f"분석 완료! {len(results)}개 페이지에서 기출 포인트를 찾았습니다.")
    # --- [Tab 2: 실시간 매칭 및 AI 요약 인터페이스] ---
with tab2:
    if st.session_state.pdf_bytes:
        # 화면을 1.1:0.9 비율로 분할 (뷰어 : 정보창)
        c1, c2 = st.columns([1.1, 0.9])
        
        with c1:
            st.subheader("📄 강의록 실시간 뷰어")
            # 페이지 입력 및 PDF 렌더링
            page_num = st.number_input("현재 강의록 페이지", min_value=1, value=1, step=1)
            
            # [해결] height=850으로 설정하여 하단 잘림 방지
            pdf_viewer(st.session_state.pdf_bytes, 
                       pages_to_render=[page_num], 
                       width=800, 
                       height=850)
        
        with c2:
            st.subheader("⚡ 실시간 족보 매칭 & AI")
            # 현재 페이지와 매칭되는 분석 결과 추출
            page_hits = [h for h in st.session_state.pre_analysis if h['page'] == page_num]
            
            if page_hits:
                for h in page_hits:
                    with st.container(border=True):
                        st.error(f"🔥 기출 적중 ({int(h['score']*100)}% 일치)")
                        st.markdown(f"**📍 관련 족보:** {h['info']}")
                        
                        st.markdown("---")
                        st.markdown("**🤖 AI 족보 브리핑**")
                        
                        # [해결] 과다 호출 및 에러 방지를 위한 '요약 버튼' 방식
                        summary_key = f"sum_cache_{page_num}"
                        if st.button("🪄 AI에게 요약 요청하기", key=f"btn_sum_{page_num}"):
                            with st.spinner("AI 분석 중..."):
                                summary_text = get_ai_summary(h['content'], api_key, ai_provider)
                                st.session_state[summary_key] = summary_text
                        
                        # 요약 결과 출력 (캐시된 내용이 있으면 바로 표시)
                        if summary_key in st.session_state:
                            st.info(st.session_state[summary_key])
                        else:
                            st.caption("위 버튼을 누르면 방대한 족보 지문을 3줄로 요약합니다.")
                        
                        # 원문은 접이식 메뉴로 숨겨서 가독성 확보
                        with st.expander("📄 족보 원문 전체 보기"):
                            st.write(h['content'])
                        
                        st.markdown("---")
                        # 개인 메모장 기능
                        user_note = st.text_area("💡 수업 내용 추가 메모", key=f"note_{page_num}")
                        
                        if st.button("📌 나만의 정리본에 저장", key=f"save_btn_{page_num}"):
                            # 요약이 안 된 상태로 저장할 경우 대비
                            final_sum = st.session_state.get(summary_key, "요약 내용 없음")
                            st.session_state.notebook.append({
                                "page": page_num,
                                "info": h['info'],
                                "summary": final_sum,
                                "note": user_note
                            })
                            st.toast("정리본 탭에 저장되었습니다!")
            else:
                st.info("이 페이지와 관련된 기출 포인트가 없습니다.")
    else:
        st.warning("먼저 '1. 사전 분석' 탭에서 강의록 PDF를 업로드해 주세요.")

# --- [Tab 3: 나만의 스마트 정리본 (단권화)] ---
with tab3:
    st.header("📝 오늘의 스마트 단권화 리포트")
    
    if not st.session_state.notebook:
        st.info("수업 중 '정리본에 저장' 버튼을 누른 항목들이 여기에 모입니다.")
    else:
        st.write(f"현재 총 {len(st.session_state.notebook)}개의 핵심 포인트가 저장되었습니다.")
        
        # 저장된 항목들을 역순(최신순)으로 표시
        for i, item in enumerate(reversed(st.session_state.notebook)):
            with st.expander(f"📔 [강의록 {item['page']}p] 관련 족보: {item['info']}", expanded=True):
                col_a, col_b = st.columns([1, 1])
                with col_a:
                    st.markdown("**🤖 AI 요약 핵심**")
                    st.info(item['summary'])
                with col_b:
                    st.markdown("**✏️ 나의 필기**")
                    st.success(item['note'] if item['note'] else "기록된 메모가 없습니다.")
                
                # 삭제 기능
                if st.button("🗑️ 삭제", key=f"del_{i}"):
                    # 인덱스 계산 주의 (reversed 사용 중이므로)
                    real_idx = len(st.session_state.notebook) - 1 - i
                    st.session_state.notebook.pop(real_idx)
                    st.rerun()

    # 정리본 전체 초기화 버튼
    if st.session_state.notebook:
        st.markdown("---")
        if st.button("⚠️ 모든 정리본 데이터 초기화"):
            st.session_state.notebook = []
            st.rerun()
