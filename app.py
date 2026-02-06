import streamlit as st
import json
import random
import time

# 실제 환경에서는 google.generativeai 또는 openai 등을 import하여 사용합니다.
# import google.generativeai as genai

st.set_page_config(layout="wide", page_title="AI 족보 분석기 (파싱 개선판)")

# --- [1] 핵심 로직: 뭉쳐있는 텍스트를 구조화하는 함수 (The FIX) ---
def parse_raw_jokbo(raw_text):
    """
    엉망으로 섞인 족보 텍스트를 입력받아
    [문제 지문, 보기, 정답, 해설]로 깔끔하게 구조화(JSON)합니다.
    """
    # 실제로는 여기서 LLM(Gemini/GPT) API를 호출해야 합니다.
    # 프롬프트 예시: 
    # "다음 텍스트에서 문제(question), 보기(choices), 정답(answer), 해설(explanation)을 추출하여 JSON으로 반환해. 
    # 텍스트가 섞여 있어도 문맥을 보고 분리해."
    
    # --- (시뮬레이션) AI가 파싱에 성공했다고 가정하고 정제된 데이터를 반환합니다 ---
    time.sleep(1.5) # AI 생각하는 시간
    
    # 입력된 텍스트에 따라 파싱 성공 여부 시뮬레이션
    if "DNA" in raw_text or "RNA" in raw_text:
        return {
            "success": True,
            "data": {
                "question": "DNA의 한 가닥(template strand)에 사이토신(C)이 20%가 있다. 이때 상보적인 가닥의 구아닌(G)의 비율은?",
                "type": "주관식/단답형",
                "choices": [],
                "answer": "20%",
                "explanation": "DNA의 상보결합 법칙(Chargaff's rule)에 따라, 주형 가닥의 C는 반대편 가닥의 G와 결합합니다. 따라서 비율은 동일하게 20%입니다."
            }
        }
    elif "uniform diameter" in raw_text:
        return {
            "success": True,
            "data": {
                "question": "Why does the DNA double helix have a uniform diameter?",
                "type": "객관식",
                "choices": ["purines pair with pyrimidines", "purines pair with purines", "sugar-phosphate backbone"],
                "answer": "1번 (purines pair with pyrimidines)",
                "explanation": "퓨린(2고리)과 피리미딘(1고리)이 결합해야 항상 일정한 폭(2nm)을 유지할 수 있습니다."
            }
        }
    else:
        # 파싱 실패 시
        return {"success": False, "error": "문제 구조를 인식할 수 없습니다."}

# --- [2] 쌍둥이 문제 생성 함수 ---
def generate_twin_problem(parsed_data):
    """
    구조화된(깔끔한) 데이터를 바탕으로 쌍둥이 문제를 만듭니다.
    """
    if not parsed_data.get("success"):
        return "원본 문제를 파싱하지 못해 변형 문제를 만들 수 없습니다."
    
    origin = parsed_data["data"]
    
    # --- (시뮬레이션) AI가 변형 문제를 생성 ---
    time.sleep(1.5)
    
    if "DNA" in origin["question"]:
        return """
        **[생성된 쌍둥이 문제]**
        Q. DNA 이중 나선에서 한 가닥의 아데닌(A) 함량이 30%일 때, 반대편 가닥의 티민(T) 함량은 얼마인가?
        
        1. 20%
        2. 30%
        3. 50%
        4. 70%
        
        **정답:** 2번
        **해설:** A와 T는 상보적으로 결합하므로 함량이 같습니다.
        """
    else:
        return "변형 문제 생성 완료 (내용 생략)"

# --- UI 구성 ---

st.title("📑 AI 강의록 분석 & 족보 매칭 시스템")
st.caption("파싱 오류 해결 버전: Raw Text -> LLM 구조화 -> 문제 생성")

col1, col2 = st.columns([1, 1])

# [왼쪽] PDF 뷰어 시뮬레이션
with col1:
    st.info("강의 PDF 파일 업로드 / 변경")
    st.markdown("### Page 13 / 44")
    
    # 이미지 속 내용 시뮬레이션
    st.markdown("""
    <div style="background-color: #f9f9f9; padding: 20px; border-radius: 10px; border: 1px solid #ddd;">
        <h3 style="color: #2c3e50;">Codons: Triplets of Nucleotides</h3>
        <ul>
            <li>During transcription, one of the two DNA strands, called the <b>template strand</b>...</li>
            <li>The template strand is always the same strand for any given gene...</li>
            <li>Each codon specifies the amino acid (one of 20)...</li>
        </ul>
        <br><br><br>
    </div>
    """, unsafe_allow_html=True)

# [오른쪽] 분석 결과 및 족보 매칭
with col2:
    tab1, tab2 = st.tabs(["📘 족보 분석", "💬 질의응답"])
    
    with tab1:
        st.subheader("🔥 관련 족보 문항")
        
        # 시나리오: DB에서 검색된 뭉쳐있는 텍스트 (사용자가 겪은 상황)
        raw_jokbo_text_1 = """
        24. DNA의 a strand(1개의 가닥)에 사이토신(C)이 20%가 있다. 
        이때 구아닌(G)의 비율은? (주관식) 정답: 알 수 없다 (왜냐하면 문제에서... 상보결합을 생각하면 안 됨 - 오답노트)
        """
        
        raw_jokbo_text_2 = """
        25. Why does the DNA double helix have a uniform diameter? (객관식, 정답 1번) (1) purines pair with pyrimidines (2) C...
        """

        # --- 문항 카드 1 ---
        with st.container(border=True):
            st.caption("출처: 누렁소_생물학2_2025 2학기 기말 족보.PDF (유사도 0.71)")
            
            # 1. 원본 텍스트 보여주기 (디버깅용, 실제론 숨겨도 됨)
            with st.expander("원본 텍스트 보기 (Raw Data)"):
                st.text(raw_jokbo_text_1)

            # 2. 파싱 및 쌍둥이 문제 생성 로직
            # 사용자가 '쌍둥이 문제 만들기'를 클릭하면 파싱을 시도함
            with st.expander("✨ 쌍둥이 문제 만들기", expanded=True):
                # (A) 파싱 단계 (Parsing Stage)
                with st.spinner("AI가 섞여있는 텍스트를 구조화하는 중..."):
                    parsed_result = parse_raw_jokbo(raw_jokbo_text_1)
                
                if parsed_result["success"]:
                    data = parsed_result["data"]
                    
                    # (B) 파싱 성공 시 구조화된 내용 표시 (사용자 확인용)
                    st.success("✅ 자동 파싱 성공!")
                    st.markdown(f"**질문:** {data['question']}")
                    st.markdown(f"**정답:** {data['answer']}")
                    
                    st.divider()
                    
                    # (C) 쌍둥이 문제 생성 요청
                    if st.button("변형 문제 생성하기", key="btn1"):
                        with st.spinner("변형 문제 생성 중..."):
                            twin_prob = generate_twin_problem(parsed_result)
                            st.markdown(twin_prob)
                else:
                    st.error("❌ 자동 파싱 실패 - 상단 내용을 참고하세요")
                    st.warning("텍스트가 너무 손상되어 문제를 추출할 수 없습니다.")

            with st.expander("✅ 해설 및 정답"):
                if parsed_result["success"]:
                    st.write(parsed_result["data"]["explanation"])
                else:
                    st.write("내용을 불러올 수 없습니다.")

        # --- 문항 카드 2 ---
        with st.container(border=True):
            st.caption("출처: 누렁소_생물학2_2025 2학기 기말 족보.PDF (유사도 0.68)")
            st.text("25. Why does the DNA double helix have a uniform...")
            
            with st.expander("✨ 쌍둥이 문제 만들기"):
                st.info("이 문제를 클릭하여 생성 시작")
