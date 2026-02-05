🩺 Med-Study OS

Med-Study OS는 의대생을 위한 스마트 학습 어시스턴트입니다.
강의록(PDF)과 족보(기출문제) 데이터를 매칭하여, 공부해야 할 핵심 포인트를 AI가 집어줍니다.

✨ 주요 기능

📂 족보 관리 및 학습

과목별로 족보 PDF를 업로드하여 AI에게 학습시킵니다.

RAG(Retrieval-Augmented Generation) 기술을 사용하여 강의 내용과 관련된 족보를 정확히 찾아냅니다.

📖 실시간 강의 분석

공부할 강의록(PDF)을 띄우면, 현재 페이지와 관련된 족보 문항을 자동으로 보여줍니다.

AI 조교가 공부 방향성, 쌍둥이 문제(변형 문제), 해설을 실시간으로 생성합니다.

전체 페이지 미리 분석(Batch Analysis) 기능을 통해 로딩 없이 쾌적하게 공부할 수 있습니다.

🎙️ 강의 녹음/분석

강의 현장에서 바로 녹음하거나 녹음 파일을 업로드하면, 텍스트로 변환 후 족보 내용과 대조하여 요약해줍니다.

💬 AI 질의응답

강의 내용에 대해 궁금한 점을 질문하면, 족보 내용을 근거로 답변해줍니다.

🛠️ 기술 스택

Frontend/Backend: Streamlit

AI Model: Google Gemini 1.5 Flash (via google-generativeai)

PDF Processing: PyMuPDF (fitz)

Vector Search: scikit-learn (Cosine Similarity)

🚀 실행 방법

1. 환경 설정

파이썬(Python 3.9 이상)이 설치된 환경에서 아래 명령어로 필수 라이브러리를 설치합니다.

pip install -r requirements.txt


2. API Key 준비

Google AI Studio에서 Gemini API Key를 발급받습니다.

앱 실행 후 사이드바 설정 메뉴에 키를 입력합니다.

3. 앱 실행

streamlit run app.py


Note: 이 프로젝트는 학습용 프로토타입입니다. 중요 데이터는 백업해두시기 바랍니다.
