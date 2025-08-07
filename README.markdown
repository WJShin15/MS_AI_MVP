# MS AI 학습 챗봇

## 개요
MS AI 학습 챗봇은 MS AI 역량강화 과정에서 1일차부터 9일차까지 학습한 Azure OpenAI, RAG(Azure AI Search 활용), 멀티모달 RAG, 파인튜닝,LangChain/LangGraph, Azure 배포 등의 내용을 바탕으로 마이크로소프트 AI 교육을 지원하기 위해 설계된 지능형 대화형 어시스턴트입니다. Azure AI Search를 활용한 RAG(Retrieval-Augmented Generation), Tavily를 통한 실시간 웹 검색, 그리고 Azure OpenAI를 이용한 자연어 처리를 결합하여 커리큘럼, 일정, 진행 상황 관련 질문에 대해 정확하고 구조화된 답변을 제공합니다.

## 주요 기능
- **벡터 검색**: Azure AI Search와 벡터 임베딩을 사용해 미리 인덱싱된 문서에서 관련 정보를 검색합니다.
- **웹 검색**: Tavily API를 통해 실시간 웹 검색 결과를 통합하여 최신 정보를 제공합니다.
- **비동기 처리**: 검색 결과를 동시에 요약하여 응답 시간을 단축합니다.
- **커리큘럼 특화 검색**: 커리큘럼, 일정, 진행 상황 관련 질문에 대해 타겟팅된 벡터 검색을 수행합니다.
- **구조화된 답변**: 개요, 세부 정보, 권장 사항 등의 섹션으로 구성된 마크다운 형식의 명확한 답변을 생성합니다.
- **사용자 세션 관리**: Streamlit의 세션 상태를 활용해 사용자 정보와 대화 기록을 저장합니다.

## 요구 사항
- Python 3.8 이상
- 필요한 Python 패키지:
  - `openai`
  - `azure-search-documents`
  - `azure-core`
  - `langchain-community`
  - `streamlit`
  - `python-dotenv`
- 환경 변수 (`.env` 파일에 설정):
  - `AZURE_AI_SEARCH_ENDPOINT`: Azure AI Search 서비스 엔드포인트
  - `AZURE_AI_SEARCH_API_KEY`: Azure AI Search API 키
  - `AZURE_AI_SEARCH_INDEX_NAME`: 일반 문서 검색용 인덱스 이름 (기본값: `rag-msai-learn03`)
  - `AZURE_AI_CURRICULUM_INDEX_NAME`: 커리큘럼 검색용 인덱스 이름 (기본값: `rag-curriculum`)
  - `AZURE_OPENAI_ENDPOINT`: Azure OpenAI 서비스 엔드포인트
  - `AZURE_OPENAI_KEY`: Azure OpenAI API 키
  - `TAVILY_API_KEY`: Tavily 웹 검색 API 키
  - `OPENAI_EMBEDDING_DEPLOYMENT_NAME`: 임베딩 모델 배포 이름
  - `AZURE_OPENAI_DEPLOYMENT_NAME`: 대화 모델 배포 이름 (기본값: `dev-gpt-4.1`)

## 설치 방법
1. 리포지토리를 클론합니다:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```
2. 가상 환경을 설정하고 활성화합니다:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows의 경우: venv\Scripts\activate
   ```
3. 필요한 패키지를 설치합니다:
   ```bash
   pip install -r requirements.txt
   ```
4. `.env` 파일을 생성하고 필요한 환경 변수를 설정합니다:
   ```plaintext
   AZURE_AI_SEARCH_ENDPOINT=<your-search-endpoint>
   AZURE_AI_SEARCH_API_KEY=<your-search-api-key>
   AZURE_AI_SEARCH_INDEX_NAME=rag-msai-learn03
   AZURE_AI_CURRICULUM_INDEX_NAME=rag-curriculum
   AZURE_OPENAI_ENDPOINT=<your-openai-endpoint>
   AZURE_OPENAI_KEY=<your-openai-key>
   TAVILY_API_KEY=<your-tavily-api-key>
   OPENAI_EMBEDDING_DEPLOYMENT_NAME=<your-embedding-deployment>
   AZURE_OPENAI_DEPLOYMENT_NAME=dev-gpt-4.1
   ```
5. 챗봇을 실행합니다:
   ```bash
   streamlit run MS_AI_Learning_Chatbot.py
   ```

## 사용 방법

### 로컬 실행
- 브라우저에서 Streamlit 애플리케이션에 접속합니다 (기본적으로 [http://localhost:8501](http://localhost:8501)).
- 최초 실행 시 사용자 ID, 이름, 수강 중인 과정(예: 1일차, 2일차 등)을 입력합니다.
- 챗봇 인터페이스에서 질문을 입력하여 대화를 시작합니다.
  - 커리큘럼, 일정, 진행 상황 관련 질문은 자동으로 커리큘럼 인덱스에서 검색됩니다.
  - 일반 질문은 문서 검색(60%)과 웹 검색(40%) 결과를 결합하여 답변합니다.
- 대화 종료를 원할 경우, 입력창에 `exit`를 입력합니다.
- 대화 기록은 세션 상태에 저장되며, 새로고침 시 초기화됩니다.

### 배포된 환경
- 배포된 MS AI 학습 챗봇은 다음 URL에서 접근할 수 있습니다: [https://swj-web-001.azurewebsites.net/](https://swj-web-001.azurewebsites.net/).
- **주의**: Streamlit 애플리케이션 실행을 위해 브라우저에서 JavaScript가 활성화되어 있어야 합니다.
- 배포된 환경에서도 사용자 ID, 이름, 과정을 입력하여 챗봇을 사용할 수 있습니다.

## 코드 구조
- **클래스**: `RAGChatbotWithWebSearch`
  - `generate_query_embedding`: 사용자 쿼리를 벡터 임베딩으로 변환.
  - `vector_search`: Azure AI Search를 통해 문서에서 벡터 검색 수행.
  - `curriculum_vector_search`: 커리큘럼 관련 질문에 특화된 검색.
  - `web_search`: Tavily API를 사용한 실시간 웹 검색.
  - `summarize_content`: 검색 결과를 비동기로 요약.
  - `process_query`: 쿼리 처리 및 검색, 요약, 답변 생성 통합.
  - `generate_final_answer`: 요약된 내용을 바탕으로 구조화된 최종 답변 생성.
- **메인 함수**: `main`
  - Streamlit UI 설정 및 사용자 입력 처리.
  - 세션 상태 관리 및 대화 기록 출력.

## 로그 및 디버깅
- 로깅은 `logging` 모듈을 통해 구현되며, `INFO` 수준으로 설정됩니다.
- 주요 작업(쿼리 처리, 검색, 요약 등)에 대한 로그가 기록됩니다.
- 오류 발생 시 상세한 오류 메시지가 로그에 기록되며, Streamlit UI에도 표시됩니다.

## 주의 사항
- 환경 변수가 누락되면 챗봇이 실행되지 않으며, Streamlit UI에 누락된 변수 목록이 표시됩니다.
- 커리큘럼 관련 질문은 `curriculum_vector_search`를 사용하며, 최대 2개의 문서를 반환하도록 제한됩니다.
- 웹 검색 결과는 최대 5개로 제한됩니다.
- 답변 생성 시 문서 정보(60%)와 웹 검색 정보(40%)를 균형 있게 사용하며, 출처는 답변 하단에 명시됩니다.
- 커리큘럼 질문의 경우 출처 표기가 생략됩니다.

## 향후 개선 계획
- 다국어 지원 추가.
- 검색 결과의 관련성 점수 계산 로직 개선.
- 사용자별 맞춤형 추천 기능 강화.
- 대화 기록의 영구 저장 및 불러오기 기능 구현.
