import asyncio
import os
from typing import List, Dict, Any
import openai
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorQuery
from langchain_community.tools.tavily_search import TavilySearchResults
import streamlit as st
from dataclasses import dataclass
import logging
from dotenv import load_dotenv
import re

load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DocumentSummary:
    source: str
    content: str
    summary: str

class RAGChatbotWithWebSearch:
    def __init__(self):
        """RAG 챗봇 초기화"""
        # Azure 설정
        self.search_endpoint = os.getenv("AZURE_AI_SEARCH_ENDPOINT")
        self.search_key = os.getenv("AZURE_AI_SEARCH_API_KEY")
        self.search_index = os.getenv("AZURE_AI_SEARCH_INDEX_NAME", "rag-msai-learn03")
        
        # OpenAI 설정 (Azure OpenAI Service)
        self.openai_client = openai.AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version="2024-12-01-preview"
        )
        
        # 검색 클라이언트 초기화
        self.search_client = SearchClient(
            endpoint=self.search_endpoint,
            index_name=self.search_index,
            credential=AzureKeyCredential(self.search_key)
        )
        
        # Tavily 웹검색 초기화
        self.tavily = TavilySearchResults(
            max_results=5,
            api_key=os.getenv("TAVILY_API_KEY")
        )
    
    def generate_query_embedding(self, query: str, deployment_name: str) -> List[float]:
        """쿼리 텍스트를 임베딩 벡터로 변환"""
        try:
            response = self.openai_client.embeddings.create(
                model=deployment_name,
                input=query
            )
            embedding = response.data[0].embedding
            return embedding
        except Exception as e:
            logger.error(f"임베딩 생성 오류: {e}")
            return []
    
    def vector_search(self, query: str, rag_params: dict, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for open-source frameworks, Azure OpenAI, RAG (using Azure AI Search) and LangChain/LangGraph information within PDF files."""
        try:
            # 쿼리 임베딩 생성
            embedding_model = os.getenv("OPENAI_EMBEDDING_DEPLOYMENT_NAME")
            embedding = self.generate_query_embedding(query, embedding_model)
            if not embedding:
                logger.error("임베딩 생성 실패")
                return []

            # 벡터 쿼리 생성 및 검색
            vector_query = {
                "vector": embedding,
                "k": top_k,
                "fields": "text_vector",  # 실제 인덱스의 벡터 필드명에 맞게 수정
                "kind": "vector"
            }
            results = self.search_client.search(
                search_text="",  # 벡터 전용 쿼리이므로 빈 문자열
                vector_queries=[vector_query]
            )
            documents = []
            for i, doc in enumerate(results):
                documents.append({
                    'id': doc.get('chunk_id', f'doc_{i+1}'),
                    'content': doc.get('chunk', ''),  # 실제 본문 필드명에 맞게 수정
                    'title': doc.get('title', ''),
                    'source': 'AzureAISearch'
                })
            logger.info(f"벡터 검색 결과: {len(documents)}개 문서 반환")
            #logger.info(f"벡터 검색 원본 doc: {documents}")
            return documents
        except Exception as e:
            logger.error(f"Azure Cognitive Search 벡터 검색 오류: {e}")
            return []
 
    def curriculum_vector_search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search information dedicated to Curriculum, Schedule, and Progress from PDF files."""
        try:
            # 쿼리 임베딩 생성
            embedding_model = os.getenv("OPENAI_EMBEDDING_DEPLOYMENT_NAME")
            embedding = self.generate_query_embedding(query, embedding_model)
            if not embedding:
                logger.error("임베딩 생성 실패")
                return []

            # curriculum 인덱스용 클라이언트 생성
            curriculum_index = os.getenv("AZURE_AI_CURRICULUM_INDEX_NAME", "rag-curriculum")
            curriculum_client = SearchClient(
                endpoint=self.search_endpoint,
                index_name=curriculum_index,
                credential=AzureKeyCredential(self.search_key)
            )
            vector_query = {
                "vector": embedding,
                "k": top_k,
                "fields": "text_vector",  # 실제 인덱스의 벡터 필드명에 맞게 수정
                "kind": "vector"
            }

            # 쿼리에서 일차 정보 추출
            day_pattern = r'(\d+)일차'
            day_match = re.search(day_pattern, query)
            target_day = day_match.group(0) if day_match else None
            
            logger.info(f"추출된 대상 일차: {target_day}")

            results = curriculum_client.search(
                search_text=query,  # 키워드 검색도 함께 활용
                vector_queries=[vector_query],
                top=top_k,
                select=["chunk_id", "chunk", "title"]  # 필요한 필드만 선택
            )
            
            documents = []
            for i, doc in enumerate(results):
                # 검색 스코어 확인 (디버깅용)
                score = getattr(doc, '@search.score', None)
                
                # 필드 매핑 확인 및 안전한 접근
                chunk_id = doc.get('chunk_id') or doc.get('id') or f'curriculum_doc_{i+1}'
                content = doc.get('chunk') or doc.get('content') or doc.get('text') or ''
                title = doc.get('title') or doc.get('document_title') or '제목 없음'
                #source = doc.get('source') or 'CurriculumIndex'
                
                if content:
                    # 일차별 필터링 로직 - 언급 횟수 기반
                    if target_day:
                        # 대상 일차 언급 횟수 계산
                        target_day_count = content.count(target_day) + title.count(target_day)
                        
                        # 다른 일차들의 언급 횟수 계산
                        other_days_pattern = r'(\d+)일차'
                        other_days = re.findall(other_days_pattern, content + title)
                        target_day_num = target_day.replace('일차', '')
                        
                        # 다른 일차별 언급 횟수 계산
                        other_days_count = {}
                        for day in other_days:
                            if day != target_day_num:
                                other_days_count[day] = other_days_count.get(day, 0) + 1
                        
                        # 가장 많이 언급된 다른 일차의 횟수
                        max_other_day_count = max(other_days_count.values()) if other_days_count else 0
                        
                        # 대상 일차 언급 비율 계산 (전체 일차 언급 중 대상 일차 비율)
                        total_day_mentions = target_day_count + sum(other_days_count.values())
                        target_day_ratio = target_day_count / total_day_mentions if total_day_mentions > 0 else 0
                        
                        # 문서에 관련성 점수 추가
                        relevance_score = target_day_count * 10 + target_day_ratio * 100
                        
                        documents.append({
                            'id': chunk_id,
                            'content': content,
                            'title': title,
                            'source': 'CurriculumIndex',
                            'target_day_count': target_day_count,
                            'max_other_day_count': max_other_day_count,
                            'target_day_ratio': target_day_ratio,
                            'relevance_score': relevance_score,
                            'relevance_reason': f"Target day mentioned {target_day_count} times, ratio: {target_day_ratio:.2f}"
                        })
                    else:
                        # target_day가 없는 경우 기본 추가
                        documents.append({
                            'id': chunk_id,
                            'content': content,
                            'title': title,
                            'source': 'CurriculumIndex',
                            'target_day_count': 0,
                            'max_other_day_count': 0,
                            'target_day_ratio': 0,
                            'relevance_score': score if score else 0,
                            'relevance_reason': 'No specific day filtering'
                        })

                # 관련성 기준으로 재정렬 (대상 일차 언급 횟수와 비율 기준)
                if target_day:
                    documents.sort(key=lambda x: (
                        x['target_day_count'],          # 1순위: 대상 일차 언급 횟수
                        x['target_day_ratio'],          # 2순위: 대상 일차 언급 비율
                        -x['max_other_day_count'],      # 3순위: 다른 일차 언급이 적은 순 (음수로 역순)
                    ), reverse=True)
                    

                # 최종 결과를 2로 제한
                documents = documents[:2]

            logger.info(f"커리큘럼 인덱스 검색 결과: {len(documents)}개 문서 반환")
            
            # # 디버깅: 검색된 문서 내용 확인
            # for i, doc in enumerate(documents[:2]):  # 상위 2개만 로깅
            #     logger.info(f"커리큘럼 문서 {i+1} - 제목: {doc['title'][:50]}...")
            #     logger.info(f"커리큘럼 문서 {i+1} - 내용 길이: {len(doc['content'])}")
            #     logger.info(f"커리큘럼 문서 {i+1} - 내용 미리보기: {doc['content'][:200]}...")
            
            return documents

        except Exception as e:
            logger.error(f"커리큘럼 인덱스 벡터 검색 오류: {e}")
            return []
        
    def web_search(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Tavily를 이용한 웹 검색"""
        try:
            # TavilySearchResults의 run 메서드를 사용하여 검색
            results = self.tavily.run(query)
            
            # 검색 결과를 문자열로 합치기
            web_content = ""
            if results:
                for i, result in enumerate(results[:max_results]):
                    title = result.get('title', 'No title')
                    content = result.get('content', 'No content')
                    url = result.get('url', 'No URL')
                    
                    web_content += f"\n=== 웹 결과 {i+1} ===\n"
                    web_content += f"제목: {title}\n"
                    web_content += f"URL: {url}\n"
                    web_content += f"내용: {content}\n"
            
            logger.info(f"웹 검색 완료: {len(results)}개 결과")
            return {
                'content': web_content,
                'source': 'Web Search Results',
                'query': query
            }
            
        except Exception as e:
            logger.error(f"웹 검색 오류: {e}")
            return {
                'content': f"웹 검색 중 오류가 발생했습니다: {e}",
                'source': 'Web Search Error',
                'query': query
            }

    async def summarize_content(self, content: str, source: str, context: str = "") -> DocumentSummary:
        """개별 문서/웹검색 결과 요약 (비동기)"""
        try:
            # 내용이 너무 짧으면 요약하지 않음
            if len(content) < 100:
                return DocumentSummary(
                    source=source,
                    content=content,
                    summary=content
                )
            
            prompt = f"""
                    다음 내용을 간결하고 핵심적으로 요약해주세요.
                    컨텍스트: {context}
                    출처: {source}

                    내용:
                    {content[:3000]}  # 토큰 제한을 위해 3000자로 제한

                    요약 시 다음 사항을 고려해주세요:
                    1. 핵심 정보만 포함
                    2. 3-5문장으로 간결하게
                    3. 출처 특성 반영 (문서 vs 웹검색)
                    4. 사용자 질문과의 관련성 중심

                    요약:"""

            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.openai_client.chat.completions.create(
                    model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "dev-gpt-4.1"),
                    messages=[
                        {"role": "system", "content": "당신은 전문적인 문서 요약 어시스턴트입니다."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=500,
                    temperature=0.4
                )
            )
            
            summary = response.choices[0].message.content.strip()
            
            return DocumentSummary(
                source=source,
                content=content,
                summary=summary
            )
            
        except Exception as e:
            logger.error(f"요약 생성 오류 ({source}): {e}")
            return DocumentSummary(
                source=source,
                content=content,
                summary=f"요약 생성 중 오류가 발생했습니다: {content[:200]}..."
            )
    

    async def process_query(self, user_query: str) -> str:
        """메인 처리 함수: 쿼리 -> 검색 -> 비동기 요약 -> 최종 답변"""
        logger.info(f"사용자 쿼리 처리 시작: {user_query}")

        # 커리큘럼/일정 관련 키워드 포함 여부 확인 및 질문 재구성
        keywords = ['커리큘럼', '일정', '스케줄', '수업', '학습', '교육', '진도']
        is_curriculum_query = any(k in user_query for k in keywords)
        course = st.session_state.user_info['course']
        new_query = user_query
        if is_curriculum_query and course:
            if '오늘' in user_query:
                new_query = user_query.replace('오늘', course + ' 커리큘럼')
            #new_query = f"{course} 커리큘럼 {user_query}"
            #logger.info(f"커리큘럼/일정 질문 재구성: {new_query}")

        if is_curriculum_query:
            # 커리큘럼/일정 질문이면 curriculum_vector_search만 실행
            logger.info("커리큘럼/일정 질문: curriculum_vector_search만 실행")
            documents = self.curriculum_vector_search(new_query, top_k=3)
            web_results = {'content': '', 'source': 'Web Search Skipped', 'query': new_query}
        else:
            # 평소에는 벡터+웹 검색 병행
            rag_params = {
                "data_sources": [
                    {
                        "type": "azure_search",
                        "parameters": {
                            "endpoint": os.getenv("AZURE_AI_SEARCH_ENDPOINT"),
                            "index_name": os.getenv("AZURE_AI_SEARCH_INDEX_NAME"),
                            "authentication": {
                                "type": "api_key",
                                "key": os.getenv("AZURE_AI_SEARCH_API_KEY")
                            },
                            "query_type": "vector",
                            "embedding_dependency": {
                                "type": "deployment_name",
                                "deployment_name": os.getenv("OPENAI_EMBEDDING_DEPLOYMENT_NAME")
                            }
                        }
                    }
                ]
            }
            logger.info("일반 질문: 벡터+웹 검색 병행")
            vector_task = asyncio.get_event_loop().run_in_executor(
                None, lambda: self.vector_search(user_query, rag_params, 5)
            )
            web_task = asyncio.get_event_loop().run_in_executor(
                None, self.web_search, user_query, 5
            )
            documents, web_results = await asyncio.gather(vector_task, web_task)

        # 2단계: 모든 내용을 비동기로 동시에 요약
        logger.info("비동기 요약 시작...")
        summarization_tasks = []

        # 벡터 검색 문서들 요약 태스크
        for i, doc in enumerate(documents):
            task = self.summarize_content(
                content=doc['content'],
                source=f"문서 {i+1}: {doc['title']}",
                context=user_query
            )
            #logger.info(f"벡터 검색 원본 제목: {doc['title']}")
            #logger.info(f"벡터 검색 원본 문서: {doc['content']}")
            summarization_tasks.append(task)

        # 웹 검색 결과 요약 태스크 (웹 검색이 스킵된 경우 빈 내용)
        web_summary_task = self.summarize_content(
            content=web_results['content'],
            source=web_results['source'],
            context=user_query
        )
        summarization_tasks.append(web_summary_task)

        # 모든 요약을 동시에 실행
        summaries = await asyncio.gather(*summarization_tasks)

        # 3단계: 요약들을 합쳐서 최종 답변 생성
        logger.info("최종 답변 생성...")
        return await self.generate_final_answer(user_query, summaries, is_curriculum_query)

    # 기존 generate_final_answer 함수를 이 코드로 교체하세요

    async def generate_final_answer(self, user_query: str, summaries: List[DocumentSummary], is_curriculum_query: bool = False) -> str:
        """요약된 내용들을 바탕으로 최종 답변 생성 (출처 표기 포함, 구조화된 형식)"""
        try:
            # 프롬프트 엔지니어링 - 출처 정보 포함
            combined_summaries = ""
            source_mapping = []
            
            for i, summary in enumerate(summaries):
                if "Web Search" in summary.source:
                    source_label = "웹 검색"
                    source_mapping.append("웹 검색: 실시간 웹 검색 결과")
                else:
                    source_label = f"문서 {i+1}"
                    # 제목 추출
                    if ':' in summary.source:
                        title = summary.source.split(':', 1)[1].strip()
                    else:
                        title = summary.source.strip()
                    source_mapping.append(f"문서 {i+1}: {title}")
                
                combined_summaries += f"\n--- {source_label} ---\n{summary.summary}\n"

            if is_curriculum_query:
                    final_prompt = f"""
                    사용자 질문: {user_query}

                    다음은 관련 정보들의 요약입니다:
                    {combined_summaries}

                    위 정보들을 종합하여 사용자의 MS AI 교육 중 질문에 대해 정확하고 도움이 되는 답변을 **구조화된 형식**으로 작성해주세요. 답변은 마크다운 형식을 사용하여 소제목, 리스트, 번호, 표 등을 활용해 가독성을 높여야 합니다.

                    답변 작성 가이드라인:
                    1. 사용자 질문에 직접적으로 답변하며, 답변은 명확하고 간결해야 함.
                    2. 커리큘럼 및 일정 관련 질문은 문서 정보만 기준으로 구조화된 형식으로 답변하고 출처표기는 생략합니다. 
                    6. 실용적이고 구체적인 조언 제공.
                    7. 한국어로 자연스럽게 작성.
                    9. **구조화된 형식**:
                    - **개요**: 질문에 대한 간단한 답변 요약 (1-2문장).
                    - **세부 정보**: 관련 정보를 리스트(예: `-` 또는 `1.`)로 정리.
                    - **권장 사항**: 사용자에게 실질적인 조언이나 다음 단계 제안 (리스트 형식 권장).

                    **구조화된 형식 답변 예시**:
                    ```markdown

                    ### 개요
                    [질문에 대한 간단한 답변 요약]

                    ### 세부 정보
                    - [항목 1]: [설명]
                    - [항목 2]: [설명]
                    - [항목 3]: [설명]

                    ### 권장 사항
                    1. [권장 사항 1]
                    2. [권장 사항 2]

                    답변:"""
            else:
                final_prompt = f"""
                사용자 질문: {user_query}

                다음은 관련 정보들의 요약입니다:
                {combined_summaries}

                위 정보들을 종합하여 사용자의 MS AI 교육 중 질문에 대해 정확하고 도움이 되는 답변을 작성해주세요.

                답변 작성 가이드라인:
                1. 사용자 질문에 직접적으로 답변하며, 답변은 명확해야 합니다. 
                2. 질문은 문서 정보 60%와 웹 검색 정보 40%로 균형 있게 활용.
                3. 신뢰할 수 있는 정보 우선 사용.
                4. 불확실한 정보는 "불확실" 또는 "추가 확인 필요"로 명시.
                5. 실용적이고 구체적인 조언 제공.
                6. 한국어로 자연스럽게 작성.
                7. 출처 규칙
                답변에서 구체적인 사실, 시간, 절차, 내용을 언급할 때는 해당 문장 끝에 출처를 표기해주세요.
                - 문서 정보 사용 시: (문서 1), (문서 2) 등
                - 웹 검색 정보 사용 시: (웹 검색)
                - 여러 출처 조합 시: (문서 1, 웹 검색), (문서 1, 문서 2) 등
                - 일반적인 설명이나 연결 문장에는 출처 표기 생략 

                답변:"""

            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.openai_client.chat.completions.create(
                    model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "dev-gpt-4.1"),
                    messages=[
                        {"role": "system", "content": "당신은 도움이 되고 정확한 정보를 제공하는 MS AI 교육 어시스턴트입니다. 구체적인 정보를 제공할 때는 반드시 출처를 명시해주세요."},
                        {"role": "user", "content": final_prompt.strip()}
                    ],
                    max_tokens=3000,
                    temperature=0.5
                )
            )
            
            final_answer = response.choices[0].message.content.strip()
            logger.info("최종 답변 생성 완료")
            
            # 하단에 출처 매핑 정보 추가 (선택사항)
            #final_answer += "\n\n" + "=" * 50
            if is_curriculum_query == False:
                final_answer += "\n\n**출처 정보**"
                for mapping in source_mapping:
                    final_answer += f"\n• {mapping}"
            
            return final_answer
            
        except Exception as e:
            logger.error(f"최종 답변 생성 오류: {e}")
            return f"답변 생성 중 오류가 발생했습니다: {e}"

def main():
    st.set_page_config(
        page_title="MS AI 학습 챗봇",
        page_icon="🤖",
        layout="wide"
    )

    st.title("🤖 MS AI 학습 챗봇")
    st.markdown("RAG 벡터 검색과 실시간 웹검색을 결합한 지능형 챗봇입니다.")

    # 환경변수 확인
    required_env_vars = [
        "AZURE_AI_SEARCH_ENDPOINT",
        "AZURE_AI_SEARCH_API_KEY", 
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_KEY",
        "TAVILY_API_KEY",
        "OPENAI_EMBEDDING_DEPLOYMENT_NAME"
    ]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        st.error(f"다음 환경변수가 설정되지 않았습니다: {', '.join(missing_vars)}")
        st.stop()

    # 사용자 정보 입력 (최초 1회)
    if 'user_info' not in st.session_state:
        with st.form("user_info_form"):
            user_id = st.text_input("User ID")
            user_name = st.text_input("User Name")
            course = st.selectbox("과정을 선택하세요", [f"{i+1}일차" for i in range(9)])
            submitted = st.form_submit_button("정보 저장")
            if submitted and user_id and user_name and course:
                st.session_state.user_info = {
                    "user_id": user_id,
                    "user_name": user_name,
                    "course": course
                }
                st.session_state.chat_messages = []
                st.session_state.chat_exit = False
                st.session_state.chatbot = RAGChatbotWithWebSearch()
                st.success("사용자 정보가 저장되었습니다.")
            elif submitted:
                st.warning("모든 정보를 입력하세요.")
        st.stop()

    # 챗봇 초기화
    if 'chatbot' not in st.session_state:
        try:
            st.session_state.chatbot = RAGChatbotWithWebSearch()
        except Exception as e:
            st.error(f"챗봇 초기화 오류: {e}")
            st.stop()
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
    if 'chat_exit' not in st.session_state:
        st.session_state.chat_exit = False

    st.markdown(f"**사용자:** {st.session_state.user_info['user_name']} ({st.session_state.user_info['user_id']}) / 과정: {st.session_state.user_info['course']}")

    # 대화 입력 및 처리 (하단에 위치)
    user_input = st.chat_input("메시지를 입력하세요. (종료하려면 'exit' 입력)", key="main_chat_input")
    if user_input:
        if user_input.lower() == "exit":
            st.session_state.chat_exit = True
            st.success("챗봇 대화를 종료합니다.")
        else:
            st.session_state.chat_messages.append({
                "user": st.session_state.user_info['user_name'],
                "query": user_input,
                "answer": None  # 답변은 아래에서 생성
            })

    # 대화 내용 출력 (상단에 위치)
    st.markdown("### 📝 대화 내용")
    if st.session_state.chat_messages:
        for i, msg in enumerate(st.session_state.chat_messages, 1):
            with st.chat_message("user"):
                st.write(msg["query"])
            # 답변이 아직 없으면 spinner를 출력
            if msg["answer"] is None and i == len(st.session_state.chat_messages):
                with st.spinner("검색하고 답변을 생성하는 중..."):
                    try:
                        answer = asyncio.run(
                            st.session_state.chatbot.process_query(msg["query"])
                        )
                        st.session_state.chat_messages[-1]["answer"] = answer
                        with st.chat_message("assistant"):
                            st.write(answer)
                    except Exception as e:
                        st.error(f"처리 중 오류가 발생했습니다: {e}")
            elif msg["answer"]:
                with st.chat_message("assistant"):
                    st.write(msg["answer"])
    else:
        st.info("아직 대화 내용이 없습니다.")

    # 대화 종료 안내
    if st.session_state.chat_exit:
        st.info("챗봇 대화가 종료되었습니다. 새로고침하면 다시 시작할 수 있습니다.")

    # 사이드바에 설정 정보
    with st.sidebar:
        st.markdown("### ⚙️ 챗봇 설정 정보")
        st.info("""
        **기능:**
        - 백터 인덱스 문서 검색 (5개)
        - 실시간 웹 검색 (5개 결과)
        - 비동기 병렬 요약 처리
        - 통합 답변 생성
        - 사용자 정보 및 대화 내용 저장
        
        **처리 과정:**
        1. 사용자 정보 입력
        2. 사용자 쿼리 분석
        3. 벡터 검색 + 웹 검색 (동시 실행)
        4. 요약 생성 (비동기)
        5. 프롬프트 엔지니어링
        6. 최종 답변 생성
        7. 대화 내용 저장 및 출력
        """)


if __name__ == "__main__":
    main()