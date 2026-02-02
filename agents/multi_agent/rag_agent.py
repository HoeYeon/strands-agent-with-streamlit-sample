"""RAG Agent - 문서 검색 전문가

OpenSearch 벡터 데이터베이스에서 스키마 문서와 도메인 지식을 검색하여
다른 에이전트에게 컨텍스트를 제공하는 전문가입니다.
"""

import logging
from typing import Any, Dict, List, Optional

from strands import Agent
from strands.tools import tool

from .base_agent import BaseMultiAgent
from .shared_context import AnalysisContext
from .vector_search import VectorSearchService, SearchResult

logger = logging.getLogger(__name__)


class RAGAgent(BaseMultiAgent):
    """RAG Agent - 문서 검색 전문가"""

    def __init__(
        self,
        model_id: str,
        opensearch_endpoint: Optional[str] = None,
        opensearch_index: str = "schema_docs",
        embedding_model: str = "amazon.titan-embed-text-v2:0",
        opensearch_username: Optional[str] = None,
        opensearch_password: Optional[str] = None,
        tools: Optional[List] = None
    ):
        self.opensearch_endpoint = opensearch_endpoint
        self.opensearch_index = opensearch_index
        self.embedding_model = embedding_model
        self.opensearch_username = opensearch_username
        self.opensearch_password = opensearch_password

        # 벡터 검색 서비스 초기화
        self._search_service = VectorSearchService(
            opensearch_endpoint=opensearch_endpoint,
            opensearch_index=opensearch_index,
            opensearch_username=opensearch_username,
            opensearch_password=opensearch_password,
            embedding_model=embedding_model
        )

        super().__init__(model_id, tools)

    def _create_search_tools(self) -> List:
        """RAG 검색 도구 생성"""
        rag_agent = self

        @tool
        def search_rag_documents(query: str) -> str:
            """RAG 문서 검색 도구

            스키마 문서와 도메인 지식을 검색합니다.

            Args:
                query: 검색할 쿼리

            Returns:
                검색된 문서 정보
            """
            logger.info(f"[RAG 검색] 쿼리: '{query}'")

            result = rag_agent.search_and_extract(query)

            if not result["success"]:
                errors = ", ".join(result["errors"]) if result["errors"] else "알 수 없는 오류"
                logger.warning(f"[RAG 검색] 실패: {errors}")
                return f"검색 실패: {errors}"

            if result["results"]:
                logger.info(f"[RAG 검색] 문서 {len(result['results'])}개 발견")
                for i, doc in enumerate(result["results"][:3], 1):
                    logger.info(f"  [{i}] score={doc.score:.3f} table={doc.metadata.get('table', 'N/A')}")
                    logger.info(f"      content: {doc.content[:100]}...")
                output = "## 검색 결과\n" + rag_agent.format_results_for_agent(result["results"])
                logger.info(f"[RAG 검색] 결과 반환 (길이: {len(output)}자)")
                return output

            logger.warning(f"[RAG 검색] 결과 없음: '{query}'")
            if result["suggestions"]:
                suggestions_text = "\n".join([f"- {s['suggestion']}" for s in result["suggestions"][:3]])
                return f"'{query}'에 대한 검색 결과가 없습니다.\n\n대안 제안:\n{suggestions_text}"
            return f"'{query}'에 대한 검색 결과가 없습니다."

        return [search_rag_documents]

    def _setup_agent(self) -> None:
        """RAG Agent 초기화"""
        search_tools = self._create_search_tools()
        all_tools = search_tools + (self.tools if self.tools else [])

        self.agent = Agent(
            name="rag_agent",
            system_prompt=self.get_system_prompt(),
            model=self.model_id,
            tools=all_tools if all_tools else None
        )

    def get_system_prompt(self) -> str:
        """RAG Agent 시스템 프롬프트"""
        return """
역할: 벡터 데이터베이스 문서 검색 전문가

────────────────────────────────────────────
주요 기능
────────────────────────────────────────────

1. **스키마 문서 검색** (search_schema_docs)
   - 사용자 쿼리와 관련된 테이블/컬럼 정보 검색
   - 테이블명, 컬럼명, 데이터 타입, 설명 추출
   - 상위 10개 관련 문서 반환

2. **도메인 지식 검색** (search_domain_knowledge)
   - 비즈니스 용어 → 데이터베이스 컬럼 매핑
   - 메트릭 정의 및 계산 공식 검색
   - 값 설명 및 비즈니스 로직 검색

────────────────────────────────────────────
검색 결과 형식
────────────────────────────────────────────

각 검색 결과에 포함:
- content: 문서 내용
- score: 관련도 점수 (0.0 ~ 1.0)
- metadata: 테이블명, 데이터베이스명 등
- source: 출처 문서 경로

────────────────────────────────────────────
handoff 규칙 (중요!)
────────────────────────────────────────────

검색 완료 후 반드시 **발견한 도메인 지식을 자연어로 상세히 전달**:

**data_expert/sql_agent로 handoff 시:**
1. 원문 질문을 그대로 포함
2. 검색에서 발견한 도메인 지식을 자연어로 설명
3. SQL 작성에 필요한 조건, 범위, 값의 의미를 명확히 기술

**예시:**
```
[원문 질문] What is the ratio of male to female patients among all those with abnormal uric acid counts?

[도메인 지식]
- UA(uric acid) 컬럼은 thrombosis_prediction.Laboratory 테이블에 있습니다.
- 정상 범위: 남성 UA > 8.0, 여성 UA > 6.5
- 따라서 abnormal(비정상)은 남성 UA <= 8.0, 여성 UA <= 6.5 입니다.
- Patient 테이블의 SEX 컬럼으로 성별 구분 ('M', 'F')
- Patient.ID와 Laboratory.ID로 조인

[SQL 작성 시 주의]
- RAG에서 제공한 정상/비정상 범위 기준을 반드시 따를 것
- 일반 상식이 아닌 검색된 도메인 지식 기준으로 조건 작성
```

────────────────────────────────────────────
오류 처리
────────────────────────────────────────────

- OpenSearch 연결 실패 → lead_agent에 보고, 기존 워크플로우 계속
- 검색 결과 없음 → 대안 검색어 제안
- 임베딩 생성 실패 → RAG 없이 진행
"""

    def get_tools(self) -> List:
        return self.tools

    def get_agent(self) -> Agent:
        if not self.agent:
            raise RuntimeError("Agent not initialized")
        return self.agent

    def search_and_extract(self, query: str, context: Optional[AnalysisContext] = None) -> Dict[str, Any]:
        """검색 및 정보 추출 통합 메서드"""
        result = {
            "success": True,
            "rag_enabled": self._search_service.is_enabled(),
            "results": [],
            "suggestions": [],
            "errors": []
        }

        if not self._search_service.is_enabled():
            result["success"] = False
            result["errors"].append("RAG가 비활성화되어 있습니다")
            return result

        try:
            search_results, search_error = self._search_service.search(query)
            if search_error:
                result["errors"].append(f"검색 실패: {search_error}")
            else:
                result["results"] = search_results
                if not search_results:
                    result["suggestions"] = self._search_service.get_alternative_suggestions(query)

            # 컨텍스트에 저장
            if context and search_results:
                self.save_to_context(context, search_results)

            result["success"] = len(result["errors"]) == 0

        except Exception as e:
            result["success"] = False
            result["errors"].append(f"검색 중 예외: {e}")

        return result

    def format_results_for_agent(self, search_results: List[SearchResult]) -> str:
        """검색 결과를 LLM 친화적 형식으로 변환"""
        if not search_results:
            return "검색 결과가 없습니다."

        lines = []
        for i, result in enumerate(search_results[:5], 1):
            lines.append(f"### {i}. (관련도: {result.score:.3f})")
            if result.metadata:
                meta = result.metadata
                if meta.get("database"):
                    lines.append(f"- DB: `{meta['database']}`")
                if meta.get("table"):
                    lines.append(f"- 테이블: `{meta['table']}`")
                if meta.get("column_name"):
                    lines.append(f"- 컬럼: `{meta['column_name']}`")
            lines.append(f"\n{result.content}\n")
            if result.source:
                lines.append(f"출처: `{result.source}`\n")
            lines.append("---")

        return "\n".join(lines)

    def save_to_context(
        self,
        context: AnalysisContext,
        search_results: Optional[List[SearchResult]] = None
    ) -> AnalysisContext:
        """검색 결과를 공유 컨텍스트에 저장"""
        if search_results:
            for result in search_results:
                context.add_rag_result({
                    "content": result.content[:500],
                    "score": result.score,
                    "metadata": result.metadata,
                    "source": result.source
                })
        return context

    def is_rag_enabled(self) -> bool:
        return self._search_service.is_enabled()

    def get_status(self) -> Dict[str, Any]:
        return {
            "rag_enabled": self._search_service.is_enabled(),
            "opensearch_endpoint": self.opensearch_endpoint,
            "opensearch_index": self.opensearch_index,
            "cache_stats": self._search_service.get_cache_stats()
        }
