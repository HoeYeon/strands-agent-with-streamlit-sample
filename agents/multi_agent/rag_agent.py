"""RAG Agent - 문서 검색 전문가

OpenSearch 벡터 데이터베이스에서 스키마 문서와 도메인 지식을 검색하여
다른 에이전트에게 컨텍스트를 제공하는 전문가입니다.

Requirements:
- 1.1: 사용자 쿼리를 임베딩으로 변환
- 1.2: 유사도 기반으로 상위 10개의 관련 스키마 문서 반환
- 1.3: 각 문서의 관련도 점수와 메타데이터 포함
- 1.4: 검색 결과가 없으면 빈 결과와 대안 검색 제안 제공
- 1.5: 스키마 문서에서 테이블명, 컬럼명, 데이터 타입, 설명 추출
- 2.1: 비즈니스 용어와 관련된 도메인 문서 검색
- 2.2: 메트릭 정의 검색 시 계산 공식, 사용 컬럼, 필터 조건 포함
- 2.3: 비즈니스 용어를 데이터베이스 컬럼명으로 변환하는 매핑 정보 제공
- 2.5: 출처 문서와 신뢰도 점수 포함
- 3.1: Swarm에 등록하고 handoff 도구 제공
- 3.4: 공유 컨텍스트에 검색 결과 저장
- 3.5: RAG 실패 시 기존 워크플로우 계속 진행
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from strands import Agent

from .base_agent import BaseMultiAgent
from .shared_context import AnalysisContext

logger = logging.getLogger(__name__)


# =============================================================================
# RAG 관련 상수
# =============================================================================

# OpenSearch 설정
DEFAULT_OPENSEARCH_INDEX = "schema_docs"
DEFAULT_TOP_K = 10

# 임베딩 모델 설정
DEFAULT_EMBEDDING_MODEL = "amazon.titan-embed-text-v2:0"
EMBEDDING_DIMENSION = 1024
MAX_EMBEDDING_TOKENS = 8192

# 타임아웃 설정
OPENSEARCH_TIMEOUT = 30  # 초
EMBEDDING_TIMEOUT = 30  # 초

# 재시도 설정
MAX_RETRY_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 1


# =============================================================================
# 데이터 모델
# =============================================================================

@dataclass
class SearchResult:
    """벡터 검색 결과 (Requirements 1.3, 2.5)"""
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = ""  # 문서 출처


@dataclass
class ColumnDetail:
    """컬럼 상세 정보 (Requirements 1.5)"""
    name: str
    type: str
    description: str = ""
    alias: str = ""
    examples: List[str] = field(default_factory=list)


@dataclass
class SchemaInfo:
    """스키마 문서에서 추출한 정보 (Requirements 1.5)"""
    table_name: str
    database: str
    columns: List[ColumnDetail] = field(default_factory=list)
    description: str = ""
    business_logic: str = ""
    source_doc: str = ""


@dataclass
class DomainMapping:
    """도메인 용어 매핑 (Requirements 2.3)"""
    business_term: str
    database_column: str
    table: str
    description: str = ""
    usage_frequency: int = 0


# =============================================================================
# RAG Agent 클래스
# =============================================================================

class RAGAgent(BaseMultiAgent):
    """RAG Agent - 문서 검색 전문가
    
    역할:
    - 사용자 쿼리 임베딩 변환 (Requirements 1.1)
    - OpenSearch k-NN 벡터 검색 (Requirements 1.2)
    - 검색 결과에 관련도 점수와 메타데이터 포함 (Requirements 1.3)
    - 빈 결과 시 대안 제안 (Requirements 1.4)
    - 스키마 정보 구조화 추출 (Requirements 1.5)
    - 도메인 지식 검색 (Requirements 2.1, 2.2, 2.3)
    - 출처 및 신뢰도 정보 포함 (Requirements 2.5)
    - Swarm 등록 및 handoff 지원 (Requirements 3.1)
    - 공유 컨텍스트에 결과 저장 (Requirements 3.4)
    - 실패 시 graceful degradation (Requirements 3.5)
    """
    
    def __init__(
        self,
        model_id: str,
        opensearch_endpoint: Optional[str] = None,
        opensearch_index: str = DEFAULT_OPENSEARCH_INDEX,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        opensearch_username: Optional[str] = None,
        opensearch_password: Optional[str] = None,
        tools: Optional[List] = None
    ):
        """RAG Agent 초기화
        
        Args:
            model_id: LLM 모델 ID
            opensearch_endpoint: OpenSearch 엔드포인트 URL
            opensearch_index: OpenSearch 인덱스 이름
            embedding_model: AWS Bedrock 임베딩 모델 ID
            opensearch_username: OpenSearch 사용자명 (Basic Auth용)
            opensearch_password: OpenSearch 비밀번호 (Basic Auth용)
            tools: 추가 도구 목록
        """
        self.opensearch_endpoint = opensearch_endpoint
        self.opensearch_index = opensearch_index
        self.embedding_model = embedding_model
        self.opensearch_username = opensearch_username
        self.opensearch_password = opensearch_password
        
        # 클라이언트 초기화
        self._opensearch_client = None
        self._bedrock_client = None
        self._rag_enabled = True
        
        # 캐시
        self._embedding_cache: Dict[str, List[float]] = {}
        
        super().__init__(model_id, tools)
        
        # OpenSearch 연결 초기화
        self._init_clients()

    
    def _init_clients(self) -> None:
        """OpenSearch 및 Bedrock 클라이언트 초기화 (Requirements 3.5)
        
        연결 실패 시 RAG를 비활성화하고 기존 워크플로우 계속 진행
        """
        # Bedrock 클라이언트 초기화
        try:
            import boto3
            self._bedrock_client = boto3.client(
                "bedrock-runtime",
                region_name="us-west-2"
            )
            logger.info("Bedrock 클라이언트 초기화 성공")
        except Exception as e:
            logger.warning(f"Bedrock 클라이언트 초기화 실패: {e}")
            self._rag_enabled = False
        
        # OpenSearch 클라이언트 초기화 (Basic Auth)
        if self.opensearch_endpoint and self.opensearch_username and self.opensearch_password:
            try:
                from opensearchpy import OpenSearch, RequestsHttpConnection
                
                # 엔드포인트에서 호스트 추출
                host = self.opensearch_endpoint.replace("https://", "").replace("http://", "")
                if host.endswith("/"):
                    host = host[:-1]
                
                # 포트 추출 (기본값 443)
                if ":" in host:
                    host, port_str = host.rsplit(":", 1)
                    port = int(port_str)
                else:
                    port = 443
                
                self._opensearch_client = OpenSearch(
                    hosts=[{"host": host, "port": port}],
                    http_auth=(self.opensearch_username, self.opensearch_password),
                    use_ssl=True,
                    verify_certs=True,
                    connection_class=RequestsHttpConnection,
                    timeout=OPENSEARCH_TIMEOUT
                )
                logger.info(f"OpenSearch 클라이언트 초기화 성공: {host}:{port}")
            except ImportError as e:
                logger.warning(f"OpenSearch 라이브러리 없음: {e}")
                self._rag_enabled = False
            except Exception as e:
                logger.warning(f"OpenSearch 클라이언트 초기화 실패: {e}")
                self._rag_enabled = False
        else:
            logger.info("OpenSearch 설정 미완료 (endpoint, username, password 필요) - RAG 비활성화")
            self._rag_enabled = False

    
    def _setup_agent(self) -> None:
        """RAG Agent 초기화 (Requirements 3.1)"""
        self.agent = Agent(
            name="rag_agent",
            system_prompt=self.get_system_prompt(),
            model=self.model_id,
            tools=self.tools if self.tools else None
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
handoff 규칙
────────────────────────────────────────────

검색 완료 후:
- data_expert: 스키마 정보 전달
- sql_agent: 도메인 지식 전달
- lead_agent: 검색 실패 또는 결과 없음 보고

────────────────────────────────────────────
오류 처리
────────────────────────────────────────────

- OpenSearch 연결 실패 → lead_agent에 보고, 기존 워크플로우 계속
- 검색 결과 없음 → 대안 검색어 제안
- 임베딩 생성 실패 → RAG 없이 진행
"""
    
    def get_tools(self) -> List:
        """RAG Agent 도구 목록"""
        return self.tools
    
    def get_agent(self) -> Agent:
        """Swarm에서 사용할 Agent 인스턴스 반환 (Requirements 3.1)"""
        if not self.agent:
            raise RuntimeError("Agent not initialized")
        return self.agent

    
    # =========================================================================
    # Requirements 1.1: 쿼리 임베딩 변환
    # =========================================================================
    
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """텍스트를 임베딩 벡터로 변환 (Requirements 1.1)
        
        AWS Bedrock Titan Embeddings V2를 사용하여 텍스트를 
        1024차원 벡터로 변환합니다.
        
        Args:
            text: 변환할 텍스트
            
        Returns:
            임베딩 벡터 (1024차원) 또는 None (실패 시)
        """
        if not self._bedrock_client:
            logger.warning("Bedrock 클라이언트 없음 - 임베딩 생성 불가")
            return None
        
        # 캐시 확인
        cache_key = text[:100]  # 처음 100자를 키로 사용
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
        
        try:
            # Titan Embeddings V2 호출
            body = json.dumps({
                "inputText": text[:MAX_EMBEDDING_TOKENS * 4],  # 대략적인 토큰 제한
                "dimensions": EMBEDDING_DIMENSION,
                "normalize": True
            })
            
            response = self._bedrock_client.invoke_model(
                modelId=self.embedding_model,
                body=body,
                contentType="application/json",
                accept="application/json"
            )
            
            response_body = json.loads(response["body"].read())
            embedding = response_body.get("embedding", [])
            
            # 캐시에 저장
            if len(self._embedding_cache) < 100:  # 최대 100개 캐시
                self._embedding_cache[cache_key] = embedding
            
            return embedding
            
        except Exception as e:
            logger.error(f"임베딩 생성 실패: {e}")
            return None

    
    # =========================================================================
    # Requirements 1.2, 1.3: 벡터 검색 및 결과 반환
    # =========================================================================
    
    def search_documents(
        self,
        query: str,
        doc_type: str = "schema",
        top_k: int = DEFAULT_TOP_K,
        filters: Optional[Dict[str, str]] = None
    ) -> List[SearchResult]:
        """OpenSearch에서 문서 검색 수행 (Requirements 1.2, 1.3)
        
        벡터 유사도 검색을 수행하고 상위 k개 결과를 반환합니다.
        
        Args:
            query: 검색 쿼리
            doc_type: 문서 타입 ("schema" 또는 "domain")
            top_k: 반환할 최대 문서 수 (기본 10개)
            filters: 필터 조건 (database, table_name)
            
        Returns:
            검색 결과 목록 (최대 10개)
        """
        if not self._rag_enabled or not self._opensearch_client:
            logger.warning("RAG 비활성화 상태 - 빈 결과 반환")
            return []
        
        # 상위 10개로 제한 (Requirements 1.2)
        top_k = min(top_k, DEFAULT_TOP_K)
        
        # 쿼리 임베딩 생성 (Requirements 1.1)
        query_embedding = self.generate_embedding(query)
        if not query_embedding:
            logger.warning("임베딩 생성 실패 - 빈 결과 반환")
            return []
        
        try:
            # 하이브리드 검색 쿼리 구성
            search_query = self._build_search_query(
                query_embedding, query, top_k, filters
            )
            
            # OpenSearch 검색 실행
            response = self._opensearch_client.search(
                index=self.opensearch_index,
                body=search_query
            )
            
            # 결과 파싱 (Requirements 1.3)
            results = self._parse_search_results(response)
            return results
            
        except Exception as e:
            logger.error(f"OpenSearch 검색 실패: {e}")
            return []

    
    def _build_search_query(
        self,
        query_embedding: List[float],
        query_text: str,
        top_k: int,
        filters: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """하이브리드 검색 쿼리 구성
        
        벡터 검색과 키워드 검색을 결합한 하이브리드 검색 쿼리를 생성합니다.
        """
        # 기본 k-NN 쿼리
        knn_query = {
            "knn": {
                "content_vector": {
                    "vector": query_embedding,
                    "k": top_k
                }
            }
        }
        
        # 키워드 검색 쿼리
        keyword_query = {
            "multi_match": {
                "query": query_text,
                "fields": ["content", "columns", "business_logic", "table_name"],
                "type": "best_fields",
                "fuzziness": "AUTO"
            }
        }
        
        # 하이브리드 쿼리 구성
        bool_query: Dict[str, Any] = {
            "should": [knn_query, keyword_query],
            "minimum_should_match": 1
        }
        
        # 필터 추가
        if filters:
            filter_clauses = []
            if "database" in filters:
                filter_clauses.append({"term": {"database": filters["database"]}})
            if "table_name" in filters:
                filter_clauses.append({"term": {"table_name": filters["table_name"]}})
            if filter_clauses:
                bool_query["filter"] = filter_clauses
        
        return {
            "size": top_k,
            "query": {"bool": bool_query},
            "_source": ["content", "table_name", "database", "columns", "business_logic"]
        }
    
    def _parse_search_results(self, response: Dict[str, Any]) -> List[SearchResult]:
        """OpenSearch 응답을 SearchResult 목록으로 변환 (Requirements 1.3)"""
        results = []
        hits = response.get("hits", {}).get("hits", [])
        
        for hit in hits:
            source = hit.get("_source", {})
            score = hit.get("_score", 0.0)
            
            # 점수 정규화 (0.0 ~ 1.0)
            normalized_score = min(score / 10.0, 1.0) if score else 0.0
            
            result = SearchResult(
                content=source.get("content", ""),
                score=normalized_score,
                metadata={
                    "table_name": source.get("table_name", ""),
                    "database": source.get("database", ""),
                    "columns": source.get("columns", ""),
                    "business_logic": source.get("business_logic", "")
                },
                source=hit.get("_id", "")
            )
            results.append(result)
        
        return results

    
    # =========================================================================
    # Requirements 1.4: 빈 결과 처리 및 대안 제안
    # =========================================================================
    
    def get_alternative_suggestions(self, query: str) -> List[str]:
        """검색 결과가 없을 때 대안 검색어 제안 (Requirements 1.4)
        
        Args:
            query: 원본 검색 쿼리
            
        Returns:
            대안 검색어 목록
        """
        suggestions = []
        
        # 쿼리 단어 분리
        words = query.lower().split()
        
        # 개별 단어로 검색 제안
        if len(words) > 1:
            suggestions.append(f"개별 키워드로 검색: {', '.join(words[:3])}")
        
        # 일반적인 검색어 제안
        common_terms = ["테이블", "스키마", "컬럼", "데이터베이스"]
        for term in common_terms:
            if term not in query.lower():
                suggestions.append(f"'{term}' 키워드 추가하여 검색")
        
        # 와일드카드 검색 제안
        if len(words) > 0:
            suggestions.append(f"유사 테이블명 검색: {words[0]}*")
        
        return suggestions[:3]  # 최대 3개 제안
    
    # =========================================================================
    # Requirements 1.5: 스키마 정보 구조화 추출
    # =========================================================================
    
    def extract_schema_info(self, documents: List[SearchResult]) -> List[SchemaInfo]:
        """스키마 문서에서 구조화된 정보 추출 (Requirements 1.5)
        
        Args:
            documents: 검색된 문서 목록
            
        Returns:
            구조화된 스키마 정보 목록
        """
        schema_infos = []
        
        for doc in documents:
            schema_info = self._parse_markdown_schema(doc)
            if schema_info:
                schema_infos.append(schema_info)
        
        return schema_infos
    
    def _parse_markdown_schema(self, doc: SearchResult) -> Optional[SchemaInfo]:
        """Markdown 형식의 스키마 문서 파싱 (Requirements 1.5)"""
        content = doc.content
        metadata = doc.metadata
        
        # 테이블명과 데이터베이스명 추출
        table_name = metadata.get("table_name", "")
        database = metadata.get("database", "")
        
        # 컬럼 정보 파싱
        columns = self._parse_columns_from_content(content)
        
        # 비즈니스 로직 추출
        business_logic = metadata.get("business_logic", "")
        
        return SchemaInfo(
            table_name=table_name,
            database=database,
            columns=columns,
            description=content[:500] if content else "",
            business_logic=business_logic,
            source_doc=doc.source
        )

    
    def _parse_columns_from_content(self, content: str) -> List[ColumnDetail]:
        """컨텐츠에서 컬럼 정보 파싱"""
        columns = []
        
        # Markdown 테이블 형식 파싱 시도
        # | Column | Alias | Type | Description |
        lines = content.split("\n")
        in_table = False
        header_found = False
        
        for line in lines:
            line = line.strip()
            
            # 테이블 헤더 감지
            if "|" in line and "Column" in line:
                in_table = True
                header_found = True
                continue
            
            # 구분선 스킵
            if in_table and line.startswith("|") and "-" in line:
                continue
            
            # 테이블 행 파싱
            if in_table and line.startswith("|"):
                parts = [p.strip() for p in line.split("|")]
                parts = [p for p in parts if p]  # 빈 문자열 제거
                
                if len(parts) >= 2:
                    col = ColumnDetail(
                        name=parts[0] if len(parts) > 0 else "",
                        alias=parts[1] if len(parts) > 1 else "",
                        type=parts[2] if len(parts) > 2 else "string",
                        description=parts[3] if len(parts) > 3 else ""
                    )
                    columns.append(col)
            
            # 테이블 종료 감지
            if in_table and header_found and not line.startswith("|") and line:
                in_table = False
        
        return columns
    
    # =========================================================================
    # Requirements 2.1, 2.2, 2.3: 도메인 지식 검색
    # =========================================================================
    
    def search_domain_knowledge(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K
    ) -> List[SearchResult]:
        """도메인 지식 검색 (Requirements 2.1)
        
        비즈니스 용어, 메트릭 정의, 값 설명 등을 검색합니다.
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 최대 문서 수
            
        Returns:
            도메인 지식 검색 결과
        """
        return self.search_documents(query, doc_type="domain", top_k=top_k)
    
    def extract_domain_mappings(
        self,
        documents: List[SearchResult]
    ) -> List[DomainMapping]:
        """도메인 용어 매핑 추출 (Requirements 2.3)
        
        Args:
            documents: 검색된 도메인 문서
            
        Returns:
            도메인 용어 매핑 목록
        """
        mappings = []
        
        for doc in documents:
            # 비즈니스 로직 섹션에서 매핑 추출
            business_logic = doc.metadata.get("business_logic", "")
            table_name = doc.metadata.get("table_name", "")
            
            # 간단한 매핑 추출 (실제로는 더 정교한 파싱 필요)
            if business_logic:
                mapping = DomainMapping(
                    business_term=doc.content[:50] if doc.content else "",
                    database_column=table_name,
                    table=doc.metadata.get("database", ""),
                    description=business_logic[:200] if business_logic else ""
                )
                mappings.append(mapping)
        
        return mappings

    
    # =========================================================================
    # Requirements 3.4: 공유 컨텍스트 저장
    # =========================================================================
    
    def save_to_context(
        self,
        context: AnalysisContext,
        schema_results: Optional[List[SearchResult]] = None,
        domain_results: Optional[List[SearchResult]] = None
    ) -> AnalysisContext:
        """검색 결과를 공유 컨텍스트에 저장 (Requirements 3.4)
        
        Args:
            context: 분석 컨텍스트
            schema_results: 스키마 검색 결과
            domain_results: 도메인 지식 검색 결과
            
        Returns:
            업데이트된 컨텍스트
        """
        # AnalysisContext에 RAG 결과 필드가 있다면 저장
        # 현재 AnalysisContext에는 해당 필드가 없으므로 
        # business_intent에 임시 저장
        rag_data = {}
        
        if schema_results:
            rag_data["rag_schema_results"] = [
                {
                    "content": r.content[:500],
                    "score": r.score,
                    "metadata": r.metadata,
                    "source": r.source
                }
                for r in schema_results
            ]
        
        if domain_results:
            rag_data["rag_domain_results"] = [
                {
                    "content": r.content[:500],
                    "score": r.score,
                    "metadata": r.metadata,
                    "source": r.source
                }
                for r in domain_results
            ]
        
        if rag_data:
            context.business_intent["rag_data"] = rag_data
        
        return context
    
    # =========================================================================
    # Requirements 3.5: 실패 시 graceful degradation
    # =========================================================================
    
    def is_rag_enabled(self) -> bool:
        """RAG 활성화 상태 확인 (Requirements 3.5)"""
        return self._rag_enabled
    
    def disable_rag(self, reason: str = "") -> None:
        """RAG 비활성화 (Requirements 3.5)
        
        Args:
            reason: 비활성화 사유
        """
        self._rag_enabled = False
        logger.warning(f"RAG 비활성화: {reason}")
    
    def enable_rag(self) -> bool:
        """RAG 활성화 시도 (Requirements 3.5)
        
        Returns:
            활성화 성공 여부
        """
        # 클라이언트 재초기화 시도
        self._init_clients()
        return self._rag_enabled

    
    # =========================================================================
    # 통합 메서드
    # =========================================================================
    
    def _build_prompt_from_context(self, context: AnalysisContext) -> str:
        """컨텍스트를 기반으로 RAG Agent 프롬프트 생성"""
        prompt_parts = [
            f"문서 검색 요청:",
            f"사용자 쿼리: {context.user_query}",
        ]
        
        if context.business_intent:
            intent_parts = []
            for key, value in context.business_intent.items():
                if value and key != "rag_data":
                    intent_parts.append(f"- {key}: {value}")
            if intent_parts:
                prompt_parts.append("파악된 의도:\n" + "\n".join(intent_parts))
        
        prompt_parts.extend([
            "",
            "수행할 작업:",
            "1. 사용자 쿼리와 관련된 스키마 문서 검색",
            "2. 도메인 지식 및 비즈니스 용어 매핑 검색",
            "3. 검색 결과를 구조화하여 반환",
            "",
            "지금 문서 검색을 시작하세요."
        ])
        
        return "\n".join(prompt_parts)
    
    def search_and_extract(
        self,
        query: str,
        context: Optional[AnalysisContext] = None
    ) -> Dict[str, Any]:
        """검색 및 정보 추출 통합 메서드
        
        Args:
            query: 검색 쿼리
            context: 분석 컨텍스트 (선택)
            
        Returns:
            검색 결과 및 추출된 정보
        """
        result = {
            "success": True,
            "rag_enabled": self._rag_enabled,
            "schema_results": [],
            "domain_results": [],
            "schema_infos": [],
            "domain_mappings": [],
            "suggestions": []
        }
        
        if not self._rag_enabled:
            result["success"] = False
            result["error"] = "RAG가 비활성화되어 있습니다"
            return result
        
        try:
            # 스키마 문서 검색
            schema_results = self.search_documents(query, doc_type="schema")
            result["schema_results"] = schema_results
            
            # 스키마 정보 추출
            if schema_results:
                result["schema_infos"] = self.extract_schema_info(schema_results)
            else:
                result["suggestions"] = self.get_alternative_suggestions(query)
            
            # 도메인 지식 검색
            domain_results = self.search_domain_knowledge(query)
            result["domain_results"] = domain_results
            
            # 도메인 매핑 추출
            if domain_results:
                result["domain_mappings"] = self.extract_domain_mappings(domain_results)
            
            # 컨텍스트에 저장
            if context:
                self.save_to_context(context, schema_results, domain_results)
            
        except Exception as e:
            logger.error(f"검색 및 추출 실패: {e}")
            result["success"] = False
            result["error"] = str(e)
        
        return result

    
    def format_results_for_agent(
        self,
        schema_infos: List[SchemaInfo],
        domain_mappings: List[DomainMapping]
    ) -> str:
        """다른 에이전트에게 전달할 형식으로 결과 포맷팅
        
        Args:
            schema_infos: 스키마 정보 목록
            domain_mappings: 도메인 매핑 목록
            
        Returns:
            포맷팅된 문자열
        """
        lines = []
        
        # 스키마 정보
        if schema_infos:
            lines.append("## 검색된 스키마 정보")
            for i, info in enumerate(schema_infos[:5], 1):
                lines.append(f"\n### {i}. {info.database}.{info.table_name}")
                if info.columns:
                    col_strs = [f"{c.name} ({c.type})" for c in info.columns[:10]]
                    lines.append(f"컬럼: {', '.join(col_strs)}")
                if info.business_logic:
                    lines.append(f"비즈니스 로직: {info.business_logic[:200]}")
                lines.append(f"출처: {info.source_doc}")
        
        # 도메인 매핑
        if domain_mappings:
            lines.append("\n## 도메인 용어 매핑")
            for mapping in domain_mappings[:5]:
                lines.append(
                    f"- {mapping.business_term} → "
                    f"{mapping.table}.{mapping.database_column}"
                )
                if mapping.description:
                    lines.append(f"  설명: {mapping.description[:100]}")
        
        return "\n".join(lines) if lines else "검색 결과가 없습니다."
    
    def get_status(self) -> Dict[str, Any]:
        """RAG Agent 상태 정보 반환
        
        Returns:
            상태 정보 딕셔너리
        """
        return {
            "rag_enabled": self._rag_enabled,
            "opensearch_endpoint": self.opensearch_endpoint,
            "opensearch_index": self.opensearch_index,
            "embedding_model": self.embedding_model,
            "opensearch_connected": self._opensearch_client is not None,
            "bedrock_connected": self._bedrock_client is not None,
            "embedding_cache_size": len(self._embedding_cache)
        }
    
    def clear_embedding_cache(self) -> None:
        """임베딩 캐시 초기화"""
        self._embedding_cache.clear()
        logger.info("임베딩 캐시 초기화 완료")
