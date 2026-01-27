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

import hashlib
import json
import logging
import signal
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

from strands import Agent
from strands.tools import tool

from .base_agent import BaseMultiAgent
from .shared_context import AnalysisContext

logger = logging.getLogger(__name__)


# =============================================================================
# 타임아웃 처리 유틸리티
# =============================================================================

T = TypeVar('T')


class TimeoutError(Exception):
    """타임아웃 예외"""
    pass


@contextmanager
def timeout_context(seconds: int, error_message: str = "작업 타임아웃"):
    """타임아웃 컨텍스트 매니저 (Requirements 1.4)

    지정된 시간 내에 작업이 완료되지 않으면 TimeoutError를 발생시킵니다.
    Unix 시스템의 메인 스레드에서만 동작합니다 (signal.SIGALRM 사용).

    Args:
        seconds: 타임아웃 시간 (초)
        error_message: 타임아웃 시 에러 메시지

    Yields:
        None

    Raises:
        TimeoutError: 타임아웃 발생 시
    """
    def timeout_handler(signum, frame):
        raise TimeoutError(error_message)

    # Unix 시스템의 메인 스레드에서만 signal 사용 가능
    try:
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    except (AttributeError, ValueError):
        # AttributeError: Windows 등 signal.SIGALRM이 없는 시스템
        # ValueError: 메인 스레드가 아닌 경우 (Streamlit 등)
        # 타임아웃 없이 실행
        yield


def with_timeout(
    func: Callable[..., T],
    timeout_seconds: int,
    *args,
    **kwargs
) -> Tuple[Optional[T], Optional[str]]:
    """함수 실행에 타임아웃 적용 (Requirements 1.4)
    
    Args:
        func: 실행할 함수
        timeout_seconds: 타임아웃 시간 (초)
        *args: 함수 인자
        **kwargs: 함수 키워드 인자
        
    Returns:
        (결과, 에러 메시지) 튜플
    """
    try:
        with timeout_context(timeout_seconds, f"{func.__name__} 타임아웃 ({timeout_seconds}초)"):
            result = func(*args, **kwargs)
            return result, None
    except TimeoutError as e:
        error_msg = str(e)
        logger.error(error_msg)
        return None, error_msg
    except Exception as e:
        error_msg = f"{func.__name__} 실행 실패: {type(e).__name__} - {str(e)}"
        logger.error(error_msg)
        return None, error_msg


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
RETRY_BACKOFF_MULTIPLIER = 2.0  # 지수 백오프 배수

# 캐시 설정
EMBEDDING_CACHE_MAX_SIZE = 100  # 최대 캐시 항목 수
EMBEDDING_CACHE_TTL_SECONDS = 3600  # 캐시 TTL (1시간)


# =============================================================================
# 데이터 모델
# =============================================================================

@dataclass
class EmbeddingCacheEntry:
    """임베딩 캐시 엔트리 (Requirements 1.1)
    
    캐시된 임베딩 벡터와 메타데이터를 저장합니다.
    """
    embedding: List[float]
    created_at: float  # timestamp
    text_hash: str  # 원본 텍스트의 해시


class LRUEmbeddingCache:
    """LRU 기반 임베딩 캐시 (Requirements 1.1)
    
    최근 사용된 임베딩을 캐시하여 중복 API 호출을 방지합니다.
    TTL과 최대 크기 제한을 지원합니다.
    """
    
    def __init__(
        self,
        max_size: int = EMBEDDING_CACHE_MAX_SIZE,
        ttl_seconds: int = EMBEDDING_CACHE_TTL_SECONDS
    ):
        """캐시 초기화
        
        Args:
            max_size: 최대 캐시 항목 수
            ttl_seconds: 캐시 항목 TTL (초)
        """
        self._cache: OrderedDict[str, EmbeddingCacheEntry] = OrderedDict()
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._hits = 0
        self._misses = 0
    
    def _compute_hash(self, text: str) -> str:
        """텍스트의 해시 계산"""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()[:32]
    
    def get(self, text: str) -> Optional[List[float]]:
        """캐시에서 임베딩 조회
        
        Args:
            text: 원본 텍스트
            
        Returns:
            캐시된 임베딩 또는 None
        """
        text_hash = self._compute_hash(text)
        
        if text_hash not in self._cache:
            self._misses += 1
            return None
        
        entry = self._cache[text_hash]
        
        # TTL 확인
        if time.time() - entry.created_at > self._ttl_seconds:
            del self._cache[text_hash]
            self._misses += 1
            return None
        
        # LRU: 최근 사용으로 이동
        self._cache.move_to_end(text_hash)
        self._hits += 1
        
        return entry.embedding
    
    def put(self, text: str, embedding: List[float]) -> None:
        """캐시에 임베딩 저장
        
        Args:
            text: 원본 텍스트
            embedding: 임베딩 벡터
        """
        text_hash = self._compute_hash(text)
        
        # 이미 존재하면 업데이트
        if text_hash in self._cache:
            self._cache.move_to_end(text_hash)
            self._cache[text_hash] = EmbeddingCacheEntry(
                embedding=embedding,
                created_at=time.time(),
                text_hash=text_hash
            )
            return
        
        # 최대 크기 초과 시 가장 오래된 항목 제거
        while len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)
        
        # 새 항목 추가
        self._cache[text_hash] = EmbeddingCacheEntry(
            embedding=embedding,
            created_at=time.time(),
            text_hash=text_hash
        )
    
    def clear(self) -> None:
        """캐시 초기화"""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """캐시 통계 반환"""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "ttl_seconds": self._ttl_seconds
        }
    
    def __len__(self) -> int:
        return len(self._cache)


@dataclass
class SearchResult:
    """벡터 검색 결과 (Requirements 1.3, 2.5)"""
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = ""  # 문서 출처


# 아래 데이터 모델들은 더 이상 사용되지 않습니다.
# OpenSearch에 이미 구조화된 형태로 저장되어 있어 별도 파싱이 불필요합니다.
# 
# @dataclass
# class ColumnDetail:
#     """컬럼 상세 정보 (Requirements 1.5)"""
#     name: str
#     type: str
#     description: str = ""
#     alias: str = ""
#     examples: List[str] = field(default_factory=list)
# 
# 
# @dataclass
# class SchemaInfo:
#     """스키마 문서에서 추출한 정보 (Requirements 1.5)"""
#     table_name: str
#     database: str
#     columns: List[ColumnDetail] = field(default_factory=list)
#     description: str = ""
#     business_logic: str = ""
#     source_doc: str = ""
# 
# 
# @dataclass
# class DomainMapping:
#     """도메인 용어 매핑 (Requirements 2.3)"""
#     business_term: str
#     database_column: str
#     table: str
#     description: str = ""
#     usage_frequency: int = 0


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
        
        # LRU 임베딩 캐시 초기화 (Requirements 1.1)
        self._embedding_cache = LRUEmbeddingCache(
            max_size=EMBEDDING_CACHE_MAX_SIZE,
            ttl_seconds=EMBEDDING_CACHE_TTL_SECONDS
        )
        
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

                # localhost 터널 사용 시 SSL 인증서 검증 비활성화
                is_localhost = host in ("localhost", "127.0.0.1")

                self._opensearch_client = OpenSearch(
                    hosts=[{"host": host, "port": port}],
                    http_auth=(self.opensearch_username, self.opensearch_password),
                    use_ssl=True,
                    verify_certs=not is_localhost,  # localhost면 검증 비활성화
                    ssl_show_warn=False,
                    connection_class=RequestsHttpConnection,
                    timeout=OPENSEARCH_TIMEOUT
                )
                logger.info(f"OpenSearch 클라이언트 초기화 성공: {host}:{port} (verify_certs={not is_localhost})")
            except ImportError as e:
                logger.warning(f"OpenSearch 라이브러리 없음: {e}")
                self._rag_enabled = False
            except Exception as e:
                logger.warning(f"OpenSearch 클라이언트 초기화 실패: {e}")
                self._rag_enabled = False
        else:
            logger.info("OpenSearch 설정 미완료 (endpoint, username, password 필요) - RAG 비활성화")
            self._rag_enabled = False

    
    def _create_search_tools(self) -> List:
        """RAG 검색 도구 생성 (Requirements 3.1)

        @tool 데코레이터를 사용하여 Agent가 호출할 수 있는 검색 도구를 생성합니다.
        기존 search_and_extract 메서드를 활용합니다.
        """
        # self를 클로저로 캡처
        rag_agent = self

        @tool
        def search_rag_documents(query: str) -> str:
            """RAG 문서 통합 검색 도구 (Requirements 1.2, 2.1)

            스키마 문서와 도메인 지식을 동시에 검색합니다.
            - 스키마 문서: 테이블/컬럼 정보, 데이터 타입, 설명
            - 도메인 지식: 비즈니스 용어 매핑, 메트릭 정의, 계산 공식

            Args:
                query: 검색할 쿼리 (예: "매출", "사용자 정보", "주문 테이블", "활성 사용자")

            Returns:
                검색된 스키마 문서와 도메인 지식 정보
            """
            # 기존 search_and_extract 메서드 활용
            result = rag_agent.search_and_extract(query)

            if not result["success"]:
                errors = ", ".join(result["errors"]) if result["errors"] else "알 수 없는 오류"
                return f"검색 실패: {errors}"

            output_parts = []

            # 스키마 검색 결과 포맷팅
            if result["schema_results"]:
                schema_text = rag_agent.format_results_for_agent(result["schema_results"])
                output_parts.append("## 스키마 문서 검색 결과\n" + schema_text)

            # 도메인 검색 결과 포맷팅
            if result["domain_results"]:
                domain_text = rag_agent.format_results_for_agent(result["domain_results"])
                output_parts.append("## 도메인 지식 검색 결과\n" + domain_text)

            # 결과가 없는 경우 대안 제안
            if not result["schema_results"] and not result["domain_results"]:
                if result["suggestions"]:
                    suggestions_text = "\n".join([
                        f"- {s['suggestion']}" for s in result["suggestions"][:3]
                    ])
                    return f"'{query}'에 대한 검색 결과가 없습니다.\n\n대안 제안:\n{suggestions_text}"
                return f"'{query}'에 대한 검색 결과가 없습니다."

            return "\n\n".join(output_parts)

        return [search_rag_documents]

    def _setup_agent(self) -> None:
        """RAG Agent 초기화 (Requirements 3.1)"""
        # RAG 검색 도구 생성
        search_tools = self._create_search_tools()

        # 기존 도구와 검색 도구 병합
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
    
    def generate_embedding(
        self,
        text: str,
        use_cache: bool = True
    ) -> Optional[List[float]]:
        """텍스트를 임베딩 벡터로 변환 (Requirements 1.1)
        
        AWS Bedrock Titan Embeddings V2를 사용하여 텍스트를 
        1024차원 벡터로 변환합니다. 재시도 로직과 캐싱을 지원합니다.
        
        Args:
            text: 변환할 텍스트
            use_cache: 캐시 사용 여부 (기본 True)
            
        Returns:
            임베딩 벡터 (1024차원) 또는 None (실패 시)
        """
        if not text or not text.strip():
            logger.warning("빈 텍스트 - 임베딩 생성 불가")
            return None
        
        if not self._bedrock_client:
            logger.warning("Bedrock 클라이언트 없음 - 임베딩 생성 불가")
            return None
        
        # 캐시 확인
        if use_cache:
            cached_embedding = self._embedding_cache.get(text)
            if cached_embedding is not None:
                logger.debug(f"캐시 히트: 텍스트 길이 {len(text)}")
                return cached_embedding
        
        # 재시도 로직으로 임베딩 생성
        embedding = self._generate_embedding_with_retry(text)
        
        # 캐시에 저장
        if embedding and use_cache:
            self._embedding_cache.put(text, embedding)
        
        return embedding
    
    def _generate_embedding_with_retry(
        self,
        text: str
    ) -> Optional[List[float]]:
        """재시도 로직을 포함한 임베딩 생성 (Requirements 1.1)
        
        최대 3회 재시도하며, 지수 백오프를 적용합니다.
        
        Args:
            text: 변환할 텍스트
            
        Returns:
            임베딩 벡터 또는 None (실패 시)
        """
        last_error: Optional[Exception] = None
        delay = RETRY_DELAY_SECONDS
        
        for attempt in range(1, MAX_RETRY_ATTEMPTS + 1):
            try:
                embedding = self._invoke_bedrock_embedding(text)
                if embedding:
                    if attempt > 1:
                        logger.info(f"임베딩 생성 성공 (시도 {attempt}/{MAX_RETRY_ATTEMPTS})")
                    return embedding
                    
            except Exception as e:
                last_error = e
                error_type = type(e).__name__
                
                # 재시도 불가능한 에러 확인
                if self._is_non_retryable_error(e):
                    logger.error(f"재시도 불가능한 에러: {error_type} - {e}")
                    break
                
                # 마지막 시도가 아니면 대기 후 재시도
                if attempt < MAX_RETRY_ATTEMPTS:
                    logger.warning(
                        f"임베딩 생성 실패 (시도 {attempt}/{MAX_RETRY_ATTEMPTS}): "
                        f"{error_type} - {e}. {delay}초 후 재시도..."
                    )
                    time.sleep(delay)
                    delay *= RETRY_BACKOFF_MULTIPLIER  # 지수 백오프
                else:
                    logger.error(
                        f"임베딩 생성 최종 실패 (시도 {attempt}/{MAX_RETRY_ATTEMPTS}): "
                        f"{error_type} - {e}"
                    )
        
        return None
    
    def _invoke_bedrock_embedding(self, text: str) -> Optional[List[float]]:
        """Bedrock API를 호출하여 임베딩 생성 (Requirements 1.1)
        
        Args:
            text: 변환할 텍스트
            
        Returns:
            임베딩 벡터 또는 None
            
        Raises:
            Exception: API 호출 실패 시
        """
        # 텍스트 길이 검증 (최대 8192 토큰, 대략 4자당 1토큰)
        max_chars = MAX_EMBEDDING_TOKENS * 4
        if len(text) > max_chars:
            logger.warning(
                f"텍스트가 너무 깁니다 ({len(text)}자). "
                f"{max_chars}자로 잘라냅니다."
            )
            text = text[:max_chars]
        
        # Titan Embeddings V2 요청 본문
        body = json.dumps({
            "inputText": text,
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
        
        if not embedding:
            logger.warning("Bedrock 응답에 임베딩이 없습니다")
            return None
        
        # 차원 검증
        if len(embedding) != EMBEDDING_DIMENSION:
            logger.warning(
                f"임베딩 차원 불일치: 예상 {EMBEDDING_DIMENSION}, "
                f"실제 {len(embedding)}"
            )
        
        return embedding
    
    def _is_non_retryable_error(self, error: Exception) -> bool:
        """재시도 불가능한 에러인지 확인
        
        Args:
            error: 발생한 예외
            
        Returns:
            재시도 불가능 여부
        """
        error_str = str(error).lower()
        
        # 재시도 불가능한 에러 패턴
        non_retryable_patterns = [
            "validationexception",  # 잘못된 요청
            "accessdenied",  # 권한 없음
            "invalidparametervalue",  # 잘못된 파라미터
            "modelnotfound",  # 모델 없음
            "resourcenotfound",  # 리소스 없음
        ]
        
        for pattern in non_retryable_patterns:
            if pattern in error_str:
                return True
        
        return False
    
    def get_embedding_cache_stats(self) -> Dict[str, Any]:
        """임베딩 캐시 통계 반환
        
        Returns:
            캐시 통계 딕셔너리
        """
        return self._embedding_cache.get_stats()

    
    # =========================================================================
    # Requirements 1.2, 1.3: 벡터 검색 및 결과 반환
    # =========================================================================
    
    def search_documents(
        self,
        query: str,
        doc_type: str = "schema",
        top_k: int = DEFAULT_TOP_K,
        filters: Optional[Dict[str, str]] = None,
        timeout_seconds: Optional[int] = None
    ) -> Tuple[List[SearchResult], Optional[str]]:
        """OpenSearch에서 문서 검색 수행 (Requirements 1.2, 1.3, 1.4)
        
        벡터 유사도 검색을 수행하고 상위 k개 결과를 반환합니다.
        검색 실패 시 에러 메시지를 함께 반환합니다.
        타임아웃을 지원합니다.
        
        Args:
            query: 검색 쿼리
            doc_type: 문서 타입 ("schema" 또는 "domain")
            top_k: 반환할 최대 문서 수 (기본 10개)
            filters: 필터 조건 (database, table_name)
            timeout_seconds: 타임아웃 시간 (초, None이면 기본값 사용)
            
        Returns:
            (검색 결과 목록, 에러 메시지) 튜플
            - 검색 결과: 최대 10개
            - 에러 메시지: 실패 시 에러 설명, 성공 시 None
        """
        # RAG 비활성화 상태 확인 (Requirements 3.5)
        if not self._rag_enabled or not self._opensearch_client:
            error_msg = "RAG가 비활성화되어 있습니다. OpenSearch 연결을 확인하세요."
            logger.warning(error_msg)
            return [], error_msg
        
        # 상위 10개로 제한 (Requirements 1.2)
        top_k = min(top_k, DEFAULT_TOP_K)
        
        # 타임아웃 설정
        if timeout_seconds is None:
            timeout_seconds = OPENSEARCH_TIMEOUT
        
        # 타임아웃 적용하여 검색 수행 (Requirements 1.4)
        try:
            with timeout_context(
                timeout_seconds,
                f"검색 타임아웃 ({timeout_seconds}초 초과)"
            ):
                return self._search_documents_internal(query, doc_type, top_k, filters)
        except TimeoutError as e:
            error_msg = str(e)
            logger.error(error_msg)
            return [], error_msg
        except Exception as e:
            error_msg = f"검색 중 예외 발생: {type(e).__name__} - {str(e)}"
            logger.error(error_msg)
            return [], error_msg
    
    def _search_documents_internal(
        self,
        query: str,
        doc_type: str,
        top_k: int,
        filters: Optional[Dict[str, str]] = None
    ) -> Tuple[List[SearchResult], Optional[str]]:
        """내부 검색 로직 (타임아웃 컨텍스트 내에서 실행)"""
        # 쿼리 임베딩 생성 (Requirements 1.1)
        query_embedding = self.generate_embedding(query)
        if not query_embedding:
            error_msg = "쿼리 임베딩 생성에 실패했습니다. Bedrock 연결을 확인하세요."
            logger.warning(error_msg)
            return [], error_msg
        
        # 재시도 로직으로 검색 수행 (Requirements 1.4)
        results, error = self._search_with_retry(
            query_embedding, query, top_k, filters
        )
        
        # 빈 결과 처리 (Requirements 1.4)
        if not results and not error:
            logger.info(f"검색 결과 없음: '{query}'")
            # 에러는 아니지만 빈 결과임을 알림
            return [], None
        
        return results, error
    
    def _search_with_retry(
        self,
        query_embedding: List[float],
        query_text: str,
        top_k: int,
        filters: Optional[Dict[str, str]] = None
    ) -> Tuple[List[SearchResult], Optional[str]]:
        """재시도 로직을 포함한 OpenSearch 검색 (Requirements 1.4)
        
        최대 3회 재시도하며, 지수 백오프를 적용합니다.
        타임아웃 및 연결 에러를 처리합니다.
        
        Args:
            query_embedding: 쿼리 임베딩 벡터
            query_text: 원본 쿼리 텍스트
            top_k: 반환할 최대 문서 수
            filters: 필터 조건
            
        Returns:
            (검색 결과 목록, 에러 메시지) 튜플
        """
        last_error: Optional[Exception] = None
        delay = RETRY_DELAY_SECONDS
        
        for attempt in range(1, MAX_RETRY_ATTEMPTS + 1):
            try:
                # 하이브리드 검색 쿼리 구성
                search_query = self._build_search_query(
                    query_embedding, query_text, top_k, filters
                )
                
                # OpenSearch 검색 실행 (타임아웃 포함)
                response = self._opensearch_client.search(
                    index=self.opensearch_index,
                    body=search_query,
                    request_timeout=OPENSEARCH_TIMEOUT
                )
                
                # 결과 파싱 (Requirements 1.3)
                results = self._parse_search_results(response)
                
                if attempt > 1:
                    logger.info(f"검색 성공 (시도 {attempt}/{MAX_RETRY_ATTEMPTS})")
                
                return results, None
                
            except Exception as e:
                last_error = e
                error_type = type(e).__name__
                
                # 재시도 불가능한 에러 확인
                if self._is_search_non_retryable_error(e):
                    error_msg = f"검색 실패 ({error_type}): {str(e)}"
                    logger.error(error_msg)
                    return [], error_msg
                
                # 마지막 시도가 아니면 대기 후 재시도
                if attempt < MAX_RETRY_ATTEMPTS:
                    logger.warning(
                        f"검색 실패 (시도 {attempt}/{MAX_RETRY_ATTEMPTS}): "
                        f"{error_type} - {e}. {delay}초 후 재시도..."
                    )
                    time.sleep(delay)
                    delay *= RETRY_BACKOFF_MULTIPLIER  # 지수 백오프
                else:
                    error_msg = (
                        f"검색 최종 실패 (시도 {attempt}/{MAX_RETRY_ATTEMPTS}): "
                        f"{error_type} - {str(e)}"
                    )
                    logger.error(error_msg)
                    return [], error_msg
        
        # 여기 도달하면 모든 재시도 실패
        error_msg = f"검색 실패: {str(last_error)}" if last_error else "알 수 없는 오류"
        return [], error_msg
    
    def _is_search_non_retryable_error(self, error: Exception) -> bool:
        """재시도 불가능한 검색 에러인지 확인 (Requirements 1.4)
        
        Args:
            error: 발생한 예외
            
        Returns:
            재시도 불가능 여부
        """
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()
        
        # 재시도 불가능한 에러 패턴
        non_retryable_patterns = [
            "indexnotfound",  # 인덱스 없음
            "authenticationexception",  # 인증 실패
            "authorizationexception",  # 권한 없음
            "invalidindexname",  # 잘못된 인덱스명
            "illegal_argument_exception",  # 잘못된 인자
            "parsing_exception",  # 쿼리 파싱 실패
        ]
        
        for pattern in non_retryable_patterns:
            if pattern in error_str or pattern in error_type:
                return True
        
        return False

    
    def _build_search_query(
        self,
        query_embedding: List[float],
        query_text: str,
        top_k: int,
        filters: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """하이브리드 검색 쿼리 구성

        벡터 검색과 키워드 검색을 결합한 하이브리드 검색 쿼리를 생성합니다.
        indexer.py의 인덱스 구조에 맞춤:
        - embedding: knn_vector (1024차원)
        - content: text
        - table, database, chunk_type, column_name, source_file: keyword
        """
        # 기본 k-NN 쿼리 (필드명: embedding)
        knn_query = {
            "knn": {
                "embedding": {
                    "vector": query_embedding,
                    "k": top_k
                }
            }
        }

        # 키워드 검색 쿼리 (indexer.py 필드 구조에 맞춤)
        keyword_query = {
            "multi_match": {
                "query": query_text,
                "fields": ["content", "table", "database", "column_name"],
                "type": "best_fields",
                "fuzziness": "AUTO"
            }
        }
        
        # 하이브리드 쿼리 구성
        bool_query: Dict[str, Any] = {
            "should": [knn_query, keyword_query],
            "minimum_should_match": 1
        }
        
        # 필터 추가 (indexer.py 필드 구조에 맞춤)
        if filters:
            filter_clauses = []
            if "database" in filters:
                filter_clauses.append({"term": {"database": filters["database"]}})
            if "table" in filters:
                filter_clauses.append({"term": {"table": filters["table"]}})
            if filter_clauses:
                bool_query["filter"] = filter_clauses

        return {
            "size": top_k,
            "query": {"bool": bool_query},
            "_source": ["content", "table", "database", "chunk_type", "column_name", "source_file"]
        }
    
    def _parse_search_results(self, response: Dict[str, Any]) -> List[SearchResult]:
        """OpenSearch 응답을 SearchResult 목록으로 변환 (Requirements 1.3)

        indexer.py 필드 구조에 맞춤:
        - content, table, database, chunk_type, column_name, source_file
        """
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
                    "table": source.get("table", ""),
                    "database": source.get("database", ""),
                    "chunk_type": source.get("chunk_type", ""),
                    "column_name": source.get("column_name", "")
                },
                source=source.get("source_file", hit.get("_id", ""))
            )
            results.append(result)

        return results

    
    # =========================================================================
    # Requirements 1.4: 빈 결과 처리 및 대안 제안
    # =========================================================================
    
    def handle_empty_results(
        self,
        query: str,
        filters: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """검색 결과가 없을 때 대안 제안 (Requirements 1.4)
        
        빈 결과에 대한 상세한 분석과 대안 검색 전략을 제공합니다.
        
        Args:
            query: 원본 검색 쿼리
            filters: 적용된 필터 조건
            
        Returns:
            대안 제안 정보 딕셔너리
        """
        suggestions = self.get_alternative_suggestions(query, filters)
        
        return {
            "empty_results": True,
            "original_query": query,
            "applied_filters": filters or {},
            "suggestions": suggestions,
            "message": (
                f"'{query}'에 대한 검색 결과가 없습니다. "
                f"다음 대안을 시도해보세요."
            )
        }
    
    def get_alternative_suggestions(
        self,
        query: str,
        filters: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, str]]:
        """검색 결과가 없을 때 대안 검색어 제안 (Requirements 1.4)
        
        Args:
            query: 원본 검색 쿼리
            filters: 적용된 필터 조건
            
        Returns:
            대안 제안 목록 (각 제안은 type, suggestion, reason 포함)
        """
        suggestions = []
        
        # 쿼리 단어 분리
        words = query.lower().split()
        
        # 1. 필터 제거 제안
        if filters:
            suggestions.append({
                "type": "remove_filters",
                "suggestion": "필터 조건을 제거하고 검색",
                "reason": f"현재 필터: {', '.join(f'{k}={v}' for k, v in filters.items())}"
            })
        
        # 2. 개별 키워드 검색 제안
        if len(words) > 1:
            key_words = [w for w in words if len(w) > 2][:3]
            if key_words:
                suggestions.append({
                    "type": "split_keywords",
                    "suggestion": f"개별 키워드로 검색: {', '.join(key_words)}",
                    "reason": "복합 쿼리를 단순화하여 더 많은 결과 찾기"
                })
        
        # 3. 일반적인 검색어 추가 제안
        common_terms = {
            "테이블": "데이터베이스 테이블 정보 검색",
            "스키마": "데이터베이스 구조 정보 검색",
            "컬럼": "테이블 컬럼 정보 검색",
            "필드": "데이터 필드 정보 검색"
        }
        
        for term, reason in common_terms.items():
            if term not in query.lower():
                suggestions.append({
                    "type": "add_keyword",
                    "suggestion": f"'{term}' 키워드 추가: {query} {term}",
                    "reason": reason
                })
                if len(suggestions) >= 5:  # 최대 5개 제안
                    break
        
        # 4. 와일드카드 검색 제안
        if len(words) > 0 and len(words[0]) > 2:
            suggestions.append({
                "type": "wildcard",
                "suggestion": f"유사 이름 검색: {words[0]}*",
                "reason": "부분 일치로 관련 테이블/컬럼 찾기"
            })
        
        # 5. 전체 카탈로그 탐색 제안
        if not suggestions or len(suggestions) < 3:
            suggestions.append({
                "type": "browse_all",
                "suggestion": "전체 데이터베이스 카탈로그 탐색",
                "reason": "사용 가능한 모든 테이블 목록 확인"
            })
        
        return suggestions[:5]  # 최대 5개 제안
    
    # =========================================================================
    # Requirements 1.5: 스키마 정보 구조화 추출 (제거됨 - OpenSearch에 이미 구조화됨)
    # =========================================================================
    # 발견: indexer.py가 이미 청킹 및 구조화 수행
    # 결과: content 필드에 이미 파싱된 텍스트 포함
    # 메타데이터: table, database, chunk_type, column_name 필드로 구조화됨
    # 결론: LLM에 검색 결과를 그대로 전달하면 됨
    
    # =========================================================================
    # Requirements 2.1, 2.2, 2.3: 도메인 지식 검색
    # =========================================================================
    
    def search_domain_knowledge(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K
    ) -> Tuple[List[SearchResult], Optional[str]]:
        """도메인 지식 검색 (Requirements 2.1)
        
        비즈니스 용어, 메트릭 정의, 값 설명 등을 검색합니다.
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 최대 문서 수
            
        Returns:
            (도메인 지식 검색 결과, 에러 메시지) 튜플
        """
        return self.search_documents(query, doc_type="domain", top_k=top_k)
    
    def format_search_results_for_llm(
        self,
        search_results: List[SearchResult],
        max_results: int = 5,
        include_score: bool = True
    ) -> str:
        """검색 결과를 LLM 친화적 형식으로 변환 (Requirements 1.3, 2.5)
        
        검색 결과를 LLM이 이해하기 쉬운 마크다운 형식으로 변환합니다.
        content와 메타데이터를 그대로 사용하며, 관련도 점수를 포함합니다.
        
        Args:
            search_results: 검색 결과 목록
            max_results: 최대 표시할 결과 수 (기본 5개)
            include_score: 관련도 점수 포함 여부 (기본 True)
            
        Returns:
            LLM 친화적 형식의 문자열
        """
        if not search_results:
            return "검색 결과가 없습니다."
        
        lines = ["# 검색된 문서\n"]
        
        for i, result in enumerate(search_results[:max_results], 1):
            # 헤더: 순번 및 관련도 점수
            if include_score:
                lines.append(f"## {i}. 문서 (관련도: {result.score:.3f})\n")
            else:
                lines.append(f"## {i}. 문서\n")
            
            # 메타데이터 (indexer.py 필드 구조에 맞춤)
            metadata = result.metadata
            if metadata:
                lines.append("**메타데이터:**")
                if "database" in metadata and metadata["database"]:
                    lines.append(f"- 데이터베이스: `{metadata['database']}`")
                if "table" in metadata and metadata["table"]:
                    lines.append(f"- 테이블: `{metadata['table']}`")
                if "chunk_type" in metadata and metadata["chunk_type"]:
                    lines.append(f"- 청크 타입: `{metadata['chunk_type']}`")
                if "column_name" in metadata and metadata["column_name"]:
                    lines.append(f"- 컬럼: `{metadata['column_name']}`")
                lines.append("")
            
            # 문서 내용
            lines.append("**내용:**")
            lines.append(f"{result.content}\n")
            
            # 출처
            if result.source:
                lines.append(f"**출처:** `{result.source}`\n")
            
            lines.append("---\n")
        
        return "\n".join(lines)

    
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
        # 스키마 검색 결과를 context.add_rag_schema_result()로 저장
        if schema_results:
            for result in schema_results:
                result_dict = {
                    "content": result.content[:500],
                    "score": result.score,
                    "metadata": result.metadata,
                    "source": result.source
                }
                context.add_rag_schema_result(result_dict)
        
        # 도메인 검색 결과를 context.add_rag_domain_result()로 저장
        if domain_results:
            for result in domain_results:
                result_dict = {
                    "content": result.content[:500],
                    "score": result.score,
                    "metadata": result.metadata,
                    "source": result.source
                }
                context.add_rag_domain_result(result_dict)
        
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
        """검색 및 정보 추출 통합 메서드 (Requirements 1.4, 3.5)
        
        스키마 문서와 도메인 지식을 검색합니다.
        에러 발생 시 graceful degradation을 적용합니다.
        
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
            "suggestions": [],
            "errors": []
        }
        
        # RAG 비활성화 상태 확인 (Requirements 3.5)
        if not self._rag_enabled:
            result["success"] = False
            result["errors"].append("RAG가 비활성화되어 있습니다")
            logger.warning("RAG 비활성화 - 검색 건너뜀")
            return result
        
        try:
            # 스키마 문서 검색 (Requirements 1.2, 1.3, 1.4)
            schema_results, schema_error = self.search_documents(
                query, doc_type="schema"
            )
            
            if schema_error:
                result["errors"].append(f"스키마 검색 실패: {schema_error}")
                logger.error(f"스키마 검색 실패: {schema_error}")
            else:
                result["schema_results"] = schema_results
                
                if schema_results:
                    logger.info(f"스키마 문서 {len(schema_results)}개 검색")
                else:
                    # 빈 결과 처리 (Requirements 1.4)
                    empty_result = self.handle_empty_results(query)
                    result["suggestions"] = empty_result["suggestions"]
                    logger.info(f"스키마 검색 결과 없음 - {len(result['suggestions'])}개 대안 제안")
            
            # 도메인 지식 검색 (Requirements 2.1)
            domain_results, domain_error = self.search_domain_knowledge(query)
            
            if domain_error:
                result["errors"].append(f"도메인 검색 실패: {domain_error}")
                logger.error(f"도메인 검색 실패: {domain_error}")
            else:
                result["domain_results"] = domain_results
                
                if domain_results:
                    logger.info(f"도메인 문서 {len(domain_results)}개 검색")
            
            # 컨텍스트에 저장 (Requirements 3.4)
            if context:
                self.save_to_context(context, schema_results, domain_results)
                logger.debug("검색 결과를 공유 컨텍스트에 저장")
            
            # 전체 성공 여부 판단
            result["success"] = len(result["errors"]) == 0
            
        except Exception as e:
            # 예상치 못한 에러 처리 (Requirements 3.5)
            error_msg = f"검색 및 추출 중 예외 발생: {type(e).__name__} - {str(e)}"
            logger.error(error_msg, exc_info=True)
            result["success"] = False
            result["errors"].append(error_msg)
        
        return result

    
    def format_results_for_agent(
        self,
        search_results: List[SearchResult]
    ) -> str:
        """다른 에이전트에게 전달할 형식으로 결과 포맷팅
        
        format_search_results_for_llm()을 사용하여 LLM 친화적 형식으로 변환합니다.
        
        Args:
            search_results: 검색 결과 목록
            
        Returns:
            포맷팅된 문자열
        """
        return self.format_search_results_for_llm(search_results)
    
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
            "embedding_cache_stats": self._embedding_cache.get_stats()
        }
    
    def clear_embedding_cache(self) -> None:
        """임베딩 캐시 초기화"""
        self._embedding_cache.clear()
        logger.info("임베딩 캐시 초기화 완료")
