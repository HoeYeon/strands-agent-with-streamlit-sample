"""공유 컨텍스트 및 데이터 모델

멀티에이전트 시스템에서 사용되는 공유 데이터 구조와 설정을 정의합니다.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ColumnInfo:
    """테이블 컬럼 정보"""
    name: str
    type: str
    description: Optional[str] = None


@dataclass
class TableInfo:
    """테이블 메타데이터 정보"""
    database: str
    table: str
    columns: List[ColumnInfo] = field(default_factory=list)
    partition_keys: List[str] = field(default_factory=list)
    relevance_score: float = 0.0


@dataclass
class AnalysisContext:
    """에이전트 간 공유되는 분석 컨텍스트"""
    user_query: str = ""
    business_intent: Dict[str, Any] = field(default_factory=dict)  # entity, metric, time, action
    identified_tables: List[TableInfo] = field(default_factory=list)
    generated_sql: Optional[str] = None
    query_execution_id: Optional[str] = None
    results: Optional[List[Dict]] = None
    error_messages: List[str] = field(default_factory=list)
    
    # RAG 관련 필드 (Requirements 3.4)
    rag_schema_results: List[Dict[str, Any]] = field(default_factory=list)
    rag_domain_results: List[Dict[str, Any]] = field(default_factory=list)
    rag_enabled: bool = True  # RAG 사용 여부
    
    def add_error(self, message: str):
        """에러 메시지 추가"""
        self.error_messages.append(message)
    
    def clear_errors(self):
        """에러 메시지 초기화"""
        self.error_messages.clear()
    
    def add_rag_schema_result(self, result: Dict[str, Any]) -> None:
        """RAG 스키마 검색 결과 추가 (Requirements 3.4)"""
        self.rag_schema_results.append(result)
    
    def add_rag_domain_result(self, result: Dict[str, Any]) -> None:
        """RAG 도메인 검색 결과 추가 (Requirements 3.4)"""
        self.rag_domain_results.append(result)
    
    def clear_rag_results(self) -> None:
        """RAG 검색 결과 초기화"""
        self.rag_schema_results.clear()
        self.rag_domain_results.clear()


@dataclass
class SwarmConfig:
    """Swarm 설정"""
    max_handoffs: int = 20
    max_iterations: int = 20
    execution_timeout: float = 900.0  # 15분
    node_timeout: float = 300.0       # 5분
    repetitive_handoff_detection_window: int = 8
    repetitive_handoff_min_unique_agents: int = 3