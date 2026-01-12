"""Multi-Agent Text2SQL System

이 패키지는 Strands Swarm 패턴을 사용하여 전문화된 에이전트들이 협업하는
멀티에이전트 text2sql 시스템을 구현합니다.

Components:
- Lead Agent: 중앙 조정자
- Data Expert Agent: 데이터 카탈로그 탐색 전문가  
- SQL Agent: 쿼리 생성/실행 전문가
- RAG Agent: 문서 검색 전문가 (OpenSearch 벡터 검색)
"""

from .lead_agent import LeadAgent
from .data_expert_agent import DataExpertAgent
from .sql_agent import SQLAgent
from .rag_agent import (
    RAGAgent,
    SearchResult,
    SchemaInfo,
    ColumnDetail,
    DomainMapping,
)
from .multi_agent_text2sql import MultiAgentText2SQL
from .shared_context import AnalysisContext, TableInfo, ColumnInfo, SwarmConfig
from .event_adapter import (
    SwarmEventAdapter,
    SwarmEventHandler,
    StreamlitSwarmUIHandler,
    SwarmEventType,
    StreamlitEventType,
    AgentStatusInfo,
    SwarmEventAdapterState,
)

__all__ = [
    "LeadAgent",
    "DataExpertAgent", 
    "SQLAgent",
    "RAGAgent",
    "MultiAgentText2SQL",
    "AnalysisContext",
    "TableInfo",
    "ColumnInfo",
    "SwarmConfig",
    # RAG Agent 관련
    "SearchResult",
    "SchemaInfo",
    "ColumnDetail",
    "DomainMapping",
    # Event Adapter
    "SwarmEventAdapter",
    "SwarmEventHandler",
    "StreamlitSwarmUIHandler",
    "SwarmEventType",
    "StreamlitEventType",
    "AgentStatusInfo",
    "SwarmEventAdapterState",
]