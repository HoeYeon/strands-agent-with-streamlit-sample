"""Multi-Agent Text2SQL System

이 패키지는 Strands Swarm 패턴을 사용하여 3개의 전문화된 에이전트가 협업하는
멀티에이전트 text2sql 시스템을 구현합니다.

Components:
- Lead Agent: 중앙 조정자
- Data Expert Agent: 데이터 카탈로그 탐색 전문가  
- SQL Agent: 쿼리 생성/실행 전문가
"""

from .lead_agent import LeadAgent
from .data_expert_agent import DataExpertAgent
from .sql_agent import SQLAgent
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
    "MultiAgentText2SQL",
    "AnalysisContext",
    "TableInfo",
    "ColumnInfo",
    "SwarmConfig",
    # Event Adapter
    "SwarmEventAdapter",
    "SwarmEventHandler",
    "StreamlitSwarmUIHandler",
    "SwarmEventType",
    "StreamlitEventType",
    "AgentStatusInfo",
    "SwarmEventAdapterState",
]