"""Lead Agent - 중앙 조정자 (LLM 기반)

사용자 요청을 받아 전체 워크플로우를 조정하고 다른 에이전트들에게 작업을 위임하는 중앙 조정자입니다.
LLM을 통해 사용자의 자연어 요청을 분석하고 적절한 에이전트에게 작업을 위임합니다.

Requirements:
- 1.1: 사용자 요청 분석 및 적절한 전문 에이전트에게 작업 위임
- 1.2: 모든 에이전트 결과 통합하여 최종 응답 제공
- 1.4: 에러 발생 시 명확한 오류 메시지와 다음 단계 제안
- 1.5: 현재 작업 중인 에이전트 상태 표시
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from strands import Agent

from .base_agent import BaseMultiAgent


class AgentType(Enum):
    """에이전트 타입 정의"""
    LEAD = "lead_agent"
    DATA_EXPERT = "data_expert"
    SQL = "sql_agent"
    RAG = "rag_agent"


class WorkflowStatus(Enum):
    """워크플로우 상태 정의"""
    IDLE = "idle"
    ANALYZING = "analyzing"
    DATA_EXPLORATION = "data_exploration"
    SQL_GENERATION = "sql_generation"
    SQL_EXECUTION = "sql_execution"
    INTEGRATING = "integrating"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class AgentResult:
    """개별 에이전트 작업 결과"""
    agent_type: AgentType
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    execution_time_ms: Optional[int] = None


@dataclass
class WorkflowState:
    """워크플로우 상태 관리 (Requirements 1.5)"""
    status: WorkflowStatus = WorkflowStatus.IDLE
    current_agent: Optional[AgentType] = None
    agent_results: List[AgentResult] = field(default_factory=list)
    progress_messages: List[str] = field(default_factory=list)
    
    def update_status(self, status: WorkflowStatus, agent: Optional[AgentType] = None):
        """상태 업데이트 및 진행 메시지 추가"""
        self.status = status
        self.current_agent = agent
        
        status_messages = {
            WorkflowStatus.ANALYZING: "사용자 요청을 분석하고 있습니다...",
            WorkflowStatus.DATA_EXPLORATION: "Data Expert Agent가 데이터 카탈로그를 탐색하고 있습니다...",
            WorkflowStatus.SQL_GENERATION: "SQL Agent가 쿼리를 생성하고 있습니다...",
            WorkflowStatus.SQL_EXECUTION: "SQL Agent가 쿼리를 실행하고 있습니다...",
            WorkflowStatus.INTEGRATING: "결과를 통합하고 있습니다...",
            WorkflowStatus.COMPLETED: "작업이 완료되었습니다.",
            WorkflowStatus.ERROR: "오류가 발생했습니다."
        }
        
        if status in status_messages:
            self.progress_messages.append(status_messages[status])
    
    def add_result(self, result: AgentResult):
        """에이전트 결과 추가"""
        self.agent_results.append(result)
    
    def get_current_status_message(self) -> str:
        """현재 상태 메시지 반환 (Requirements 1.5)"""
        if self.current_agent:
            agent_names = {
                AgentType.LEAD: "Lead Agent",
                AgentType.DATA_EXPERT: "Data Expert Agent",
                AgentType.SQL: "SQL Agent",
                AgentType.RAG: "RAG Agent"
            }
            return f"현재 작업 중: {agent_names.get(self.current_agent, 'Unknown')}"
        return f"상태: {self.status.value}"





class LeadAgent(BaseMultiAgent):
    """Lead Agent - 중앙 조정자 (LLM 기반)
    
    역할:
    - LLM을 통한 사용자 자연어 요청 분석 및 의도 파악 (Requirements 1.1)
    - 적절한 전문 에이전트로 작업 위임 (Requirements 1.1)
    - 에이전트 간 협업 조정
    - 최종 결과 통합 및 사용자 응답 생성 (Requirements 1.2)
    - 오류 처리 및 상태 관리 (Requirements 1.4, 1.5)
    """
    
    def __init__(self, model_id: str, tools: Optional[List] = None):
        self.workflow_state = WorkflowState()
        super().__init__(model_id, tools)
    
    def _setup_agent(self):
        """Lead Agent 초기화"""
        self.agent = Agent(
            name="lead_agent",
            system_prompt=self.get_system_prompt(),
            model=self.model_id,
            tools=self.tools if self.tools else None
        )
    
    def get_system_prompt(self) -> str:
        """Lead Agent 시스템 프롬프트"""
        return """
역할: 멀티에이전트 Text2SQL 시스템 중앙 조정자

협업 에이전트:
- rag_agent: 스키마 문서 및 도메인 지식 검색 (기본 활성화)
- data_expert: 데이터 카탈로그 탐색, 테이블 식별
- sql_agent: SQL 생성 및 Athena 실행

────────────────────────────────────────────
기본 워크플로우
────────────────────────────────────────────
**데이터 분석 요청** (SQL 필요):
1. rag_agent → 도메인 지식/스키마 문서 검색
2. data_expert → 테이블 식별 (RAG 결과 활용)
3. sql_agent → SQL 생성 및 실행

**정보 조회 요청** (SQL 불필요):
- 키워드: "테이블 목록", "스키마", "컬럼 정보", "어떤 데이터"
- 처리: data_expert → 결과 받으면 직접 응답 (RAG/sql_agent 생략 가능)

────────────────────────────────────────────
RAG 우선 원칙
────────────────────────────────────────────
**RAG를 먼저 호출하는 이유:**
- 비즈니스 용어 → DB 컬럼 매핑 (예: "활성 사용자" → status='active')
- 메트릭 정의 확인 (예: "매출" → gross_sales vs net_sales)
- 컬럼 값 의미 파악 (예: status='A'의 의미)
- 테이블 간 관계 및 조인 조건 파악

**RAG 생략 가능한 경우:**
- 단순 테이블/컬럼 목록 조회
- 이전 대화에서 이미 도메인 지식 확보
- 사용자가 명시적으로 테이블/컬럼명 제공

**RAG 실패 시:**
- 오류 로그 기록 후 기존 워크플로우 계속 진행
- data_expert → sql_agent 순서로 fallback
- 사용자에게 "도메인 지식 없이 진행" 알림

────────────────────────────────────────────
handoff 규칙 (중요: 원문 질문 필수 포함!)
────────────────────────────────────────────
rag_agent 위임 (기본):
  handoff_to_agent(agent_name="rag_agent", message="
    [원문 질문] {사용자의 원래 질문 그대로 복사}
    [검색 키워드] {핵심 비즈니스 용어}, {엔티티/메트릭}
  ")

data_expert 위임:
  handoff_to_agent(agent_name="data_expert", message="
    [원문 질문] {사용자의 원래 질문 그대로 복사}
    [RAG 결과] {발견된 도메인 지식 요약}
    [탐색 요청] {필요한 테이블/컬럼}
  ")

sql_agent 위임:
  handoff_to_agent(agent_name="sql_agent", message="
    [원문 질문] {사용자의 원래 질문 그대로 복사}
    [도메인 지식] {비즈니스 용어}: {컬럼} = '{값}'
    [테이블 정보] {database}.{table}
  ")

⚠️ 금지:
- 동일 에이전트 2회 연속 위임
- 정보 조회 요청을 sql_agent로 위임

────────────────────────────────────────────
응답 규칙
────────────────────────────────────────────
정보 조회 결과 ("[정보 조회 완료]" 수신 시):
- 테이블/컬럼 정보를 사용자 친화적으로 정리
- 가능한 분석 예시 제안

데이터 분석 결과:
- 실행된 SQL 표시
- 결과 데이터 요약
- 추가 분석 제안

오류 발생 시:
1. 무엇이 실패했는지
2. 가능한 원인
3. 사용자가 할 수 있는 조치
"""
    
    def get_tools(self) -> List:
        """Lead Agent 도구 목록"""
        # handoff_to_agent 도구는 Swarm에서 자동으로 제공됨
        return self.tools
    
    def get_agent(self) -> Agent:
        """Swarm에서 사용할 Agent 인스턴스 반환"""
        return self.agent

    # =========================================================================
    # Requirements 1.5: 작업 상태 표시
    # =========================================================================

    def get_current_status(self) -> Dict[str, Any]:
        """현재 작업 상태 정보 반환 (Requirements 1.5)
        
        Returns:
            상태 정보 딕셔너리:
            - status: 현재 워크플로우 상태
            - current_agent: 현재 작업 중인 에이전트
            - message: 상태 메시지
            - progress: 진행 메시지 목록
        """
        return {
            "status": self.workflow_state.status.value,
            "current_agent": (
                self.workflow_state.current_agent.value 
                if self.workflow_state.current_agent else None
            ),
            "message": self.workflow_state.get_current_status_message(),
            "progress": self.workflow_state.progress_messages.copy()
        }
    
    def update_agent_status(
        self, 
        agent_type: AgentType, 
        status: WorkflowStatus
    ) -> None:
        """에이전트 작업 상태 업데이트 (Requirements 1.5)
        
        Args:
            agent_type: 에이전트 타입
            status: 새로운 상태
        """
        self.workflow_state.update_status(status, agent_type)

    def reset_workflow_state(self) -> None:
        """워크플로우 상태 초기화"""
        self.workflow_state = WorkflowState()
