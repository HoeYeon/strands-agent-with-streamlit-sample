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
from .shared_context import AnalysisContext, TableInfo


class AgentType(Enum):
    """에이전트 타입 정의"""
    LEAD = "lead_agent"
    DATA_EXPERT = "data_expert"
    SQL = "sql_agent"


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
                AgentType.SQL: "SQL Agent"
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
- data_expert: 데이터 카탈로그 탐색, 테이블 식별
- sql_agent: SQL 생성 및 Athena 실행

────────────────────────────────────────────
요청 유형 판단
────────────────────────────────────────────
**정보 조회** (SQL 불필요):
- 키워드: "테이블 목록", "스키마", "컬럼 정보", "어떤 데이터"
- 처리: data_expert → 결과 받으면 직접 응답 (sql_agent 위임 안함)

**데이터 분석** (SQL 필요):
- 키워드: "합계", "평균", "통계", "추이", "가장 많은/적은"
- 처리: data_expert → sql_agent → 결과 통합

**테이블 정보 이미 있는 경우**:
- data_expert 생략 → sql_agent 직접 위임

────────────────────────────────────────────
handoff 규칙
────────────────────────────────────────────
data_expert 위임:
  handoff_to_agent(agent_name="data_expert", message="탐색 요청: [요구사항]")

sql_agent 위임:
  handoff_to_agent(agent_name="sql_agent", message="SQL 요청: [테이블 정보 + 요구사항]")

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

    
    def _build_prompt_from_context(self, context: AnalysisContext) -> str:
        """컨텍스트를 기반으로 Lead Agent 프롬프트 생성"""
        prompt_parts = [
            f"사용자 요청: {context.user_query}",
        ]
        
        if context.business_intent:
            prompt_parts.append(f"파악된 의도: {context.business_intent}")
        
        if context.identified_tables:
            tables_info = []
            for table in context.identified_tables:
                tables_info.append(
                    f"- {table.database}.{table.table} (관련성: {table.relevance_score})"
                )
            prompt_parts.append(f"식별된 테이블:\n" + "\n".join(tables_info))
        
        if context.generated_sql:
            prompt_parts.append(f"생성된 SQL:\n{context.generated_sql}")
        
        if context.results:
            prompt_parts.append(f"쿼리 결과: {len(context.results)}행")
        
        if context.error_messages:
            prompt_parts.append(f"발생한 오류:\n" + "\n".join(context.error_messages))
        
        return "\n\n".join(prompt_parts)
    
    # =========================================================================
    # Requirements 1.1: 사용자 요청 분석 및 의도 파악 (LLM 기반)
    # =========================================================================
    
    def analyze_user_request(self, user_query: str) -> Dict[str, Any]:
        """사용자 요청 분석 및 의도 파악 (Requirements 1.1) - LLM 기반
        
        LLM을 통해 자연어 쿼리를 분석하고 적절한 에이전트에게 위임할 작업을 결정합니다.
        규칙 기반 키워드 매칭 대신 LLM이 직접 의도를 파악합니다.
        
        Args:
            user_query: 사용자의 자연어 쿼리
            
        Returns:
            분석 결과 딕셔너리:
            - success: 분석 성공 여부
            - context: 분석 컨텍스트
            - delegation_target: 위임할 에이전트
            - delegation_message: 위임 메시지
        """
        self.workflow_state.update_status(
            WorkflowStatus.ANALYZING, 
            AgentType.LEAD
        )
        
        context = AnalysisContext(user_query=user_query)
        
        # 위임 대상 결정 (LLM이 의도를 파악하므로 단순화)
        delegation = self._determine_delegation(context)
        
        return {
            "success": True,
            "context": context,
            "delegation_target": delegation["target"],
            "delegation_message": delegation["message"]
        }
    
    def _determine_delegation(self, context: AnalysisContext) -> Dict[str, Any]:
        """작업 위임 대상 결정 (Requirements 1.1) - LLM 기반
        
        현재 컨텍스트를 기반으로 어떤 에이전트에게 작업을 위임할지 결정합니다.
        LLM이 의도를 파악하므로 규칙 기반 의도 추출은 제거되었습니다.
        
        Args:
            context: 현재 분석 컨텍스트
            
        Returns:
            위임 정보 딕셔너리:
            - target: 위임 대상 에이전트
            - message: 위임 메시지
        """
        # 테이블 정보가 없으면 Data Expert로 위임
        if not context.identified_tables:
            return {
                "target": AgentType.DATA_EXPERT,
                "message": self._build_data_expert_message(context)
            }
        
        # 테이블 정보가 있으면 SQL Agent로 위임
        return {
            "target": AgentType.SQL,
            "message": self._build_sql_agent_message(context)
        }
    
    def _build_data_expert_message(self, context: AnalysisContext) -> str:
        """Data Expert Agent로 전달할 메시지 생성 (LLM 기반)
        
        LLM이 직접 의도를 파악하므로 사용자 요청을 그대로 전달합니다.
        """
        message_parts = [
            f"사용자 요청: {context.user_query}",
            "",
            "요청 작업:",
            "1. 사용자 요청을 분석하여 비즈니스 의도 파악",
            "2. AWS Athena 카탈로그에서 관련 데이터베이스 탐색",
            "3. 비즈니스 요구사항에 맞는 테이블 식별",
            "4. 테이블 메타데이터 분석 및 적합성 판단",
            "5. SQL 최적화를 위한 힌트 제공 (파티션 키, 날짜 컬럼)"
        ]
        
        return "\n".join(message_parts)
    
    def _build_sql_agent_message(self, context: AnalysisContext) -> str:
        """SQL Agent로 전달할 메시지 생성 (LLM 기반)
        
        LLM이 직접 의도를 파악하므로 사용자 요청과 테이블 정보를 전달합니다.
        """
        message_parts = [
            f"사용자 요청: {context.user_query}",
        ]
        
        if context.identified_tables:
            message_parts.append("\n추천 테이블:")
            for table in context.identified_tables[:3]:
                message_parts.append(
                    f"- {table.database}.{table.table} (관련성: {table.relevance_score:.2f})"
                )
        
        message_parts.extend([
            "",
            "요청 작업:",
            "1. 사용자 요청의 비즈니스 의도 파악",
            "2. 제공된 테이블 정보를 기반으로 최적화된 SQL 쿼리 생성",
            "3. Athena에서 쿼리 실행",
            "4. 결과 조회 및 포맷팅"
        ])
        
        return "\n".join(message_parts)

    
    # =========================================================================
    # Requirements 1.2: 결과 통합 및 최종 응답 생성
    # =========================================================================
    
    def integrate_results(self, context: AnalysisContext) -> str:
        """모든 에이전트의 결과를 통합하여 최종 응답 생성 (Requirements 1.2)
        
        Args:
            context: 모든 에이전트 작업이 완료된 분석 컨텍스트
            
        Returns:
            통합된 최종 응답 문자열
        """
        self.workflow_state.update_status(
            WorkflowStatus.INTEGRATING, 
            AgentType.LEAD
        )
        
        # 에러가 있는 경우
        if context.error_messages:
            return self._generate_error_response(context)
        
        # 결과가 없는 경우
        if not context.results:
            return self._generate_no_results_response(context)
        
        # 성공적인 결과
        return self._generate_success_response(context)
    
    def integrate_agent_results(
        self, 
        agent_results: List[AgentResult],
        context: AnalysisContext
    ) -> Dict[str, Any]:
        """개별 에이전트 결과들을 통합 (Requirements 1.2)
        
        Args:
            agent_results: 각 에이전트의 작업 결과 목록
            context: 분석 컨텍스트
            
        Returns:
            통합된 결과 딕셔너리
        """
        integrated = {
            "success": True,
            "data_exploration": None,
            "sql_execution": None,
            "errors": [],
            "summary": ""
        }
        
        for result in agent_results:
            if result.agent_type == AgentType.DATA_EXPERT:
                integrated["data_exploration"] = {
                    "success": result.success,
                    "tables_found": result.data.get("tables_count", 0),
                    "databases_explored": result.data.get("databases_count", 0),
                    "optimization_hints": result.data.get("optimization_hints", [])
                }
                if not result.success:
                    integrated["errors"].append(result.error_message)
                    
            elif result.agent_type == AgentType.SQL:
                integrated["sql_execution"] = {
                    "success": result.success,
                    "query": result.data.get("sql_query"),
                    "execution_id": result.data.get("execution_id"),
                    "row_count": result.data.get("row_count", 0),
                    "execution_time_ms": result.execution_time_ms
                }
                if not result.success:
                    integrated["errors"].append(result.error_message)
        
        # 전체 성공 여부 판단
        integrated["success"] = len(integrated["errors"]) == 0
        
        # 요약 생성
        integrated["summary"] = self._generate_integration_summary(integrated, context)
        
        return integrated
    
    def _generate_integration_summary(
        self, 
        integrated: Dict[str, Any],
        context: AnalysisContext
    ) -> str:
        """통합 결과 요약 생성"""
        parts = []
        
        if integrated["data_exploration"]:
            de = integrated["data_exploration"]
            if de["success"]:
                parts.append(
                    f"데이터 탐색: {de['databases_explored']}개 DB에서 "
                    f"{de['tables_found']}개 테이블 식별"
                )
            else:
                parts.append("데이터 탐색: 실패")
        
        if integrated["sql_execution"]:
            se = integrated["sql_execution"]
            if se["success"]:
                parts.append(
                    f"SQL 실행: {se['row_count']}행 결과 "
                    f"({se['execution_time_ms']}ms)"
                )
            else:
                parts.append("SQL 실행: 실패")
        
        return " | ".join(parts) if parts else "작업 결과 없음"
    
    def _generate_error_response(self, context: AnalysisContext) -> str:
        """오류 상황에 대한 응답 생성 (Requirements 1.4)
        
        에러 발생 시 명확한 오류 메시지와 다음 단계를 제안합니다.
        """
        self.workflow_state.update_status(WorkflowStatus.ERROR)
        
        response_parts = [
            f"## 요약",
            f"'{context.user_query}' 처리 중 오류가 발생했습니다.",
            "",
            "## 발생한 오류"
        ]
        
        for i, error in enumerate(context.error_messages, 1):
            response_parts.append(f"{i}. {error}")
        
        # 오류 유형별 다음 단계 제안 (Requirements 1.4)
        response_parts.extend([
            "",
            "## 다음 단계 제안"
        ])
        
        suggestions = self._get_error_suggestions(context.error_messages)
        for suggestion in suggestions:
            response_parts.append(f"- {suggestion}")
        
        return "\n".join(response_parts)
    
    def _get_error_suggestions(self, error_messages: List[str]) -> List[str]:
        """오류 메시지에 따른 다음 단계 제안 생성 (Requirements 1.4)"""
        suggestions = []
        error_text = " ".join(error_messages).lower()
        
        if "권한" in error_text or "permission" in error_text or "access" in error_text:
            suggestions.append("AWS IAM 권한 설정을 확인해보세요 (athena:* 관련)")
            suggestions.append("AWS 프로파일 설정이 올바른지 확인해보세요")
        
        if "테이블" in error_text or "table" in error_text:
            suggestions.append("요청에 테이블 이름을 더 구체적으로 명시해보세요")
            suggestions.append("사용 가능한 테이블 목록을 먼저 조회해보세요")
        
        if "데이터베이스" in error_text or "database" in error_text:
            suggestions.append("데이터베이스 이름을 확인해보세요")
            suggestions.append("AWS Athena 콘솔에서 데이터베이스 존재 여부를 확인해보세요")
        
        if "쿼리" in error_text or "query" in error_text or "sql" in error_text:
            suggestions.append("요청을 더 구체적으로 다시 작성해보세요")
            suggestions.append("시간 범위나 조건을 조정해보세요")
        
        if "타임아웃" in error_text or "timeout" in error_text:
            suggestions.append("쿼리 범위를 축소해보세요 (시간 범위, LIMIT 등)")
            suggestions.append("파티션 키를 활용한 필터링을 추가해보세요")
        
        # 기본 제안
        if not suggestions:
            suggestions = [
                "요청을 더 구체적으로 다시 작성해보세요",
                "AWS 연결 상태를 확인해보세요",
                "데이터베이스 및 테이블 접근 권한을 확인해보세요"
            ]
        
        return suggestions
    
    def _generate_no_results_response(self, context: AnalysisContext) -> str:
        """결과가 없는 경우 응답 생성"""
        self.workflow_state.update_status(WorkflowStatus.COMPLETED)
        
        response_parts = [
            "## 요약",
            f"'{context.user_query}'에 대한 분석을 완료했지만 결과가 없습니다.",
            "",
            "## 수행된 작업"
        ]
        
        # 각 단계별 상태
        if context.identified_tables:
            response_parts.append(
                f"1. 데이터 탐색: ✅ {len(context.identified_tables)}개 테이블 식별"
            )
        else:
            response_parts.append("1. 데이터 탐색: ❌ 관련 테이블을 찾지 못함")
        
        if context.generated_sql:
            response_parts.append("2. SQL 생성: ✅ 완료")
        else:
            response_parts.append("2. SQL 생성: ❌ 실패")
        
        if context.query_execution_id:
            response_parts.append("3. 쿼리 실행: ✅ 완료 (결과 0행)")
        else:
            response_parts.append("3. 쿼리 실행: ❌ 실행되지 않음")
        
        response_parts.extend([
            "",
            "## 다음 단계 제안",
            "- 검색 조건을 완화해보세요",
            "- 다른 시간 범위로 시도해보세요",
            "- 사용 가능한 데이터를 먼저 확인해보세요"
        ])
        
        return "\n".join(response_parts)
    
    def _generate_success_response(self, context: AnalysisContext) -> str:
        """성공적인 결과에 대한 응답 생성 (Requirements 1.2)"""
        self.workflow_state.update_status(WorkflowStatus.COMPLETED)
        
        response_parts = [
            "## 요약",
            f"'{context.user_query}'에 대한 분석을 성공적으로 완료했습니다.",
            "",
            "## 수행된 작업"
        ]
        
        # 데이터 탐색 결과
        if context.identified_tables:
            tables_summary = ", ".join([
                f"{t.database}.{t.table}" 
                for t in context.identified_tables[:3]
            ])
            response_parts.append(
                f"1. 데이터 탐색: {len(context.identified_tables)}개 테이블 식별 "
                f"({tables_summary})"
            )
        
        # SQL 실행 결과
        if context.generated_sql and context.results:
            response_parts.append(
                f"2. SQL 생성 및 실행: {len(context.results)}행 결과"
            )
        
        # 최종 결과
        response_parts.extend([
            "",
            "## 최종 결과",
            f"총 {len(context.results)}행의 데이터를 조회했습니다."
        ])
        
        # 결과 미리보기 (최대 5행)
        if context.results and len(context.results) > 0:
            response_parts.append("\n### 결과 미리보기")
            preview_count = min(5, len(context.results))
            for i, row in enumerate(context.results[:preview_count], 1):
                if isinstance(row, dict):
                    row_str = ", ".join([f"{k}: {v}" for k, v in list(row.items())[:5]])
                else:
                    row_str = str(row)
                response_parts.append(f"{i}. {row_str}")
            
            if len(context.results) > preview_count:
                response_parts.append(f"... 외 {len(context.results) - preview_count}행")
        
        # 다음 단계 제안
        response_parts.extend([
            "",
            "## 다음 단계 제안",
            "- 결과를 더 자세히 분석해보세요",
            "- 다른 조건으로 추가 분석을 시도해보세요",
            "- 결과를 시각화해보세요"
        ])
        
        return "\n".join(response_parts)

    
    # =========================================================================
    # Requirements 1.5: 작업 상태 표시
    # =========================================================================
    
    def get_workflow_state(self) -> WorkflowState:
        """현재 워크플로우 상태 반환 (Requirements 1.5)"""
        return self.workflow_state
    
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
    
    def record_agent_result(
        self,
        agent_type: AgentType,
        success: bool,
        data: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        execution_time_ms: Optional[int] = None
    ) -> None:
        """에이전트 작업 결과 기록 (Requirements 1.5)
        
        Args:
            agent_type: 에이전트 타입
            success: 성공 여부
            data: 결과 데이터
            error_message: 오류 메시지 (실패 시)
            execution_time_ms: 실행 시간 (밀리초)
        """
        result = AgentResult(
            agent_type=agent_type,
            success=success,
            data=data or {},
            error_message=error_message,
            execution_time_ms=execution_time_ms
        )
        self.workflow_state.add_result(result)
    
    def reset_workflow_state(self) -> None:
        """워크플로우 상태 초기화"""
        self.workflow_state = WorkflowState()
    
    # =========================================================================
    # 유틸리티 메서드
    # =========================================================================
    
    def validate_context_for_sql(self, context: AnalysisContext) -> Dict[str, Any]:
        """SQL 생성을 위한 컨텍스트 유효성 검증
        
        Args:
            context: 분석 컨텍스트
            
        Returns:
            검증 결과 딕셔너리:
            - valid: 유효 여부
            - missing: 누락된 정보 목록
            - suggestions: 제안 사항
        """
        missing = []
        suggestions = []
        
        if not context.identified_tables:
            missing.append("테이블 정보")
            suggestions.append("먼저 Data Expert Agent를 통해 테이블을 식별해야 합니다")
        
        if not context.business_intent:
            missing.append("비즈니스 의도")
            suggestions.append("사용자 요청을 분석하여 의도를 파악해야 합니다")
        
        return {
            "valid": len(missing) == 0,
            "missing": missing,
            "suggestions": suggestions
        }
    
    def format_handoff_context(self, context: AnalysisContext) -> Dict[str, Any]:
        """에이전트 간 handoff를 위한 컨텍스트 포맷팅
        
        Args:
            context: 분석 컨텍스트
            
        Returns:
            handoff용 컨텍스트 딕셔너리
        """
        return {
            "user_query": context.user_query,
            "business_intent": context.business_intent,
            "identified_tables": [
                {
                    "database": t.database,
                    "table": t.table,
                    "columns": [
                        {"name": c.name, "type": c.type}
                        for c in t.columns
                    ],
                    "partition_keys": t.partition_keys,
                    "relevance_score": t.relevance_score
                }
                for t in context.identified_tables
            ],
            "generated_sql": context.generated_sql,
            "query_execution_id": context.query_execution_id,
            "has_results": context.results is not None and len(context.results) > 0,
            "result_count": len(context.results) if context.results else 0,
            "errors": context.error_messages
        }
    
    def create_status_event(self, message: str) -> Dict[str, Any]:
        """상태 이벤트 생성 (UI 업데이트용)
        
        Args:
            message: 상태 메시지
            
        Returns:
            상태 이벤트 딕셔너리
        """
        return {
            "type": "agent_status",
            "agent": self.workflow_state.current_agent.value if self.workflow_state.current_agent else "lead_agent",
            "status": self.workflow_state.status.value,
            "message": message
        }
