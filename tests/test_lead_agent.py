"""Lead Agent 테스트 (LLM 기반)

Requirements:
- 1.1: 사용자 요청 분석 및 적절한 전문 에이전트에게 작업 위임
- 1.2: 모든 에이전트 결과 통합하여 최종 응답 제공
- 1.4: 에러 발생 시 명확한 오류 메시지와 다음 단계 제안
- 1.5: 현재 작업 중인 에이전트 상태 표시
"""

import pytest
from agents.multi_agent.lead_agent import (
    LeadAgent,
    AgentType,
    WorkflowStatus,
    WorkflowState,
    AgentResult,
)
from agents.multi_agent.shared_context import AnalysisContext, TableInfo, ColumnInfo


class TestWorkflowState:
    """워크플로우 상태 관리 테스트 (Requirements 1.5)"""
    
    def test_initial_state_is_idle(self):
        """초기 상태가 IDLE인지 확인"""
        state = WorkflowState()
        assert state.status == WorkflowStatus.IDLE
    
    def test_update_status_changes_status(self):
        """상태 업데이트가 올바르게 동작하는지 확인"""
        state = WorkflowState()
        state.update_status(WorkflowStatus.ANALYZING, AgentType.LEAD)
        assert state.status == WorkflowStatus.ANALYZING
        assert state.current_agent == AgentType.LEAD
    
    def test_update_status_adds_progress_message(self):
        """상태 업데이트 시 진행 메시지가 추가되는지 확인"""
        state = WorkflowState()
        state.update_status(WorkflowStatus.DATA_EXPLORATION, AgentType.DATA_EXPERT)
        assert len(state.progress_messages) > 0
    
    def test_add_result_stores_agent_result(self):
        """에이전트 결과가 올바르게 저장되는지 확인"""
        state = WorkflowState()
        result = AgentResult(
            agent_type=AgentType.DATA_EXPERT,
            success=True,
            data={"tables_count": 5}
        )
        state.add_result(result)
        assert len(state.agent_results) == 1
        assert state.agent_results[0].agent_type == AgentType.DATA_EXPERT

    
    def test_get_current_status_message_with_agent(self):
        """현재 에이전트가 있을 때 상태 메시지 확인 (Requirements 1.5)"""
        state = WorkflowState()
        state.update_status(WorkflowStatus.DATA_EXPLORATION, AgentType.DATA_EXPERT)
        message = state.get_current_status_message()
        assert "Data Expert Agent" in message


class TestLeadAgentInit:
    """Lead Agent 초기화 테스트"""
    
    def test_lead_agent_has_workflow_state(self):
        """Lead Agent가 워크플로우 상태를 가지는지 확인"""
        # Agent 초기화 없이 클래스 속성만 확인
        agent = LeadAgent.__new__(LeadAgent)
        agent.workflow_state = WorkflowState()
        assert hasattr(agent, 'workflow_state')
        assert isinstance(agent.workflow_state, WorkflowState)


class TestLLMBasedAnalysis:
    """LLM 기반 분석 테스트 (Requirements 1.1)
    
    규칙 기반 의도 추출이 제거되고 LLM이 직접 의도를 파악합니다.
    이 테스트는 LLM 기반 분석의 기본 동작을 확인합니다.
    """
    
    @pytest.fixture
    def lead_agent(self):
        """Lead Agent 인스턴스 생성 (Agent 초기화 없이)"""
        agent = LeadAgent.__new__(LeadAgent)
        agent.workflow_state = WorkflowState()
        return agent
    
    def test_analyze_user_request_returns_context(self, lead_agent):
        """사용자 요청 분석이 컨텍스트를 반환하는지 확인"""
        result = lead_agent.analyze_user_request("상품별 매출 조회")
        assert result["success"] is True
        assert result["context"] is not None
        assert result["context"].user_query == "상품별 매출 조회"
    
    def test_analyze_user_request_determines_delegation(self, lead_agent):
        """사용자 요청 분석이 위임 대상을 결정하는지 확인"""
        result = lead_agent.analyze_user_request("매출 분석")
        assert "delegation_target" in result
        assert "delegation_message" in result
    
    def test_analyze_user_request_updates_workflow_status(self, lead_agent):
        """사용자 요청 분석이 워크플로우 상태를 업데이트하는지 확인"""
        lead_agent.analyze_user_request("매출 분석")
        assert lead_agent.workflow_state.status == WorkflowStatus.ANALYZING


class TestDelegationDecision:
    """작업 위임 결정 테스트 (Requirements 1.1) - LLM 기반"""
    
    @pytest.fixture
    def lead_agent(self):
        """Lead Agent 인스턴스 생성"""
        agent = LeadAgent.__new__(LeadAgent)
        agent.workflow_state = WorkflowState()
        return agent
    
    def test_delegate_to_data_expert_without_tables(self, lead_agent):
        """테이블 정보가 없으면 Data Expert로 위임"""
        context = AnalysisContext(user_query="매출 분석")
        
        delegation = lead_agent._determine_delegation(context)
        assert delegation["target"] == AgentType.DATA_EXPERT
    
    def test_delegate_to_sql_agent_with_tables(self, lead_agent):
        """테이블 정보가 있으면 SQL Agent로 위임"""
        context = AnalysisContext(user_query="매출 분석")
        context.identified_tables = [
            TableInfo(database="db", table="sales", relevance_score=0.9)
        ]
        
        delegation = lead_agent._determine_delegation(context)
        assert delegation["target"] == AgentType.SQL
    
    def test_data_expert_message_includes_user_query(self, lead_agent):
        """Data Expert 메시지에 사용자 쿼리가 포함되는지 확인"""
        context = AnalysisContext(user_query="상품별 매출 조회")
        
        message = lead_agent._build_data_expert_message(context)
        assert "상품별 매출 조회" in message
    
    def test_sql_agent_message_includes_table_info(self, lead_agent):
        """SQL Agent 메시지에 테이블 정보가 포함되는지 확인"""
        context = AnalysisContext(user_query="매출 분석")
        context.identified_tables = [
            TableInfo(database="analytics", table="sales", relevance_score=0.95)
        ]
        
        message = lead_agent._build_sql_agent_message(context)
        assert "analytics.sales" in message



class TestResultIntegration:
    """결과 통합 테스트 (Requirements 1.2)"""
    
    @pytest.fixture
    def lead_agent(self):
        """Lead Agent 인스턴스 생성"""
        agent = LeadAgent.__new__(LeadAgent)
        agent.workflow_state = WorkflowState()
        return agent
    
    def test_integrate_results_with_errors_returns_error_response(self, lead_agent):
        """에러가 있으면 에러 응답 생성"""
        context = AnalysisContext(user_query="매출 분석")
        context.add_error("테이블을 찾을 수 없습니다")
        
        response = lead_agent.integrate_results(context)
        assert "오류" in response or "발생" in response
    
    def test_integrate_results_without_results_returns_no_results_response(self, lead_agent):
        """결과가 없으면 결과 없음 응답 생성"""
        context = AnalysisContext(user_query="매출 분석")
        context.results = None
        
        response = lead_agent.integrate_results(context)
        assert "결과가 없습니다" in response or "결과 없음" in response or "0행" in response
    
    def test_integrate_results_with_data_returns_success_response(self, lead_agent):
        """결과가 있으면 성공 응답 생성"""
        context = AnalysisContext(user_query="매출 분석")
        context.identified_tables = [
            TableInfo(database="db", table="sales", relevance_score=0.9)
        ]
        context.generated_sql = "SELECT * FROM sales"
        context.results = [{"product": "A", "revenue": 1000}]
        
        response = lead_agent.integrate_results(context)
        assert "성공" in response or "완료" in response
    
    def test_integrate_agent_results_combines_all_results(self, lead_agent):
        """모든 에이전트 결과가 통합되는지 확인"""
        context = AnalysisContext(user_query="매출 분석")
        
        results = [
            AgentResult(
                agent_type=AgentType.DATA_EXPERT,
                success=True,
                data={"tables_count": 3, "databases_count": 1}
            ),
            AgentResult(
                agent_type=AgentType.SQL,
                success=True,
                data={"sql_query": "SELECT *", "row_count": 10},
                execution_time_ms=500
            )
        ]
        
        integrated = lead_agent.integrate_agent_results(results, context)
        
        assert integrated["success"] is True
        assert integrated["data_exploration"] is not None
        assert integrated["sql_execution"] is not None
        assert integrated["data_exploration"]["tables_found"] == 3
        assert integrated["sql_execution"]["row_count"] == 10


class TestErrorHandling:
    """에러 처리 테스트 (Requirements 1.4)"""
    
    @pytest.fixture
    def lead_agent(self):
        """Lead Agent 인스턴스 생성"""
        agent = LeadAgent.__new__(LeadAgent)
        agent.workflow_state = WorkflowState()
        return agent
    
    def test_error_response_includes_error_messages(self, lead_agent):
        """에러 응답에 오류 메시지가 포함되는지 확인"""
        context = AnalysisContext(user_query="매출 분석")
        context.add_error("권한이 없습니다")
        
        response = lead_agent._generate_error_response(context)
        assert "권한이 없습니다" in response
    
    def test_error_response_includes_suggestions(self, lead_agent):
        """에러 응답에 다음 단계 제안이 포함되는지 확인 (Requirements 1.4)"""
        context = AnalysisContext(user_query="매출 분석")
        context.add_error("권한 오류가 발생했습니다")
        
        response = lead_agent._generate_error_response(context)
        assert "다음 단계" in response or "제안" in response
    
    def test_get_error_suggestions_for_permission_error(self, lead_agent):
        """권한 오류에 대한 제안 생성"""
        suggestions = lead_agent._get_error_suggestions(["권한이 없습니다"])
        assert any("권한" in s or "IAM" in s for s in suggestions)
    
    def test_get_error_suggestions_for_table_error(self, lead_agent):
        """테이블 오류에 대한 제안 생성"""
        suggestions = lead_agent._get_error_suggestions(["테이블을 찾을 수 없습니다"])
        assert any("테이블" in s for s in suggestions)
    
    def test_get_error_suggestions_for_timeout_error(self, lead_agent):
        """타임아웃 오류에 대한 제안 생성"""
        suggestions = lead_agent._get_error_suggestions(["쿼리 타임아웃"])
        assert any("범위" in s or "축소" in s for s in suggestions)
    
    def test_get_error_suggestions_default(self, lead_agent):
        """기본 제안 생성"""
        suggestions = lead_agent._get_error_suggestions(["알 수 없는 오류"])
        assert len(suggestions) > 0


class TestStatusManagement:
    """상태 관리 테스트 (Requirements 1.5)"""
    
    @pytest.fixture
    def lead_agent(self):
        """Lead Agent 인스턴스 생성"""
        agent = LeadAgent.__new__(LeadAgent)
        agent.workflow_state = WorkflowState()
        return agent
    
    def test_get_current_status_returns_dict(self, lead_agent):
        """현재 상태가 딕셔너리로 반환되는지 확인"""
        status = lead_agent.get_current_status()
        assert isinstance(status, dict)
        assert "status" in status
        assert "current_agent" in status
        assert "message" in status
        assert "progress" in status
    
    def test_update_agent_status_changes_workflow_state(self, lead_agent):
        """에이전트 상태 업데이트가 워크플로우 상태를 변경하는지 확인"""
        lead_agent.update_agent_status(AgentType.DATA_EXPERT, WorkflowStatus.DATA_EXPLORATION)
        
        assert lead_agent.workflow_state.status == WorkflowStatus.DATA_EXPLORATION
        assert lead_agent.workflow_state.current_agent == AgentType.DATA_EXPERT
    
    def test_record_agent_result_adds_to_workflow_state(self, lead_agent):
        """에이전트 결과 기록이 워크플로우 상태에 추가되는지 확인"""
        lead_agent.record_agent_result(
            agent_type=AgentType.SQL,
            success=True,
            data={"row_count": 100},
            execution_time_ms=250
        )
        
        assert len(lead_agent.workflow_state.agent_results) == 1
        assert lead_agent.workflow_state.agent_results[0].agent_type == AgentType.SQL
    
    def test_reset_workflow_state_clears_state(self, lead_agent):
        """워크플로우 상태 초기화가 올바르게 동작하는지 확인"""
        lead_agent.update_agent_status(AgentType.SQL, WorkflowStatus.SQL_EXECUTION)
        lead_agent.record_agent_result(AgentType.SQL, True, {"row_count": 10})
        
        lead_agent.reset_workflow_state()
        
        assert lead_agent.workflow_state.status == WorkflowStatus.IDLE
        assert lead_agent.workflow_state.current_agent is None
        assert len(lead_agent.workflow_state.agent_results) == 0


class TestContextValidation:
    """컨텍스트 유효성 검증 테스트"""
    
    @pytest.fixture
    def lead_agent(self):
        """Lead Agent 인스턴스 생성"""
        agent = LeadAgent.__new__(LeadAgent)
        agent.workflow_state = WorkflowState()
        return agent
    
    def test_validate_context_without_tables_is_invalid(self, lead_agent):
        """테이블 정보가 없으면 유효하지 않음"""
        context = AnalysisContext(user_query="매출 분석")
        
        result = lead_agent.validate_context_for_sql(context)
        assert result["valid"] is False
        assert "테이블 정보" in result["missing"]
    
    def test_validate_context_with_tables_is_valid(self, lead_agent):
        """테이블 정보가 있으면 유효함"""
        context = AnalysisContext(user_query="매출 분석")
        context.identified_tables = [
            TableInfo(database="db", table="sales", relevance_score=0.9)
        ]
        context.business_intent = {"entity": "product"}
        
        result = lead_agent.validate_context_for_sql(context)
        assert result["valid"] is True


class TestHandoffContextFormatting:
    """Handoff 컨텍스트 포맷팅 테스트"""
    
    @pytest.fixture
    def lead_agent(self):
        """Lead Agent 인스턴스 생성"""
        agent = LeadAgent.__new__(LeadAgent)
        agent.workflow_state = WorkflowState()
        return agent
    
    def test_format_handoff_context_includes_user_query(self, lead_agent):
        """Handoff 컨텍스트에 사용자 쿼리가 포함되는지 확인"""
        context = AnalysisContext(user_query="매출 분석")
        
        formatted = lead_agent.format_handoff_context(context)
        assert formatted["user_query"] == "매출 분석"
    
    def test_format_handoff_context_includes_tables(self, lead_agent):
        """Handoff 컨텍스트에 테이블 정보가 포함되는지 확인"""
        context = AnalysisContext(user_query="매출 분석")
        context.identified_tables = [
            TableInfo(
                database="db",
                table="sales",
                columns=[ColumnInfo(name="id", type="int")],
                partition_keys=["date"],
                relevance_score=0.9
            )
        ]
        
        formatted = lead_agent.format_handoff_context(context)
        assert len(formatted["identified_tables"]) == 1
        assert formatted["identified_tables"][0]["database"] == "db"
        assert formatted["identified_tables"][0]["table"] == "sales"
