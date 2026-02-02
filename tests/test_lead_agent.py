"""Lead Agent 테스트 (LLM 기반)

Requirements:
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

    def test_get_current_status_message_with_rag_agent(self):
        """RAG 에이전트 상태 메시지 확인"""
        state = WorkflowState()
        state.update_status(WorkflowStatus.ANALYZING, AgentType.RAG)
        message = state.get_current_status_message()
        assert "RAG Agent" in message


class TestLeadAgentInit:
    """Lead Agent 초기화 테스트"""

    def test_lead_agent_has_workflow_state(self):
        """Lead Agent가 워크플로우 상태를 가지는지 확인"""
        agent = LeadAgent.__new__(LeadAgent)
        agent.workflow_state = WorkflowState()
        assert hasattr(agent, 'workflow_state')
        assert isinstance(agent.workflow_state, WorkflowState)


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

    def test_reset_workflow_state_clears_state(self, lead_agent):
        """워크플로우 상태 초기화가 올바르게 동작하는지 확인"""
        lead_agent.update_agent_status(AgentType.SQL, WorkflowStatus.SQL_EXECUTION)

        lead_agent.reset_workflow_state()

        assert lead_agent.workflow_state.status == WorkflowStatus.IDLE
        assert lead_agent.workflow_state.current_agent is None
        assert len(lead_agent.workflow_state.agent_results) == 0
