"""전체 워크플로우 통합 테스트

Requirements:
- 1.1: 사용자 요청 분석 및 적절한 전문 에이전트에게 작업 위임
- 1.2: 모든 에이전트 결과 통합하여 최종 응답 제공
- 1.3: 공유 컨텍스트를 통해 정보 전달
- 1.4: 에러 발생 시 명확한 오류 메시지와 다음 단계 제안
- 1.5: 현재 작업 중인 에이전트 상태 표시

이 테스트 모듈은 멀티에이전트 Text2SQL 시스템의 전체 워크플로우를 검증합니다.
실제 AWS Athena 환경 없이도 시스템의 통합 동작을 테스트합니다.
"""

import pytest
import queue
from unittest.mock import patch
from typing import Dict, Any, List

from agents.multi_agent.lead_agent import (
    LeadAgent,
    AgentType,
    WorkflowStatus,
    WorkflowState,
    AgentResult,
)
from agents.multi_agent.shared_context import (
    AnalysisContext,
    TableInfo,
    ColumnInfo,
    SwarmConfig,
)
from agents.multi_agent.event_adapter import SwarmEventAdapter


class TestSharedContextPropagation:
    """공유 컨텍스트 전파 테스트 (Requirements 1.3)"""
    
    @pytest.fixture
    def context(self):
        return AnalysisContext(user_query="매출 분석")
    
    def test_context_business_intent_propagation(self, context):
        """비즈니스 의도가 컨텍스트에 전파되는지 확인"""
        context.business_intent = {
            "entity": "product",
            "metric": "revenue"
        }
        
        assert context.business_intent["entity"] == "product"
        assert context.business_intent["metric"] == "revenue"
    
    def test_context_table_info_propagation(self, context):
        """테이블 정보가 컨텍스트에 전파되는지 확인"""
        table = TableInfo(
            database="db",
            table="sales",
            columns=[ColumnInfo(name="id", type="string")],
            partition_keys=["date"],
            relevance_score=0.9
        )
        context.identified_tables = [table]
        
        assert len(context.identified_tables) == 1
        assert context.identified_tables[0].database == "db"
    
    def test_context_sql_propagation(self, context):
        """SQL 쿼리가 컨텍스트에 전파되는지 확인"""
        context.generated_sql = "SELECT * FROM sales"
        
        assert context.generated_sql == "SELECT * FROM sales"
    
    def test_context_results_propagation(self, context):
        """쿼리 결과가 컨텍스트에 전파되는지 확인"""
        context.results = [{"id": 1}, {"id": 2}]
        
        assert len(context.results) == 2
    
    def test_context_error_propagation(self, context):
        """에러 메시지가 컨텍스트에 전파되는지 확인"""
        context.add_error("테스트 에러")
        
        assert len(context.error_messages) == 1
        assert "테스트 에러" in context.error_messages


class TestWorkflowStatusTracking:
    """작업 상태 추적 테스트 (Requirements 1.5)"""
    
    @pytest.fixture
    def lead_agent(self):
        agent = LeadAgent.__new__(LeadAgent)
        agent.workflow_state = WorkflowState()
        return agent
    
    def test_status_tracking_through_workflow(self, lead_agent):
        """워크플로우 전체에서 상태 추적"""
        # 초기 상태
        assert lead_agent.workflow_state.status == WorkflowStatus.IDLE
        
        # 분석 시작
        lead_agent.update_agent_status(AgentType.LEAD, WorkflowStatus.ANALYZING)
        assert lead_agent.workflow_state.status == WorkflowStatus.ANALYZING
        assert lead_agent.workflow_state.current_agent == AgentType.LEAD
        
        # 데이터 탐색
        lead_agent.update_agent_status(AgentType.DATA_EXPERT, WorkflowStatus.DATA_EXPLORATION)
        assert lead_agent.workflow_state.status == WorkflowStatus.DATA_EXPLORATION
        assert lead_agent.workflow_state.current_agent == AgentType.DATA_EXPERT
        
        # SQL 생성
        lead_agent.update_agent_status(AgentType.SQL, WorkflowStatus.SQL_GENERATION)
        assert lead_agent.workflow_state.status == WorkflowStatus.SQL_GENERATION
        assert lead_agent.workflow_state.current_agent == AgentType.SQL
        
        # 완료
        lead_agent.update_agent_status(AgentType.LEAD, WorkflowStatus.COMPLETED)
        assert lead_agent.workflow_state.status == WorkflowStatus.COMPLETED
    
    def test_progress_messages_accumulate(self, lead_agent):
        """진행 메시지가 누적되는지 확인"""
        lead_agent.update_agent_status(AgentType.LEAD, WorkflowStatus.ANALYZING)
        lead_agent.update_agent_status(AgentType.DATA_EXPERT, WorkflowStatus.DATA_EXPLORATION)
        lead_agent.update_agent_status(AgentType.SQL, WorkflowStatus.SQL_GENERATION)
        
        assert len(lead_agent.workflow_state.progress_messages) >= 3
    
    def test_agent_results_recorded(self, lead_agent):
        """에이전트 결과가 기록되는지 확인"""
        lead_agent.record_agent_result(
            agent_type=AgentType.DATA_EXPERT,
            success=True,
            data={"tables_count": 3},
            execution_time_ms=500
        )
        
        lead_agent.record_agent_result(
            agent_type=AgentType.SQL,
            success=True,
            data={"row_count": 100},
            execution_time_ms=1500
        )
        
        assert len(lead_agent.workflow_state.agent_results) == 2
        assert lead_agent.workflow_state.agent_results[0].agent_type == AgentType.DATA_EXPERT
        assert lead_agent.workflow_state.agent_results[1].agent_type == AgentType.SQL
    
    def test_get_current_status_returns_complete_info(self, lead_agent):
        """현재 상태 조회가 완전한 정보를 반환하는지 확인"""
        lead_agent.update_agent_status(AgentType.SQL, WorkflowStatus.SQL_EXECUTION)
        
        status = lead_agent.get_current_status()
        
        assert "status" in status
        assert "current_agent" in status
        assert "message" in status
        assert "progress" in status
        assert status["status"] == "sql_execution"
        assert status["current_agent"] == "sql_agent"


class TestSwarmConfigValidation:
    """Swarm 설정 검증 테스트"""
    
    def test_default_swarm_config(self):
        """기본 Swarm 설정 확인"""
        config = SwarmConfig()
        
        assert config.max_handoffs == 20
        assert config.max_iterations == 20
        assert config.execution_timeout == 900.0  # 15분
        assert config.node_timeout == 300.0  # 5분
        assert config.repetitive_handoff_detection_window == 8
        assert config.repetitive_handoff_min_unique_agents == 3
    
    def test_custom_swarm_config(self):
        """커스텀 Swarm 설정"""
        config = SwarmConfig(
            max_handoffs=10,
            execution_timeout=600.0
        )
        
        assert config.max_handoffs == 10
        assert config.execution_timeout == 600.0


class TestEventAdapterIntegration:
    """이벤트 어댑터 통합 테스트"""
    
    @pytest.fixture
    def event_queue(self):
        return queue.Queue()
    
    @pytest.fixture
    def event_adapter(self, event_queue):
        from agents.events.registry import EventRegistry
        registry = EventRegistry()
        return SwarmEventAdapter(
            event_queue=event_queue,
            event_registry=registry
        )
    
    def test_event_conversion_for_node_start(self, event_adapter):
        """노드 시작 이벤트 변환"""
        swarm_event = {
            "type": "multiagent_node_start",
            "node_id": "data_expert"
        }
        
        converted = event_adapter.convert_event(swarm_event)
        
        assert converted is not None
        assert converted.get("type") == "agent_status"
    
    def test_event_conversion_for_handoff(self, event_adapter):
        """핸드오프 이벤트 변환"""
        swarm_event = {
            "type": "multiagent_handoff",
            "from_node_id": "lead_agent",
            "to_node_ids": ["data_expert"]
        }
        
        converted = event_adapter.convert_event(swarm_event)
        
        assert converted is not None
    
    def test_workflow_status_tracking(self, event_adapter):
        """워크플로우 상태 추적"""
        # 노드 시작 이벤트 처리
        event_adapter.process_event({
            "type": "multiagent_node_start",
            "node_id": "lead_agent"
        })
        
        status = event_adapter.get_current_status()
        
        assert status is not None
        assert "current_agent" in status


class TestPerformanceAndTimeout:
    """성능 및 타임아웃 테스트"""
    
    def test_swarm_config_timeout_values(self):
        """Swarm 타임아웃 설정 확인"""
        config = SwarmConfig()
        
        # 전체 실행 타임아웃: 15분
        assert config.execution_timeout == 900.0
        
        # 개별 노드 타임아웃: 5분
        assert config.node_timeout == 300.0
    
    def test_polling_configuration(self):
        """폴링 설정 확인"""
        from agents.multi_agent.sql_agent import (
            POLLING_INTERVAL_SECONDS,
            MAX_POLLING_ATTEMPTS
        )
        
        # 5초 간격
        assert POLLING_INTERVAL_SECONDS == 5
        
        # 최대 5회
        assert MAX_POLLING_ATTEMPTS == 5
        
        # 최대 대기 시간: 25초
        max_wait = POLLING_INTERVAL_SECONDS * MAX_POLLING_ATTEMPTS
        assert max_wait == 25
    
    def test_result_row_limit(self):
        """결과 행 수 제한 확인"""
        from agents.multi_agent.sql_agent import MAX_QUERY_RESULTS
        
        assert MAX_QUERY_RESULTS == 1000


class TestRAGAgentIntegration:
    """RAG Agent 통합 테스트 (Requirements 3.1, 3.2, 3.3, 3.4, 3.5)"""
    
    @pytest.fixture
    def rag_agent(self):
        """RAG Agent 인스턴스"""
        from agents.multi_agent.rag_agent import RAGAgent
        return RAGAgent(
            model_id="test-model",
            opensearch_endpoint="https://test.opensearch.com",
            opensearch_index="test_index",
            opensearch_username="test_user",
            opensearch_password="test_pass"
        )
    
    @pytest.fixture
    def lead_agent_with_rag(self):
        """RAG Agent가 포함된 Lead Agent"""
        agent = LeadAgent.__new__(LeadAgent)
        agent.workflow_state = WorkflowState()
        return agent
    
    @pytest.fixture
    def sample_context_with_rag(self):
        """RAG 결과가 포함된 분석 컨텍스트"""
        context = AnalysisContext(user_query="지난달 매출 상위 5개 상품")
        context.rag_enabled = True
        return context
    
    def test_rag_agent_swarm_registration(self, rag_agent):
        """RAG Agent가 Swarm에 등록되는지 확인 (Requirements 3.1)"""
        # RAG Agent가 BaseMultiAgent를 상속받아 get_agent() 메서드 제공
        agent = rag_agent.get_agent()
        
        assert agent is not None
        assert agent.name == "rag_agent"
        assert agent.model is not None
    
    def test_rag_search_and_context_save(self, sample_context_with_rag, rag_agent):
        """RAG 검색 및 컨텍스트 저장 테스트"""
        from agents.multi_agent.vector_search import SearchResult

        mock_results = [
            SearchResult(
                content="Table: sales_transactions\nColumns: product_id, revenue",
                score=0.9,
                metadata={"table": "sales_transactions", "database": "analytics"},
                source="sales_transactions.md"
            )
        ]

        with patch.object(
            rag_agent._search_service,
            'search',
            return_value=(mock_results, None)
        ):
            result = rag_agent.search_and_extract(
                "매출 상품",
                context=sample_context_with_rag
            )

            assert result["success"] is True
            assert len(result["results"]) == 1
            assert len(sample_context_with_rag.rag_results) == 1
            assert sample_context_with_rag.rag_results[0]["metadata"]["table"] == "sales_transactions"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
