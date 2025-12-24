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
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from agents.multi_agent.lead_agent import (
    LeadAgent,
    AgentType,
    WorkflowStatus,
    WorkflowState,
    AgentResult,
)
from agents.multi_agent.data_expert_agent import DataExpertAgent
from agents.multi_agent.sql_agent import SQLAgent
from agents.multi_agent.shared_context import (
    AnalysisContext,
    TableInfo,
    ColumnInfo,
    SwarmConfig,
)
from agents.multi_agent.event_adapter import SwarmEventAdapter


class TestEndToEndWorkflow:
    """End-to-End 워크플로우 테스트 (Requirements 1.1, 1.2, 1.3)"""
    
    @pytest.fixture
    def lead_agent(self):
        """Lead Agent 인스턴스 (Agent 초기화 없이)"""
        agent = LeadAgent.__new__(LeadAgent)
        agent.workflow_state = WorkflowState()
        return agent
    
    @pytest.fixture
    def data_expert(self):
        """Data Expert Agent 인스턴스"""
        return DataExpertAgent(model_id="test-model")
    
    @pytest.fixture
    def sql_agent(self):
        """SQL Agent 인스턴스"""
        return SQLAgent(model_id="test-model")
    
    @pytest.fixture
    def sample_context(self):
        """샘플 분석 컨텍스트"""
        return AnalysisContext(
            user_query="지난달 매출 상위 5개 상품"
        )
    
    @pytest.fixture
    def sample_table(self):
        """샘플 테이블 정보"""
        return TableInfo(
            database="analytics",
            table="sales_transactions",
            columns=[
                ColumnInfo(name="product_id", type="string"),
                ColumnInfo(name="product_name", type="string"),
                ColumnInfo(name="amount", type="decimal"),
                ColumnInfo(name="revenue", type="double"),
                ColumnInfo(name="transaction_date", type="timestamp"),
            ],
            partition_keys=["year", "month"],
            relevance_score=0.9
        )
    
    def test_workflow_request_analysis(self, lead_agent, sample_context):
        """워크플로우 1단계: 사용자 요청 분석 (Requirements 1.1) - LLM 기반"""
        # Lead Agent가 사용자 요청을 분석 (LLM 기반이므로 규칙 기반 의도 추출 없음)
        result = lead_agent.analyze_user_request(sample_context.user_query)
        
        assert result["success"] is True
        assert result["delegation_target"] is not None
        assert result["context"] is not None
        
        # LLM 기반이므로 사용자 쿼리가 컨텍스트에 저장되었는지 확인
        assert result["context"].user_query == sample_context.user_query
    
    def test_workflow_delegation_to_data_expert(self, lead_agent, sample_context):
        """워크플로우 2단계: Data Expert로 위임 (Requirements 1.1)"""
        # 테이블 정보가 없으면 Data Expert로 위임
        result = lead_agent.analyze_user_request(sample_context.user_query)
        
        assert result["delegation_target"] == AgentType.DATA_EXPERT
        assert "데이터" in result["delegation_message"] or "탐색" in result["delegation_message"]
    
    def test_workflow_delegation_to_sql_agent(self, lead_agent, sample_context, sample_table):
        """워크플로우 3단계: SQL Agent로 위임 (Requirements 1.1) - LLM 기반"""
        # 테이블 정보가 있으면 SQL Agent로 위임
        sample_context.identified_tables = [sample_table]
        
        # LLM 기반이므로 _determine_delegation은 context만 받음
        delegation = lead_agent._determine_delegation(sample_context)
        
        assert delegation["target"] == AgentType.SQL
        assert "SQL" in delegation["message"] or "쿼리" in delegation["message"]
    
    def test_workflow_data_exploration(self, data_expert, sample_context):
        """워크플로우: 데이터 탐색 (Requirements 2.1, 2.2, 2.3)"""
        # 비즈니스 의도 설정
        sample_context.business_intent = {
            "entity": "product",
            "metric": "revenue",
            "time": "last_month",
            "action": "top_k"
        }
        
        # 테이블 데이터 시뮬레이션
        mock_tables = [
            {
                "database": "analytics",
                "name": "sales_transactions",
                "columns": [
                    {"name": "product_id", "type": "string"},
                    {"name": "revenue", "type": "double"},
                    {"name": "sale_date", "type": "timestamp"},
                ],
                "partition_keys": ["year", "month"]
            }
        ]
        
        # 테이블 매칭 테스트
        matched = data_expert._match_tables_to_requirements(mock_tables, sample_context)
        
        assert len(matched) > 0
        assert matched[0].relevance_score > 0.3
        assert matched[0].database == "analytics"
    
    def test_workflow_sql_generation(self, sql_agent, sample_context, sample_table):
        """워크플로우: SQL 생성 준비 (Requirements 3.1, 3.2 - LLM 기반)"""
        # 컨텍스트 설정
        sample_context.identified_tables = [sample_table]
        sample_context.business_intent = {
            "entity": "product",
            "metric": "revenue",
            "time": "last_month",
            "action": "top_k",
            "raw_keywords": ["5"]
        }
        
        # LLM 기반 SQL 생성 준비 (카탈로그 컨텍스트 업데이트)
        result = sql_agent.generate_and_execute_sql(sample_context)
        
        assert result["success"] is True
        assert result["ready_for_execution"] is True
        # 카탈로그 컨텍스트에 테이블 정보가 포함되어야 함
        catalog_context = sql_agent.get_catalog_context()
        assert "analytics.sales_transactions" in catalog_context
        assert "product_id" in catalog_context
    
    def test_workflow_result_integration(self, lead_agent, sample_context, sample_table):
        """워크플로우: 결과 통합 (Requirements 1.2)"""
        # 완료된 컨텍스트 설정
        sample_context.identified_tables = [sample_table]
        sample_context.generated_sql = "SELECT product_id, SUM(revenue) FROM sales GROUP BY product_id"
        sample_context.query_execution_id = "test-execution-id"
        sample_context.results = [
            {"product_id": "P001", "total_revenue": 10000},
            {"product_id": "P002", "total_revenue": 8000},
            {"product_id": "P003", "total_revenue": 6000},
        ]
        
        # 결과 통합
        response = lead_agent.integrate_results(sample_context)
        
        assert "성공" in response or "완료" in response
        assert "3" in response  # 결과 행 수


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


class TestErrorScenarios:
    """에러 상황 및 복구 시나리오 테스트 (Requirements 1.4)"""
    
    @pytest.fixture
    def lead_agent(self):
        agent = LeadAgent.__new__(LeadAgent)
        agent.workflow_state = WorkflowState()
        return agent
    
    def test_error_response_for_permission_error(self, lead_agent):
        """권한 오류 시 적절한 응답 생성"""
        context = AnalysisContext(user_query="매출 분석")
        context.add_error("Access Denied: 권한이 없습니다")
        
        response = lead_agent._generate_error_response(context)
        
        assert "오류" in response or "발생" in response
        assert "다음 단계" in response or "제안" in response
        suggestions = lead_agent._get_error_suggestions(context.error_messages)
        assert any("권한" in s or "IAM" in s for s in suggestions)
    
    def test_error_response_for_table_not_found(self, lead_agent):
        """테이블 없음 오류 시 적절한 응답 생성"""
        context = AnalysisContext(user_query="매출 분석")
        context.add_error("테이블을 찾을 수 없습니다")
        
        response = lead_agent._generate_error_response(context)
        
        assert "오류" in response or "발생" in response
        suggestions = lead_agent._get_error_suggestions(context.error_messages)
        assert any("테이블" in s for s in suggestions)
    
    def test_error_response_for_query_timeout(self, lead_agent):
        """쿼리 타임아웃 시 적절한 응답 생성"""
        context = AnalysisContext(user_query="대용량 데이터 분석")
        context.add_error("쿼리 실행 타임아웃")
        
        response = lead_agent._generate_error_response(context)
        
        suggestions = lead_agent._get_error_suggestions(context.error_messages)
        assert any("범위" in s or "축소" in s for s in suggestions)
    
    def test_error_response_for_sql_syntax_error(self, lead_agent):
        """SQL 구문 오류 시 적절한 응답 생성"""
        context = AnalysisContext(user_query="매출 분석")
        context.add_error("SQL 쿼리 구문 오류")
        
        response = lead_agent._generate_error_response(context)
        
        suggestions = lead_agent._get_error_suggestions(context.error_messages)
        assert len(suggestions) > 0
    
    def test_multiple_errors_handling(self, lead_agent):
        """다중 에러 처리"""
        context = AnalysisContext(user_query="매출 분석")
        context.add_error("첫 번째 오류")
        context.add_error("두 번째 오류")
        
        response = lead_agent._generate_error_response(context)
        
        assert "첫 번째 오류" in response
        assert "두 번째 오류" in response


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


class TestNaturalLanguageQueryScenarios:
    """다양한 자연어 쿼리 시나리오 테스트"""
    
    @pytest.fixture
    def lead_agent(self):
        agent = LeadAgent.__new__(LeadAgent)
        agent.workflow_state = WorkflowState()
        return agent
    
    @pytest.fixture
    def sql_agent(self):
        return SQLAgent(model_id="test-model")
    
    def test_top_k_query_scenario(self, lead_agent, sql_agent):
        """Top-K 쿼리 시나리오 (LLM 기반)"""
        query = "지난달 매출 상위 10개 상품"
        
        # Lead Agent 분석 (LLM 기반이므로 규칙 기반 의도 추출 없음)
        result = lead_agent.analyze_user_request(query)
        assert result["success"] is True
        assert result["context"].user_query == query
        
        # SQL Agent는 LLM 기반으로 동작하므로 카탈로그 컨텍스트 업데이트 테스트
        sample_table = TableInfo(
            database="analytics",
            table="sales",
            columns=[ColumnInfo(name="product_id", type="string")],
            partition_keys=[],
            relevance_score=0.9
        )
        sql_agent.update_catalog_context([sample_table])
        catalog_context = sql_agent.get_catalog_context()
        assert "analytics.sales" in catalog_context
    
    def test_trend_analysis_scenario(self, lead_agent, sql_agent):
        """추이 분석 시나리오 (LLM 기반)"""
        query = "최근 매출 추이 분석"
        
        # LLM 기반이므로 사용자 요청 분석만 테스트
        result = lead_agent.analyze_user_request(query)
        assert result["success"] is True
        assert result["context"].user_query == query
    
    def test_comparison_scenario(self, lead_agent, sql_agent):
        """비교 분석 시나리오 (LLM 기반)"""
        query = "카테고리별 매출 비교"
        
        # LLM 기반이므로 사용자 요청 분석만 테스트
        result = lead_agent.analyze_user_request(query)
        assert result["success"] is True
        assert result["context"].user_query == query
    
    def test_customer_analysis_scenario(self, lead_agent, sql_agent):
        """고객 분석 시나리오 (LLM 기반)"""
        query = "이번달 고객 주문 건수"
        
        # LLM 기반이므로 사용자 요청 분석만 테스트
        result = lead_agent.analyze_user_request(query)
        assert result["success"] is True
        assert result["context"].user_query == query
    
    def test_event_analysis_scenario(self, lead_agent, sql_agent):
        """이벤트 분석 시나리오 (LLM 기반)"""
        query = "어제 방문자 클릭 이벤트 통계"
        
        # LLM 기반이므로 사용자 요청 분석만 테스트
        result = lead_agent.analyze_user_request(query)
        assert result["success"] is True
        assert result["context"].user_query == query


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


class TestAgentResultIntegration:
    """에이전트 결과 통합 테스트 (Requirements 1.2)"""
    
    @pytest.fixture
    def lead_agent(self):
        agent = LeadAgent.__new__(LeadAgent)
        agent.workflow_state = WorkflowState()
        return agent
    
    def test_integrate_successful_results(self, lead_agent):
        """성공적인 결과 통합"""
        context = AnalysisContext(user_query="매출 분석")
        
        results = [
            AgentResult(
                agent_type=AgentType.DATA_EXPERT,
                success=True,
                data={"tables_count": 3, "databases_count": 1},
                execution_time_ms=500
            ),
            AgentResult(
                agent_type=AgentType.SQL,
                success=True,
                data={"sql_query": "SELECT *", "row_count": 100, "execution_id": "test-id"},
                execution_time_ms=1500
            )
        ]
        
        integrated = lead_agent.integrate_agent_results(results, context)
        
        assert integrated["success"] is True
        assert integrated["data_exploration"]["tables_found"] == 3
        assert integrated["sql_execution"]["row_count"] == 100
        assert len(integrated["errors"]) == 0
    
    def test_integrate_partial_failure(self, lead_agent):
        """부분 실패 결과 통합"""
        context = AnalysisContext(user_query="매출 분석")
        
        results = [
            AgentResult(
                agent_type=AgentType.DATA_EXPERT,
                success=True,
                data={"tables_count": 3, "databases_count": 1}
            ),
            AgentResult(
                agent_type=AgentType.SQL,
                success=False,
                error_message="쿼리 실행 실패"
            )
        ]
        
        integrated = lead_agent.integrate_agent_results(results, context)
        
        assert integrated["success"] is False
        assert len(integrated["errors"]) == 1
        assert "쿼리 실행 실패" in integrated["errors"]
    
    def test_integration_summary_generation(self, lead_agent):
        """통합 요약 생성"""
        context = AnalysisContext(user_query="매출 분석")
        
        results = [
            AgentResult(
                agent_type=AgentType.DATA_EXPERT,
                success=True,
                data={"tables_count": 3, "databases_count": 2}
            ),
            AgentResult(
                agent_type=AgentType.SQL,
                success=True,
                data={"row_count": 50},
                execution_time_ms=1000
            )
        ]
        
        integrated = lead_agent.integrate_agent_results(results, context)
        
        assert integrated["summary"] != ""
        assert "데이터 탐색" in integrated["summary"] or "DB" in integrated["summary"]
        assert "SQL" in integrated["summary"] or "행" in integrated["summary"]


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
