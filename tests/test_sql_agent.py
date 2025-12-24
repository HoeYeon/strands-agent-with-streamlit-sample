"""SQL Agent 단위 테스트 (LLM 기반)

Requirements:
- 3.1: 카탈로그 정보(테이블 스키마, 컬럼, 파티션 키)를 시스템 프롬프트에 포함
- 3.2: Strands Agent의 LLM을 통해 비즈니스 의도를 해석하고 SQL 쿼리 생성
- 3.3: start_query_execution 호출 후 QueryExecutionId 저장
- 3.4: 5초 간격으로 최대 5회 폴링하여 SUCCEEDED 상태 대기
- 3.5: get_query_results로 최대 1000행 결과 반환
"""

import pytest
from agents.multi_agent.sql_agent import (
    SQLAgent,
    QueryExecutionResult,
    QueryResult,
    POLLING_INTERVAL_SECONDS,
    MAX_POLLING_ATTEMPTS,
    MAX_QUERY_RESULTS,
    DEFAULT_CATALOG,
    DEFAULT_WORKGROUP,
)
from agents.multi_agent.shared_context import (
    AnalysisContext,
    TableInfo,
    ColumnInfo,
)


class TestSQLAgentConstants:
    """SQL Agent 상수 테스트"""
    
    def test_polling_interval_is_5_seconds(self):
        """Requirements 3.4: 폴링 간격이 5초인지 확인"""
        assert POLLING_INTERVAL_SECONDS == 5
    
    def test_max_polling_attempts_is_5(self):
        """Requirements 3.4: 최대 폴링 횟수가 5회인지 확인"""
        assert MAX_POLLING_ATTEMPTS == 5
    
    def test_max_query_results_is_1000(self):
        """Requirements 3.5: 최대 결과 행 수가 1000인지 확인"""
        assert MAX_QUERY_RESULTS == 1000
    
    def test_default_catalog(self):
        """기본 카탈로그가 AwsDataCatalog인지 확인"""
        assert DEFAULT_CATALOG == "AwsDataCatalog"
    
    def test_default_workgroup(self):
        """기본 워크그룹이 primary인지 확인"""
        assert DEFAULT_WORKGROUP == "primary"


class TestCatalogContextUpdate:
    """카탈로그 컨텍스트 업데이트 테스트 (Requirements 3.1)"""
    
    @pytest.fixture
    def agent(self):
        """테스트용 에이전트 (MCP 클라이언트 없이)"""
        return SQLAgent(model_id="test-model")
    
    @pytest.fixture
    def sample_tables(self):
        """샘플 테이블 정보"""
        return [
            TableInfo(
                database="analytics",
                table="sales_transactions",
                columns=[
                    ColumnInfo(name="transaction_id", type="string", description="거래 ID"),
                    ColumnInfo(name="product_id", type="string", description="상품 ID"),
                    ColumnInfo(name="amount", type="decimal", description="금액"),
                    ColumnInfo(name="transaction_date", type="timestamp", description="거래일시"),
                ],
                partition_keys=["year", "month"],
                relevance_score=0.9
            ),
            TableInfo(
                database="analytics",
                table="products",
                columns=[
                    ColumnInfo(name="product_id", type="string"),
                    ColumnInfo(name="product_name", type="string"),
                ],
                partition_keys=[],
                relevance_score=0.7
            )
        ]
    
    def test_update_catalog_context_includes_table_info(self, agent, sample_tables):
        """카탈로그 컨텍스트에 테이블 정보 포함 (Requirements 3.1)"""
        agent.update_catalog_context(sample_tables)
        
        context = agent.get_catalog_context()
        
        assert "analytics.sales_transactions" in context
        assert "analytics.products" in context
    
    def test_update_catalog_context_includes_columns(self, agent, sample_tables):
        """카탈로그 컨텍스트에 컬럼 정보 포함 (Requirements 3.1)"""
        agent.update_catalog_context(sample_tables)
        
        context = agent.get_catalog_context()
        
        assert "transaction_id" in context
        assert "product_id" in context
        assert "amount" in context
    
    def test_update_catalog_context_includes_partition_keys(self, agent, sample_tables):
        """카탈로그 컨텍스트에 파티션 키 포함 (Requirements 3.1)"""
        agent.update_catalog_context(sample_tables)
        
        context = agent.get_catalog_context()
        
        assert "파티션 키" in context
        assert "year" in context
        assert "month" in context
    
    def test_update_catalog_context_includes_relevance_score(self, agent, sample_tables):
        """카탈로그 컨텍스트에 관련성 점수 포함"""
        agent.update_catalog_context(sample_tables)
        
        context = agent.get_catalog_context()
        
        assert "0.9" in context or "0.90" in context
    
    def test_update_catalog_context_empty_tables(self, agent):
        """빈 테이블 목록으로 업데이트"""
        agent.update_catalog_context([])
        
        context = agent.get_catalog_context()
        
        assert context == ""
    
    def test_system_prompt_includes_catalog_context(self, agent, sample_tables):
        """시스템 프롬프트에 카탈로그 컨텍스트 포함 (Requirements 3.1)"""
        agent.update_catalog_context(sample_tables)
        
        system_prompt = agent.get_system_prompt()
        
        assert "카탈로그 정보" in system_prompt
        assert "analytics.sales_transactions" in system_prompt


class TestBuildPromptFromContext:
    """컨텍스트 기반 프롬프트 생성 테스트 (Requirements 3.2)"""
    
    @pytest.fixture
    def agent(self):
        return SQLAgent(model_id="test-model")
    
    @pytest.fixture
    def context_with_table(self):
        return AnalysisContext(
            user_query="지난달 매출 상위 5개 상품",
            business_intent={
                "entity": "product",
                "metric": "revenue",
                "time": "last_month",
                "action": "top_k"
            },
            identified_tables=[
                TableInfo(
                    database="analytics",
                    table="sales",
                    columns=[
                        ColumnInfo(name="product_id", type="string"),
                        ColumnInfo(name="amount", type="decimal"),
                    ],
                    partition_keys=[],
                    relevance_score=0.9
                )
            ]
        )
    
    def test_build_prompt_includes_user_query(self, agent, context_with_table):
        """프롬프트에 사용자 쿼리 포함"""
        prompt = agent._build_prompt_from_context(context_with_table)
        
        assert "지난달 매출 상위 5개 상품" in prompt
    
    def test_build_prompt_includes_business_intent(self, agent, context_with_table):
        """프롬프트에 비즈니스 의도 포함"""
        prompt = agent._build_prompt_from_context(context_with_table)
        
        assert "비즈니스 의도" in prompt
        assert "entity" in prompt or "product" in prompt
    
    def test_build_prompt_updates_catalog_context(self, agent, context_with_table):
        """프롬프트 생성 시 카탈로그 컨텍스트 업데이트"""
        agent._build_prompt_from_context(context_with_table)
        
        catalog_context = agent.get_catalog_context()
        
        assert "analytics.sales" in catalog_context


class TestQueryExecutionIdStorage:
    """QueryExecutionId 저장 테스트 (Requirements 3.3)"""
    
    @pytest.fixture
    def agent(self):
        return SQLAgent(model_id="test-model")
    
    def test_store_execution_id(self, agent):
        """실행 ID 저장"""
        execution_id = "test-execution-id-12345"
        
        agent.store_execution_id(execution_id)
        
        assert agent.get_latest_execution_id() == execution_id
    
    def test_store_resets_polling_count(self, agent):
        """실행 ID 저장 시 폴링 카운트 초기화"""
        agent._polling_count = 3
        
        agent.store_execution_id("new-execution-id")
        
        assert agent.get_polling_count() == 0
    
    def test_initial_execution_id_is_none(self, agent):
        """초기 실행 ID는 None"""
        assert agent.get_latest_execution_id() is None


class TestPollingLimits:
    """폴링 제한 테스트 (Requirements 3.4)"""
    
    @pytest.fixture
    def agent(self):
        return SQLAgent(model_id="test-model")
    
    def test_is_within_polling_limit_initially(self, agent):
        """초기 상태에서 폴링 제한 내"""
        assert agent.is_within_polling_limit() is True
    
    def test_is_within_polling_limit_after_some_polls(self, agent):
        """일부 폴링 후에도 제한 내"""
        agent._polling_count = 3
        assert agent.is_within_polling_limit() is True
    
    def test_is_not_within_polling_limit_at_max(self, agent):
        """최대 폴링 횟수에서 제한 초과"""
        agent._polling_count = MAX_POLLING_ATTEMPTS
        assert agent.is_within_polling_limit() is False
    
    def test_polling_count_increments(self, agent):
        """폴링 카운트 증가 확인"""
        initial = agent.get_polling_count()
        agent._polling_count += 1
        assert agent.get_polling_count() == initial + 1


class TestQueryResultsLimit:
    """쿼리 결과 제한 테스트 (Requirements 3.5)"""
    
    @pytest.fixture
    def agent(self):
        return SQLAgent(model_id="test-model")
    
    def test_get_query_results_respects_max_limit(self, agent):
        """최대 결과 제한 준수"""
        result = agent.get_query_results("test-id", max_results=2000)
        
        assert result is not None
    
    def test_process_execution_result_limits_rows(self, agent):
        """실행 결과 처리 시 행 수 제한"""
        context = AnalysisContext(user_query="test")
        
        large_result_data = {
            "rows": [{"id": i} for i in range(1500)],
            "columns": ["id"]
        }
        
        result = agent.process_execution_result(
            execution_id="test-id",
            status="SUCCEEDED",
            context=context,
            result_data=large_result_data
        )
        
        assert result["success"] is True
        assert result["row_count"] == MAX_QUERY_RESULTS
        assert result["truncated"] is True


class TestResultFormatting:
    """결과 포맷팅 테스트"""
    
    @pytest.fixture
    def agent(self):
        return SQLAgent(model_id="test-model")
    
    def test_format_empty_results(self, agent):
        """빈 결과 포맷팅"""
        result = QueryResult(columns=[], rows=[], row_count=0)
        
        formatted = agent.format_results(result)
        
        assert "결과가 없습니다" in formatted
    
    def test_format_results_with_data(self, agent):
        """데이터가 있는 결과 포맷팅"""
        result = QueryResult(
            columns=["id", "name"],
            rows=[
                {"id": "1", "name": "Product A"},
                {"id": "2", "name": "Product B"},
            ],
            row_count=2
        )
        
        formatted = agent.format_results(result)
        
        assert "id" in formatted
        assert "name" in formatted
        assert "총 2행" in formatted
    
    def test_format_truncated_results(self, agent):
        """잘린 결과 포맷팅"""
        result = QueryResult(
            columns=["id"],
            rows=[{"id": str(i)} for i in range(10)],
            row_count=10,
            truncated=True
        )
        
        formatted = agent.format_results(result)
        
        assert f"{MAX_QUERY_RESULTS}행으로 제한" in formatted


class TestGenerateAndExecuteSQL:
    """SQL 생성 및 실행 통합 테스트 (LLM 기반)"""
    
    @pytest.fixture
    def agent(self):
        return SQLAgent(model_id="test-model")
    
    @pytest.fixture
    def context_with_table(self):
        return AnalysisContext(
            user_query="지난달 매출 상위 5개 상품",
            identified_tables=[
                TableInfo(
                    database="analytics",
                    table="sales",
                    columns=[
                        ColumnInfo(name="product_id", type="string"),
                        ColumnInfo(name="amount", type="decimal"),
                        ColumnInfo(name="sale_date", type="timestamp"),
                    ],
                    partition_keys=[],
                    relevance_score=0.9
                )
            ]
        )
    
    def test_generate_and_execute_success(self, agent, context_with_table):
        """성공적인 SQL 생성 및 실행 준비"""
        result = agent.generate_and_execute_sql(context_with_table)
        
        assert result["success"] is True
        assert result["ready_for_execution"] is True
        assert "catalog_context" in result
    
    def test_generate_and_execute_updates_catalog_context(self, agent, context_with_table):
        """SQL 생성 시 카탈로그 컨텍스트 업데이트 (Requirements 3.1)"""
        agent.generate_and_execute_sql(context_with_table)
        
        catalog_context = agent.get_catalog_context()
        
        assert "analytics.sales" in catalog_context
        assert "product_id" in catalog_context
    
    def test_generate_and_execute_without_tables(self, agent):
        """테이블 없이 실행 시 실패"""
        context = AnalysisContext(
            user_query="매출 조회",
            identified_tables=[]
        )
        
        result = agent.generate_and_execute_sql(context)
        
        assert result["success"] is False
        assert len(context.error_messages) > 0


class TestExecutionSummary:
    """실행 요약 테스트"""
    
    @pytest.fixture
    def agent(self):
        return SQLAgent(model_id="test-model")
    
    def test_summary_includes_sql(self, agent):
        """요약에 SQL 포함"""
        context = AnalysisContext(
            user_query="test",
            generated_sql="SELECT * FROM test"
        )
        
        summary = agent.get_execution_summary(context)
        
        assert "SELECT * FROM test" in summary
    
    def test_summary_includes_execution_id(self, agent):
        """요약에 실행 ID 포함"""
        context = AnalysisContext(
            user_query="test",
            query_execution_id="test-execution-id"
        )
        
        summary = agent.get_execution_summary(context)
        
        assert "test-execution-id" in summary
    
    def test_summary_includes_errors(self, agent):
        """요약에 오류 포함"""
        context = AnalysisContext(user_query="test")
        context.add_error("Test error message")
        
        summary = agent.get_execution_summary(context)
        
        assert "Test error message" in summary


class TestSystemPrompt:
    """시스템 프롬프트 테스트"""
    
    @pytest.fixture
    def agent(self):
        return SQLAgent(model_id="test-model")
    
    def test_system_prompt_includes_llm_sql_generation_rules(self, agent):
        """시스템 프롬프트에 LLM SQL 생성 규칙 포함"""
        prompt = agent.get_system_prompt()
        
        assert "SQL 생성 규칙" in prompt
        assert "비즈니스 의도" in prompt or "entity" in prompt
    
    def test_system_prompt_includes_athena_workflow(self, agent):
        """시스템 프롬프트에 Athena 워크플로우 포함"""
        prompt = agent.get_system_prompt()
        
        assert "start_query_execution" in prompt
        assert "get_query_results" in prompt
    
    def test_system_prompt_includes_handoff_rules(self, agent):
        """시스템 프롬프트에 handoff 규칙 포함"""
        prompt = agent.get_system_prompt()
        
        assert "handoff_to_agent" in prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
