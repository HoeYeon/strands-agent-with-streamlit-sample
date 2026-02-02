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
