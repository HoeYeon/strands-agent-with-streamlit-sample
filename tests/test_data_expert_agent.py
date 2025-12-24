"""Data Expert Agent 단위 테스트 (LLM 기반)

Requirements:
- 2.1: MCP 도구를 사용하여 AWS Athena 카탈로그 조회
- 2.2: 최대 50개/DB 테이블 메타데이터 수집
- 2.3: Strands Agent의 LLM을 통해 테이블 스키마 분석 및 적합한 테이블 추천
- 2.5: 파티션 키 및 최적화 힌트 제공
"""

import pytest
from agents.multi_agent.data_expert_agent import (
    DataExpertAgent,
    MAX_TABLES_PER_DATABASE,
)
from agents.multi_agent.shared_context import (
    AnalysisContext,
    TableInfo,
    ColumnInfo,
)


class TestDataExpertAgentConstants:
    """Data Expert Agent 상수 테스트"""
    
    def test_max_tables_per_database_is_50(self):
        """Requirements 2.2: 최대 테이블 수가 50인지 확인"""
        assert MAX_TABLES_PER_DATABASE == 50


class TestCatalogInfoUpdate:
    """카탈로그 정보 업데이트 테스트 (Requirements 2.1, 2.3)"""
    
    @pytest.fixture
    def agent(self):
        """테스트용 에이전트 (MCP 클라이언트 없이)"""
        return DataExpertAgent(model_id="test-model")
    
    def test_update_catalog_info(self, agent):
        """카탈로그 정보 업데이트"""
        catalog_info = "테스트 카탈로그 정보"
        
        agent.update_catalog_info(catalog_info)
        
        assert agent.get_catalog_info() == catalog_info
    
    def test_system_prompt_includes_catalog_info(self, agent):
        """시스템 프롬프트에 카탈로그 정보 포함 (Requirements 2.3)"""
        catalog_info = "analytics.sales_transactions 테이블"
        agent.update_catalog_info(catalog_info)
        
        system_prompt = agent.get_system_prompt()
        
        assert "카탈로그 정보" in system_prompt
        assert "analytics.sales_transactions" in system_prompt
    
    def test_initial_catalog_info_is_empty(self, agent):
        """초기 카탈로그 정보는 빈 문자열"""
        assert agent.get_catalog_info() == ""


class TestBuildPromptFromContext:
    """컨텍스트 기반 프롬프트 생성 테스트"""
    
    @pytest.fixture
    def agent(self):
        return DataExpertAgent(model_id="test-model")
    
    def test_build_prompt_includes_user_query(self, agent):
        """프롬프트에 사용자 쿼리 포함"""
        context = AnalysisContext(user_query="지난달 매출 상위 5개 상품")
        
        prompt = agent._build_prompt_from_context(context)
        
        assert "지난달 매출 상위 5개 상품" in prompt
    
    def test_build_prompt_includes_business_intent(self, agent):
        """프롬프트에 비즈니스 의도 포함"""
        context = AnalysisContext(
            user_query="매출 분석",
            business_intent={
                "entity": "product",
                "metric": "revenue"
            }
        )
        
        prompt = agent._build_prompt_from_context(context)
        
        assert "파악된 의도" in prompt
        assert "entity" in prompt or "product" in prompt


class TestColumnInfoExtraction:
    """컬럼 정보 추출 테스트"""
    
    @pytest.fixture
    def agent(self):
        return DataExpertAgent(model_id="test-model")
    
    def test_extract_from_dict_columns(self, agent):
        """딕셔너리 형태 컬럼에서 정보 추출"""
        table_data = {
            "columns": [
                {"name": "id", "type": "string"},
                {"name": "amount", "type": "decimal", "description": "금액"},
            ]
        }
        
        columns = agent._extract_column_info(table_data)
        
        assert len(columns) == 2
        assert columns[0].name == "id"
        assert columns[0].type == "string"
        assert columns[1].description == "금액"
    
    def test_extract_from_string_columns(self, agent):
        """문자열 형태 컬럼에서 정보 추출"""
        table_data = {
            "columns": ["id", "name", "amount"]
        }
        
        columns = agent._extract_column_info(table_data)
        
        assert len(columns) == 3
        assert columns[0].name == "id"
        assert columns[0].type == "string"  # 기본값


class TestPartitionKeyExtraction:
    """파티션 키 추출 테스트 (Requirements 2.5)"""
    
    @pytest.fixture
    def agent(self):
        return DataExpertAgent(model_id="test-model")
    
    def test_extract_string_partition_keys(self, agent):
        """문자열 형태 파티션 키 추출"""
        table_data = {
            "partition_keys": ["year", "month", "day"]
        }
        
        partition_keys = agent._extract_partition_keys(table_data)
        
        assert partition_keys == ["year", "month", "day"]
    
    def test_extract_dict_partition_keys(self, agent):
        """딕셔너리 형태 파티션 키 추출"""
        table_data = {
            "partition_keys": [
                {"name": "year", "type": "int"},
                {"name": "month", "type": "int"},
            ]
        }
        
        partition_keys = agent._extract_partition_keys(table_data)
        
        assert partition_keys == ["year", "month"]
    
    def test_empty_partition_keys(self, agent):
        """파티션 키가 없는 경우"""
        table_data = {}
        
        partition_keys = agent._extract_partition_keys(table_data)
        
        assert partition_keys == []


class TestOptimizationHints:
    """최적화 힌트 생성 테스트 (Requirements 2.5)"""
    
    @pytest.fixture
    def agent(self):
        return DataExpertAgent(model_id="test-model")
    
    def test_generate_partition_hints(self, agent):
        """파티션 키 기반 힌트 생성"""
        tables = [
            TableInfo(
                database="analytics",
                table="sales",
                columns=[
                    ColumnInfo(name="amount", type="decimal"),
                    ColumnInfo(name="sale_date", type="timestamp"),
                ],
                partition_keys=["year", "month"],
                relevance_score=0.8
            )
        ]
        
        hints = agent._generate_optimization_hints(tables)
        
        assert len(hints) == 1
        assert len(hints[0]["partition_hints"]) > 0
        assert "year" in hints[0]["partition_hints"][0]
    
    def test_generate_date_column_hints(self, agent):
        """날짜 컬럼 기반 힌트 생성"""
        tables = [
            TableInfo(
                database="analytics",
                table="events",
                columns=[
                    ColumnInfo(name="event_id", type="string"),
                    ColumnInfo(name="created_at", type="timestamp"),
                ],
                partition_keys=[],
                relevance_score=0.7
            )
        ]
        
        hints = agent._generate_optimization_hints(tables)
        
        assert len(hints) == 1
        assert len(hints[0]["date_column_hints"]) > 0


class TestDateColumnDetection:
    """날짜 컬럼 감지 테스트"""
    
    @pytest.fixture
    def agent(self):
        return DataExpertAgent(model_id="test-model")
    
    def test_detect_timestamp_type(self, agent):
        """timestamp 타입 컬럼 감지"""
        columns = [
            ColumnInfo(name="event_time", type="timestamp"),
            ColumnInfo(name="id", type="string"),
        ]
        
        date_columns = agent._find_date_columns(columns)
        
        assert "event_time" in date_columns
    
    def test_detect_date_name_pattern(self, agent):
        """날짜 이름 패턴 컬럼 감지"""
        columns = [
            ColumnInfo(name="created_at", type="string"),
            ColumnInfo(name="updated_date", type="string"),
            ColumnInfo(name="id", type="string"),
        ]
        
        date_columns = agent._find_date_columns(columns)
        
        assert "created_at" in date_columns
        assert "updated_date" in date_columns
        assert "id" not in date_columns


class TestNumericColumnDetection:
    """숫자형 컬럼 감지 테스트"""
    
    @pytest.fixture
    def agent(self):
        return DataExpertAgent(model_id="test-model")
    
    def test_detect_numeric_types(self, agent):
        """숫자형 타입 컬럼 감지"""
        columns = [
            ColumnInfo(name="amount", type="decimal"),
            ColumnInfo(name="count", type="bigint"),
            ColumnInfo(name="price", type="double"),
            ColumnInfo(name="name", type="string"),
        ]
        
        numeric_columns = agent._find_numeric_columns(columns)
        
        assert "amount" in numeric_columns
        assert "count" in numeric_columns
        assert "price" in numeric_columns
        assert "name" not in numeric_columns


class TestTableMatching:
    """테이블 매칭 테스트 (Requirements 2.3 - LLM 기반)"""
    
    @pytest.fixture
    def agent(self):
        return DataExpertAgent(model_id="test-model")
    
    def test_match_returns_max_5_tables(self, agent):
        """최대 5개 테이블만 반환하는지 확인"""
        tables = [
            {
                "database": f"db{i}",
                "name": f"sales_table_{i}",
                "columns": [
                    {"name": "amount", "type": "decimal"},
                    {"name": "date", "type": "timestamp"},
                ],
                "relevance_score": 0.8
            }
            for i in range(10)
        ]
        
        context = AnalysisContext(user_query="매출 분석")
        
        matched = agent._match_tables_to_requirements(tables, context)
        
        assert len(matched) <= 5
    
    def test_match_includes_relevance_score(self, agent):
        """매칭 결과에 관련성 점수가 포함되는지 확인"""
        tables = [
            {
                "database": "analytics",
                "name": "sales",
                "columns": [{"name": "amount", "type": "decimal"}],
                "relevance_score": 0.9
            }
        ]
        
        context = AnalysisContext(user_query="매출 분석")
        
        matched = agent._match_tables_to_requirements(tables, context)
        
        if matched:
            assert hasattr(matched[0], "relevance_score")
            assert 0.0 <= matched[0].relevance_score <= 1.0


class TestExploreCatalog:
    """카탈로그 탐색 테스트 (LLM 기반)"""
    
    @pytest.fixture
    def agent(self):
        return DataExpertAgent(model_id="test-model")
    
    def test_explore_catalog_returns_prompt(self, agent):
        """카탈로그 탐색이 프롬프트를 반환하는지 확인"""
        context = AnalysisContext(user_query="매출 분석")
        
        result = agent.explore_catalog(context)
        
        assert result["success"] is True
        assert result["ready_for_exploration"] is True
        assert "prompt" in result


class TestProcessCatalogResults:
    """카탈로그 결과 처리 테스트"""
    
    @pytest.fixture
    def agent(self):
        return DataExpertAgent(model_id="test-model")
    
    def test_process_catalog_results_success(self, agent):
        """카탈로그 결과 처리 성공"""
        context = AnalysisContext(user_query="매출 분석")
        tables_data = [
            {
                "database": "analytics",
                "name": "sales",
                "columns": [
                    {"name": "amount", "type": "decimal"},
                    {"name": "sale_date", "type": "timestamp"},
                ],
                "partition_keys": ["year", "month"],
                "relevance_score": 0.9
            }
        ]
        
        result = agent.process_catalog_results(tables_data, context)
        
        assert result["success"] is True
        assert len(result["tables"]) == 1
        assert len(context.identified_tables) == 1
        assert "optimization_hints" in result


class TestFormatTableInfo:
    """테이블 정보 포맷팅 테스트"""
    
    @pytest.fixture
    def agent(self):
        return DataExpertAgent(model_id="test-model")
    
    def test_format_table_info_includes_columns(self, agent):
        """포맷팅된 정보에 컬럼 포함"""
        tables = [
            TableInfo(
                database="analytics",
                table="sales",
                columns=[
                    ColumnInfo(name="amount", type="decimal"),
                    ColumnInfo(name="product_id", type="string"),
                ],
                partition_keys=["year"],
                relevance_score=0.9
            )
        ]
        
        formatted = agent.format_table_info_for_sql_agent(tables)
        
        assert "analytics.sales" in formatted
        assert "amount" in formatted
        assert "product_id" in formatted
    
    def test_format_table_info_includes_partition_keys(self, agent):
        """포맷팅된 정보에 파티션 키 포함"""
        tables = [
            TableInfo(
                database="analytics",
                table="sales",
                columns=[ColumnInfo(name="amount", type="decimal")],
                partition_keys=["year", "month"],
                relevance_score=0.9
            )
        ]
        
        formatted = agent.format_table_info_for_sql_agent(tables)
        
        assert "파티션 키" in formatted
        assert "year" in formatted
        assert "month" in formatted
    
    def test_format_empty_tables(self, agent):
        """빈 테이블 목록 포맷팅"""
        formatted = agent.format_table_info_for_sql_agent([])
        
        assert formatted == ""


class TestSystemPrompt:
    """시스템 프롬프트 테스트"""
    
    @pytest.fixture
    def agent(self):
        return DataExpertAgent(model_id="test-model")
    
    def test_system_prompt_includes_llm_matching_rules(self, agent):
        """시스템 프롬프트에 LLM 기반 매칭 규칙 포함"""
        prompt = agent.get_system_prompt()
        
        assert "LLM 기반 테이블 매칭" in prompt
        assert "entity" in prompt
        assert "metric" in prompt
    
    def test_system_prompt_includes_metadata_rules(self, agent):
        """시스템 프롬프트에 메타데이터 탐색 규칙 포함"""
        prompt = agent.get_system_prompt()
        
        assert "list_databases" in prompt
        assert "list_table_metadata" in prompt
        assert "max_results=50" in prompt
    
    def test_system_prompt_includes_handoff_rules(self, agent):
        """시스템 프롬프트에 handoff 규칙 포함"""
        prompt = agent.get_system_prompt()
        
        assert "handoff_to_agent" in prompt
        assert "sql_agent" in prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
