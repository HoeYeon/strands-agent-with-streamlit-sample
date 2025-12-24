"""Data Expert Agent - 데이터 카탈로그 탐색 전문가 (LLM 기반)

AWS Athena 데이터 카탈로그를 탐색하고 LLM을 통해 비즈니스 요구사항에 적합한 테이블을 식별합니다.

Requirements:
- 2.1: MCP 도구를 사용하여 AWS Athena 카탈로그의 데이터베이스 조회
- 2.2: 각 데이터베이스에서 최대 50개씩 테이블 메타데이터 수집
- 2.3: Strands Agent의 LLM을 통해 테이블 스키마와 컬럼 정보를 분석하여 가장 적합한 테이블 추천
- 2.5: 파티션 키와 날짜 컬럼 정보를 식별하여 SQL 최적화 힌트 제공
"""

from typing import Any, Dict, List, Optional

from strands import Agent

from .base_agent import BaseMultiAgent
from .shared_context import AnalysisContext, TableInfo, ColumnInfo
from .constants import (
    MAX_TABLES_PER_DATABASE,
    DATE_TIME_TYPES,
    DATE_COLUMN_PATTERNS,
    NUMERIC_TYPES,
)



class DataExpertAgent(BaseMultiAgent):
    """Data Expert Agent - 데이터 카탈로그 탐색 전문가 (LLM 기반)
    
    역할:
    - MCP 도구를 사용한 데이터베이스 목록 조회 (Requirements 2.1)
    - 테이블 메타데이터 수집 - 최대 50개/DB (Requirements 2.2)
    - LLM을 통한 테이블 스키마 분석 및 적합성 판단 (Requirements 2.3)
    - 파티션 키 및 최적화 힌트 제공 (Requirements 2.5)
    """
    
    def __init__(self, model_id: str, tools: Optional[List] = None):
        self._catalog_info: str = ""  # 수집된 카탈로그 정보
        super().__init__(model_id, tools)
    
    def _setup_agent(self):
        """Data Expert Agent 초기화"""
        self.agent = Agent(
            name="data_expert",
            system_prompt=self.get_system_prompt(),
            model=self.model_id,
            tools=self.tools if self.tools else None
        )
    
    def get_system_prompt(self) -> str:
        """Data Expert Agent 시스템 프롬프트 (LLM 기반 테이블 매칭)"""
        base_prompt = """
역할: AWS Athena 데이터 카탈로그 탐색 전문가

⚠️ 핵심 규칙: MCP 도구로 탐색 완료 후에만 handoff_to_agent 호출 가능

작업 순서:
1. list_databases → 데이터베이스 목록 조회
2. list_tables (max_results=50) → 테이블 메타데이터 수집
3. get_table → 컬럼명과 타입 상세 조회  ← 추가
4. 사용자 요청 분석 → 적합한 테이블 식별
5. handoff_to_agent 호출

테이블 매칭 기준:
- 테이블/컬럼명과 요청의 연관성
- 숫자형 컬럼 (metric용), 날짜 컬럼 (필터용) 존재 여부
- 파티션 키 (성능 최적화)

handoff 규칙:
- sql_agent: 추천 테이블 + 컬럼 정보 확보 완료 시
- lead_agent: 탐색 완료 또는 3회 이상 오류 시

⚠️ sql_agent 전달 시 필수 포맷:
테이블: database.table_name
컬럼 정보:
- column_name (타입: string/bigint/timestamp 등)
- date_column (타입: string) ← 날짜지만 string인 경우 명시
파티션 키: partition_col (타입)
"""
        
        return base_prompt
    
    def get_tools(self) -> List:
        """Data Expert Agent 도구 목록"""
        return self.tools
    
    def get_agent(self) -> Agent:
        """Swarm에서 사용할 Agent 인스턴스 반환"""
        return self.agent
    
    def update_catalog_info(self, catalog_info: str) -> None:
        """수집된 카탈로그 정보 업데이트
        
        LLM이 테이블 적합성을 판단할 때 사용할 컨텍스트를 설정합니다.
        """
        self._catalog_info = catalog_info
        self._setup_agent()  # 새 컨텍스트로 에이전트 재초기화
    
    def get_catalog_info(self) -> str:
        """현재 카탈로그 정보 반환 (테스트용)"""
        return self._catalog_info


    def _build_prompt_from_context(self, context: AnalysisContext) -> str:
        """컨텍스트를 기반으로 Data Expert Agent 프롬프트 생성"""
        prompt_parts = [
            "비즈니스 요구사항 분석:",
            f"사용자 요청: {context.user_query}",
        ]
        
        if context.business_intent:
            intent_parts = []
            for key, value in context.business_intent.items():
                if value:
                    intent_parts.append(f"- {key}: {value}")
            if intent_parts:
                prompt_parts.append("파악된 의도:\n" + "\n".join(intent_parts))
        
        prompt_parts.extend([
            "",
            "수행할 작업:",
            "1. AWS Athena 카탈로그에서 관련 데이터베이스 탐색",
            "2. 비즈니스 요구사항에 맞는 테이블 식별 (최대 50개/DB)",
            "3. 테이블 메타데이터 분석 및 적합성 판단",
            "4. SQL 최적화를 위한 힌트 제공 (파티션 키, 날짜 컬럼)",
            "",
            "지금 데이터 탐색을 시작하세요."
        ])
        
        return "\n".join(prompt_parts)
    
    def explore_catalog(self, context: AnalysisContext) -> Dict[str, Any]:
        """데이터 카탈로그 탐색 및 테이블 식별 (LLM 기반)
        
        Requirements:
        - 2.1: MCP 도구를 사용한 AWS Athena 카탈로그 조회
        - 2.2: 최대 50개/DB 테이블 메타데이터 수집
        - 2.3: LLM을 통한 테이블 적합성 판단
        - 2.5: 파티션 키 및 최적화 힌트 제공
        
        실제 카탈로그 조회 및 테이블 선택은 Strands Agent가 MCP 도구를 통해 수행합니다.
        """
        try:
            # 프롬프트 생성
            prompt = self._build_prompt_from_context(context)
            
            return {
                "success": True,
                "context": context,
                "prompt": prompt,
                "ready_for_exploration": True
            }
            
        except Exception as e:
            context.add_error(f"데이터 카탈로그 탐색 실패: {str(e)}")
            return {"success": False, "context": context, "error": str(e)}
    
    def process_catalog_results(
        self,
        tables_data: List[Dict[str, Any]],
        context: AnalysisContext
    ) -> Dict[str, Any]:
        """카탈로그 조회 결과 처리
        
        LLM이 선택한 테이블 정보를 TableInfo 객체로 변환합니다.
        """
        try:
            relevant_tables = []
            
            for table_data in tables_data:
                columns = self._extract_column_info(table_data)
                partition_keys = self._extract_partition_keys(table_data)
                
                table_info = TableInfo(
                    database=table_data.get("database", ""),
                    table=table_data.get("name", ""),
                    columns=columns,
                    partition_keys=partition_keys,
                    relevance_score=table_data.get("relevance_score", 0.8)
                )
                relevant_tables.append(table_info)
            
            context.identified_tables = relevant_tables
            
            # 최적화 힌트 생성
            optimization_hints = self._generate_optimization_hints(relevant_tables)
            
            return {
                "success": True,
                "context": context,
                "tables": relevant_tables,
                "optimization_hints": optimization_hints
            }
            
        except Exception as e:
            context.add_error(f"카탈로그 결과 처리 실패: {str(e)}")
            return {"success": False, "context": context, "error": str(e)}
    
    def _match_tables_to_requirements(
        self, 
        tables: List[Dict[str, Any]], 
        context: AnalysisContext
    ) -> List[TableInfo]:
        """테이블 정보를 TableInfo 객체로 변환
        
        LLM 기반 방식에서는 LLM이 이미 적합한 테이블을 선택했으므로,
        이 메서드는 단순히 데이터 변환만 수행합니다.
        """
        relevant_tables = []
        
        for table_data in tables:
            columns = self._extract_column_info(table_data)
            partition_keys = self._extract_partition_keys(table_data)
            
            table_info = TableInfo(
                database=table_data.get("database", ""),
                table=table_data.get("name", ""),
                columns=columns,
                partition_keys=partition_keys,
                relevance_score=table_data.get("relevance_score", 0.8)
            )
            relevant_tables.append(table_info)
        
        return relevant_tables[:5]  # 상위 5개만 반환
    
    def _extract_column_info(self, table_data: Dict[str, Any]) -> List[ColumnInfo]:
        """테이블 데이터에서 컬럼 정보 추출"""
        columns = []
        raw_columns = table_data.get("columns", [])
        
        for col in raw_columns:
            if isinstance(col, str):
                columns.append(ColumnInfo(name=col, type="string"))
            elif isinstance(col, dict):
                columns.append(ColumnInfo(
                    name=col.get("name", ""),
                    type=col.get("type", "string"),
                    description=col.get("description")
                ))
            elif hasattr(col, "name"):
                columns.append(ColumnInfo(
                    name=col.name,
                    type=getattr(col, "type", "string"),
                    description=getattr(col, "description", None)
                ))
        
        return columns
    
    def _extract_partition_keys(self, table_data: Dict[str, Any]) -> List[str]:
        """테이블 데이터에서 파티션 키 추출 (Requirements 2.5)"""
        partition_keys = table_data.get("partition_keys", [])
        
        if isinstance(partition_keys, list):
            result = []
            for pk in partition_keys:
                if isinstance(pk, str):
                    result.append(pk)
                elif isinstance(pk, dict):
                    result.append(pk.get("name", ""))
                elif hasattr(pk, "name"):
                    result.append(pk.name)
            return result
        
        return []


    def _generate_optimization_hints(
        self, 
        tables: List[TableInfo]
    ) -> List[Dict[str, Any]]:
        """SQL 최적화 힌트 생성 (Requirements 2.5)
        
        파티션 키와 날짜 컬럼 정보를 기반으로 최적화 힌트 제공
        """
        hints = []
        
        for table in tables:
            table_hints = {
                "table": f"{table.database}.{table.table}",
                "partition_hints": [],
                "date_column_hints": [],
                "performance_tips": []
            }
            
            # 파티션 키 힌트
            if table.partition_keys:
                for pk in table.partition_keys:
                    table_hints["partition_hints"].append(
                        f"WHERE {pk} = '<value>' -- 파티션 필터링으로 스캔 범위 축소"
                    )
                table_hints["performance_tips"].append(
                    f"파티션 키({', '.join(table.partition_keys)})를 WHERE 절에 포함하여 성능 최적화"
                )
            
            # 날짜 컬럼 힌트
            date_columns = self._find_date_columns(table.columns)
            for date_col in date_columns:
                table_hints["date_column_hints"].append(
                    f"WHERE {date_col} >= date_trunc('month', current_date - interval '1' month)"
                )
            
            if date_columns:
                table_hints["performance_tips"].append(
                    f"날짜 컬럼({', '.join(date_columns)})으로 시간 범위 필터링 권장"
                )
            
            # 일반 성능 팁
            if len(table.columns) > 20:
                table_hints["performance_tips"].append(
                    "컬럼 수가 많으므로 SELECT * 대신 필요한 컬럼만 명시 권장"
                )
            
            hints.append(table_hints)
        
        return hints
    
    def _find_date_columns(self, columns: List[ColumnInfo]) -> List[str]:
        """날짜/시간 컬럼 찾기"""
        date_columns = []
        
        for col in columns:
            col_type = col.type.lower() if col.type else ""
            col_name = col.name.lower() if col.name else ""
            
            # 타입 기반 확인
            is_date_type = any(dt in col_type for dt in DATE_TIME_TYPES)
            
            # 이름 패턴 기반 확인
            is_date_name = DATE_COLUMN_PATTERNS.search(col_name) is not None
            
            if is_date_type or is_date_name:
                date_columns.append(col.name)
        
        return date_columns
    
    def _find_numeric_columns(self, columns: List[ColumnInfo]) -> List[str]:
        """숫자형 컬럼 찾기"""
        numeric_columns = []
        
        for col in columns:
            col_type = col.type.lower() if col.type else ""
            
            if any(num_type in col_type for num_type in NUMERIC_TYPES):
                numeric_columns.append(col.name)
        
        return numeric_columns
    
    def analyze_table_for_query(
        self, 
        table: TableInfo, 
        context: AnalysisContext
    ) -> Dict[str, Any]:
        """특정 테이블에 대한 상세 분석 및 쿼리 힌트 생성
        
        SQL Agent에게 전달할 정보 준비
        """
        analysis = {
            "table_info": {
                "database": table.database,
                "table": table.table,
                "relevance_score": table.relevance_score
            },
            "schema": {
                "columns": [
                    {"name": col.name, "type": col.type, "description": col.description}
                    for col in table.columns
                ],
                "partition_keys": table.partition_keys
            },
            "query_hints": {
                "date_columns": self._find_date_columns(table.columns),
                "numeric_columns": self._find_numeric_columns(table.columns),
                "suggested_filters": []
            }
        }
        
        # 파티션 키 기반 필터 제안
        if table.partition_keys:
            analysis["query_hints"]["suggested_filters"].append({
                "type": "partition",
                "columns": table.partition_keys,
                "hint": "파티션 키를 WHERE 절에 포함하여 성능 최적화"
            })
        
        # 날짜 컬럼 기반 필터 제안
        date_cols = analysis["query_hints"]["date_columns"]
        if date_cols:
            analysis["query_hints"]["suggested_filters"].append({
                "type": "date_range",
                "columns": date_cols,
                "hint": "날짜 범위 필터링으로 데이터 스캔 범위 축소"
            })
        
        return analysis
    
    def format_table_info_for_sql_agent(self, tables: List[TableInfo]) -> str:
        """SQL Agent에게 전달할 테이블 정보 포맷팅"""
        if not tables:
            return ""
        
        lines = ["추천 테이블 정보:"]
        
        for i, table in enumerate(tables, 1):
            lines.append(f"\n{i}. {table.database}.{table.table}")
            
            # 컬럼 정보
            if table.columns:
                col_strs = []
                for col in table.columns:
                    col_str = f"{col.name} ({col.type})"
                    if col.description:
                        col_str += f" - {col.description}"
                    col_strs.append(col_str)
                lines.append(f"   컬럼: {', '.join(col_strs)}")
            
            # 파티션 키
            if table.partition_keys:
                lines.append(f"   파티션 키: {', '.join(table.partition_keys)}")
                lines.append("   ⚠️ 파티션 키를 WHERE 절에 사용하면 성능이 향상됩니다")
        
        return "\n".join(lines)
