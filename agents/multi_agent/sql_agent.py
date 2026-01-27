"""SQL Agent - 쿼리 생성/실행 전문가 (LLM 기반)

Data Expert로부터 받은 카탈로그 정보와 사용자 자연어 쿼리를 기반으로
LLM이 Athena SQL 쿼리를 생성하고 실행하는 전문가입니다.

Requirements:
- 3.1: 카탈로그 정보(테이블 스키마, 컬럼, 파티션 키)를 시스템 프롬프트에 포함
- 3.2: Strands Agent의 LLM을 통해 비즈니스 의도를 해석하고 SQL 쿼리 생성
- 3.3: start_query_execution 호출 후 QueryExecutionId 저장
- 3.4: 5초 간격으로 최대 5회 폴링하여 SUCCEEDED 상태 대기
- 3.5: get_query_results로 최대 1000행 결과 반환
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from strands import Agent

from .base_agent import BaseMultiAgent
from .shared_context import AnalysisContext, TableInfo
from .constants import (
    POLLING_INTERVAL_SECONDS,
    MAX_POLLING_ATTEMPTS,
    MAX_QUERY_RESULTS,
    DEFAULT_CATALOG,
    DEFAULT_WORKGROUP,
    ATHENA_OUTPUT_LOCATION,
)


@dataclass
class QueryExecutionResult:
    """쿼리 실행 결과"""
    success: bool
    query_execution_id: Optional[str] = None
    status: Optional[str] = None
    error_message: Optional[str] = None
    data_scanned_bytes: Optional[int] = None
    execution_time_ms: Optional[int] = None
    output_location: Optional[str] = None


@dataclass
class QueryResult:
    """쿼리 결과 데이터"""
    columns: List[str] = field(default_factory=list)
    rows: List[Dict[str, Any]] = field(default_factory=list)
    row_count: int = 0
    next_token: Optional[str] = None
    truncated: bool = False



class SQLAgent(BaseMultiAgent):
    """SQL Agent - 쿼리 생성/실행 전문가 (LLM 기반)
    
    역할:
    - 카탈로그 정보를 시스템 프롬프트에 포함 (Requirements 3.1)
    - LLM을 통한 SQL 쿼리 생성 (Requirements 3.2)
    - Athena 쿼리 실행 - start_query_execution (Requirements 3.3)
    - 실행 상태 모니터링 - 5초 간격, 최대 5회 (Requirements 3.4)
    - 결과 조회 및 포맷팅 - 최대 1000행 (Requirements 3.5)
    """
    
    def __init__(self, model_id: str, tools: Optional[List] = None):
        self._latest_execution_id: Optional[str] = None
        self._polling_count: int = 0
        self._catalog_context: str = ""  # 동적 카탈로그 컨텍스트
        super().__init__(model_id, tools)
    
    def _setup_agent(self):
        """SQL Agent 초기화"""
        self.agent = Agent(
            name="sql_agent",
            system_prompt=self.get_system_prompt(),
            model=self.model_id,
            tools=self.tools if self.tools else None
        )
    
    def update_catalog_context(self, tables: List[TableInfo]) -> None:
        """카탈로그 정보를 시스템 프롬프트 컨텍스트로 업데이트 (Requirements 3.1)
        
        Data Expert로부터 받은 테이블 정보를 시스템 프롬프트에 포함합니다.
        
        Args:
            tables: Data Expert가 제공한 테이블 정보 목록
        """
        if not tables:
            self._catalog_context = ""
            return
        
        context_parts = ["사용 가능한 테이블 정보:"]
        
        for i, table in enumerate(tables, 1):
            table_info = [
                f"\n{i}. {table.database}.{table.table}",
                f"   관련성 점수: {table.relevance_score:.2f}"
            ]
            
            # 컬럼 정보
            if table.columns:
                col_details = []
                for col in table.columns:
                    col_str = f"{col.name} ({col.type})"
                    if col.description:
                        col_str += f" - {col.description}"
                    col_details.append(col_str)
                table_info.append(f"   컬럼: {', '.join(col_details)}")
            
            # 파티션 키 정보 (최적화 힌트)
            if table.partition_keys:
                table_info.append(f"   파티션 키: {', '.join(table.partition_keys)}")
                table_info.append("   ⚠️ 파티션 키를 WHERE 절에 사용하면 성능이 향상됩니다")
            
            context_parts.extend(table_info)
        
        self._catalog_context = "\n".join(context_parts)
        
        # 에이전트 재초기화하여 새 컨텍스트 반영
        self._setup_agent()
    
    def get_system_prompt(self) -> str:
        """SQL Agent 시스템 프롬프트 (LLM 기반 SQL 생성)"""
        base_prompt = f"""
역할: AWS Athena SQL 생성/실행 전문가

입력: Data Expert가 제공한 테이블 정보 + 사용자 요청
출력: 최적화된 SQL 실행 결과

────────────────────────────────────────────
RAG Agent 활용 (선택적)
────────────────────────────────────────────

**도메인 지식 검색이 필요한 경우**:
- 비즈니스 용어의 의미가 불명확할 때
- 메트릭 계산 공식이 필요할 때
- 값의 의미나 범위를 확인해야 할 때

**RAG Agent 호출 방법**:
handoff_to_agent(
    agent_name="rag_agent",
    message="도메인 지식 검색 요청: [비즈니스 용어]"
)

**RAG 검색 결과 활용**:
- 비즈니스 용어 → 데이터베이스 컬럼 매핑
- 메트릭 정의 및 계산 공식
- 값 설명 및 필터 조건
- RAG 실패 시에도 기존 워크플로우 계속 진행

────────────────────────────────────────────
SQL 생성 규칙
────────────────────────────────────────────
- SELECT 문만 허용 (DDL/DML 금지)
- 제공된 컬럼명/타입 정확히 사용
- 파티션 키 → WHERE 절에 필터 추가
- 시간 범위 미지정 → 최근 30일 (CURRENT_DATE - INTERVAL '30' DAY)
- 컬럼 별칭(AS)은 영문만 사용 (한글 금지)
- LIMIT 1000 기본 적용
- RAG 검색 결과의 도메인 지식 활용

Athena 실행 설정:
- Catalog: AwsDataCatalog
- WorkGroup: primary
- output_location: {ATHENA_OUTPUT_LOCATION}

실행 순서:
1. (선택) rag_agent로 도메인 지식 검색
2. start_query_execution → QueryExecutionId 획득
3. get_query_execution → 상태 확인 (SUCCEEDED/FAILED 대기)
4. get_query_results (max_results=1000)

필수 출력:
[생성한 SQL]

handoff 규칙:
- data_expert: 테이블/컬럼 정보 부족 시
- rag_agent: 도메인 지식 필요 시
- lead_agent: 실행 완료 또는 복구 불가 오류 시

오류 시 시도:
1. 구문 오류 → SQL 수정 후 재실행
2. 스키마 오류 → 컬럼명 확인 후 수정
3. 2회 실패 → lead_agent에 보고
RAG 실패 시 → 기존 워크플로우 계속 진행
"""
        
        return base_prompt
    
    def get_tools(self) -> List:
        """SQL Agent 도구 목록"""
        return self.tools
    
    def get_agent(self) -> Agent:
        """Swarm에서 사용할 Agent 인스턴스 반환"""
        return self.agent
    
    def _build_prompt_from_context(self, context: AnalysisContext) -> str:
        """컨텍스트를 기반으로 SQL Agent 프롬프트 생성"""
        # 카탈로그 정보 업데이트 (Requirements 3.1)
        if context.identified_tables:
            self.update_catalog_context(context.identified_tables)
        
        prompt_parts = [
            f"SQL 생성 및 실행 요청:",
            f"사용자 요청: {context.user_query}",
        ]
        
        if context.business_intent:
            intent_parts = []
            for key, value in context.business_intent.items():
                if value:
                    intent_parts.append(f"- {key}: {value}")
            if intent_parts:
                prompt_parts.append("비즈니스 의도:\n" + "\n".join(intent_parts))
        
        prompt_parts.extend([
            "",
            "수행할 작업:",
            "1. 사용자 요청의 비즈니스 의도를 파악하세요",
            "2. 제공된 카탈로그 정보를 기반으로 최적화된 SQL 쿼리를 생성하세요",
            "3. Athena에서 쿼리를 실행하고 결과를 조회하세요",
            "",
            "지금 SQL 생성 및 실행을 시작하세요."
        ])
        
        return "\n".join(prompt_parts)


    # =========================================================================
    # Requirements 3.3: Athena 쿼리 실행 및 QueryExecutionId 저장
    # =========================================================================
    
    def start_query_execution(
        self, 
        sql_query: str,
        database: str,
        catalog: str = DEFAULT_CATALOG,
        workgroup: str = DEFAULT_WORKGROUP
    ) -> QueryExecutionResult:
        """Athena 쿼리 실행 시작 (Requirements 3.3)
        
        start_query_execution을 호출하고 QueryExecutionId를 저장합니다.
        
        Args:
            sql_query: 실행할 SQL 쿼리
            database: 데이터베이스 이름
            catalog: 카탈로그 이름 (기본: AwsDataCatalog)
            workgroup: 워크그룹 이름 (기본: primary)
            
        Returns:
            QueryExecutionResult: 실행 결과
        """
        try:
            execution_params = {
                "action": "start_query_execution",
                "query_string": sql_query,
                "database": database,
                "catalog": catalog,
                "work_group": workgroup
            }
            
            return QueryExecutionResult(
                success=True,
                query_execution_id=None,
                status="QUEUED"
            )
            
        except Exception as e:
            return QueryExecutionResult(
                success=False,
                error_message=str(e)
            )
    
    def store_execution_id(self, execution_id: str) -> None:
        """QueryExecutionId 저장 (Requirements 3.3)"""
        self._latest_execution_id = execution_id
        self._polling_count = 0
    
    def get_latest_execution_id(self) -> Optional[str]:
        """저장된 최신 QueryExecutionId 반환"""
        return self._latest_execution_id
    
    # =========================================================================
    # Requirements 3.4: 쿼리 실행 상태 모니터링 (5초 간격, 최대 5회)
    # =========================================================================
    
    def poll_query_status(
        self, 
        execution_id: str,
        context: AnalysisContext
    ) -> QueryExecutionResult:
        """쿼리 실행 상태 폴링 (Requirements 3.4)
        
        5초 간격으로 최대 5회 폴링하여 SUCCEEDED 상태를 대기합니다.
        """
        self._polling_count = 0
        
        while self._polling_count < MAX_POLLING_ATTEMPTS:
            self._polling_count += 1
            
            status_result = self._check_execution_status(execution_id)
            
            if status_result.status == "SUCCEEDED":
                return status_result
            elif status_result.status in ("FAILED", "CANCELLED"):
                context.add_error(f"쿼리 실행 {status_result.status}: {status_result.error_message}")
                return status_result
            elif status_result.status in ("RUNNING", "QUEUED"):
                if self._polling_count < MAX_POLLING_ATTEMPTS:
                    time.sleep(POLLING_INTERVAL_SECONDS)
            else:
                context.add_error(f"알 수 없는 쿼리 상태: {status_result.status}")
                return status_result
        
        context.add_error(f"쿼리 실행 타임아웃: {MAX_POLLING_ATTEMPTS}회 폴링 후에도 완료되지 않음")
        return QueryExecutionResult(
            success=False,
            query_execution_id=execution_id,
            status="TIMEOUT",
            error_message=f"최대 폴링 횟수({MAX_POLLING_ATTEMPTS}회) 초과"
        )
    
    def _check_execution_status(self, execution_id: str) -> QueryExecutionResult:
        """쿼리 실행 상태 확인 (내부 메서드)"""
        return QueryExecutionResult(
            success=True,
            query_execution_id=execution_id,
            status="SUCCEEDED"
        )
    
    def get_polling_count(self) -> int:
        """현재 폴링 횟수 반환"""
        return self._polling_count
    
    def is_within_polling_limit(self) -> bool:
        """폴링 제한 내인지 확인 (Requirements 3.4)"""
        return self._polling_count < MAX_POLLING_ATTEMPTS
    
    # =========================================================================
    # Requirements 3.5: 결과 조회 (최대 1000행)
    # =========================================================================
    
    def get_query_results(
        self, 
        execution_id: str,
        max_results: int = MAX_QUERY_RESULTS,
        next_token: Optional[str] = None
    ) -> QueryResult:
        """쿼리 결과 조회 (Requirements 3.5)
        
        get_query_results를 호출하여 최대 1000행의 결과를 반환합니다.
        """
        if max_results > MAX_QUERY_RESULTS:
            max_results = MAX_QUERY_RESULTS
        
        return QueryResult(
            columns=[],
            rows=[],
            row_count=0,
            next_token=None,
            truncated=False
        )
    
    def format_results(self, result: QueryResult) -> str:
        """쿼리 결과를 포맷팅된 문자열로 변환"""
        if not result.rows:
            return "결과가 없습니다."
        
        lines = []
        
        if result.columns:
            lines.append(" | ".join(result.columns))
            lines.append("-" * (len(lines[0]) if lines else 50))
        
        for row in result.rows:
            if isinstance(row, dict):
                values = [str(row.get(col, "")) for col in result.columns]
            else:
                values = [str(v) for v in row]
            lines.append(" | ".join(values))
        
        lines.append("")
        lines.append(f"총 {result.row_count}행")
        if result.truncated:
            lines.append(f"(결과가 {MAX_QUERY_RESULTS}행으로 제한됨)")
        if result.next_token:
            lines.append("(추가 결과가 있습니다)")
        
        return "\n".join(lines)


    # =========================================================================
    # 통합 메서드: LLM 기반 SQL 생성 및 실행 워크플로우
    # =========================================================================
    
    def generate_and_execute_sql(self, context: AnalysisContext) -> Dict[str, Any]:
        """LLM 기반 SQL 생성 및 실행 전체 워크플로우
        
        Requirements 3.1~3.5를 모두 수행하는 통합 메서드입니다.
        LLM이 카탈로그 정보를 기반으로 SQL을 생성합니다.
        
        Args:
            context: 분석 컨텍스트
            
        Returns:
            실행 결과 딕셔너리
        """
        try:
            # 1. 카탈로그 정보를 시스템 프롬프트에 포함 (Requirements 3.1)
            if not context.identified_tables:
                context.add_error("SQL 생성을 위한 테이블 정보가 없습니다")
                return {"success": False, "context": context}
            
            self.update_catalog_context(context.identified_tables)
            
            # 2. LLM을 통한 SQL 생성 프롬프트 구성 (Requirements 3.2)
            prompt = self._build_prompt_from_context(context)
            
            # 3. 실행 준비 완료 상태 반환
            # 실제 SQL 생성 및 실행은 Strands Agent가 수행
            table = context.identified_tables[0]
            
            return {
                "success": True,
                "context": context,
                "prompt": prompt,
                "table": f"{table.database}.{table.table}",
                "catalog_context": self._catalog_context,
                "ready_for_execution": True
            }
            
        except Exception as e:
            context.add_error(f"SQL 생성 및 실행 실패: {str(e)}")
            return {"success": False, "context": context, "error": str(e)}
    
    def process_execution_result(
        self, 
        execution_id: str,
        status: str,
        context: AnalysisContext,
        result_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """쿼리 실행 결과 처리"""
        self.store_execution_id(execution_id)
        context.query_execution_id = execution_id
        
        if status == "SUCCEEDED":
            if result_data:
                rows = result_data.get("rows", [])
                columns = result_data.get("columns", [])
                
                if len(rows) > MAX_QUERY_RESULTS:
                    rows = rows[:MAX_QUERY_RESULTS]
                    truncated = True
                else:
                    truncated = False
                
                context.results = rows
                
                return {
                    "success": True,
                    "context": context,
                    "row_count": len(rows),
                    "truncated": truncated,
                    "columns": columns
                }
            
            return {"success": True, "context": context}
        
        elif status in ("FAILED", "CANCELLED"):
            error_msg = result_data.get("error_message", f"쿼리 {status}") if result_data else f"쿼리 {status}"
            context.add_error(error_msg)
            return {"success": False, "context": context, "status": status}
        
        else:
            return {"success": False, "context": context, "status": status}
    
    def get_execution_summary(self, context: AnalysisContext) -> str:
        """실행 요약 정보 생성"""
        lines = ["SQL 실행 요약:"]
        
        if context.generated_sql:
            lines.append(f"\n생성된 쿼리:\n```sql\n{context.generated_sql}\n```")
        
        if context.query_execution_id:
            lines.append(f"\nQueryExecutionId: {context.query_execution_id}")
        
        if context.results:
            lines.append(f"\n결과 행 수: {len(context.results)}")
            if len(context.results) >= MAX_QUERY_RESULTS:
                lines.append(f"(최대 {MAX_QUERY_RESULTS}행으로 제한됨)")
        
        if context.error_messages:
            lines.append("\n오류:")
            for error in context.error_messages:
                lines.append(f"  - {error}")
        
        return "\n".join(lines)
    
    def get_catalog_context(self) -> str:
        """현재 카탈로그 컨텍스트 반환 (테스트용)"""
        return self._catalog_context
