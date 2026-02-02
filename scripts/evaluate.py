"""Multi-Agent Text2SQL Evaluation Script

answer_sheet_filter.md의 질문들을 멀티에이전트 시스템에 보내고
결과를 기대값과 비교하여 정확도를 측정합니다.
"""

import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 프로젝트 루트를 path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_env_file(env_path: str = None) -> None:
    """환경변수 파일 로드 (.env 형식)"""
    if env_path is None:
        env_path = project_root / ".env"
    else:
        env_path = Path(env_path)

    if not env_path.exists():
        return

    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value


# .env 파일에서 환경변수 로드
load_env_file()

from agents.multi_agent.multi_agent_text2sql import MultiAgentText2SQL


@dataclass
class EvaluationCase:
    """평가 케이스"""
    question: str
    expected: Any
    actual: Optional[Any] = None
    passed: bool = False
    error: Optional[str] = None
    execution_time_s: float = 0.0
    sql_query: Optional[str] = None


@dataclass
class EvaluationResult:
    """전체 평가 결과"""
    total: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    cases: List[EvaluationCase] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0


def parse_answer_sheet(file_path: str) -> List[Tuple[str, Any]]:
    """answer_sheet_filter.md 파일 파싱

    형식: "1. 질문 : [결과]"
    """
    questions = []

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 각 줄 파싱
    for line in content.strip().split('\n'):
        line = line.strip()
        if not line:
            continue

        # "숫자. 질문 : [결과]" 패턴 매칭
        match = re.match(r'^\d+\.\s*(.+?)\s*:\s*(\[.+\])$', line)
        if match:
            question = match.group(1).strip()
            expected_str = match.group(2).strip()

            try:
                expected = json.loads(expected_str)
            except json.JSONDecodeError:
                expected = expected_str

            questions.append((question, expected))

    return questions


def extract_sql_query(response_text: str) -> Optional[str]:
    """응답 텍스트에서 SQL 쿼리 추출"""
    if not response_text:
        return None

    # 패턴 1: ```sql ... ``` 블록
    sql_block_match = re.search(r'```sql\s*(.*?)\s*```', response_text, re.DOTALL | re.IGNORECASE)
    if sql_block_match:
        return sql_block_match.group(1).strip()

    # 패턴 2: SELECT ... 문장 (FROM 포함)
    select_match = re.search(r'(SELECT\s+.+?(?:FROM|from).+?)(?:```|$|\n\n)', response_text, re.DOTALL | re.IGNORECASE)
    if select_match:
        query = select_match.group(1).strip()
        # 쿼리 끝 정리
        if query.endswith('```'):
            query = query[:-3].strip()
        return query

    return None


def extract_sql_result(response_text: str) -> Optional[Any]:
    """응답 텍스트에서 SQL 실행 결과만 추출

    패턴:
    - "답변: X개" 또는 "결과: X" 형태
    - [결과] 또는 결과: 뒤의 JSON 배열
    - JSON 배열 형태 [...]
    """
    if not response_text:
        return None

    # 패턴 0: "답변:" 뒤의 숫자 (최우선)
    answer_patterns = [
        r'답변[:\s]*\**(\d+)',  # "답변: **69개" or "답변: 69"
        r'Answer[:\s]*\**(\d+)',  # "Answer: 69"
    ]

    for pattern in answer_patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            try:
                val = int(match.group(1))
                return [val]
            except:
                pass

    # 패턴 1: "N개 지역", "N개", "수: N" 형태 (한국어 결과)
    count_patterns = [
        r'결과[:\s]*\**(\d+)',  # "결과: 68" or "결과: **68**"
        r'[:\s]+(\d+)개',  # "68개 지역", "수: 68개"
        r'[:\s]+(\d+)\s*(?:개|건|명|행)',  # "68 개", "68 건"
        r'총[:\s]*(\d+)',  # "총 68"
    ]

    for pattern in count_patterns:
        matches = re.findall(pattern, response_text)
        if matches:
            try:
                # 첫 번째 매칭된 숫자 사용 (보통 핵심 결과)
                val = int(matches[0])
                return [val]
            except:
                pass

    # 패턴 2: JSON 배열 결과 표시
    json_patterns = [
        r'결과[:\s]*(\[.+?\])',
        r'답[:\s]*(\[.+?\])',
        r'answer[:\s]*(\[.+?\])',
        r'result[:\s]*(\[.+?\])',
    ]

    for pattern in json_patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            try:
                return json.loads(match.group(1))
            except:
                pass

    # 패턴 3: 독립적인 JSON 배열 찾기
    json_matches = re.findall(r'\[[\d\.\,\s\"\'\-]+\]', response_text)
    if json_matches:
        try:
            return json.loads(json_matches[-1])
        except:
            pass

    # 패턴 4: 문자열 배열 (["name1", "name2"])
    str_array_matches = re.findall(r'\[["\'][^]]+["\']\]', response_text)
    if str_array_matches:
        try:
            return json.loads(str_array_matches[-1])
        except:
            pass

    return None


def compare_results(expected: Any, actual: Any, tolerance: float = 0.01) -> bool:
    """결과 비교 (숫자는 tolerance 허용)"""
    if expected is None or actual is None:
        return False

    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return False

        for e, a in zip(expected, actual):
            if not compare_single(e, a, tolerance):
                return False
        return True

    return compare_single(expected, actual, tolerance)


def compare_single(expected: Any, actual: Any, tolerance: float) -> bool:
    """단일 값 비교"""
    # 둘 다 숫자인 경우
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if expected == 0:
            return abs(actual) < tolerance
        return abs(expected - actual) / abs(expected) < tolerance

    # 문자열 비교 (대소문자 무시)
    if isinstance(expected, str) and isinstance(actual, str):
        return expected.lower() == actual.lower()

    # 그 외 동일 비교
    return expected == actual


def run_single_evaluation(
    agent: MultiAgentText2SQL,
    question: str,
    expected: Any,
    verbose: bool = False
) -> EvaluationCase:
    """단일 질문 평가 실행"""
    case = EvaluationCase(question=question, expected=expected)

    start_time = time.time()

    try:
        # 에이전트 실행 (스트리밍 결과 수집)
        full_response = ""

        for event in agent.stream_response(question):
            event_type = event.get("type", "")

            # 텍스트 데이터 수집
            if "data" in event:
                full_response += event.get("data", "")

            # 완료 이벤트
            if event_type == "complete":
                result = event.get("result")
                if result:
                    # SwarmResult에서 최종 응답 추출
                    final_text = extract_final_text(result)
                    if final_text:
                        full_response = final_text
                break

            # 에러 이벤트
            if event_type == "force_stop":
                case.error = event.get("force_stop_reason", "Unknown error")
                break

        case.execution_time_s = time.time() - start_time

        # 디버그: 전체 응답 출력
        if verbose:
            print(f"\n--- Full Response ---")
            print(full_response[:2000] if len(full_response) > 2000 else full_response)
            print(f"--- End Response ---\n")

        # 결과 추출 및 비교
        case.actual = extract_sql_result(full_response)
        case.sql_query = extract_sql_query(full_response)
        case.passed = compare_results(expected, case.actual)

    except Exception as e:
        case.error = str(e)
        case.execution_time_s = time.time() - start_time

    return case


def extract_final_text(swarm_result) -> Optional[str]:
    """SwarmResult에서 최종 텍스트 추출 (lead_agent의 최종 응답만)"""
    try:
        # node_results에서 lead_agent 응답 추출
        if hasattr(swarm_result, 'node_results') and swarm_result.node_results:
            node_results = swarm_result.node_results
            # lead_agent의 결과 찾기
            if 'lead_agent' in node_results:
                lead_result = node_results['lead_agent']
                if hasattr(lead_result, 'result') and lead_result.result:
                    agent_result = lead_result.result
                    if hasattr(agent_result, 'message'):
                        msg = agent_result.message
                        if isinstance(msg, dict) and 'content' in msg:
                            content = msg['content']
                            if isinstance(content, list):
                                texts = []
                                for block in content:
                                    if isinstance(block, dict) and 'text' in block:
                                        texts.append(block['text'])
                                return ''.join(texts)
                            return str(content)

        # 기존 방식으로 시도
        if hasattr(swarm_result, 'result') and swarm_result.result:
            result = swarm_result.result
            if hasattr(result, 'message'):
                msg = result.message
                if hasattr(msg, 'content'):
                    if isinstance(msg.content, list):
                        texts = []
                        for block in msg.content:
                            if hasattr(block, 'text'):
                                texts.append(block.text)
                            elif isinstance(block, dict) and 'text' in block:
                                texts.append(block['text'])
                        return ''.join(texts)
                    return str(msg.content)

        # 최후의 수단: str(swarm_result)는 사용하지 않음
        # 대신 None 반환하여 full_response 사용하도록
        return None
    except:
        return None


def run_evaluation(
    answer_sheet_path: str,
    model_id: str = "global.anthropic.claude-haiku-4-5-20251001-v1:0",
    limit: Optional[int] = None,
    verbose: bool = True
) -> EvaluationResult:
    """전체 평가 실행"""

    # 질문 로드
    questions = parse_answer_sheet(answer_sheet_path)
    if limit:
        questions = questions[:limit]

    print(f"\n{'='*60}")
    print(f"Multi-Agent Text2SQL Evaluation")
    print(f"{'='*60}")
    print(f"Questions: {len(questions)}")
    print(f"Model: {model_id}")
    print(f"{'='*60}\n")

    # 에이전트 초기화
    print("Initializing agent...")
    agent = MultiAgentText2SQL(model_id)
    print("Agent initialized.\n")

    # 평가 결과
    result = EvaluationResult(total=len(questions))

    # 각 질문 실행
    for i, (question, expected) in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {question[:50]}...")

        case = run_single_evaluation(agent, question, expected, verbose=verbose)
        result.cases.append(case)

        if case.error:
            result.errors += 1
            status = "ERROR"
        elif case.passed:
            result.passed += 1
            status = "PASS"
        else:
            result.failed += 1
            status = "FAIL"

        if verbose:
            print(f"  Status: {status}")
            print(f"  Expected: {expected}")
            print(f"  Actual: {case.actual}")
            print(f"  Time: {case.execution_time_s:.2f}s")
            if case.error:
                print(f"  Error: {case.error}")
            print()

        # 컨텍스트 리셋
        agent.reset_context()

    # 결과 출력
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Total: {result.total}")
    print(f"Passed: {result.passed}")
    print(f"Failed: {result.failed}")
    print(f"Errors: {result.errors}")
    print(f"Accuracy: {result.accuracy:.1%}")
    print(f"{'='*60}\n")

    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Multi-Agent Text2SQL Evaluation")
    parser.add_argument(
        "--answer-sheet",
        default="assets/answer_sheet_filter.md",
        help="Path to answer sheet file"
    )
    parser.add_argument(
        "--model",
        default="global.anthropic.claude-haiku-4-5-20251001-v1:0",
        help="Model ID to use"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of questions to evaluate"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Save results to JSON file"
    )

    args = parser.parse_args()

    # 경로 처리
    answer_sheet_path = args.answer_sheet
    if not os.path.isabs(answer_sheet_path):
        answer_sheet_path = str(project_root / answer_sheet_path)

    # 평가 실행
    result = run_evaluation(
        answer_sheet_path=answer_sheet_path,
        model_id=args.model,
        limit=args.limit,
        verbose=not args.quiet
    )

    # 결과 저장 (자동)
    eval_results_dir = project_root / "eval_results"
    eval_results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_file = eval_results_dir / f"{timestamp}.json"

    output_data = {
        "timestamp": timestamp,
        "model_id": args.model,
        "total": result.total,
        "passed": result.passed,
        "failed": result.failed,
        "errors": result.errors,
        "accuracy": result.accuracy,
        "cases": [
            {
                "question": c.question,
                "expected": c.expected,
                "actual": c.actual,
                "passed": c.passed,
                "error": c.error,
                "execution_time_s": c.execution_time_s,
                "sql_query": c.sql_query
            }
            for c in result.cases
        ]
    }

    # 자동 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"Results saved to: {output_file}")

    # 추가 저장 경로 지정 시
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"Also saved to: {args.output}")

    # 실패 시 exit code 1
    sys.exit(0 if result.failed == 0 and result.errors == 0 else 1)


if __name__ == "__main__":
    main()
