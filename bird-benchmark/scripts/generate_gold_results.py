"""SQLite Gold SQL 실행 결과를 JSON 파일로 저장"""

import json
import sqlite3
import sys
from pathlib import Path
from typing import Optional, List, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    MINI_DEV_DATA,
    get_db_path,
)


def parse_gold_sql(file_path: Path) -> List[tuple]:
    """Gold SQL 파일 파싱"""
    results = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.rsplit("\t", 1)
            if len(parts) == 2:
                results.append((parts[0].strip(), parts[1].strip()))
    return results


def execute_sqlite(db_path: Path, sql: str) -> Optional[List]:
    """SQLite에서 SQL 실행"""
    try:
        conn = sqlite3.connect(str(db_path), timeout=60)
        cursor = conn.cursor()
        cursor.execute(sql)
        result = cursor.fetchall()
        conn.close()
        return result
    except Exception as e:
        print(f"SQLite Error: {e}", file=sys.stderr)
        return None


def serialize_result(result: Optional[List]) -> Any:
    """결과를 JSON 직렬화 가능한 형태로 변환"""
    if result is None:
        return None
    return [list(row) for row in result]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=MINI_DEV_DATA / "mini_dev_sqlite_gold.sql")
    parser.add_argument("--output", type=Path, default=MINI_DEV_DATA / "mini_dev_sqlite_gold_results.json")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    # Gold SQL 로드
    gold_sqls = parse_gold_sql(args.input)
    print(f"Loaded {len(gold_sqls)} gold queries")

    if args.limit:
        gold_sqls = gold_sqls[:args.limit]

    # 실행 및 결과 저장
    results = []
    for i, (sql, db_id) in enumerate(gold_sqls):
        db_path = get_db_path(db_id)

        result = execute_sqlite(db_path, sql)

        results.append({
            "index": i,
            "db_id": db_id,
            "sql": sql,
            "result": serialize_result(result),
            "error": result is None
        })

        if (i + 1) % 50 == 0:
            print(f"Progress: {i+1}/{len(gold_sqls)}")

    # JSON 저장
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    error_count = sum(1 for r in results if r["error"])
    print(f"\nDone! Saved {len(results)} results to {args.output}")
    print(f"Errors: {error_count}")


if __name__ == "__main__":
    main()
