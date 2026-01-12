#!/usr/bin/env python3
"""BIRD 데이터를 S3에 업로드하는 스크립트

SQLite 데이터베이스를 Parquet으로 변환하고 S3에 업로드합니다.
Glue Catalog 등록을 위한 DDL도 생성합니다.

Usage:
    python upload_to_s3.py --bucket your-bucket-name

Prerequisites:
    pip install pandas pyarrow boto3
"""

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple

import boto3
import pandas as pd

# 데이터 경로
BIRD_ROOT = Path(__file__).parent
MINI_DEV_DATA = BIRD_ROOT / "data" / "mini_dev" / "llm" / "mini_dev_data"
MINI_DEV_DATABASES = MINI_DEV_DATA / "dev_databases"
MINI_DEV_SQLITE_JSON = MINI_DEV_DATA / "mini_dev_sqlite.json"


def get_all_db_ids() -> List[str]:
    """데이터셋에서 사용되는 모든 DB ID 추출"""
    with open(MINI_DEV_SQLITE_JSON) as f:
        data = json.load(f)
    return sorted(set(item["db_id"] for item in data))


def get_sqlite_tables(db_path: Path) -> List[str]:
    """SQLite DB의 테이블 목록 반환"""
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()
    return tables


def get_table_schema(db_path: Path, table_name: str) -> List[Tuple[str, str]]:
    """테이블 스키마 정보 반환"""
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info('{table_name}');")
    columns = [(row[1], row[2]) for row in cursor.fetchall()]
    conn.close()
    return columns


def sqlite_to_athena_type(sqlite_type: str) -> str:
    """SQLite 타입을 Athena 타입으로 변환"""
    sqlite_type = sqlite_type.upper()
    if "INT" in sqlite_type:
        return "BIGINT"
    elif any(t in sqlite_type for t in ["REAL", "FLOAT", "DOUBLE"]):
        return "DOUBLE"
    elif "BLOB" in sqlite_type:
        return "BINARY"
    elif "BOOL" in sqlite_type:
        return "BOOLEAN"
    return "STRING"


def table_to_parquet(db_path: Path, table_name: str) -> bytes:
    """테이블을 Parquet 바이트로 변환"""
    conn = sqlite3.connect(str(db_path))
    df = pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn)
    conn.close()

    # Parquet으로 변환
    import io
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False, compression='snappy')
    buffer.seek(0)
    return buffer.read()


def generate_athena_ddl(db_id: str, table_name: str, columns: List[Tuple[str, str]], s3_location: str) -> str:
    """Athena CREATE TABLE DDL 생성"""
    # 예약어 목록
    reserved_words = {"date", "order", "group", "key", "index", "table", "column", "select", "from", "where"}

    col_defs = []
    for col_name, col_type in columns:
        athena_type = sqlite_to_athena_type(col_type)
        safe_name = f"`{col_name}`" if col_name.lower() in reserved_words else col_name
        col_defs.append(f"  {safe_name} {athena_type}")

    return f"""CREATE EXTERNAL TABLE IF NOT EXISTS bird_benchmark.{db_id}__{table_name} (
{','.join(col_defs)}
)
STORED AS PARQUET
LOCATION '{s3_location}/{db_id}/{table_name}/'
TBLPROPERTIES ('parquet.compression'='SNAPPY');
"""


def main():
    parser = argparse.ArgumentParser(description="Upload BIRD data to S3 for Athena")
    parser.add_argument("--bucket", type=str, default="text2sql-data-bucket-897729106229", help="S3 bucket name")
    parser.add_argument("--prefix", type=str, default="bird-benchmark", help="S3 prefix")
    parser.add_argument("--profile", type=str, default=None, help="AWS profile name")
    parser.add_argument("--region", type=str, default="us-east-1", help="AWS region")
    parser.add_argument("--db-id", type=str, default=None, help="Process specific database only")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be uploaded without uploading")
    args = parser.parse_args()

    # AWS 세션
    session_kwargs = {"region_name": args.region}
    if args.profile:
        session_kwargs["profile_name"] = args.profile

    session = boto3.Session(**session_kwargs)
    s3 = session.client("s3")

    # 처리할 DB 목록
    db_ids = [args.db_id] if args.db_id else get_all_db_ids()

    print(f"=== BIRD Benchmark S3 Upload ===")
    print(f"Bucket: s3://{args.bucket}/{args.prefix}/")
    print(f"Databases: {len(db_ids)}")
    print(f"Dry run: {args.dry_run}")
    print()

    all_ddl = [
        "-- BIRD Benchmark Athena DDL",
        "-- Auto-generated\n",
        "CREATE DATABASE IF NOT EXISTS bird_benchmark;\n",
    ]

    total_tables = 0
    total_size = 0

    for db_id in db_ids:
        db_path = MINI_DEV_DATABASES / db_id / f"{db_id}.sqlite"
        if not db_path.exists():
            print(f"[SKIP] {db_id}: not found")
            continue

        tables = get_sqlite_tables(db_path)
        print(f"\n[{db_id}] {len(tables)} tables")

        for table_name in tables:
            s3_key = f"{args.prefix}/{db_id}/{table_name}/data.parquet"

            # 스키마 정보
            columns = get_table_schema(db_path, table_name)

            # DDL 생성
            s3_location = f"s3://{args.bucket}/{args.prefix}"
            ddl = generate_athena_ddl(db_id, table_name, columns, s3_location)
            all_ddl.append(ddl)

            if args.dry_run:
                print(f"  - {table_name} -> s3://{args.bucket}/{s3_key}")
            else:
                # Parquet 변환 및 업로드
                parquet_data = table_to_parquet(db_path, table_name)
                s3.put_object(
                    Bucket=args.bucket,
                    Key=s3_key,
                    Body=parquet_data,
                    ContentType="application/octet-stream",
                )
                size_mb = len(parquet_data) / 1024 / 1024
                total_size += size_mb
                print(f"  - {table_name} ({size_mb:.1f}MB) ✓")

            total_tables += 1

    # DDL 파일 저장
    ddl_content = "\n".join(all_ddl)
    ddl_path = BIRD_ROOT / "athena_create_tables.sql"
    with open(ddl_path, "w") as f:
        f.write(ddl_content)

    # DDL도 S3에 업로드
    if not args.dry_run:
        s3.put_object(
            Bucket=args.bucket,
            Key=f"{args.prefix}/ddl/create_tables.sql",
            Body=ddl_content,
            ContentType="text/plain",
        )

    print(f"\n=== Summary ===")
    print(f"Databases: {len(db_ids)}")
    print(f"Tables: {total_tables}")
    if not args.dry_run:
        print(f"Total size: {total_size:.1f}MB")
    print(f"\nDDL saved to: {ddl_path}")
    print(f"DDL on S3: s3://{args.bucket}/{args.prefix}/ddl/create_tables.sql")

    print(f"\n=== Next Steps ===")
    print(f"1. Run the DDL in Athena console to create tables")
    print(f"2. Or use AWS CLI:")
    print(f"   aws s3 cp s3://{args.bucket}/{args.prefix}/ddl/create_tables.sql - | head -50")


if __name__ == "__main__":
    main()
