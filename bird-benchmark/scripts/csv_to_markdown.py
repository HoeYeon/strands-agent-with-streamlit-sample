#!/usr/bin/env python3
"""
CSV to Markdown 변환 스크립트

BIRD benchmark의 database_description CSV 파일들을
RAG에 적합한 Markdown 형식으로 변환합니다.
"""

import csv
import os
from pathlib import Path
from typing import Optional


def parse_csv_file(csv_path: str) -> list[dict]:
    """CSV 파일을 파싱하여 컬럼 정보 리스트를 반환합니다."""
    columns = []

    # 유니코드 특수문자 -> 표준 문자 매핑
    UNICODE_REPLACEMENTS = {
        '\u00A0': ' ',    # non-breaking space
        '\u00B7': '-',    # middle dot (·)
        '\u2022': '-',    # bullet point (•)
        '\u2013': '-',    # en dash (–)
        '\u2014': '-',    # em dash (—)
        '\u2018': "'",    # left single quote (')
        '\u2019': "'",    # right single quote (')
        '\u201C': '"',    # left double quote (")
        '\u201D': '"',    # right double quote (")
        '\u2026': '...',  # ellipsis (…)
    }

    def clean_text(text: str) -> str:
        """유니코드 특수문자를 일반 문자로 대체합니다."""
        if not text:
            return text
        for old_char, new_char in UNICODE_REPLACEMENTS.items():
            text = text.replace(old_char, new_char)
        return text

    # 여러 인코딩 시도
    encodings = ['utf-8-sig', 'utf-8', 'cp1252', 'latin-1']

    for encoding in encodings:
        try:
            with open(csv_path, 'r', encoding=encoding) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # 빈 행 스킵
                    original_name = (row.get('original_column_name') or '').strip()
                    if not original_name:
                        continue

                    columns.append({
                        'original_name': clean_text(original_name),
                        'column_name': clean_text((row.get('column_name') or '').strip()),
                        'description': clean_text((row.get('column_description') or '').strip()),
                        'data_type': clean_text((row.get('data_format') or '').strip()),
                        'value_description': clean_text((row.get('value_description') or '').strip())
                    })
            return columns
        except UnicodeDecodeError:
            continue

    raise ValueError(f"Unable to decode file: {csv_path}")


def generate_markdown(
    table_name: str,
    database_name: str,
    columns: list[dict]
) -> str:
    """컬럼 정보를 Markdown 형식으로 변환합니다."""

    lines = []

    # 헤더
    lines.append(f"# Table: {table_name}")
    lines.append(f"")
    lines.append(f"**Database**: {database_name}")
    lines.append(f"")

    # 컬럼 테이블
    lines.append("## Columns")
    lines.append("")
    lines.append("| Column | Alias | Type | Description |")
    lines.append("|--------|-------|------|-------------|")

    business_logic = []

    for col in columns:
        # original_column_name이 실제 DB 컬럼명
        original_name = col['original_name']
        # column_name은 별칭 (없거나 원본과 같으면 빈칸)
        alias = col['column_name'] if col['column_name'] and col['column_name'] != original_name else ''
        data_type = col['data_type'] if col['data_type'] else '-'
        description = col['description'] if col['description'] else '-'

        # 테이블 내 파이프 문자 이스케이프
        description = description.replace('|', '\\|')
        alias = alias.replace('|', '\\|')

        lines.append(f"| {original_name} | {alias} | {data_type} | {description} |")

        # value_description이 있고 "NOT USEFUL"이 아니면 비즈니스 로직에 추가
        if col['value_description'] and 'NOT USEFUL' not in col['value_description'].upper():
            # 비즈니스 로직에서는 alias가 있으면 함께 표시
            display_name = f"{original_name} ({alias})" if alias else original_name
            business_logic.append({
                'column': display_name,
                'logic': col['value_description']
            })

    # 비즈니스 로직 섹션
    if business_logic:
        lines.append("")
        lines.append("## Business Logic & Value Descriptions")
        lines.append("")

        for item in business_logic:
            # 멀티라인 처리: 줄바꿈을 유지하면서 들여쓰기
            logic_text = item['logic'].strip()
            # 줄바꿈을 bullet point 하위 항목으로 변환
            logic_lines = logic_text.split('\n')

            lines.append(f"### {item['column']}")
            lines.append("")
            for logic_line in logic_lines:
                logic_line = logic_line.strip()
                if logic_line:
                    lines.append(f"- {logic_line}")
            lines.append("")

    return '\n'.join(lines)


def convert_database_descriptions(
    input_base_path: str,
    output_base_path: str
) -> list[str]:
    """
    모든 database_description CSV 파일들을 Markdown으로 변환합니다.

    Args:
        input_base_path: dev_databases 폴더 경로
        output_base_path: 마크다운 파일을 저장할 경로

    Returns:
        생성된 마크다운 파일 경로 리스트
    """
    input_path = Path(input_base_path)
    output_path = Path(output_base_path)
    output_path.mkdir(parents=True, exist_ok=True)

    generated_files = []

    # 각 데이터베이스 폴더 순회
    for db_folder in sorted(input_path.iterdir()):
        if not db_folder.is_dir():
            continue

        db_name = db_folder.name
        desc_folder = db_folder / 'database_description'

        if not desc_folder.exists():
            continue

        # 데이터베이스별 출력 폴더 생성
        db_output_folder = output_path / db_name
        db_output_folder.mkdir(parents=True, exist_ok=True)

        # 각 CSV 파일 처리
        for csv_file in sorted(desc_folder.glob('*.csv')):
            table_name = csv_file.stem  # 확장자 제외한 파일명

            try:
                columns = parse_csv_file(str(csv_file))

                if not columns:
                    print(f"  [SKIP] {db_name}/{table_name}: 빈 파일")
                    continue

                markdown_content = generate_markdown(table_name, db_name, columns)

                # 마크다운 파일 저장
                output_file = db_output_folder / f"{table_name}.md"
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(markdown_content)

                generated_files.append(str(output_file))
                print(f"  [OK] {db_name}/{table_name}.md")

            except Exception as e:
                print(f"  [ERROR] {db_name}/{table_name}: {e}")

    return generated_files


def main():
    # 경로 설정
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    input_path = project_root / 'data' / 'mini_dev' / 'llm' / 'mini_dev_data' / 'dev_databases'
    output_path = project_root / 'data' / 'mini_dev' / 'markdown_descriptions'

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print("-" * 50)

    if not input_path.exists():
        print(f"Error: Input path not found: {input_path}")
        return

    generated_files = convert_database_descriptions(str(input_path), str(output_path))

    print("-" * 50)
    print(f"총 {len(generated_files)}개 파일 생성 완료")
    print(f"출력 경로: {output_path}")


if __name__ == '__main__':
    main()
