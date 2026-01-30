import re
import json
import argparse
import boto3
from pathlib import Path
from dataclasses import dataclass, asdict
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth


@dataclass
class Chunk:
    table: str
    database: str
    chunk_type: str
    column_name: str | None
    content: str
    source_file: str


def parse_markdown(file_path: Path) -> list[Chunk]:
    text = file_path.read_text()
    chunks = []
    
    table_match = re.search(r'^# Table: (.+)$', text, re.MULTILINE)
    db_match = re.search(r'\*\*Database\*\*: (.+)$', text, re.MULTILINE)
    table = table_match.group(1) if table_match else "unknown"
    database = db_match.group(1) if db_match else "unknown"
    
    columns_match = re.search(r'## Columns\n\n(\|.+?\|)\n(\|[-| ]+\|)\n((?:\|.+?\|\n)+)', text)
    column_rows = []
    if columns_match:
        for row in columns_match.group(3).strip().split('\n'):
            cols = [c.strip() for c in row.split('|')[1:-1]]
            if len(cols) >= 4:
                column_rows.append({'name': cols[0], 'alias': cols[1], 'type': cols[2], 'desc': cols[3]})
    
    business_logic = {}
    bl_section = re.search(r'## Business Logic & Value Descriptions\n\n(.+)', text, re.DOTALL)
    if bl_section:
        bl_text = bl_section.group(1)
        col_sections = re.split(r'### ', bl_text)
        for section in col_sections:
            if not section.strip():
                continue
            lines = section.strip().split('\n')
            header = lines[0]
            col_name = re.match(r'^(\w+)', header)
            if col_name:
                business_logic[col_name.group(1)] = '\n'.join(lines[1:]).strip()
    
    col_summary = '\n'.join([f"- {c['name']} ({c['type']}): {c['desc']}" for c in column_rows])
    overview = f"# Table: {table}\nDatabase: {database}\n\n## Columns\n{col_summary}"
    chunks.append(Chunk(table, database, 'table_overview', None, overview, str(file_path)))
    
    for col in column_rows:
        if col['name'] in business_logic:
            content = f"# Table: {table} > Column: {col['name']}\n"
            content += f"Database: {database}\n"
            content += f"Type: {col['type']}\n"
            if col['alias']:
                content += f"Alias: {col['alias']}\n"
            content += f"Description: {col['desc']}\n\n"
            content += f"## Business Logic\n{business_logic[col['name']]}"
            chunks.append(Chunk(table, database, 'column_detail', col['name'], content, str(file_path)))
    
    return chunks


def get_embedding(bedrock, text: str) -> list[float]:
    response = bedrock.invoke_model(
        modelId='amazon.titan-embed-text-v2:0',
        body=json.dumps({'inputText': text})
    )
    return json.loads(response['body'].read())['embedding']


def create_index(client: OpenSearch, index_name: str):
    if client.indices.exists(index=index_name):
        print(f"인덱스 {index_name} 이미 존재")
        return
    
    client.indices.create(index=index_name, body={
        'settings': {'index': {'knn': True}},
        'mappings': {
            'properties': {
                'embedding': {
                    'type': 'knn_vector',
                    'dimension': 1024,
                    'method': {'name': 'hnsw', 'engine': 'faiss'}
                },
                'content': {'type': 'text'},
                'table': {'type': 'keyword'},
                'database': {'type': 'keyword'},
                'chunk_type': {'type': 'keyword'},
                'column_name': {'type': 'keyword'},
                'source_file': {'type': 'keyword'}
            }
        }
    })
    print(f"인덱스 {index_name} 생성 완료")


def index_chunks(client: OpenSearch, bedrock, chunks: list[Chunk], index_name: str):
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(bedrock, chunk.content)
        doc = asdict(chunk)
        doc['embedding'] = embedding
        client.index(index=index_name, body=doc)
        if (i + 1) % 10 == 0:
            print(f"진행: {i + 1}/{len(chunks)}")
    print(f"총 {len(chunks)}개 청크 인덱싱 완료")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', help='마크다운 디렉토리 (로컬 또는 s3://)')
    parser.add_argument('--opensearch-host', default='localhost')
    parser.add_argument('--opensearch-port', type=int, default=9443)
    parser.add_argument('--index-name', default='bird-description')
    parser.add_argument('--region', default='us-east-1')
    parser.add_argument('--profile', default=None)
    parser.add_argument('--recreate', action='store_true', help='기존 인덱스 삭제 후 재생성')
    args = parser.parse_args()
    
    session = boto3.Session(profile_name=args.profile)
    bedrock = session.client('bedrock-runtime', region_name=args.region)
    
    # localhost 터널 사용 시 기본 인증, VPC 직접 접근 시 IAM 인증
    if args.opensearch_host == 'localhost':
        client = OpenSearch(
            hosts=[{'host': args.opensearch_host, 'port': args.opensearch_port}],
            http_auth=('admin', input('OpenSearch admin 비밀번호: ')),
            use_ssl=True,
            verify_certs=False,
            ssl_show_warn=False
        )
    else:
        credentials = session.get_credentials()
        awsauth = AWS4Auth(
            credentials.access_key,
            credentials.secret_key,
            args.region,
            'es',
            session_token=credentials.token
        )
        client = OpenSearch(
            hosts=[{'host': args.opensearch_host, 'port': 443}],
            http_auth=awsauth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection
        )
    
    # S3 또는 로컬에서 청킹
    if args.input_dir.startswith('s3://'):
        s3 = session.client('s3')
        bucket = args.input_dir.replace('s3://', '').split('/')[0]
        prefix = '/'.join(args.input_dir.replace('s3://', '').split('/')[1:])
        
        import tempfile
        local_dir = Path(tempfile.mkdtemp())
        
        paginator = s3.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get('Contents', []):
                if obj['Key'].endswith('.md'):
                    local_path = local_dir / obj['Key'].replace(prefix, '').lstrip('/')
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    s3.download_file(bucket, obj['Key'], str(local_path))
        input_path = local_dir
    else:
        input_path = Path(args.input_dir)
    
    chunks = []
    for md_file in input_path.rglob('*.md'):
        chunks.extend(parse_markdown(md_file))
    print(f"총 {len(chunks)}개 청크 생성")

    if args.recreate and client.indices.exists(index=args.index_name):
        client.indices.delete(index=args.index_name)
        print(f"인덱스 {args.index_name} 삭제됨")

    create_index(client, args.index_name)
    index_chunks(client, bedrock, chunks, args.index_name)


if __name__ == '__main__':
    main()
