
import streamlit as st
from app.main import StreamlitChatApp
from app.config import AppConfig
from agents.multi_agent.multi_agent_text2sql import MultiAgentText2SQL


def create_multi_agent(model_id: str) -> MultiAgentText2SQL:
    """멀티에이전트 Text2SQL 팩토리 함수
    
    기존 MyCustomAgent를 대체하는 새로운 멀티에이전트 시스템입니다.
    Strands Swarm 패턴을 사용하여 3개의 전문화된 에이전트가 협업합니다:
    - Lead Agent: 사용자 요청 분석 및 워크플로우 조정
    - Data Expert Agent: AWS Athena 카탈로그 탐색 및 테이블 식별
    - SQL Agent: SQL 쿼리 생성 및 실행

    ID: Hello
    pw: 12341234
    
    Requirements:
    - 5.1: 기존 MyCustomAgent 인터페이스의 stream_response 메서드 제공
    - 5.2: 기존 get_ui_state 메서드 제공
    - 5.3: 기존 이벤트 시스템과 호환되는 콜백 제공
    """
    return MultiAgentText2SQL(model_id=model_id)


def main():
    """메인 애플리케이션 실행
    
    멀티에이전트 Text2SQL 시스템을 사용하는 Streamlit 채팅 애플리케이션입니다.
    기존 MyCustomAgent와 동일한 인터페이스를 제공하여 호환성을 유지합니다.
    
    Requirements:
    - 5.1: stream_response 메서드 제공
    - 5.2: get_ui_state 메서드 제공
    - 5.3: 기존 이벤트 시스템과 호환
    - 5.4: 디버그 정보 통합
    - 5.5: MCP 클라이언트 접근 관리
    """
    
    # 멀티에이전트 설정 생성
    config = AppConfig(
        # 페이지 설정
        page_config={
            "page_title": "Multi-Agent Text2SQL Chat",
            "page_icon": "🤖",
            "layout": "wide",
        },
        
        # UI 설정
        app_title="🤖 Multi-Agent Text2SQL Chat",
        sidebar_header="🎛️ Model Settings",
        chat_input_placeholder="자연어로 데이터 분석을 요청해보세요...",
        
        # 사용 가능한 모델들
        available_models=[
            "us.anthropic.claude-sonnet-4-20250514-v1:0",
            "us.amazon.nova-pro-v1:0",
            "us.amazon.nova-premier-v1:0", 
            "openai.gpt-oss-20b-1:0"
        ],
        
        # 기본 모델 (Claude가 Swarm과 가장 호환성이 좋음)
        default_model="us.anthropic.claude-sonnet-4-20250514-v1:0",
        
        # 멀티에이전트 팩토리 설정 (Requirements 5.1, 5.2, 5.3)
        agent_factory=create_multi_agent
    )
    
    # 앱 실행
    app = StreamlitChatApp(config)
    app.run()


if __name__ == "__main__":
    main()