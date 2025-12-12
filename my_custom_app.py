"""커스텀 에이전트를 사용하는 Streamlit 앱

이 파일은 새로 만든 MyCustomAgent를 사용하는 방법을 보여줍니다.
"""

import streamlit as st
from app.main import StreamlitChatApp
from app.config import AppConfig
from agents.my_custom_agent import MyCustomAgent


def create_custom_agent(model_id: str) -> MyCustomAgent:
    """커스텀 에이전트 팩토리 함수"""
    return MyCustomAgent(model_id=model_id)


def main():
    """메인 애플리케이션 실행"""
    
    # 커스텀 설정 생성
    config = AppConfig(
        # 페이지 설정
        page_config={
            "page_title": "My Custom Agent Chat",
            "page_icon": "🔧",
            "layout": "wide",
        },
        
        # UI 설정
        app_title="🔧 My Custom Agent Chat",
        sidebar_header="🎛️ Model Settings",
        chat_input_placeholder="텍스트 분석이나 번역을 요청해보세요...",
        
        # 사용 가능한 모델들
        available_models=[
            "openai.gpt-oss-120b-1:0",
            "us.amazon.nova-pro-v1:0",
            "us.amazon.nova-premier-v1:0", 
            "us.anthropic.claude-sonnet-4-20250514-v1:0",
        ],
        
        # 기본 모델
        default_model="openai.gpt-oss-120b-1:0",
        
        # 커스텀 에이전트 팩토리 설정
        agent_factory=create_custom_agent
    )
    
    # 앱 실행
    app = StreamlitChatApp(config)
    app.run()


if __name__ == "__main__":
    main()