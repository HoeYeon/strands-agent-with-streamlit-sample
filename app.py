"""Streamlit application entry point."""

import os
from pathlib import Path
from app.main import StreamlitChatApp
from app.config import AppConfig
from agents.my_custom_agent import MyCustomAgent


def create_custom_agent(model_id: str) -> MyCustomAgent:
    """커스텀 에이전트 팩토리 함수"""
    return MyCustomAgent(model_id=model_id)

def main() -> None:
    """Main entry point for the Streamlit application."""
    model_id = "openai.gpt-oss-120b-1:0"
    config = AppConfig(agent_factory=create_custom_agent)
    app = StreamlitChatApp(config)
    app.run()


if __name__ == "__main__":
    main()
