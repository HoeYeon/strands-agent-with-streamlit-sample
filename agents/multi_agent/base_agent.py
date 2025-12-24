"""기본 에이전트 인터페이스

모든 멀티에이전트 시스템의 에이전트가 상속받을 기본 클래스를 정의합니다.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from strands import Agent

from .shared_context import AnalysisContext


class BaseMultiAgent(ABC):
    """멀티에이전트 시스템의 기본 에이전트 클래스"""
    
    def __init__(self, model_id: str, tools: Optional[List] = None):
        self.model_id = model_id
        self.tools = tools or []
        self.agent: Optional[Agent] = None
        self._setup_agent()
    
    @abstractmethod
    def _setup_agent(self):
        """에이전트 초기화 - 각 에이전트에서 구현"""
        pass
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """에이전트별 시스템 프롬프트 반환"""
        pass
    
    @abstractmethod
    def get_tools(self) -> list:
        """에이전트별 도구 목록 반환"""
        pass
    
    def process_context(self, context: AnalysisContext) -> Dict[str, Any]:
        """공유 컨텍스트를 처리하고 결과 반환"""
        if not self.agent:
            raise RuntimeError("Agent not initialized")
        
        # 컨텍스트를 기반으로 프롬프트 생성
        prompt = self._build_prompt_from_context(context)
        
        try:
            result = self.agent(prompt)
            return {"success": True, "result": result, "context": context}
        except Exception as e:
            context.add_error(f"{self.__class__.__name__} error: {str(e)}")
            return {"success": False, "error": str(e), "context": context}
    
    @abstractmethod
    def _build_prompt_from_context(self, context: AnalysisContext) -> str:
        """컨텍스트를 기반으로 에이전트별 프롬프트 생성"""
        pass
    
    def get_agent(self) -> Agent:
        """Swarm에서 사용할 Agent 인스턴스 반환"""
        if not self.agent:
            raise RuntimeError("Agent not initialized")
        return self.agent