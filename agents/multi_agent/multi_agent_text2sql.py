"""Multi-Agent Text2SQL System

Strands Swarm 패턴을 사용하여 3개의 전문화된 에이전트가 협업하는 멀티에이전트 시스템입니다.
기존 MyCustomAgent와 동일한 인터페이스를 제공하여 호환성을 유지합니다.

Requirements:
- 4.1: Strands Swarm 패턴을 사용하여 에이전트 간 협업 구성
- 4.2: handoff_to_agent 도구를 사용하여 다음 에이전트로 제어 이동
- 4.3: invocation_state를 통해 컨텍스트와 설정 전파
- 5.1: 기존 MyCustomAgent 인터페이스의 stream_response 메서드 제공
- 5.2: 기존 get_ui_state 메서드 제공
- 5.3: 기존 이벤트 시스템과 호환되는 콜백 제공
- 5.4: 모든 에이전트의 디버그 정보 통합
- 5.5: MCP 클라이언트 접근 관리
"""

import asyncio
import queue
import sys
import threading
import time
import os
from typing import Any, Callable, Dict, Generator, List, Optional

from strands import Agent
from strands.multiagent import Swarm
from strands.tools.mcp.mcp_client import MCPClient
from mcp import stdio_client, StdioServerParameters

from agents.events.registry import EventRegistry
from agents.events.lifecycle import (
    DebugHandler,
    LifecycleHandler,
    LoggingHandler,
    ReasoningHandler,
)
from agents.events.ui import StreamlitUIState

from .lead_agent import LeadAgent, AgentType, WorkflowStatus
from .data_expert_agent import DataExpertAgent
from .sql_agent import SQLAgent
from .shared_context import AnalysisContext, SwarmConfig
from .event_adapter import (
    SwarmEventAdapter,
    SwarmEventHandler,
    StreamlitSwarmUIHandler,
)


class MultiAgentText2SQL:
    """멀티에이전트 Text2SQL 시스템
    
    기존 MyCustomAgent와 동일한 인터페이스를 제공하면서
    내부적으로는 Swarm 패턴을 사용한 멀티에이전트 협업을 수행합니다.
    
    Requirements:
    - 5.1: stream_response 메서드 제공 (Swarm 실행 래핑)
    - 5.2: get_ui_state 메서드 제공
    - 5.3: 기존 이벤트 시스템과 호환되는 콜백 제공
    - 5.4: 모든 에이전트의 디버그 정보 통합
    - 5.5: MCP 클라이언트 접근 관리
    """
    
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.event_queue = queue.Queue()
        self.event_registry = EventRegistry()
        self.ui_state = StreamlitUIState()
        
        # 디버그 모드 상태 (Requirements 5.4)
        self._debug_enabled = False
        self._debug_handler: Optional[DebugHandler] = None
        
        # 이벤트 핸들러 설정 (Requirements 5.3)
        self._setup_handlers()
        
        # MCP 클라이언트 설정 (Requirements 5.5)
        self.mcp_client = self._setup_mcp_client()
        
        # Swarm 및 에이전트들 초기화
        self.swarm = self._create_swarm()
        
        # 공유 컨텍스트
        self.analysis_context = AnalysisContext()
        
        # 외부 콜백 핸들러 (Requirements 5.3)
        self._external_callback: Optional[Callable] = None
        
        # Swarm 이벤트 어댑터 설정 (Requirements 1.5, 5.3)
        self._event_adapter = SwarmEventAdapter(
            event_queue=self.event_queue,
            event_registry=self.event_registry,
        )
        
        # Swarm 이벤트 핸들러 등록 (Requirements 5.3)
        self._swarm_event_handler = SwarmEventHandler(self._event_adapter)
        self.event_registry.register(self._swarm_event_handler)
        
        # Streamlit Swarm UI 핸들러 (Requirements 1.5)
        self._swarm_ui_handler = StreamlitSwarmUIHandler(self._event_adapter, self.ui_state)
        self.event_registry.register(self._swarm_ui_handler)
        
        # 현재 활성 에이전트 추적 (lead_agent 응답만 UI에 표시)
        self._current_agent: str = "lead_agent"
    
    def _setup_handlers(self):
        """핵심 핸들러들을 등록합니다. (Requirements 5.3)
        
        기존 이벤트 시스템과 호환되는 핸들러들을 설정합니다.
        """
        self.event_registry.register(LifecycleHandler())
        self.event_registry.register(ReasoningHandler())
        self.event_registry.register(LoggingHandler(log_level="INFO"))
        
        # 디버그 핸들러 참조 저장 (Requirements 5.4)
        self._debug_handler = DebugHandler(debug_enabled=self._debug_enabled)
        self.event_registry.register(self._debug_handler)
    
    def _setup_mcp_client(self) -> MCPClient:
        """MCP 클라이언트 설정 (Requirements 5.5)
        
        AWS Athena 데이터 처리를 위한 MCP 서버 연결을 설정합니다.
        MCP 클라이언트는 중앙에서 관리되며 모든 에이전트가 공유합니다.
        """
        # MCP 서버 환경 변수 설정
        mcp_env = {
            "FASTMCP_LOG_LEVEL": "ERROR",
            "LOGURU_LEVEL": "ERROR",
            "LOG_LEVEL": "ERROR",
            "AWS_PROFILE": os.environ.get("AWS_PROFILE", "demo"),
            "AWS_DEFAULT_REGION": os.environ.get("AWS_DEFAULT_REGION", "us-west-2"),
        }
        
        # Athena 출력 위치 설정 (선택적)
        athena_output = os.environ.get("ATHENA_OUTPUT_LOCATION")
        if athena_output:
            mcp_env["ATHENA_OUTPUT_LOCATION"] = athena_output
        
        mcp_client = MCPClient(
            lambda: stdio_client(
                StdioServerParameters(
                    command="uv",
                    args=["run", "awslabs.aws-dataprocessing-mcp-server"],
                    env=mcp_env,
                ),
            ),
            tool_filters={"allowed": [
                "manage_aws_athena_query_executions",
                "manage_aws_athena_data_catalogs",
                "manage_aws_athena_databases_and_tables",
                "manage_aws_athena_workgroups"
            ]}
        )
        mcp_client.start()
        return mcp_client
    
    def get_mcp_client(self) -> Optional[MCPClient]:
        """MCP 클라이언트 접근 (Requirements 5.5)
        
        AWS 데이터 처리 도구에 대한 접근을 관리합니다.
        
        Returns:
            MCPClient 인스턴스 또는 None
        """
        return self.mcp_client
    
    def is_mcp_client_active(self) -> bool:
        """MCP 클라이언트 활성 상태 확인 (Requirements 5.5)
        
        Returns:
            MCP 클라이언트가 활성 상태인지 여부
        """
        return self.mcp_client is not None
    
    def _get_mcp_tools(self) -> List:
        """MCP 클라이언트에서 도구 목록 가져오기
        
        Returns:
            MCP 도구 목록
        """
        if self.mcp_client:
            try:
                return self.mcp_client.list_tools_sync()
            except Exception:
                return []
        return []
    
    def _filter_tools_by_name(self, tools: List, allowed_names: List[str]) -> List:
        """도구 목록에서 허용된 이름의 도구만 필터링
        
        Args:
            tools: 전체 도구 목록
            allowed_names: 허용할 도구 이름 목록
            
        Returns:
            필터링된 도구 목록
        """
        if not tools:
            return []
        
        filtered = []
        for tool in tools:
            # MCPAgentTool은 tool_name 속성을 사용
            tool_name = getattr(tool, 'tool_name', None)
            # 일반 도구는 name 속성 사용
            if tool_name is None:
                tool_name = getattr(tool, 'name', None)
            # dict인 경우
            if tool_name is None and isinstance(tool, dict):
                tool_name = tool.get('name')
            
            if tool_name and tool_name in allowed_names:
                filtered.append(tool)
        
        return filtered
    
    def _create_swarm(self) -> Swarm:
        """Swarm 및 에이전트들 생성 (Requirements 4.1, 4.2, 4.3)
        
        Strands Swarm 패턴을 사용하여 에이전트 간 협업을 구성합니다.
        - handoff_to_agent 도구는 Swarm에서 자동으로 각 에이전트에 제공됩니다
        - invocation_state를 통해 MCP 클라이언트와 설정을 공유합니다
        """
        # MCP 도구 가져오기 및 에이전트별 필터링
        mcp_tools = self._get_mcp_tools()
        
        # 디버그: MCP 도구 목록 출력
        print(f"\n🔧 [MCP Tools] 총 {len(mcp_tools)}개 도구 로드됨", file=sys.stderr)
        for tool in mcp_tools:
            tool_name = getattr(tool, 'name', None) or (tool.get('name') if isinstance(tool, dict) else str(tool))
            print(f"   - {tool_name}", file=sys.stderr)
        
        # 에이전트별 도구 필터링
        data_expert_tools = self._filter_tools_by_name(
            mcp_tools, 
            ["manage_aws_athena_data_catalogs", "manage_aws_athena_databases_and_tables"]
        )
        sql_agent_tools = self._filter_tools_by_name(
            mcp_tools,
            ["manage_aws_athena_query_executions", "manage_aws_athena_workgroups"]
        )
        
        # 개별 에이전트 생성 (필터링된 도구 전달)
        self.lead_agent = LeadAgent(self.model_id, tools=[])
        self.data_expert = DataExpertAgent(self.model_id, tools=data_expert_tools)
        self.sql_agent = SQLAgent(self.model_id, tools=sql_agent_tools)
        
        # 각 에이전트에 별도의 callback_handler 설정 (Requirements 5.3 - UI 이벤트 전달)
        # data_expert는 터미널에만 로깅하는 핸들러 사용
        self.lead_agent.agent.callback_handler = self._create_callback_handler("lead_agent")
        self.data_expert.agent.callback_handler = self._create_callback_handler("data_expert")
        self.sql_agent.agent.callback_handler = self._create_callback_handler("sql_agent")
        
        # Swarm 설정 (Requirements 4.1)
        config = SwarmConfig()
        
        # invocation_state 설정 (Requirements 4.3 - 에이전트 간 공유 상태)
        # 이 상태는 LLM에 노출되지 않고 도구와 훅에서만 접근 가능
        self._invocation_state = {
            "mcp_client": self.mcp_client,
            "aws_config": {
                "region": os.environ.get("AWS_DEFAULT_REGION", "us-west-2"),
                "profile": os.environ.get("AWS_PROFILE", "default")
            },
            "debug_mode": False,
            "session_id": f"session_{int(time.time())}",
            "analysis_context": None  # 분석 컨텍스트 공유용
        }
        
        # Swarm 생성 (Requirements 4.1)
        # - entry_point: Lead Agent가 진입점
        # - handoff_to_agent 도구가 자동으로 각 에이전트에 추가됨 (Requirements 4.2)
        # 주의: 에이전트 리스트는 위치 인자로 전달해야 함 (agents= 키워드 사용 불가)
        swarm = Swarm(
            [self.lead_agent.agent, self.data_expert.agent, self.sql_agent.agent],
            entry_point=self.lead_agent.agent,
            max_handoffs=config.max_handoffs,
            max_iterations=config.max_iterations,
            execution_timeout=config.execution_timeout,
            node_timeout=config.node_timeout,
            repetitive_handoff_detection_window=config.repetitive_handoff_detection_window,
            repetitive_handoff_min_unique_agents=config.repetitive_handoff_min_unique_agents
        )
        
        return swarm
    
    def _create_callback_handler(self, agent_name: str):
        """에이전트별 callback handler 생성
        
        Args:
            agent_name: 에이전트 이름 (lead_agent, data_expert, sql_agent)
            
        Returns:
            해당 에이전트용 callback handler 함수
        """
        def handler(**kwargs):
            # 터미널 로깅 (모든 에이전트)
            self._log_agent_event_to_terminal(kwargs, agent_name)
            
            # data_expert의 이벤트는 UI에 표시하지 않음
            if agent_name == "data_expert":
                return
            
            # 텍스트 스트리밍 이벤트를 큐에 추가
            if "data" in kwargs:
                text = kwargs.get("data", "")
                if text:
                    self.event_queue.put({"data": text})
            
            # 도구 사용 이벤트
            elif "current_tool_use" in kwargs:
                self.event_queue.put({"current_tool_use": kwargs["current_tool_use"]})
            
            # 도구 결과 이벤트
            elif "tool_result" in kwargs:
                self.event_queue.put({"tool_result": kwargs["tool_result"]})
            
            # 추론 이벤트
            elif "reasoningText" in kwargs:
                self.event_queue.put({"reasoningText": kwargs["reasoningText"]})
        
        return handler
    
    def _log_agent_event_to_terminal(self, event: Dict[str, Any], agent_name: str = "") -> None:
        """에이전트 간 대화 이벤트를 터미널에 로깅합니다.
        
        UI에는 표시하지 않고 터미널에서만 에이전트 간 대화를 확인할 수 있습니다.
        """
        # 이벤트 타입 추출
        event_type = event.get("type", "")
        
        # 에이전트 상태 이벤트 (node_start, node_stop, handoff)
        if "multiagent_node_start" in str(event) or event_type == "multiagent_node_start":
            node_id = event.get("node_id", "unknown")
            print(f"\n🚀 [Agent Start] {node_id}", file=sys.stderr)
        
        elif "multiagent_node_stop" in str(event) or event_type == "multiagent_node_stop":
            node_id = event.get("node_id", "unknown")
            print(f"\n✅ [Agent Stop] {node_id}", file=sys.stderr)
        
        elif "multiagent_handoff" in str(event) or event_type == "multiagent_handoff":
            from_agents = event.get("from_node_ids", [])
            to_agents = event.get("to_node_ids", [])
            from_str = from_agents[0] if from_agents else "unknown"
            to_str = to_agents[0] if to_agents else "unknown"
            print(f"\n🔀 [Handoff] {from_str} → {to_str}", file=sys.stderr)
        
        # # 텍스트 스트리밍 이벤트
        # elif "data" in event:
        #     text = event.get("data", "")
        #     if text:
        #         # 줄바꿈 없이 스트리밍 출력
        #         print(text, end="", flush=True, file=sys.stderr)
        
        # 도구 사용 이벤트 (새 도구 호출 시작 시에만 로깅)
        elif "current_tool_use" in event:
            tool_info = event.get("current_tool_use", {})
            tool_use_id = tool_info.get("toolUseId", "")
            tool_name = tool_info.get("name", "")
            # 도구 이름이 있고, 새로운 도구 호출인 경우에만 로깅
            if tool_name and tool_use_id:
                if not hasattr(self, "_logged_tool_ids"):
                    self._logged_tool_ids = set()
                if tool_use_id not in self._logged_tool_ids:
                    self._logged_tool_ids.add(tool_use_id)
                    print(f"\n🔧 [Tool Call] {tool_name}", file=sys.stderr)
        
        # 도구 결과 이벤트
        elif "tool_result" in event:
            tool_result = event.get("tool_result", {})
            status = tool_result.get("status", "unknown")
            print(f"\n📋 [Tool Result] status={status}", file=sys.stderr)
        
        # # 추론 이벤트
        # elif "reasoningText" in event:
        #     reasoning = event.get("reasoningText", "")
        #     if reasoning:
        #         print(f"\n💭 [Reasoning] {reasoning[:100]}...", file=sys.stderr)
        
        # 완료 이벤트
        elif event_type == "complete" or "complete" in event:
            print(f"\n🏁 [Complete]", file=sys.stderr)
            # 완료 시 로깅된 도구 ID 초기화
            if hasattr(self, "_logged_tool_ids"):
                self._logged_tool_ids.clear()
    
    def set_callback_handler(self, callback: Callable) -> None:
        """외부 콜백 핸들러 설정 (Requirements 5.3)
        
        기존 이벤트 시스템과 호환되는 콜백을 설정합니다.
        
        Args:
            callback: 이벤트를 받을 콜백 함수
        """
        self._external_callback = callback
    
    def remove_callback_handler(self) -> None:
        """외부 콜백 핸들러 제거 (Requirements 5.3)"""
        self._external_callback = None
    
    def stream_response(self, user_input: str) -> Generator[Dict[str, Any], None, None]:
        """사용자 입력에 대한 스트리밍 응답을 생성합니다 (Requirements 5.1)
        
        이 메서드는 Streamlit 프론트엔드에서 필수로 요구됩니다.
        기존 MyCustomAgent와 동일한 인터페이스를 제공합니다.
        Swarm을 백그라운드에서 실행하고 실시간으로 이벤트를 스트리밍합니다.
        
        Args:
            user_input: 사용자의 자연어 입력
            
        Yields:
            이벤트 딕셔너리 (기존 MyCustomAgent와 동일한 형식)
        """
        
        # UI 상태 초기화
        self.ui_state.reset()
        
        # 현재 에이전트 초기화 (lead_agent부터 시작)
        self._current_agent = "lead_agent"
        
        # 이벤트 큐 비우기
        while not self.event_queue.empty():
            try:
                self.event_queue.get_nowait()
            except queue.Empty:
                break
        
        # 분석 컨텍스트 초기화
        self.analysis_context = AnalysisContext(user_query=user_input)
        self._invocation_state["analysis_context"] = self.analysis_context
        
        # 결과 저장용
        swarm_result = None
        swarm_error = None
        swarm_complete = threading.Event()
        
        def run_swarm():
            """백그라운드에서 Swarm 동기 실행"""
            nonlocal swarm_result, swarm_error
            try:
                swarm_result = self.swarm(
                    user_input,
                    invocation_state=self._invocation_state
                )
            except Exception as e:
                swarm_error = str(e)
            finally:
                # 완료 신호
                self.event_queue.put({"type": "_swarm_complete"})
                swarm_complete.set()
        
        # 백그라운드 스레드에서 Swarm 실행
        thread = threading.Thread(target=run_swarm)
        thread.start()
        
        # 시작 이벤트
        yield {"type": "start"}
        
        # 실시간으로 이벤트 큐에서 가져와서 yield
        while not swarm_complete.is_set() or not self.event_queue.empty():
            try:
                # 짧은 타임아웃으로 큐 폴링
                event = self.event_queue.get(timeout=0.1)
                
                # 내부 완료 신호는 스킵
                if event.get("type") == "_swarm_complete":
                    continue
                
                yield event
                
            except queue.Empty:
                # 큐가 비어있으면 계속 대기
                continue
        
        # 스레드 완료 대기 (안전장치)
        thread.join(timeout=10)
        
        # 에러 처리
        if swarm_error:
            yield {"type": "force_stop", "force_stop_reason": swarm_error}
            return
        
        # 완료 이벤트
        yield {"type": "complete", "result": swarm_result}
    
    def _extract_final_response(self, swarm_result) -> str:
        """SwarmResult에서 최종 응답 텍스트 추출"""
        try:
            # SwarmResult의 마지막 에이전트 결과에서 텍스트 추출
            if hasattr(swarm_result, 'result') and swarm_result.result:
                result = swarm_result.result
                # AgentResult에서 메시지 추출
                if hasattr(result, 'message'):
                    msg = result.message
                    if hasattr(msg, 'content'):
                        # content가 리스트인 경우
                        if isinstance(msg.content, list):
                            texts = []
                            for block in msg.content:
                                if hasattr(block, 'text'):
                                    texts.append(block.text)
                                elif isinstance(block, dict) and 'text' in block:
                                    texts.append(block['text'])
                            return ''.join(texts)
                        return str(msg.content)
                # 문자열로 변환 시도
                return str(result)
            
            # results 딕셔너리에서 마지막 결과 추출
            if hasattr(swarm_result, 'results') and swarm_result.results:
                last_result = list(swarm_result.results.values())[-1]
                if hasattr(last_result, 'result'):
                    return self._extract_final_response(last_result)
            
            return str(swarm_result)
        except Exception:
            return str(swarm_result) if swarm_result else ""
    
    def _convert_swarm_event(self, swarm_event: Dict[str, Any]) -> Dict[str, Any]:
        """Swarm 이벤트를 기존 이벤트 형식으로 변환
        
        이벤트 어댑터를 사용하여 Swarm 이벤트를 Streamlit 이벤트로 변환합니다.
        
        Requirements:
        - 1.5: 작업 진행 상황 표시
        - 5.3: 기존 이벤트 시스템과 호환
        """
        # 이벤트 어댑터를 통해 변환 (Requirements 1.5, 5.3)
        converted_event = self._event_adapter.convert_event(swarm_event)
        
        # Lead Agent 상태 업데이트 (Requirements 1.5)
        event_type = swarm_event.get("type", "")
        if event_type in ("multiagent_node_start", "multiagent_handoff"):
            agent_name = swarm_event.get("node_id")
            if not agent_name and event_type == "multiagent_handoff":
                to_agents = swarm_event.get("to_node_ids", [])
                agent_name = to_agents[0] if to_agents else None
            if agent_name:
                self._update_lead_agent_status(agent_name)
        
        return converted_event
    
    def _update_lead_agent_status(self, agent_name: str) -> None:
        """Lead Agent의 워크플로우 상태 업데이트 (Requirements 1.5)"""
        if not hasattr(self, 'lead_agent'):
            return
        
        agent_type_map = {
            "lead_agent": AgentType.LEAD,
            "data_expert": AgentType.DATA_EXPERT,
            "sql_agent": AgentType.SQL
        }
        
        status_map = {
            "lead_agent": WorkflowStatus.ANALYZING,
            "data_expert": WorkflowStatus.DATA_EXPLORATION,
            "sql_agent": WorkflowStatus.SQL_GENERATION
        }
        
        agent_type = agent_type_map.get(agent_name)
        status = status_map.get(agent_name, WorkflowStatus.ANALYZING)
        
        if agent_type:
            self.lead_agent.update_agent_status(agent_type, status)
    
    def get_ui_state(self) -> StreamlitUIState:
        """현재 UI 상태를 반환합니다 (Requirements 5.2)
        
        이 메서드는 Streamlit 프론트엔드에서 필수로 요구됩니다.
        기존 MyCustomAgent와 동일한 인터페이스를 제공합니다.
        
        Returns:
            StreamlitUIState 인스턴스
        """
        return self.ui_state
    
    async def stream_response_async(self, user_input: str):
        """비동기 스트리밍 응답 생성
        
        Swarm의 stream_async를 사용하여 실시간 이벤트를 스트리밍합니다.
        
        Requirements:
        - 4.1: Swarm 패턴으로 에이전트 협업
        - 4.2: handoff_to_agent로 에이전트 간 제어 이동
        - 4.3: invocation_state로 컨텍스트 전파
        """
        # 분석 컨텍스트 초기화
        self.analysis_context = AnalysisContext(user_query=user_input)
        self._invocation_state["analysis_context"] = self.analysis_context
        
        # 시작 이벤트
        yield {"type": "start"}
        
        try:
            # Swarm 스트리밍 실행 (Requirements 4.1, 4.2, 4.3)
            async for event in self.swarm.stream_async(
                user_input,
                invocation_state=self._invocation_state
            ):
                # 이벤트 변환 및 전달
                converted_event = self._convert_swarm_event(event)
                yield converted_event
                
                # 최종 결과 처리
                if event.get("type") == "multiagent_result":
                    break
                    
        except Exception as e:
            yield {"type": "force_stop", "reason": str(e)}
    
    def enable_debug_mode(self, enabled: bool = True):
        """디버그 모드를 토글합니다. (Requirements 5.4)
        
        모든 에이전트의 디버그 정보를 통합하여 표시합니다.
        
        Args:
            enabled: 디버그 모드 활성화 여부
        """
        self._debug_enabled = enabled
        
        # 디버그 핸들러 업데이트
        if self._debug_handler:
            self._debug_handler.debug_enabled = enabled
        
        # 이벤트 레지스트리의 모든 디버그 핸들러 업데이트
        for handler in self.event_registry._handlers:
            if isinstance(handler, DebugHandler):
                handler.debug_enabled = enabled
        
        # invocation_state에도 반영 (Requirements 4.3)
        self._invocation_state["debug_mode"] = enabled
    
    def is_debug_enabled(self) -> bool:
        """디버그 모드 활성화 상태 확인 (Requirements 5.4)
        
        Returns:
            디버그 모드 활성화 여부
        """
        return self._debug_enabled
    
    def get_debug_info(self) -> Dict[str, Any]:
        """모든 에이전트의 디버그 정보 통합 반환 (Requirements 5.4)
        
        Returns:
            통합된 디버그 정보 딕셔너리
        """
        debug_info = {
            "debug_enabled": self._debug_enabled,
            "event_log": [],
            "agents": {},
            "workflow_status": self.get_workflow_status(),
            "analysis_context": {
                "user_query": self.analysis_context.user_query,
                "business_intent": self.analysis_context.business_intent,
                "tables_count": len(self.analysis_context.identified_tables),
                "has_sql": self.analysis_context.generated_sql is not None,
                "has_results": self.analysis_context.results is not None,
                "error_count": len(self.analysis_context.error_messages)
            }
        }
        
        # 디버그 핸들러의 이벤트 로그 추가
        if self._debug_handler and self._debug_handler.debug_enabled:
            debug_info["event_log"] = self._debug_handler.event_log.copy()
        
        # 각 에이전트의 상태 정보 추가
        if hasattr(self, 'lead_agent'):
            debug_info["agents"]["lead_agent"] = {
                "status": self.lead_agent.workflow_state.status.value,
                "current_agent": (
                    self.lead_agent.workflow_state.current_agent.value 
                    if self.lead_agent.workflow_state.current_agent else None
                ),
                "results_count": len(self.lead_agent.workflow_state.agent_results)
            }
        
        if hasattr(self, 'data_expert'):
            debug_info["agents"]["data_expert"] = {
                "initialized": self.data_expert.agent is not None
            }
        
        if hasattr(self, 'sql_agent'):
            debug_info["agents"]["sql_agent"] = {
                "initialized": self.sql_agent.agent is not None
            }
        
        return debug_info
    
    def get_analysis_context(self) -> AnalysisContext:
        """현재 분석 컨텍스트를 반환합니다."""
        return self.analysis_context
    
    def reset_context(self):
        """분석 컨텍스트를 초기화합니다."""
        self.analysis_context = AnalysisContext()
        if hasattr(self, 'lead_agent'):
            self.lead_agent.reset_workflow_state()
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """현재 워크플로우 상태를 반환합니다 (Requirements 1.5)"""
        if hasattr(self, 'lead_agent'):
            return self.lead_agent.get_current_status()
        return {
            "status": "idle",
            "current_agent": None,
            "message": "대기 중",
            "progress": []
        }
    
    def get_event_registry(self) -> EventRegistry:
        """이벤트 레지스트리 반환 (Requirements 5.3)
        
        기존 이벤트 시스템과의 호환성을 위해 이벤트 레지스트리에 접근합니다.
        
        Returns:
            EventRegistry 인스턴스
        """
        return self.event_registry
    
    def register_event_handler(self, handler) -> None:
        """이벤트 핸들러 등록 (Requirements 5.3)
        
        기존 이벤트 시스템과 호환되는 핸들러를 등록합니다.
        
        Args:
            handler: EventHandler 인스턴스
        """
        self.event_registry.register(handler)
    
    def get_event_adapter(self) -> SwarmEventAdapter:
        """이벤트 어댑터 반환 (Requirements 1.5, 5.3)
        
        Swarm 이벤트를 Streamlit 이벤트로 변환하는 어댑터에 접근합니다.
        
        Returns:
            SwarmEventAdapter 인스턴스
        """
        return self._event_adapter
    
    def get_swarm_workflow_status(self) -> Dict[str, Any]:
        """Swarm 워크플로우 상태 반환 (Requirements 1.5)
        
        이벤트 어댑터를 통해 현재 워크플로우 상태를 반환합니다.
        
        Returns:
            워크플로우 상태 딕셔너리
        """
        return self._event_adapter.get_current_status()
    
    def get_agent_progress(self) -> List[Dict[str, Any]]:
        """에이전트 진행 상황 반환 (Requirements 1.5)
        
        이벤트 어댑터를 통해 에이전트 진행 상황을 반환합니다.
        
        Returns:
            에이전트 진행 상황 목록
        """
        return self._event_adapter.get_agent_progress()
    
    def set_status_placeholder(self, placeholder) -> None:
        """상태 표시용 placeholder 설정 (Requirements 1.5)
        
        Streamlit UI에서 에이전트 상태를 표시할 placeholder를 설정합니다.
        
        Args:
            placeholder: Streamlit placeholder 객체
        """
        self._swarm_ui_handler.set_status_placeholder(placeholder)
    
    def __del__(self):
        """소멸자 - MCP 클라이언트 정리 (Requirements 5.5)"""
        if hasattr(self, 'mcp_client') and self.mcp_client:
            try:
                self.mcp_client.stop()
            except:
                pass