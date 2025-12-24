"""Swarm 이벤트를 Streamlit 이벤트로 변환하는 어댑터

Strands Swarm의 멀티에이전트 이벤트를 기존 Streamlit UI 이벤트 시스템과
호환되는 형식으로 변환합니다.

Requirements:
- 1.5: 작업 진행 상황 확인 시 현재 어떤 에이전트가 작업 중인지 상태 표시
- 5.3: 기존 이벤트 시스템과 호환되는 콜백 제공
"""

import queue
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from agents.events.registry import EventRegistry, EventHandler, EventType


class SwarmEventType(Enum):
    """Swarm 멀티에이전트 이벤트 타입"""
    # 멀티에이전트 이벤트
    NODE_START = "multiagent_node_start"
    NODE_STREAM = "multiagent_node_stream"
    NODE_STOP = "multiagent_node_stop"
    HANDOFF = "multiagent_handoff"
    RESULT = "multiagent_result"
    
    # 기본 에이전트 이벤트
    DATA = "data"
    DELTA = "delta"
    CURRENT_TOOL_USE = "current_tool_use"
    TOOL_RESULT = "tool_result"
    REASONING = "reasoning"
    REASONING_TEXT = "reasoningText"
    
    # 라이프사이클 이벤트
    INIT_EVENT_LOOP = "init_event_loop"
    START_EVENT_LOOP = "start_event_loop"
    MESSAGE = "message"
    COMPLETE = "complete"
    FORCE_STOP = "force_stop"
    RESULT_LEGACY = "result"


class StreamlitEventType(Enum):
    """Streamlit UI 이벤트 타입"""
    # 에이전트 상태 이벤트
    AGENT_STATUS = "agent_status"
    AGENT_HANDOFF = "agent_handoff"
    
    # 텍스트 이벤트
    TEXT_DELTA = "text_delta"
    TEXT_COMPLETE = "text_complete"
    
    # 도구 이벤트
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    
    # 추론 이벤트
    REASONING = "reasoning"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    
    # 라이프사이클 이벤트
    START = "start"
    COMPLETE = "complete"
    FORCE_STOP = "force_stop"
    ERROR = "error"


@dataclass
class AgentStatusInfo:
    """에이전트 상태 정보"""
    agent_name: str
    status: str  # "idle", "working", "completed", "error"
    message: str = ""
    progress: float = 0.0


@dataclass
class SwarmEventAdapterState:
    """어댑터 상태 관리"""
    current_agent: Optional[str] = None
    agent_history: List[str] = field(default_factory=list)
    agent_statuses: Dict[str, AgentStatusInfo] = field(default_factory=dict)
    tool_calls: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    accumulated_text: str = ""
    is_completed: bool = False
    error_message: Optional[str] = None


class SwarmEventAdapter:
    """Swarm 이벤트를 Streamlit 이벤트로 변환하는 어댑터
    
    Requirements:
    - 1.5: 작업 진행 상황 확인 시 현재 어떤 에이전트가 작업 중인지 상태 표시
    - 5.3: 기존 이벤트 시스템과 호환되는 콜백 제공
    """
    
    # 에이전트 이름 매핑 (내부 이름 -> 표시 이름)
    AGENT_DISPLAY_NAMES = {
        "lead_agent": "Lead Agent (조정자)",
        "data_expert": "Data Expert (데이터 전문가)",
        "sql_agent": "SQL Agent (쿼리 전문가)",
    }
    
    # 에이전트별 상태 메시지
    AGENT_STATUS_MESSAGES = {
        "lead_agent": {
            "working": "사용자 요청을 분석하고 있습니다...",
            "completed": "분석 완료",
        },
        "data_expert": {
            "working": "데이터 카탈로그를 탐색하고 있습니다...",
            "completed": "테이블 식별 완료",
        },
        "sql_agent": {
            "working": "SQL 쿼리를 생성하고 실행하고 있습니다...",
            "completed": "쿼리 실행 완료",
        },
    }
    
    def __init__(
        self,
        event_queue: Optional[queue.Queue] = None,
        event_registry: Optional[EventRegistry] = None,
        external_callback: Optional[Callable] = None,
    ):
        """어댑터 초기화
        
        Args:
            event_queue: 변환된 이벤트를 저장할 큐
            event_registry: 기존 이벤트 레지스트리
            external_callback: 외부 콜백 함수
        """
        self.event_queue = event_queue or queue.Queue()
        self.event_registry = event_registry
        self.external_callback = external_callback
        self.state = SwarmEventAdapterState()
    
    def reset(self) -> None:
        """어댑터 상태 초기화"""
        self.state = SwarmEventAdapterState()
        # 이벤트 큐 비우기
        while not self.event_queue.empty():
            try:
                self.event_queue.get_nowait()
            except queue.Empty:
                break
    
    def convert_event(self, swarm_event: Dict[str, Any]) -> Dict[str, Any]:
        """Swarm 이벤트를 Streamlit 이벤트로 변환
        
        Args:
            swarm_event: Swarm에서 발생한 이벤트
            
        Returns:
            Streamlit UI와 호환되는 이벤트
        """
        event_type = swarm_event.get("type", self._infer_event_type(swarm_event))
        
        # 멀티에이전트 이벤트 변환
        if event_type == SwarmEventType.NODE_START.value:
            return self._convert_node_start(swarm_event)
        elif event_type == SwarmEventType.NODE_STREAM.value:
            return self._convert_node_stream(swarm_event)
        elif event_type == SwarmEventType.NODE_STOP.value:
            return self._convert_node_stop(swarm_event)
        elif event_type == SwarmEventType.HANDOFF.value:
            return self._convert_handoff(swarm_event)
        elif event_type == SwarmEventType.RESULT.value:
            return self._convert_result(swarm_event)
        
        # 기본 에이전트 이벤트 변환
        elif event_type == SwarmEventType.DATA.value or "data" in swarm_event:
            return self._convert_data(swarm_event)
        elif event_type == SwarmEventType.CURRENT_TOOL_USE.value or "current_tool_use" in swarm_event:
            return self._convert_tool_use(swarm_event)
        elif event_type == SwarmEventType.TOOL_RESULT.value or "tool_result" in swarm_event:
            return self._convert_tool_result(swarm_event)
        elif event_type in (SwarmEventType.REASONING.value, SwarmEventType.REASONING_TEXT.value) or "reasoningText" in swarm_event:
            return self._convert_reasoning(swarm_event)
        
        # 라이프사이클 이벤트 변환
        elif event_type == SwarmEventType.COMPLETE.value or event_type == "complete":
            return self._convert_complete(swarm_event)
        elif event_type == SwarmEventType.FORCE_STOP.value or "force_stop" in swarm_event:
            return self._convert_force_stop(swarm_event)
        elif event_type == SwarmEventType.RESULT_LEGACY.value or "result" in swarm_event:
            return self._convert_legacy_result(swarm_event)
        
        # 알 수 없는 이벤트는 그대로 전달
        return swarm_event
    
    def process_event(self, swarm_event: Dict[str, Any]) -> Dict[str, Any]:
        """이벤트를 변환하고 큐에 추가 및 핸들러에 전달
        
        Args:
            swarm_event: Swarm에서 발생한 이벤트
            
        Returns:
            변환된 Streamlit 이벤트
        """
        converted_event = self.convert_event(swarm_event)
        
        # 이벤트 큐에 추가
        self.event_queue.put(converted_event)
        
        # 이벤트 레지스트리를 통해 핸들러들에게 전달 (Requirements 5.3)
        if self.event_registry:
            self.event_registry.process_event(converted_event)
        
        # 외부 콜백 호출 (Requirements 5.3)
        if self.external_callback:
            try:
                self.external_callback(**converted_event)
            except Exception:
                pass  # 외부 콜백 오류는 무시
        
        return converted_event
    
    def _infer_event_type(self, event: Dict[str, Any]) -> str:
        """이벤트 타입 추론"""
        # 우선순위 기반 타입 추론
        priority_keys = [
            "multiagent_node_start", "multiagent_node_stream", "multiagent_node_stop",
            "multiagent_handoff", "multiagent_result",
            "data", "current_tool_use", "tool_result",
            "reasoningText", "reasoning",
            "result", "force_stop", "complete"
        ]
        
        for key in priority_keys:
            if key in event:
                return key
        
        return event.get("type", "unknown")
    
    def _convert_node_start(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """에이전트 시작 이벤트 변환 (Requirements 1.5)"""
        node_id = event.get("node_id", "unknown")
        node_type = event.get("node_type", "agent")
        
        # 상태 업데이트
        self.state.current_agent = node_id
        if node_id not in self.state.agent_history:
            self.state.agent_history.append(node_id)
        
        # 에이전트 상태 정보 업데이트
        display_name = self.AGENT_DISPLAY_NAMES.get(node_id, node_id)
        status_messages = self.AGENT_STATUS_MESSAGES.get(node_id, {})
        message = status_messages.get("working", f"{display_name}가 작업을 시작합니다...")
        
        self.state.agent_statuses[node_id] = AgentStatusInfo(
            agent_name=node_id,
            status="working",
            message=message,
        )
        
        return {
            "type": StreamlitEventType.AGENT_STATUS.value,
            "agent": node_id,
            "agent_display_name": display_name,
            "node_type": node_type,
            "status": "working",
            "message": message,
            "agent_history": self.state.agent_history.copy(),
        }
    
    def _convert_node_stream(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """에이전트 스트리밍 이벤트 변환"""
        node_id = event.get("node_id", self.state.current_agent or "unknown")
        inner_event = event.get("event", {})
        
        # 내부 이벤트 타입에 따라 변환
        if "data" in inner_event:
            text = inner_event.get("data", "")
            self.state.accumulated_text += text
            return {
                "type": StreamlitEventType.TEXT_DELTA.value,
                "data": text,
                "text": text,
                "agent": node_id,
                "accumulated_text": self.state.accumulated_text,
            }
        elif "current_tool_use" in inner_event:
            return self._convert_tool_use(inner_event, agent=node_id)
        elif "tool_result" in inner_event:
            return self._convert_tool_result(inner_event, agent=node_id)
        elif "reasoningText" in inner_event:
            return self._convert_reasoning(inner_event, agent=node_id)
        
        # 기본적으로 내부 이벤트에 에이전트 정보 추가
        inner_event["agent"] = node_id
        return inner_event
    
    def _convert_node_stop(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """에이전트 종료 이벤트 변환 (Requirements 1.5)"""
        node_id = event.get("node_id", self.state.current_agent or "unknown")
        node_result = event.get("node_result", {})
        
        # 에이전트 상태 업데이트
        display_name = self.AGENT_DISPLAY_NAMES.get(node_id, node_id)
        status_messages = self.AGENT_STATUS_MESSAGES.get(node_id, {})
        message = status_messages.get("completed", f"{display_name} 작업 완료")
        
        if node_id in self.state.agent_statuses:
            self.state.agent_statuses[node_id].status = "completed"
            self.state.agent_statuses[node_id].message = message
        
        return {
            "type": StreamlitEventType.AGENT_STATUS.value,
            "agent": node_id,
            "agent_display_name": display_name,
            "status": "completed",
            "message": message,
            "node_result": node_result,
            "agent_history": self.state.agent_history.copy(),
        }
    
    def _convert_handoff(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """에이전트 전환 이벤트 변환 (Requirements 1.5)"""
        from_node_ids = event.get("from_node_ids", [])
        to_node_ids = event.get("to_node_ids", [])
        handoff_message = event.get("message", "")
        
        from_agent = from_node_ids[0] if from_node_ids else self.state.current_agent
        to_agent = to_node_ids[0] if to_node_ids else "unknown"
        
        # 상태 업데이트
        self.state.current_agent = to_agent
        if to_agent not in self.state.agent_history:
            self.state.agent_history.append(to_agent)
        
        # 이전 에이전트 상태 업데이트
        if from_agent and from_agent in self.state.agent_statuses:
            self.state.agent_statuses[from_agent].status = "completed"
        
        # 새 에이전트 상태 설정
        to_display_name = self.AGENT_DISPLAY_NAMES.get(to_agent, to_agent)
        status_messages = self.AGENT_STATUS_MESSAGES.get(to_agent, {})
        message = status_messages.get("working", f"{to_display_name}로 작업을 전달합니다...")
        
        self.state.agent_statuses[to_agent] = AgentStatusInfo(
            agent_name=to_agent,
            status="working",
            message=message,
        )
        
        return {
            "type": StreamlitEventType.AGENT_HANDOFF.value,
            "from_agent": from_agent,
            "to_agent": to_agent,
            "from_agents": from_node_ids,
            "to_agents": to_node_ids,
            "handoff_message": handoff_message,
            "agent_display_name": to_display_name,
            "status": "working",
            "message": message,
            "agent_history": self.state.agent_history.copy(),
        }
    
    def _convert_result(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """최종 결과 이벤트 변환"""
        result = event.get("result")
        self.state.is_completed = True
        
        return {
            "type": StreamlitEventType.COMPLETE.value,
            "result": result,
            "status": "completed",
            "agent_history": self.state.agent_history.copy(),
            "final_agent": self.state.current_agent,
        }
    
    def _convert_data(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """텍스트 데이터 이벤트 변환"""
        text = event.get("data", "")
        self.state.accumulated_text += text
        
        return {
            "type": StreamlitEventType.TEXT_DELTA.value,
            "data": text,
            "text": text,
            "agent": self.state.current_agent,
        }
    
    def _convert_tool_use(self, event: Dict[str, Any], agent: Optional[str] = None) -> Dict[str, Any]:
        """도구 사용 이벤트 변환"""
        tool_info = event.get("current_tool_use", {})
        tool_use_id = tool_info.get("toolUseId") or tool_info.get("tool_use_id", "")
        tool_name = tool_info.get("name", "unknown")
        tool_input = tool_info.get("input", {})
        
        # 도구 호출 추적
        self.state.tool_calls[tool_use_id] = {
            "name": tool_name,
            "input": tool_input,
            "status": "running",
        }
        
        return {
            "type": StreamlitEventType.TOOL_CALL.value,
            "current_tool_use": tool_info,
            "tool_use_id": tool_use_id,
            "tool_name": tool_name,
            "arguments": tool_input,
            "status": "running",
            "agent": agent or self.state.current_agent,
        }
    
    def _convert_tool_result(self, event: Dict[str, Any], agent: Optional[str] = None) -> Dict[str, Any]:
        """도구 결과 이벤트 변환"""
        tool_result = event.get("tool_result", {})
        tool_use_id = tool_result.get("toolUseId") or tool_result.get("tool_use_id", "")
        result_content = tool_result.get("content", tool_result.get("result", ""))
        status = tool_result.get("status", "success")
        
        # 도구 호출 상태 업데이트
        if tool_use_id in self.state.tool_calls:
            self.state.tool_calls[tool_use_id]["status"] = "completed"
            self.state.tool_calls[tool_use_id]["result"] = result_content
        
        return {
            "type": StreamlitEventType.TOOL_RESULT.value,
            "tool_result": tool_result,
            "tool_use_id": tool_use_id,
            "result": result_content,
            "status": status,
            "agent": agent or self.state.current_agent,
        }
    
    def _convert_reasoning(self, event: Dict[str, Any], agent: Optional[str] = None) -> Dict[str, Any]:
        """추론 이벤트 변환"""
        reasoning_text = event.get("reasoningText", event.get("reasoning", ""))
        
        return {
            "type": StreamlitEventType.REASONING.value,
            "reasoningText": reasoning_text,
            "reasoning": reasoning_text,
            "agent": agent or self.state.current_agent,
        }
    
    def _convert_complete(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """완료 이벤트 변환"""
        self.state.is_completed = True
        result = event.get("result")
        
        return {
            "type": StreamlitEventType.COMPLETE.value,
            "result": result,
            "status": "completed",
            "agent_history": self.state.agent_history.copy(),
        }
    
    def _convert_force_stop(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """강제 중단 이벤트 변환"""
        reason = event.get("force_stop_reason", event.get("reason", "Unknown error"))
        self.state.error_message = reason
        self.state.is_completed = True
        
        # 현재 에이전트 상태를 에러로 업데이트
        if self.state.current_agent and self.state.current_agent in self.state.agent_statuses:
            self.state.agent_statuses[self.state.current_agent].status = "error"
            self.state.agent_statuses[self.state.current_agent].message = f"오류: {reason}"
        
        return {
            "type": StreamlitEventType.FORCE_STOP.value,
            "force_stop": True,
            "force_stop_reason": reason,
            "reason": reason,
            "agent": self.state.current_agent,
            "agent_history": self.state.agent_history.copy(),
        }
    
    def _convert_legacy_result(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """레거시 결과 이벤트 변환"""
        result = event.get("result")
        self.state.is_completed = True
        
        return {
            "type": StreamlitEventType.COMPLETE.value,
            "result": result,
            "status": "completed",
        }
    
    def get_current_status(self) -> Dict[str, Any]:
        """현재 워크플로우 상태 반환 (Requirements 1.5)"""
        return {
            "current_agent": self.state.current_agent,
            "current_agent_display_name": self.AGENT_DISPLAY_NAMES.get(
                self.state.current_agent, self.state.current_agent
            ) if self.state.current_agent else None,
            "agent_history": self.state.agent_history.copy(),
            "agent_statuses": {
                name: {
                    "agent_name": info.agent_name,
                    "status": info.status,
                    "message": info.message,
                    "progress": info.progress,
                }
                for name, info in self.state.agent_statuses.items()
            },
            "is_completed": self.state.is_completed,
            "error_message": self.state.error_message,
            "tool_calls_count": len(self.state.tool_calls),
        }
    
    def get_agent_progress(self) -> List[Dict[str, Any]]:
        """에이전트 진행 상황 목록 반환 (Requirements 1.5)"""
        progress = []
        for agent_name in self.state.agent_history:
            status_info = self.state.agent_statuses.get(agent_name)
            if status_info:
                progress.append({
                    "agent": agent_name,
                    "display_name": self.AGENT_DISPLAY_NAMES.get(agent_name, agent_name),
                    "status": status_info.status,
                    "message": status_info.message,
                    "is_current": agent_name == self.state.current_agent,
                })
        return progress



class SwarmEventHandler(EventHandler):
    """Swarm 이벤트를 처리하는 이벤트 핸들러
    
    기존 EventRegistry 시스템과 통합되어 Swarm 이벤트를 처리합니다.
    
    Requirements:
    - 5.3: 기존 이벤트 시스템과 호환되는 콜백 제공
    """
    
    # Swarm 관련 이벤트 타입들
    SWARM_EVENT_TYPES = {
        SwarmEventType.NODE_START.value,
        SwarmEventType.NODE_STREAM.value,
        SwarmEventType.NODE_STOP.value,
        SwarmEventType.HANDOFF.value,
        SwarmEventType.RESULT.value,
        StreamlitEventType.AGENT_STATUS.value,
        StreamlitEventType.AGENT_HANDOFF.value,
    }
    
    def __init__(self, adapter: SwarmEventAdapter):
        """핸들러 초기화
        
        Args:
            adapter: SwarmEventAdapter 인스턴스
        """
        self.adapter = adapter
    
    @property
    def priority(self) -> int:
        """핸들러 우선순위 (낮을수록 먼저 실행)"""
        return 5  # UI 핸들러보다 먼저 실행
    
    def can_handle(self, event_type: str) -> bool:
        """이벤트 처리 가능 여부 확인"""
        return event_type in self.SWARM_EVENT_TYPES
    
    def handle(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """이벤트 처리
        
        Args:
            event: 처리할 이벤트
            
        Returns:
            처리 결과 또는 None
        """
        event_type = event.get("type", "")
        
        # 에이전트 상태 이벤트 처리
        if event_type in (StreamlitEventType.AGENT_STATUS.value, StreamlitEventType.AGENT_HANDOFF.value):
            return {
                "swarm_event_processed": True,
                "event_type": event_type,
                "agent": event.get("agent"),
                "status": event.get("status"),
            }
        
        # 멀티에이전트 이벤트 처리
        if event_type.startswith("multiagent_"):
            return {
                "swarm_event_processed": True,
                "event_type": event_type,
                "node_id": event.get("node_id"),
            }
        
        return None


class StreamlitSwarmUIHandler(EventHandler):
    """Swarm 이벤트를 Streamlit UI로 렌더링하는 핸들러
    
    Requirements:
    - 1.5: 작업 진행 상황 확인 시 현재 어떤 에이전트가 작업 중인지 상태 표시
    - 5.3: 기존 이벤트 시스템과 호환되는 콜백 제공
    """
    
    def __init__(self, adapter: SwarmEventAdapter, ui_state=None):
        """핸들러 초기화
        
        Args:
            adapter: SwarmEventAdapter 인스턴스
            ui_state: StreamlitUIState 인스턴스 (선택적)
        """
        self.adapter = adapter
        self.ui_state = ui_state
        self._status_placeholder = None
    
    @property
    def priority(self) -> int:
        """핸들러 우선순위"""
        return 8  # 일반 UI 핸들러(10)보다 약간 먼저 실행
    
    def set_status_placeholder(self, placeholder) -> None:
        """상태 표시용 placeholder 설정"""
        self._status_placeholder = placeholder
    
    def can_handle(self, event_type: str) -> bool:
        """이벤트 처리 가능 여부 확인"""
        return event_type in (
            StreamlitEventType.AGENT_STATUS.value,
            StreamlitEventType.AGENT_HANDOFF.value,
        )
    
    def handle(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """이벤트 처리 및 UI 업데이트
        
        Args:
            event: 처리할 이벤트
            
        Returns:
            처리 결과 또는 None
        """
        event_type = event.get("type", "")
        
        if event_type == StreamlitEventType.AGENT_STATUS.value:
            self._update_agent_status_ui(event)
        elif event_type == StreamlitEventType.AGENT_HANDOFF.value:
            self._update_handoff_ui(event)
        
        return {"ui_updated": True, "event_type": event_type}
    
    def _update_agent_status_ui(self, event: Dict[str, Any]) -> None:
        """에이전트 상태 UI 업데이트 (Requirements 1.5)"""
        if not self._status_placeholder:
            return
        
        agent = event.get("agent_display_name", event.get("agent", "Unknown"))
        status = event.get("status", "working")
        message = event.get("message", "")
        
        # 상태에 따른 아이콘
        status_icons = {
            "working": "🔄",
            "completed": "✅",
            "error": "❌",
            "idle": "⏸️",
        }
        icon = status_icons.get(status, "🔄")
        
        try:
            self._status_placeholder.markdown(f"{icon} **{agent}**: {message}")
        except Exception:
            pass  # Streamlit placeholder 오류 무시
    
    def _update_handoff_ui(self, event: Dict[str, Any]) -> None:
        """에이전트 전환 UI 업데이트 (Requirements 1.5)"""
        if not self._status_placeholder:
            return
        
        to_agent = event.get("agent_display_name", event.get("to_agent", "Unknown"))
        message = event.get("message", f"{to_agent}로 작업을 전달합니다...")
        
        try:
            self._status_placeholder.markdown(f"🔀 **{to_agent}**: {message}")
        except Exception:
            pass  # Streamlit placeholder 오류 무시
    
    def render_progress(self) -> None:
        """전체 진행 상황 렌더링 (Requirements 1.5)"""
        if not self._status_placeholder:
            return
        
        progress = self.adapter.get_agent_progress()
        if not progress:
            return
        
        lines = []
        for item in progress:
            status = item.get("status", "idle")
            is_current = item.get("is_current", False)
            display_name = item.get("display_name", item.get("agent", "Unknown"))
            
            # 상태 아이콘
            if is_current and status == "working":
                icon = "🔄"
            elif status == "completed":
                icon = "✅"
            elif status == "error":
                icon = "❌"
            else:
                icon = "⏸️"
            
            # 현재 에이전트 강조
            if is_current:
                lines.append(f"{icon} **{display_name}** ← 현재")
            else:
                lines.append(f"{icon} {display_name}")
        
        try:
            self._status_placeholder.markdown("\n".join(lines))
        except Exception:
            pass
