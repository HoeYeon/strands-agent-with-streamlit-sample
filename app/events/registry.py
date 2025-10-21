"""Event handler architecture for streaming callbacks.

This module provides the core event handling system including:
- EventType: Enumeration of supported event types
- EventHandler: Abstract base class for event handlers
- EventRegistry: Event routing and handler management
"""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional


class EventType(Enum):
    """Supported Strands Agent event types."""
    # Text Generation Events
    DATA = "data"
    DELTA = "delta"
    
    # Tool Events
    CURRENT_TOOL_USE = "current_tool_use"
    TOOL_RESULT = "tool_result"  # Custom event
    
    # Lifecycle Events
    INIT_EVENT_LOOP = "init_event_loop"
    START_EVENT_LOOP = "start_event_loop"
    START = "start"
    MESSAGE = "message"
    EVENT = "event"
    COMPLETE = "complete"
    FORCE_STOP = "force_stop"
    FORCE_STOP_REASON = "force_stop_reason"
    RESULT = "result"
    
    # Reasoning Events
    REASONING = "reasoning"
    REASONING_TEXT = "reasoningText"
    REASONING_SIGNATURE = "reasoning_signature"
    REDACTED_CONTENT = "redactedContent"


class EventHandler(ABC):
    """Interface for event handlers."""
    
    @abstractmethod
    def can_handle(self, event_type: str) -> bool:
        """Return True if the handler can process the event type."""
        pass
    
    @abstractmethod
    def handle(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle an event and optionally return structured data."""
        pass
    
    @property
    def priority(self) -> int:
        """Handler priority (lower is executed earlier)."""
        return 100


class EventRegistry:
    """Registry that routes events to the appropriate handlers."""
    
    def __init__(self):
        self._handlers: List[EventHandler] = []
    
    def register(self, handler: EventHandler) -> None:
        """Register a handler and keep order by priority."""
        self._handlers.append(handler)
        self._handlers.sort(key=lambda h: h.priority)
    
    def get_handlers(self, event_type: str) -> List[EventHandler]:
        """Return handlers that can process the given event type."""
        return [h for h in self._handlers if h.can_handle(event_type)]
    
    def process_event(self, event: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Dispatch an event and collect handler outputs."""
        results = []
        event_type = self._extract_event_type(event)
        
        for handler in self.get_handlers(event_type):
            try:
                result = handler.handle(event)
                if result:
                    results.append(result)
            except Exception as e:
                # Surface handler errors without raising exceptions again
                handler_name = type(handler).__name__
                error_result = {
                    "handler_error": {
                        "handler": handler_name,
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "event_type": event_type
                    }
                }
                results.append(error_result)

        return results
    
    def _extract_event_type(self, event: Dict[str, Any]) -> str:
        """Infer the event type from the payload."""
        # Priority: data > current_tool_use > reasoningText > fallback to first key
        priority_events = ["data", "current_tool_use", "tool_result", "reasoning", "reasoningText", "redactedContent", "result", "force_stop"]
        
        for priority_event in priority_events:
            if priority_event in event:
                return priority_event
        
        # Fall back to the first key if no known type is present
        for key in event.keys():
            return key
        
        return "unknown"
