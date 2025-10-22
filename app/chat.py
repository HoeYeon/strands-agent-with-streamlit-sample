"""Chat handling logic for Streamlit.

This module manages chat interactions, user input processing,
and streaming response handling with the agent backend.
"""

from typing import Dict, Any
import streamlit as st

from .session import SessionManager
from .ui import PlaceholderManager, ErrorHandler


class ChatHandler:
    """Handle chat interactions and streaming."""

    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager
        self.placeholder_manager = PlaceholderManager()
        self.error_handler = ErrorHandler()

    def handle_user_input(self, prompt: str) -> None:
        """Process user input and generate assistant response."""
        # Add user message to session
        self.session_manager.add_message("user", prompt)

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Handle assistant response
        with st.chat_message("assistant"):
            self._handle_assistant_response(prompt)

    def _handle_assistant_response(self, prompt: str) -> None:
        """Handle the assistant response generation and streaming."""
        # Create placeholders
        message_container = st.container()
        status_ph, tool_ph, chain_ph, response_ph = (
            self.placeholder_manager.create_chat_placeholders(message_container)
        )

        # Setup UI state
        agent = self.session_manager.agent
        ui_state = self.session_manager.get_agent_ui_state()
        if ui_state:
            ui_state.reset()
            ui_state.message_container = message_container

            # Setup handler placeholders
            self.placeholder_manager.setup_ui_handler_placeholders(
                agent, status_ph, tool_ph, chain_ph, response_ph
            )

        try:
            # Stream the response
            self._stream_response(prompt, agent, status_ph, chain_ph, response_ph)
        except Exception as error:
            # Handle streaming error
            assistant_message = self.error_handler.handle_streaming_error(
                error, status_ph, response_ph, chain_ph
            )
            self.session_manager.add_message("assistant", assistant_message)

    def _is_stream_complete(self, event: Dict[str, Any]) -> bool:
        """Check if event signals stream completion.
        
        Supports both standard format (type: complete/force_stop) and
        legacy format (result/force_stop fields).
        """
        # Standard format
        if event.get("type") in ("complete", "force_stop"):
            return True
        
        # Legacy format (backward compatibility)
        if event.get("result") or event.get("force_stop"):
            return True
        
        return False

    def _stream_response(self, prompt: str, agent, status_ph, chain_ph, response_ph) -> None:
        """Stream the agent response and handle events."""
        # Stream events and process them
        stream = agent.stream_response(prompt)
        for event in stream:
            try:
                # Process events on the main thread
                results = agent.event_registry.process_event(event)

                # Handle any handler errors
                self.error_handler.handle_handler_errors(results, status_ph)

            except Exception as handler_error:
                # Display handler error but continue streaming
                self.error_handler.display_handler_error(handler_error, status_ph)

            # Stop streaming once the agent reports completion
            if self._is_stream_complete(event):
                break

        # Finalize and persist the response
        self._finalize_response(agent)

    def _finalize_response(self, agent) -> None:
        """Finalize the assistant response and add to session."""
        for handler in agent.event_registry._handlers:
            if hasattr(handler, "finalize_response"):
                assistant_message = handler.finalize_response()
                self.session_manager.add_message("assistant", assistant_message)
                break