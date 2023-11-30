"""Streaming callback manager."""
from llama_index.callbacks.base_handler import BaseCallbackHandler
from llama_index.callbacks.schema import CBEventType

from typing import Optional, Dict, Any, List, Callable

STORAGE_DIR = "./storage"  # directory to cache the generated index
DATA_DIR = "./data"  # directory containing the documents to index


class StreamlitFunctionsCallbackHandler(BaseCallbackHandler):
    """Callback handler that outputs streamlit components given events."""

    def __init__(self, msg_handler: Callable[[str], Any]) -> None:
        """Initialize the base callback handler."""
        self.msg_handler = msg_handler
        super().__init__([], [])

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        """Run when an event starts and return id of event."""
        if event_type == CBEventType.FUNCTION_CALL:
            if payload is None:
                raise ValueError("Payload cannot be None")
            arguments_str = payload["function_call"]
            tool_str = payload["tool"].name
            print_str = f"Calling function: {tool_str} with args: {arguments_str}\n\n"
            self.msg_handler(print_str)
        else:
            pass
        return event_id

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        """Run when an event ends."""
        pass
        # TODO: currently we don't need to do anything here
        # if event_type == CBEventType.FUNCTION_CALL:
        #     response = payload["function_call_response"]
        #     # Add this to queue
        #     print_str = (
        #         f"\n\nGot output: {response}\n"
        #         "========================\n\n"
        #     )
        # elif event_type == CBEventType.AGENT_STEP:
        #     # put response into queue
        #     self._queue.put(payload["response"])

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        """Run when an overall trace is launched."""
        pass

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """Run when an overall trace is exited."""
        pass
