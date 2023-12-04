"""Multimodal agent builder."""

from llama_index.llms import ChatMessage
from typing import List, cast, Optional
from core.builder_config import BUILDER_LLM
from typing import Dict, Any
import uuid
from core.constants import AGENT_CACHE_DIR

from core.param_cache import ParamCache, RAGParams
from core.utils import (
    load_data,
    construct_mm_agent,
)
from core.agent_builder.registry import AgentCacheRegistry
from core.agent_builder.base import GEN_SYS_PROMPT_TMPL, BaseRAGAgentBuilder

from llama_index.chat_engine.types import BaseChatEngine

from llama_index.callbacks import trace_method
from llama_index.query_engine.multi_modal import SimpleMultiModalQueryEngine
from llama_index.chat_engine.types import (
    AGENT_CHAT_RESPONSE_TYPE,
    StreamingAgentChatResponse,
    AgentChatResponse,
)
from llama_index.llms.base import ChatResponse
from typing import Generator


class MultimodalChatEngine(BaseChatEngine):
    """Multimodal chat engine.

    This chat engine is a light wrapper around a query engine.
    Offers no real 'chat' functionality, is a beta feature.

    """

    def __init__(self, mm_query_engine: SimpleMultiModalQueryEngine) -> None:
        """Init params."""
        self._mm_query_engine = mm_query_engine

    def reset(self) -> None:
        """Reset conversation state."""
        pass

    @trace_method("chat")
    def chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> AGENT_CHAT_RESPONSE_TYPE:
        """Main chat interface."""
        # just return the top-k results
        response = self._mm_query_engine.query(message)
        return AgentChatResponse(response=str(response))

    @trace_method("chat")
    def stream_chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> StreamingAgentChatResponse:
        """Stream chat interface."""
        response = self._mm_query_engine.query(message)

        def _chat_stream(response: str) -> Generator[ChatResponse, None, None]:
            yield ChatResponse(message=ChatMessage(role="assistant", content=response))

        chat_stream = _chat_stream(str(response))
        return StreamingAgentChatResponse(chat_stream=chat_stream)

    @trace_method("chat")
    async def achat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> AGENT_CHAT_RESPONSE_TYPE:
        """Async version of main chat interface."""
        response = await self._mm_query_engine.aquery(message)
        return AgentChatResponse(response=str(response))

    @trace_method("chat")
    async def astream_chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> StreamingAgentChatResponse:
        """Async version of main chat interface."""
        return self.stream_chat(message, chat_history)


class MultimodalRAGAgentBuilder(BaseRAGAgentBuilder):
    """Multimodal RAG Agent builder.

    Contains a set of functions to construct a RAG agent, including:
    - setting system prompts
    - loading data
    - adding web search
    - setting parameters (e.g. top-k)

    Must pass in a cache. This cache will be modified as the agent is built.

    """

    def __init__(
        self,
        cache: Optional[ParamCache] = None,
        agent_registry: Optional[AgentCacheRegistry] = None,
    ) -> None:
        """Init params."""
        self._cache = cache or ParamCache()
        self._agent_registry = agent_registry or AgentCacheRegistry(
            str(AGENT_CACHE_DIR)
        )

    @property
    def cache(self) -> ParamCache:
        """Cache."""
        return self._cache

    @property
    def agent_registry(self) -> AgentCacheRegistry:
        """Agent registry."""
        return self._agent_registry

    def create_system_prompt(self, task: str) -> str:
        """Create system prompt for another agent given an input task."""
        llm = BUILDER_LLM
        fmt_messages = GEN_SYS_PROMPT_TMPL.format_messages(task=task)
        response = llm.chat(fmt_messages)
        self._cache.system_prompt = response.message.content

        return f"System prompt created: {response.message.content}"

    def load_data(
        self,
        file_names: Optional[List[str]] = None,
        directory: Optional[str] = None,
    ) -> str:
        """Load data for a given task.

        Only ONE of file_names or directory should be specified.
        **NOTE**: urls not supported in multi-modal setting.

        Args:
            file_names (Optional[List[str]]): List of file names to load.
                Defaults to None.
            directory (Optional[str]): Directory to load files from.

        """
        file_names = file_names or []
        directory = directory or ""
        docs = load_data(file_names=file_names, directory=directory)
        self._cache.docs = docs
        self._cache.file_names = file_names
        self._cache.directory = directory
        return "Data loaded successfully."

    def get_rag_params(self) -> Dict:
        """Get parameters used to configure the RAG pipeline.

        Should be called before `set_rag_params` so that the agent is aware of the
        schema.

        """
        rag_params = self._cache.rag_params
        return rag_params.dict()

    def set_rag_params(self, **rag_params: Dict) -> str:
        """Set RAG parameters.

        These parameters will then be used to actually initialize the agent.
        Should call `get_rag_params` first to get the schema of the input dictionary.

        Args:
            **rag_params (Dict): dictionary of RAG parameters.

        """
        new_dict = self._cache.rag_params.dict()
        new_dict.update(rag_params)
        rag_params_obj = RAGParams(**new_dict)
        self._cache.rag_params = rag_params_obj
        return "RAG parameters set successfully."

    def create_agent(self, agent_id: Optional[str] = None) -> str:
        """Create an agent.

        There are no parameters for this function because all the
        functions should have already been called to set up the agent.

        """
        if self._cache.system_prompt is None:
            raise ValueError("Must set system prompt before creating agent.")

        # construct additional tools
        agent, extra_info = construct_mm_agent(
            cast(str, self._cache.system_prompt),
            cast(RAGParams, self._cache.rag_params),
            self._cache.docs,
        )

        # if agent_id not specified, randomly generate one
        agent_id = agent_id or self._cache.agent_id or f"Agent_{str(uuid.uuid4())}"
        self._cache.builder_type = "multimodal"
        self._cache.vector_index = extra_info["vector_index"]
        self._cache.agent_id = agent_id
        self._cache.agent = agent

        # save the cache to disk
        self._agent_registry.add_new_agent_cache(agent_id, self._cache)
        return "Agent created successfully."

    def update_agent(
        self,
        agent_id: str,
        system_prompt: Optional[str] = None,
        include_summarization: Optional[bool] = None,
        top_k: Optional[int] = None,
        chunk_size: Optional[int] = None,
        embed_model: Optional[str] = None,
        llm: Optional[str] = None,
        additional_tools: Optional[List] = None,
    ) -> None:
        """Update agent.

        Delete old agent by ID and create a new one.
        Optionally update the system prompt and RAG parameters.

        NOTE: Currently is manually called, not meant for agent use.

        """
        self._agent_registry.delete_agent_cache(self.cache.agent_id)

        # set agent id
        self.cache.agent_id = agent_id

        # set system prompt
        if system_prompt is not None:
            self.cache.system_prompt = system_prompt
        # get agent_builder
        # We call set_rag_params and create_agent, which will
        # update the cache
        # TODO: decouple functions from tool functions exposed to the agent
        rag_params_dict: Dict[str, Any] = {}
        if include_summarization is not None:
            rag_params_dict["include_summarization"] = include_summarization
        if top_k is not None:
            rag_params_dict["top_k"] = top_k
        if chunk_size is not None:
            rag_params_dict["chunk_size"] = chunk_size
        if embed_model is not None:
            rag_params_dict["embed_model"] = embed_model
        if llm is not None:
            rag_params_dict["llm"] = llm

        self.set_rag_params(**rag_params_dict)

        # update tools
        if additional_tools is not None:
            self.cache.tools = additional_tools

        # this will update the agent in the cache
        self.create_agent()
