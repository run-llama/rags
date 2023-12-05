"""Agent builder."""

from llama_index.llms import ChatMessage
from llama_index.prompts import ChatPromptTemplate
from typing import List, cast, Optional
from core.builder_config import BUILDER_LLM
from typing import Dict, Any
import uuid
from core.constants import AGENT_CACHE_DIR
from abc import ABC, abstractmethod

from core.param_cache import ParamCache, RAGParams
from core.utils import (
    load_data,
    get_tool_objects,
    construct_agent,
)
from core.agent_builder.registry import AgentCacheRegistry


# System prompt tool
GEN_SYS_PROMPT_STR = """\
Task information is given below. 

Given the task, please generate a system prompt for an OpenAI-powered bot \
to solve this task: 
{task} \

Make sure the system prompt obeys the following requirements:
- Tells the bot to ALWAYS use tools given to solve the task. \
NEVER give an answer without using a tool.
- Does not reference a specific data source. \
The data source is implicit in any queries to the bot, \
and telling the bot to analyze a specific data source might confuse it given a \
user query.

"""

gen_sys_prompt_messages = [
    ChatMessage(
        role="system",
        content="You are helping to build a system prompt for another bot.",
    ),
    ChatMessage(role="user", content=GEN_SYS_PROMPT_STR),
]

GEN_SYS_PROMPT_TMPL = ChatPromptTemplate(gen_sys_prompt_messages)


class BaseRAGAgentBuilder(ABC):
    """Base RAG Agent builder class."""

    @property
    @abstractmethod
    def cache(self) -> ParamCache:
        """Cache."""

    @property
    @abstractmethod
    def agent_registry(self) -> AgentCacheRegistry:
        """Agent registry."""


class RAGAgentBuilder(BaseRAGAgentBuilder):
    """RAG Agent builder.

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
        urls: Optional[List[str]] = None,
    ) -> str:
        """Load data for a given task.

        Only ONE of file_names or directory or urls should be specified.

        Args:
            file_names (Optional[List[str]]): List of file names to load.
                Defaults to None.
            directory (Optional[str]): Directory to load files from.
            urls (Optional[List[str]]): List of urls to load.
                Defaults to None.

        """
        file_names = file_names or []
        urls = urls or []
        directory = directory or ""
        docs = load_data(file_names=file_names, directory=directory, urls=urls)
        self._cache.docs = docs
        self._cache.file_names = file_names
        self._cache.urls = urls
        self._cache.directory = directory
        return "Data loaded successfully."

    def add_web_tool(self) -> str:
        """Add a web tool to enable agent to solve a task."""
        # TODO: make this not hardcoded to a web tool
        # Set up Metaphor tool
        if "web_search" in self._cache.tools:
            return "Web tool already added."
        else:
            self._cache.tools.append("web_search")
        return "Web tool added successfully."

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
        additional_tools = get_tool_objects(self.cache.tools)
        agent, extra_info = construct_agent(
            cast(str, self._cache.system_prompt),
            cast(RAGParams, self._cache.rag_params),
            self._cache.docs,
            additional_tools=additional_tools,
        )

        # if agent_id not specified, randomly generate one
        agent_id = agent_id or self._cache.agent_id or f"Agent_{str(uuid.uuid4())}"
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
