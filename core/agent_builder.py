"""Agent builder."""

from llama_index.llms import ChatMessage
from llama_index.prompts import ChatPromptTemplate
from typing import List, cast, Optional
from llama_index.tools import FunctionTool
from llama_index.agent.types import BaseAgent
from core.builder_config import BUILDER_LLM
from typing import Dict, Tuple, Any, Callable, Union
import streamlit as st
from pathlib import Path
import json
import uuid
from core.constants import AGENT_CACHE_DIR
import shutil

from core.param_cache import ParamCache, RAGParams
from core.utils import (
    load_data,
    get_tool_objects,
    construct_agent,
    load_meta_agent,
)


class AgentCacheRegistry:
    """Registry for agent caches, in disk.

    Can register new agent caches, load agent caches, delete agent caches, etc.

    """

    def __init__(self, dir: Union[str, Path]) -> None:
        """Init params."""
        self._dir = dir

    def _add_agent_id_to_directory(self, agent_id: str) -> None:
        """Save agent id to directory."""
        full_path = Path(self._dir) / "agent_ids.json"
        if not full_path.exists():
            with open(full_path, "w") as f:
                json.dump({"agent_ids": [agent_id]}, f)
        else:
            with open(full_path, "r") as f:
                agent_ids = json.load(f)["agent_ids"]
            if agent_id in agent_ids:
                raise ValueError(f"Agent id {agent_id} already exists.")
            agent_ids_set = set(agent_ids)
            agent_ids_set.add(agent_id)
            with open(full_path, "w") as f:
                json.dump({"agent_ids": list(agent_ids_set)}, f)

    def add_new_agent_cache(self, agent_id: str, cache: ParamCache) -> None:
        """Register agent."""
        # save the cache to disk
        agent_cache_path = f"{self._dir}/{agent_id}"
        cache.save_to_disk(agent_cache_path)
        # save to agent ids
        self._add_agent_id_to_directory(agent_id)

    def get_agent_ids(self) -> List[str]:
        """Get agent ids."""
        full_path = Path(self._dir) / "agent_ids.json"
        if not full_path.exists():
            return []
        with open(full_path, "r") as f:
            agent_ids = json.load(f)["agent_ids"]

        return agent_ids

    def get_agent_cache(self, agent_id: str) -> ParamCache:
        """Get agent cache."""
        full_path = Path(self._dir) / f"{agent_id}"
        if not full_path.exists():
            raise ValueError(f"Cache for agent {agent_id} does not exist.")
        cache = ParamCache.load_from_disk(str(full_path))
        return cache

    def delete_agent_cache(self, agent_id: str) -> None:
        """Delete agent cache."""
        # modify / resave agent_ids
        agent_ids = self.get_agent_ids()
        new_agent_ids = [id for id in agent_ids if id != agent_id]
        full_path = Path(self._dir) / "agent_ids.json"
        with open(full_path, "w") as f:
            json.dump({"agent_ids": new_agent_ids}, f)

        # remove agent cache
        full_path = Path(self._dir) / f"{agent_id}"
        if full_path.exists():
            # recursive delete
            shutil.rmtree(full_path)


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


class RAGAgentBuilder:
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
        self, file_names: Optional[List[str]] = None, urls: Optional[List[str]] = None
    ) -> str:
        """Load data for a given task.

        Only ONE of file_names or urls should be specified.

        Args:
            file_names (Optional[List[str]]): List of file names to load.
                Defaults to None.
            urls (Optional[List[str]]): List of urls to load.
                Defaults to None.

        """
        file_names = file_names or []
        urls = urls or []
        docs = load_data(file_names=file_names, urls=urls)
        self._cache.docs = docs
        self._cache.file_names = file_names
        self._cache.urls = urls
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


####################
#### META Agent ####
####################

RAG_BUILDER_SYS_STR = """\
You are helping to construct an agent given a user-specified task. 
You should generally use the tools in this rough order to build the agent.

1) Create system prompt tool: to create the system prompt for the agent.
2) Load in user-specified data (based on file paths they specify).
3) Decide whether or not to add additional tools.
4) Set parameters for the RAG pipeline.
5) Build the agent

This will be a back and forth conversation with the user. You should
continue asking users if there's anything else they want to do until
they say they're done. To help guide them on the process, 
you can give suggestions on parameters they can set based on the tools they
have available (e.g. "Do you want to set the number of documents to retrieve?")

"""


### DEFINE Agent ####
# NOTE: here we define a function that is dependent on the LLM,
# please make sure to update the LLM above if you change the function below


def _get_builder_agent_tools(agent_builder: RAGAgentBuilder) -> List[FunctionTool]:
    """Get list of builder agent tools to pass to the builder agent."""
    # see if metaphor api key is set, otherwise don't add web tool
    # TODO: refactor this later

    if "metaphor_key" in st.secrets:
        fns: List[Callable] = [
            agent_builder.create_system_prompt,
            agent_builder.load_data,
            agent_builder.add_web_tool,
            agent_builder.get_rag_params,
            agent_builder.set_rag_params,
            agent_builder.create_agent,
        ]
    else:
        fns = [
            agent_builder.create_system_prompt,
            agent_builder.load_data,
            agent_builder.get_rag_params,
            agent_builder.set_rag_params,
            agent_builder.create_agent,
        ]

    fn_tools: List[FunctionTool] = [FunctionTool.from_defaults(fn=fn) for fn in fns]
    return fn_tools


# define agent
# @st.cache_resource
def load_meta_agent_and_tools(
    cache: Optional[ParamCache] = None,
    agent_registry: Optional[AgentCacheRegistry] = None,
) -> Tuple[BaseAgent, RAGAgentBuilder]:

    # think of this as tools for the agent to use
    agent_builder = RAGAgentBuilder(cache, agent_registry=agent_registry)

    fn_tools = _get_builder_agent_tools(agent_builder)

    builder_agent = load_meta_agent(
        fn_tools, llm=BUILDER_LLM, system_prompt=RAG_BUILDER_SYS_STR, verbose=True
    )

    return builder_agent, agent_builder
