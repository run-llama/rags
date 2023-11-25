from llama_index.llms import OpenAI, ChatMessage, Anthropic, Replicate
from llama_index.llms.base import LLM
from llama_index.llms.utils import resolve_llm
from pydantic import BaseModel, Field
import os
from llama_index.agent import OpenAIAgent, ReActAgent
from llama_index.agent.react.prompts import REACT_CHAT_SYSTEM_HEADER
from llama_index import (
    VectorStoreIndex,
    SummaryIndex,
    ServiceContext,
    StorageContext,
    Document,
    load_index_from_storage,
)
from llama_index.prompts import ChatPromptTemplate
from typing import List, cast, Optional
from llama_index import SimpleDirectoryReader
from llama_index.embeddings.utils import resolve_embed_model
from llama_index.tools import QueryEngineTool, ToolMetadata, FunctionTool
from llama_index.agent.types import BaseAgent
from llama_index.chat_engine.types import BaseChatEngine
from llama_index.agent.react.formatter import ReActChatFormatter
from llama_index.llms.openai_utils import is_function_calling_model
from llama_index.chat_engine import CondensePlusContextChatEngine
from builder_config import BUILDER_LLM
from typing import Dict, Tuple, Any, Callable
import streamlit as st
from pathlib import Path
import json
import uuid
from constants import AGENT_CACHE_DIR
import shutil


def _resolve_llm(llm_str: str) -> LLM:
    """Resolve LLM."""
    # TODO: make this less hardcoded with if-else statements
    # see if there's a prefix
    # - if there isn't, assume it's an OpenAI model
    # - if there is, resolve it
    tokens = llm_str.split(":")
    if len(tokens) == 1:
        os.environ["OPENAI_API_KEY"] = st.secrets.openai_key
        llm: LLM = OpenAI(model=llm_str)
    elif tokens[0] == "local":
        llm = resolve_llm(llm_str)
    elif tokens[0] == "openai":
        os.environ["OPENAI_API_KEY"] = st.secrets.openai_key
        llm = OpenAI(model=tokens[1])
    elif tokens[0] == "anthropic":
        os.environ["ANTHROPIC_API_KEY"] = st.secrets.anthropic_key
        llm = Anthropic(model=tokens[1])
    elif tokens[0] == "replicate":
        os.environ["REPLICATE_API_KEY"] = st.secrets.replicate_key
        llm = Replicate(model=tokens[1])
    else:
        raise ValueError(f"LLM {llm_str} not recognized.")
    return llm


####################
#### META TOOLS ####
####################


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


class RAGParams(BaseModel):
    """RAG parameters.

    Parameters used to configure a RAG pipeline.

    """

    include_summarization: bool = Field(
        default=False,
        description=(
            "Whether to include summarization in the RAG pipeline. (only for GPT-4)"
        ),
    )
    top_k: int = Field(
        default=2, description="Number of documents to retrieve from vector store."
    )
    chunk_size: int = Field(default=1024, description="Chunk size for vector store.")
    embed_model: str = Field(
        default="default", description="Embedding model to use (default is OpenAI)"
    )
    llm: str = Field(
        default="gpt-4-1106-preview", description="LLM to use for summarization."
    )


def load_data(
    file_names: Optional[List[str]] = None, urls: Optional[List[str]] = None
) -> List[Document]:
    """Load data."""
    file_names = file_names or []
    urls = urls or []
    if not file_names and not urls:
        raise ValueError("Must specify either file_names or urls.")
    elif file_names and urls:
        raise ValueError("Must specify only one of file_names or urls.")
    elif file_names:
        reader = SimpleDirectoryReader(input_files=file_names)
        docs = reader.load_data()
    elif urls:
        from llama_hub.web.simple_web.base import SimpleWebPageReader

        # use simple web page reader from llamahub
        loader = SimpleWebPageReader()
        docs = loader.load_data(urls=urls)
    else:
        raise ValueError("Must specify either file_names or urls.")

    return docs


def load_agent(
    tools: List,
    llm: LLM,
    system_prompt: str,
    extra_kwargs: Optional[Dict] = None,
    **kwargs: Any,
) -> BaseChatEngine:
    """Load agent."""
    extra_kwargs = extra_kwargs or {}
    if isinstance(llm, OpenAI) and is_function_calling_model(llm.model):
        # get OpenAI Agent
        agent: BaseChatEngine = OpenAIAgent.from_tools(
            tools=tools, llm=llm, system_prompt=system_prompt, **kwargs
        )
    else:
        if "vector_index" not in extra_kwargs:
            raise ValueError(
                "Must pass in vector index for CondensePlusContextChatEngine."
            )
        vector_index = cast(VectorStoreIndex, extra_kwargs["vector_index"])
        rag_params = cast(RAGParams, extra_kwargs["rag_params"])
        # use condense + context chat engine
        agent = CondensePlusContextChatEngine.from_defaults(
            vector_index.as_retriever(similarity_top_k=rag_params.top_k),
        )

    return agent


def load_meta_agent(
    tools: List,
    llm: LLM,
    system_prompt: str,
    extra_kwargs: Optional[Dict] = None,
    **kwargs: Any,
) -> BaseAgent:
    """Load meta agent.

    TODO: consolidate with load_agent.

    The meta-agent *has* to perform tool-use.

    """
    extra_kwargs = extra_kwargs or {}
    if isinstance(llm, OpenAI) and is_function_calling_model(llm.model):
        # get OpenAI Agent
        agent: BaseAgent = OpenAIAgent.from_tools(
            tools=tools, llm=llm, system_prompt=system_prompt, **kwargs
        )
    else:
        agent = ReActAgent.from_tools(
            tools=tools,
            llm=llm,
            react_chat_formatter=ReActChatFormatter(
                system_header=system_prompt + "\n" + REACT_CHAT_SYSTEM_HEADER,
            ),
            **kwargs,
        )

    return agent


def construct_agent(
    system_prompt: str,
    rag_params: RAGParams,
    docs: List[Document],
    vector_index: Optional[VectorStoreIndex] = None,
    additional_tools: Optional[List] = None,
) -> Tuple[BaseChatEngine, Dict]:
    """Construct agent from docs / parameters / indices."""
    extra_info = {}
    additional_tools = additional_tools or []

    # first resolve llm and embedding model
    embed_model = resolve_embed_model(rag_params.embed_model)
    # llm = resolve_llm(rag_params.llm)
    # TODO: use OpenAI for now
    # llm = OpenAI(model=rag_params.llm)
    llm = _resolve_llm(rag_params.llm)

    # first let's index the data with the right parameters
    service_context = ServiceContext.from_defaults(
        chunk_size=rag_params.chunk_size,
        llm=llm,
        embed_model=embed_model,
    )

    if vector_index is None:
        vector_index = VectorStoreIndex.from_documents(
            docs, service_context=service_context
        )
    else:
        pass

    extra_info["vector_index"] = vector_index

    vector_query_engine = vector_index.as_query_engine(
        similarity_top_k=rag_params.top_k
    )
    all_tools = []
    vector_tool = QueryEngineTool(
        query_engine=vector_query_engine,
        metadata=ToolMetadata(
            name="vector_tool",
            description=("Use this tool to answer any user question over any data."),
        ),
    )
    all_tools.append(vector_tool)
    if rag_params.include_summarization:
        summary_index = SummaryIndex.from_documents(
            docs, service_context=service_context
        )
        summary_query_engine = summary_index.as_query_engine()
        summary_tool = QueryEngineTool(
            query_engine=summary_query_engine,
            metadata=ToolMetadata(
                name="summary_tool",
                description=(
                    "Use this tool for any user questions that ask "
                    "for a summarization of content"
                ),
            ),
        )
        all_tools.append(summary_tool)

    # then we add tools
    all_tools.extend(additional_tools)

    # build agent
    if system_prompt is None:
        return "System prompt not set yet. Please set system prompt first."

    agent = load_agent(
        all_tools,
        llm=llm,
        system_prompt=system_prompt,
        verbose=True,
        extra_kwargs={"vector_index": vector_index, "rag_params": rag_params},
    )
    return agent, extra_info


class ParamCache(BaseModel):
    """Cache for RAG agent builder.

    Created a wrapper class around a dict in case we wanted to more explicitly
    type different items in the cache.

    """

    # arbitrary types
    class Config:
        arbitrary_types_allowed = True

    # system prompt
    system_prompt: Optional[str] = Field(
        default=None, description="System prompt for RAG agent."
    )
    # data
    file_names: List[str] = Field(
        default_factory=list, description="File names as data source (if specified)"
    )
    urls: List[str] = Field(
        default_factory=list, description="URLs as data source (if specified)"
    )
    docs: List = Field(default_factory=list, description="Documents for RAG agent.")
    # tools
    tools: List = Field(
        default_factory=list, description="Additional tools for RAG agent (e.g. web)"
    )
    # RAG params
    rag_params: RAGParams = Field(
        default_factory=RAGParams, description="RAG parameters for RAG agent."
    )

    # agent params
    vector_index: Optional[VectorStoreIndex] = Field(
        default=None, description="Vector index for RAG agent."
    )
    agent_id: str = Field(
        default_factory=lambda: f"Agent_{str(uuid.uuid4())}",
        description="Agent ID for RAG agent.",
    )
    agent: Optional[BaseChatEngine] = Field(default=None, description="RAG agent.")

    def save_to_disk(self, save_dir: str) -> None:
        """Save cache to disk."""
        # NOTE: more complex than just calling dict() because we want to
        # only store serializable fields and be space-efficient

        dict_to_serialize = {
            "system_prompt": self.system_prompt,
            "file_names": self.file_names,
            "urls": self.urls,
            # TODO: figure out tools
            # "tools": [],
            "rag_params": self.rag_params.dict(),
            "agent_id": self.agent_id,
        }
        # store the vector store within the agent
        if self.vector_index is None:
            raise ValueError("Must specify vector index in order to save.")
        self.vector_index.storage_context.persist(Path(save_dir) / "storage")

        # if save_path directories don't exist, create it
        if not Path(save_dir).exists():
            Path(save_dir).mkdir(parents=True)
        with open(Path(save_dir) / "cache.json", "w") as f:
            json.dump(dict_to_serialize, f)

    @classmethod
    def load_from_disk(
        cls,
        save_dir: str,
    ) -> "ParamCache":
        """Load cache from disk."""
        storage_context = StorageContext.from_defaults(
            persist_dir=str(Path(save_dir) / "storage")
        )
        vector_index = cast(VectorStoreIndex, load_index_from_storage(storage_context))

        with open(Path(save_dir) / "cache.json", "r") as f:
            cache_dict = json.load(f)

        # replace rag params with RAGParams object
        cache_dict["rag_params"] = RAGParams(**cache_dict["rag_params"])

        # add in the missing fields
        # load docs
        cache_dict["docs"] = load_data(
            file_names=cache_dict["file_names"], urls=cache_dict["urls"]
        )
        # load agent from index
        agent, _ = construct_agent(
            cache_dict["system_prompt"],
            cache_dict["rag_params"],
            cache_dict["docs"],
            vector_index=vector_index,
            # TODO: figure out tools
        )
        cache_dict["vector_index"] = vector_index
        cache_dict["agent"] = agent

        return cls(**cache_dict)


def add_agent_id_to_directory(dir: str, agent_id: str) -> None:
    """Save agent id to directory."""
    full_path = Path(dir) / "agent_ids.json"
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


def load_agent_ids_from_directory(dir: str) -> List[str]:
    """Load agent ids file."""
    full_path = Path(dir) / "agent_ids.json"
    if not full_path.exists():
        return []
    with open(full_path, "r") as f:
        agent_ids = json.load(f)["agent_ids"]

    return agent_ids


def load_cache_from_directory(
    dir: str,
    agent_id: str,
) -> ParamCache:
    """Load cache from directory."""
    full_path = Path(dir) / f"{agent_id}"
    if not full_path.exists():
        raise ValueError(f"Cache for agent {agent_id} does not exist.")
    cache = ParamCache.load_from_disk(str(full_path))
    return cache


def remove_agent_from_directory(
    dir: str,
    agent_id: str,
) -> None:
    """Remove agent from directory."""

    # modify / resave agent_ids
    agent_ids = load_agent_ids_from_directory(dir)
    new_agent_ids = [id for id in agent_ids if id != agent_id]
    full_path = Path(dir) / "agent_ids.json"
    with open(full_path, "w") as f:
        json.dump({"agent_ids": new_agent_ids}, f)

    # remove agent cache
    full_path = Path(dir) / f"{agent_id}"
    if full_path.exists():
        # recursive delete
        shutil.rmtree(full_path)


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
        self, cache: Optional[ParamCache] = None, cache_dir: Optional[str] = None
    ) -> None:
        """Init params."""
        self._cache = cache or ParamCache()
        self._cache_dir = cache_dir or AGENT_CACHE_DIR

    @property
    def cache(self) -> ParamCache:
        """Cache."""
        return self._cache

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

    # NOTE: unused
    def add_web_tool(self) -> str:
        """Add a web tool to enable agent to solve a task."""
        # TODO: make this not hardcoded to a web tool
        # Set up Metaphor tool
        from llama_hub.tools.metaphor.base import MetaphorToolSpec

        # TODO: set metaphor API key
        metaphor_tool = MetaphorToolSpec(
            api_key=os.environ["METAPHOR_API_KEY"],
        )
        metaphor_tool_list = metaphor_tool.to_tool_list()

        self._cache.tools.extend(metaphor_tool_list)
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

        agent, extra_info = construct_agent(
            cast(str, self._cache.system_prompt),
            cast(RAGParams, self._cache.rag_params),
            self._cache.docs,
            additional_tools=self._cache.tools,
        )

        # if agent_id not specified, randomly generate one
        agent_id = agent_id or self._cache.agent_id or f"Agent_{str(uuid.uuid4())}"
        self._cache.vector_index = extra_info["vector_index"]
        self._cache.agent_id = agent_id
        self._cache.agent = agent

        # save the cache to disk
        agent_cache_path = f"{self._cache_dir}/{agent_id}"
        self._cache.save_to_disk(agent_cache_path)
        # save to agent ids
        add_agent_id_to_directory(str(self._cache_dir), agent_id)

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
    ) -> None:
        """Update agent.

        Delete old agent by ID and create a new one.
        Optionally update the system prompt and RAG parameters.

        NOTE: Currently is manually called, not meant for agent use.

        """
        # remove saved agent from directory, since we'll be re-saving
        remove_agent_from_directory(str(AGENT_CACHE_DIR), self.cache.agent_id)

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


# define agent
# @st.cache_resource
def load_meta_agent_and_tools(
    cache: Optional[ParamCache] = None,
) -> Tuple[BaseAgent, RAGAgentBuilder]:

    # think of this as tools for the agent to use
    agent_builder = RAGAgentBuilder(cache)

    fns: List[Callable] = [
        agent_builder.create_system_prompt,
        agent_builder.load_data,
        # add_web_tool,
        agent_builder.get_rag_params,
        agent_builder.set_rag_params,
        agent_builder.create_agent,
    ]
    fn_tools = [FunctionTool.from_defaults(fn=fn) for fn in fns]

    builder_agent = load_meta_agent(
        fn_tools, llm=BUILDER_LLM, system_prompt=RAG_BUILDER_SYS_STR, verbose=True
    )

    return builder_agent, agent_builder
