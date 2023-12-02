"""Param cache."""

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
from core.builder_config import BUILDER_LLM
from typing import Dict, Tuple, Any, Callable
import streamlit as st
from pathlib import Path
import json
import uuid
from core.constants import AGENT_CACHE_DIR
import shutil
from core.utils import (
    load_data,
    get_tool_objects,
    construct_agent
    
)

from llama_index.callbacks import CallbackManager
from callback_manager import StreamlitFunctionsCallbackHandler


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
            "tools": self.tools,
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
        additional_tools = get_tool_objects(cache_dict["tools"])
        agent, _ = construct_agent(
            cache_dict["system_prompt"],
            cache_dict["rag_params"],
            cache_dict["docs"],
            vector_index=vector_index,
            additional_tools=additional_tools,
            # TODO: figure out tools
        )
        cache_dict["vector_index"] = vector_index
        cache_dict["agent"] = agent

        return cls(**cache_dict)
