"""Param cache."""

from pydantic import BaseModel, Field
from llama_index import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from typing import List, cast, Optional
from llama_index.chat_engine.types import BaseChatEngine
from pathlib import Path
import json
import uuid
from core.utils import (
    load_data,
    get_tool_objects,
    construct_agent,
    RAGParams,
    construct_mm_agent,
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
    directory: Optional[str] = Field(
        default=None, description="Directory as data source (if specified)"
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
    builder_type: str = Field(
        default="default", description="Builder type (default, multimodal)."
    )
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
            "directory": self.directory,
            # TODO: figure out tools
            "tools": self.tools,
            "rag_params": self.rag_params.dict(),
            "builder_type": self.builder_type,
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
        with open(Path(save_dir) / "cache.json", "r") as f:
            cache_dict = json.load(f)

        storage_context = StorageContext.from_defaults(
            persist_dir=str(Path(save_dir) / "storage")
        )
        if cache_dict["builder_type"] == "multimodal":
            from llama_index.indices.multi_modal.base import MultiModalVectorStoreIndex

            vector_index: VectorStoreIndex = cast(
                MultiModalVectorStoreIndex, load_index_from_storage(storage_context)
            )
        else:
            vector_index = cast(
                VectorStoreIndex, load_index_from_storage(storage_context)
            )

        # replace rag params with RAGParams object
        cache_dict["rag_params"] = RAGParams(**cache_dict["rag_params"])

        # add in the missing fields
        # load docs
        cache_dict["docs"] = load_data(
            file_names=cache_dict["file_names"],
            urls=cache_dict["urls"],
            directory=cache_dict["directory"],
        )
        # load agent from index
        additional_tools = get_tool_objects(cache_dict["tools"])

        if cache_dict["builder_type"] == "multimodal":
            vector_index = cast(MultiModalVectorStoreIndex, vector_index)
            agent, _ = construct_mm_agent(
                cache_dict["system_prompt"],
                cache_dict["rag_params"],
                cache_dict["docs"],
                mm_vector_index=vector_index,
            )
        else:
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
