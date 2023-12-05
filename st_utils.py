"""Streamlit utils."""
from core.agent_builder.loader import (
    load_meta_agent_and_tools,
    AgentCacheRegistry,
)
from core.agent_builder.base import BaseRAGAgentBuilder
from core.param_cache import ParamCache
from core.constants import (
    AGENT_CACHE_DIR,
)
from typing import Optional, cast
from pydantic import BaseModel

from llama_index.agent.types import BaseAgent
import streamlit as st


def update_selected_agent_with_id(selected_id: Optional[str] = None) -> None:
    """Update selected agent with id."""
    # set session state
    st.session_state.selected_id = (
        selected_id if selected_id != "Create a new agent" else None
    )

    # clear agent builder and builder agent
    st.session_state.builder_agent = None
    st.session_state.agent_builder = None

    # clear selected cache
    st.session_state.selected_cache = None


## handler for sidebar specifically
def update_selected_agent() -> None:
    """Update selected agent."""
    selected_id = st.session_state.agent_selector

    update_selected_agent_with_id(selected_id)


def get_cached_is_multimodal() -> bool:
    """Get default multimodal st."""
    if (
        "selected_cache" not in st.session_state.keys()
        or st.session_state.selected_cache is None
    ):
        default_val = False
    else:
        selected_cache = cast(ParamCache, st.session_state.selected_cache)
        default_val = True if selected_cache.builder_type == "multimodal" else False
    return default_val


def get_is_multimodal() -> bool:
    """Get is multimodal."""
    if "is_multimodal_st" not in st.session_state.keys():
        st.session_state.is_multimodal_st = False
    return st.session_state.is_multimodal_st


def add_builder_config() -> None:
    """Add builder config."""
    with st.expander("Builder Config (Advanced)"):
        # add a few options - openai api key, and
        if (
            "selected_cache" not in st.session_state.keys()
            or st.session_state.selected_cache is None
        ):
            is_locked = False
        else:
            is_locked = True

        st.checkbox(
            "Enable multimodal search (beta)",
            key="is_multimodal_st",
            on_change=update_selected_agent,
            value=get_cached_is_multimodal(),
            disabled=is_locked,
        )


def add_sidebar() -> None:
    """Add sidebar."""
    with st.sidebar:
        agent_registry = cast(AgentCacheRegistry, st.session_state.agent_registry)
        st.session_state.cur_agent_ids = agent_registry.get_agent_ids()
        choices = ["Create a new agent"] + st.session_state.cur_agent_ids

        # by default, set index to 0. if value is in selected_id, set index to that
        index = 0
        if "selected_id" in st.session_state.keys():
            if st.session_state.selected_id is not None:
                index = choices.index(st.session_state.selected_id)
        # display buttons
        st.radio(
            "Agents",
            choices,
            index=index,
            on_change=update_selected_agent,
            key="agent_selector",
        )


class CurrentSessionState(BaseModel):
    """Current session state."""

    # arbitrary types
    class Config:
        arbitrary_types_allowed = True

    agent_registry: AgentCacheRegistry
    selected_id: Optional[str]
    selected_cache: Optional[ParamCache]
    agent_builder: BaseRAGAgentBuilder
    cache: ParamCache
    builder_agent: BaseAgent


def get_current_state() -> CurrentSessionState:
    """Get current state.

    This includes current state stored in session state and derived from it, e.g.
    - agent registry
    - selected agent
    - selected cache
    - agent builder
    - builder agent

    """
    # get agent registry
    agent_registry = AgentCacheRegistry(str(AGENT_CACHE_DIR))
    if "agent_registry" not in st.session_state.keys():
        st.session_state.agent_registry = agent_registry

    if "cur_agent_ids" not in st.session_state.keys():
        st.session_state.cur_agent_ids = agent_registry.get_agent_ids()

    if "selected_id" not in st.session_state.keys():
        st.session_state.selected_id = None

    # set selected cache if doesn't exist
    if (
        "selected_cache" not in st.session_state.keys()
        or st.session_state.selected_cache is None
    ):
        # update selected cache
        if st.session_state.selected_id is None:
            st.session_state.selected_cache = None
        else:
            # load agent from directory
            agent_registry = cast(AgentCacheRegistry, st.session_state.agent_registry)
            agent_cache = agent_registry.get_agent_cache(st.session_state.selected_id)
            st.session_state.selected_cache = agent_cache

    # set builder agent / agent builder
    if (
        "builder_agent" not in st.session_state.keys()
        or st.session_state.builder_agent is None
        or "agent_builder" not in st.session_state.keys()
        or st.session_state.agent_builder is None
    ):
        if (
            "selected_cache" in st.session_state.keys()
            and st.session_state.selected_cache is not None
        ):
            # create builder agent / tools from selected cache
            builder_agent, agent_builder = load_meta_agent_and_tools(
                cache=st.session_state.selected_cache,
                agent_registry=st.session_state.agent_registry,
                # NOTE: we will probably generalize this later into different
                # builder configs
                is_multimodal=get_cached_is_multimodal(),
            )
        else:
            # create builder agent / tools from new cache
            builder_agent, agent_builder = load_meta_agent_and_tools(
                agent_registry=st.session_state.agent_registry,
                is_multimodal=get_is_multimodal(),
            )

        st.session_state.builder_agent = builder_agent
        st.session_state.agent_builder = agent_builder

    return CurrentSessionState(
        agent_registry=st.session_state.agent_registry,
        selected_id=st.session_state.selected_id,
        selected_cache=st.session_state.selected_cache,
        agent_builder=st.session_state.agent_builder,
        cache=st.session_state.agent_builder.cache,
        builder_agent=st.session_state.builder_agent,
    )
