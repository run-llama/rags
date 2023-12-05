"""Streamlit page showing builder config."""
import streamlit as st

from core.param_cache import (
    RAGParams,
)
from core.agent_builder.loader import (
    RAGAgentBuilder,
    AgentCacheRegistry,
)
from st_utils import update_selected_agent_with_id, get_current_state, add_sidebar
from typing import cast


####################
#### STREAMLIT #####
####################


def update_agent() -> None:
    """Update agent."""
    if (
        "agent_builder" in st.session_state.keys()
        and st.session_state.agent_builder is not None
    ):
        additional_tools = st.session_state.additional_tools_st.strip().split(",")
        if additional_tools == [""]:
            additional_tools = []
        agent_builder = cast(RAGAgentBuilder, st.session_state.agent_builder)
        ### Update the agent
        agent_builder.update_agent(
            st.session_state.agent_id_st,
            system_prompt=st.session_state.sys_prompt_st,
            include_summarization=st.session_state.include_summarization_st,
            top_k=st.session_state.top_k_st,
            chunk_size=st.session_state.chunk_size_st,
            embed_model=st.session_state.embed_model_st,
            llm=st.session_state.llm_st,
            additional_tools=additional_tools,
        )

        # Update Radio Buttons: update selected agent to the new id
        update_selected_agent_with_id(agent_builder.cache.agent_id)
    else:
        raise ValueError("Agent builder is None. Cannot update agent.")


def delete_agent() -> None:
    """Delete agent."""
    if (
        "agent_builder" in st.session_state.keys()
        and st.session_state.agent_builder is not None
        and "agent_registry" in st.session_state.keys()
    ):
        agent_builder = cast(RAGAgentBuilder, st.session_state.agent_builder)
        agent_registry = cast(AgentCacheRegistry, st.session_state.agent_registry)
        ### Delete agent
        # remove saved agent from directory
        agent_registry.delete_agent_cache(agent_builder.cache.agent_id)
        # Update Radio Buttons: update selected agent to the new id
        update_selected_agent_with_id(None)
    else:
        raise ValueError("Agent builder is None. Cannot delete agent.")


st.set_page_config(
    page_title="RAG Pipeline Config",
    page_icon="🦙",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)
st.title("RAG Pipeline Config")

current_state = get_current_state()
add_sidebar()


if current_state.agent_builder is not None:

    st.info(f"Viewing config for agent: {current_state.cache.agent_id}", icon="ℹ️")

    agent_id_st = st.text_input(
        "Agent ID", value=current_state.cache.agent_id, key="agent_id_st"
    )

    if current_state.cache.system_prompt is None:
        system_prompt = ""
    else:
        system_prompt = current_state.cache.system_prompt
    sys_prompt_st = st.text_area(
        "System Prompt", value=system_prompt, key="sys_prompt_st"
    )

    rag_params = cast(RAGParams, current_state.cache.rag_params)

    with st.expander("Loaded Data (Expand to view)"):
        file_names = st.text_input(
            "File names (not editable)",
            value=",".join(current_state.cache.file_names),
            disabled=True,
        )
        directory = st.text_input(
            "Directory (not editable)",
            value=current_state.cache.directory,
            disabled=True,
        )
        urls = st.text_input(
            "URLs (not editable)",
            value=",".join(current_state.cache.urls),
            disabled=True,
        )

    include_summarization_st = st.checkbox(
        "Include Summarization (only works for GPT-4)",
        value=rag_params.include_summarization,
        key="include_summarization_st",
    )

    # add web tool
    additional_tools_st = st.text_input(
        "Additional tools (currently only supports 'web_search')",
        value=",".join(current_state.cache.tools),
        key="additional_tools_st",
    )

    top_k_st = st.number_input("Top K", value=rag_params.top_k, key="top_k_st")
    chunk_size_st = st.number_input(
        "Chunk Size", value=rag_params.chunk_size, key="chunk_size_st"
    )
    embed_model_st = st.text_input(
        "Embed Model", value=rag_params.embed_model, key="embed_model_st"
    )
    llm_st = st.text_input("LLM", value=rag_params.llm, key="llm_st")
    if current_state.cache.agent is not None:
        st.button("Update Agent", on_click=update_agent)
        st.button(":red[Delete Agent]", on_click=delete_agent)
    else:
        # show text saying "agent not created"
        st.info("Agent not created. Please create an agent in the above section.")

else:
    st.info("No agent builder found. Please create an agent in the above section.")
