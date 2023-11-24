"""Streamlit page showing builder config."""
import streamlit as st
import openai
from streamlit_pills import pills
from typing import cast

from agent_utils import (
    RAGParams,
    RAGAgentBuilder,
    ParamCache,
    remove_agent_from_directory
)
from st_utils import update_selected_agent_with_id
from llama_index.agent.types import BaseAgent
from constants import (
    AGENT_CACHE_DIR
)
from st_utils import add_sidebar


####################
#### STREAMLIT #####
####################



st.set_page_config(page_title="RAG Pipeline Config", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
st.title("RAG Pipeline Config")
add_sidebar()

# first, pick the cache: this is preloaded from an existing agent, or is part of the current one being
# created
if "selected_cache" in st.session_state.keys() and st.session_state.selected_cache is not None:
    cache = cast(ParamCache, st.session_state.selected_cache)
    agent_builder = RAGAgentBuilder(cache)
elif "agent_builder" in st.session_state.keys():
    agent_builder = cast(RAGAgentBuilder, st.session_state.agent_builder)
else:
    agent_builder = None

if agent_builder is not None:

    st.info(
        f"Viewing config for agent: {agent_builder.cache.agent_id}", icon="ℹ️"
    )


    agent_id_st = st.text_input("Agent ID", value=agent_builder.cache.agent_id)

    if agent_builder.cache.system_prompt is None:
        system_prompt = ""
    else:
        system_prompt = agent_builder.cache.system_prompt
    sys_prompt_st = st.text_area("System Prompt", value=system_prompt)

    rag_params = cast(RAGParams, agent_builder.cache.rag_params)
    file_names = st.text_input(
        "File names (not editable)", 
        value=",".join(agent_builder.cache.file_names),
        disabled=True
    )
    urls = st.text_input(
        "URLs (not editable)",
        value=",".join(agent_builder.cache.urls),
        disabled=True
    )
    include_summarization_st = st.checkbox("Include Summarization (only works for GPT-4)", value=rag_params.include_summarization)
    top_k_st = st.number_input("Top K", value=rag_params.top_k)
    chunk_size_st = st.number_input("Chunk Size", value=rag_params.chunk_size)
    embed_model_st = st.text_input("Embed Model", value=rag_params.embed_model)
    llm_st = st.text_input("LLM", value=rag_params.llm)
    if agent_builder.cache.agent is not None:
        if st.button("Update Agent"):
            ### Update the agent

            # remove saved agent from directory, since we'll be re-saving
            remove_agent_from_directory(AGENT_CACHE_DIR, agent_builder.cache.agent_id)

            # set agent id
            agent_builder.cache.agent_id = agent_id_st

            # set system prompt
            agent_builder.cache.system_prompt = sys_prompt_st
            # get agent_builder
            # We call set_rag_params and create_agent, which will
            # update the cache
            # TODO: decouple functions from tool functions exposed to the agent
            agent_builder.set_rag_params(
                include_summarization=include_summarization_st,
                top_k=top_k_st,
                chunk_size=chunk_size_st,
                embed_model=embed_model_st,
                llm=llm_st,
            )
            # this will update the agent in the cache
            agent_builder.create_agent()

            # update selected agent to the new id
            update_selected_agent_with_id(agent_builder.cache.agent_id)

            # trigger refresh
            st.rerun()
    else:
        # show text saying "agent not created"
        st.info("Agent not created. Please create an agent in the above section.")

else:
    st.info("No agent builder found. Please create an agent in the above section.")