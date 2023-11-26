"""Streamlit utils."""
from agent_utils import (
    load_agent_ids_from_directory,
    load_cache_from_directory,
)
from constants import (
    AGENT_CACHE_DIR,
)
from typing import Optional

import streamlit as st


def update_selected_agent_with_id(selected_id: Optional[str] = None) -> None:
    """Update selected agent with id."""
    # set session state
    st.session_state.selected_id = (
        selected_id if selected_id != "Create a new agent" else None
    )
    if st.session_state.selected_id is None:
        st.session_state.selected_cache = None
    else:
        # load agent from directory
        agent_cache = load_cache_from_directory(
            str(AGENT_CACHE_DIR), st.session_state.selected_id
        )
        st.session_state.selected_cache = agent_cache


## handler for sidebar specifically
def update_selected_agent() -> None:
    """Update selected agent."""
    selected_id = st.session_state.agent_selector

    update_selected_agent_with_id(selected_id)


def add_sidebar() -> None:
    """Add sidebar."""
    with st.sidebar:
        st.session_state.cur_agent_ids = load_agent_ids_from_directory(
            str(AGENT_CACHE_DIR)
        )
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
