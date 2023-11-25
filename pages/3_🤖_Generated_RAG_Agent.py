"""Streamlit page showing builder config."""
import streamlit as st
from typing import cast, Optional
from agent_utils import RAGAgentBuilder, ParamCache
from st_utils import add_sidebar


####################
#### STREAMLIT #####
####################


st.set_page_config(
    page_title="Generated RAG Agent",
    page_icon="🦙",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)
st.title("Generated RAG Agent")

add_sidebar()

if (
    "agent_messages" not in st.session_state.keys()
):  # Initialize the chat messages history
    st.session_state.agent_messages = [
        {"role": "assistant", "content": "Ask me a question!"}
    ]


def add_to_message_history(role: str, content: str) -> None:
    message = {"role": role, "content": str(content)}
    st.session_state.agent_messages.append(message)  # Add response to message history


# first, pick the cache: this is preloaded from an existing agent,
# or is part of the current one being created
agent = None
if (
    "selected_cache" in st.session_state.keys()
    and st.session_state.selected_cache is not None
):
    cache: Optional[ParamCache] = cast(ParamCache, st.session_state.selected_cache)
elif "agent_builder" in st.session_state.keys():
    agent_builder = cast(RAGAgentBuilder, st.session_state.agent_builder)
    cache = agent_builder.cache
else:
    cache = None
    st.info("Agent not created. Please create an agent in the above section.")

# if agent is created, then we can chat with it
if cache is not None and cache.agent is not None:
    st.info(f"Viewing config for agent: {cache.agent_id}", icon="ℹ️")
    agent = cache.agent
    for message in st.session_state.agent_messages:  # Display the prior chat messages
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # don't process selected for now
    if prompt := st.chat_input(
        "Your question"
    ):  # Prompt for user input and save to chat history
        add_to_message_history("user", prompt)
        with st.chat_message("user"):
            st.write(prompt)

    # If last message is not from assistant, generate a new response
    if st.session_state.agent_messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = agent.chat(str(prompt))
                st.write(str(response))
                add_to_message_history("assistant", str(response))
else:
    st.info("Agent not created. Please create an agent in the above section.")
