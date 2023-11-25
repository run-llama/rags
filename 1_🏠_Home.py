import streamlit as st
from streamlit_pills import pills

from agent_utils import (
    load_meta_agent_and_tools,
    load_agent_ids_from_directory,
)
from st_utils import add_sidebar
from constants import (
    AGENT_CACHE_DIR,
)


####################
#### STREAMLIT #####
####################


st.set_page_config(
    page_title="Build a RAGs bot, powered by LlamaIndex",
    page_icon="🦙",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)
st.title("Build a RAGs bot, powered by LlamaIndex 💬🦙")
st.info(
    "Use this page to build your RAG bot over your data! "
    "Once the agent is finished creating, check out the `RAG Config` and "
    "`Generated RAG Agent` pages.\n"
    "To build a new agent, please make sure that 'Create a new agent' is selected.",
    icon="ℹ️",
)


add_sidebar()


if (
    "selected_cache" in st.session_state.keys()
    and st.session_state.selected_cache is not None
):
    # create builder agent / tools from selected cache
    builder_agent, agent_builder = load_meta_agent_and_tools(
        cache=st.session_state.selected_cache
    )
else:
    # create builder agent / tools from new cache
    builder_agent, agent_builder = load_meta_agent_and_tools()


st.info(f"Currently building/editing agent: {agent_builder.cache.agent_id}", icon="ℹ️")


if "builder_agent" not in st.session_state.keys():
    st.session_state.builder_agent = builder_agent
if "agent_builder" not in st.session_state.keys():
    st.session_state.agent_builder = agent_builder

# add pills
selected = pills(
    "Outline your task!",
    [
        "I want to analyze this PDF file (data/invoices.pdf)",
        "I want to search over my CSV documents.",
    ],
    clearable=True,
    index=None,
)

if "messages" not in st.session_state.keys():  # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "What RAG bot do you want to build?"}
    ]


def add_to_message_history(role: str, content: str) -> None:
    message = {"role": role, "content": str(content)}
    st.session_state.messages.append(message)  # Add response to message history


for message in st.session_state.messages:  # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# handle user input
if prompt := st.chat_input(
    "Your question"
):  # Prompt for user input and save to chat history
    add_to_message_history("user", prompt)
    with st.chat_message("user"):
        st.write(prompt)

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.builder_agent.chat(prompt)
            st.write(str(response))
            add_to_message_history("assistant", str(response))

            # check agent_ids again, if it doesn't match, add to directory and refresh
            agent_ids = load_agent_ids_from_directory(str(AGENT_CACHE_DIR))
            # check diff between agent_ids and cur agent ids
            diff_ids = list(set(agent_ids) - set(st.session_state.cur_agent_ids))
            if len(diff_ids) > 0:
                # clear streamlit cache, to allow you to generate a new agent
                st.cache_resource.clear()

                # trigger refresh
                st.rerun()

else:
    pass
