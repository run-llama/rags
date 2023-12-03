"""Streamlit page showing builder config."""
import streamlit as st
from st_utils import add_sidebar, get_current_state


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

current_state = get_current_state()
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


def display_messages() -> None:
    """Display messages."""
    for message in st.session_state.agent_messages:  # Display the prior chat messages
        with st.chat_message(message["role"]):
            msg_type = message["msg_type"] if "msg_type" in message.keys() else "text"
            if msg_type == "text":
                st.write(message["content"])
            elif msg_type == "info":
                st.info(message["content"], icon="ℹ️")
            else:
                raise ValueError(f"Unknown message type: {msg_type}")


# if agent is created, then we can chat with it
if current_state.cache is not None and current_state.cache.agent is not None:
    st.info(f"Viewing config for agent: {current_state.cache.agent_id}", icon="ℹ️")
    agent = current_state.cache.agent

    # display prior messages
    display_messages()

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
