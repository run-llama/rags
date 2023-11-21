import streamlit as st
from streamlit_pills import pills

from agent_utils import (
    load_meta_agent_and_tools,
)


####################
#### STREAMLIT #####
####################


st.set_page_config(page_title="Build a RAGs bot, powered by LlamaIndex", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
st.title("Build a RAGs bot, powered by LlamaIndex 💬🦙")
st.info(
    "Use this page to build your RAG bot over your data! "
    "Once the agent is finished creating, check out the `RAG Config` and `Generated RAG Agent` pages.", 
    icon="ℹ️"
)

# TODO: noodle on this
# with st.sidebar:
#     openai_api_key_st = st.text_input("OpenAI API Key (optional, not needed if you filled in secrets.toml)", value="", type="password")
#     if st.button("Save"):
#         # save api key
#         st.session_state.openai_api_key = openai_api_key_st

#### load builder agent and its tool spec (the agent_builder)
builder_agent, agent_builder = load_meta_agent_and_tools()

if "builder_agent" not in st.session_state.keys():
    st.session_state.builder_agent = builder_agent
if "agent_builder" not in st.session_state.keys():
    st.session_state.agent_builder = agent_builder

# add pills
selected = pills(
    "Outline your task!", 
    ["I want to analyze this PDF file (data/invoices.pdf)", 
     "I want to search over my CSV documents."
    ], clearable=True, index=None
)

if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "What RAG bot do you want to build?"}
    ]

def add_to_message_history(role, content):
    message = {"role": role, "content": str(content)}
    st.session_state.messages.append(message) # Add response to message history

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# handle user input
if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    add_to_message_history("user", prompt)
    with st.chat_message("user"):
        st.write(prompt)

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.builder_agent.chat(prompt)
            st.write(str(response))
            add_to_message_history("assistant", response)

# # check cache
print(st.session_state.agent_builder.cache)
# if "agent" in cache:
#     st.session_state.agent = cache["agent"]
