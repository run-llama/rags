"""Configuration."""
import streamlit as st
import os

### DEFINE BUILDER_LLM #####
## Uncomment the LLM you want to use to construct the meta agent

## OpenAI
from llama_index.llms import OpenAI

# set OpenAI Key - use Streamlit secrets
os.environ["OPENAI_API_KEY"] = st.secrets.openai_key
# load LLM
BUILDER_LLM = OpenAI(model="gpt-4-1106-preview")

# # Anthropic (make sure you `pip install anthropic`)
# from llama_index.llms import Anthropic
# # set Anthropic key
# os.environ["ANTHROPIC_API_KEY"] = st.secrets.anthropic_key
# BUILDER_LLM = Anthropic()
