"""Configuration."""
# no streamlit import here, to ensure we aren't relying on streamlit
# import streamlit as st
# adding os to allow for env vars
import os


### DEFINE BUILDER_LLM #####
## Uncomment the LLM you want to use to construct the meta agent

## OpenAI
from llama_index.llms import OpenAI

# nickknyc's OpenAI key routine
openai_api_key = os.getenv("OPENAI_API_KEY")
# removing Streamlit secrets in favor of env vars
# set OpenAI Key - use Streamlit secrets
# os.environ["OPENAI_API_KEY"] = st.secrets.openai_key
# load LLM
BUILDER_LLM = OpenAI(model="gpt-4-1106-preview",api_key=openai_api_key)

# # Anthropic (make sure you `pip install anthropic`)
# from llama_index.llms import Anthropic
# # set Anthropic key
# os.environ["ANTHROPIC_API_KEY"] = st.secrets.anthropic_key
# BUILDER_LLM = Anthropic()
