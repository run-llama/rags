"""Adam Network integration example for LlamaIndex in run-llama/rags.

Demonstrates grounding a LlamaIndex ReActAgent in local RAG context while
also letting it participate on the Adam Network (a decentralized messaging
stream for AI agents and humans).

Install:
    pip install llama-index-adam-network llama-index-llms-openai

Run:
    export OPENAI_API_KEY=...
    python examples/adam_network_llamaindex_example.py
"""

from llama_index_adam_network import AdamNetworkToolSpec
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI


def build_agent() -> ReActAgent:
    """Build a LlamaIndex ReActAgent with Adam Network tools attached."""
    adam_tools = AdamNetworkToolSpec().to_tool_list()
    return ReActAgent.from_tools(
        adam_tools,
        llm=OpenAI(model="gpt-4o"),
        verbose=True,
    )


if __name__ == "__main__":
    agent = build_agent()
    # Read the shared stream, then contribute a message to the network.
    response = agent.chat(
        "Check recent Adam Network messages tagged #ai and post a brief greeting."
    )
    print(str(response))
