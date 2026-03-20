"""Tests for MiniMax LLM provider integration."""

import os
import sys
from unittest.mock import MagicMock, patch, call

import pytest


# Mock all heavy dependencies before importing core.utils
_mock_modules = {}
for mod_name in [
    "streamlit", "streamlit_pills",
    "llama_index", "llama_index.llms", "llama_index.llms.base",
    "llama_index.llms.utils", "llama_index.llms.openai_utils",
    "llama_index.agent", "llama_index.agent.types",
    "llama_index.agent.react", "llama_index.agent.react.prompts",
    "llama_index.agent.react.formatter",
    "llama_index.embeddings", "llama_index.embeddings.utils",
    "llama_index.tools", "llama_index.schema",
    "llama_index.callbacks",
    "llama_index.indices", "llama_index.indices.multi_modal",
    "llama_index.indices.multi_modal.base",
    "llama_index.indices.multi_modal.retriever",
    "llama_index.multi_modal_llms", "llama_index.multi_modal_llms.openai",
    "llama_index.query_engine", "llama_index.query_engine.multi_modal",
    "llama_index.chat_engine", "llama_index.chat_engine.types",
    "llama_index.llms.openai",
    "llama_hub", "llama_hub.web", "llama_hub.web.simple_web",
    "llama_hub.web.simple_web.base",
    "llama_hub.tools", "llama_hub.tools.metaphor",
    "llama_hub.tools.metaphor.base",
    "core.callback_manager",
]:
    if mod_name not in sys.modules:
        _mock_modules[mod_name] = MagicMock()
        sys.modules[mod_name] = _mock_modules[mod_name]

# Set up streamlit mock with secrets
st_mock = sys.modules["streamlit"]
st_mock.secrets = MagicMock()
st_mock.secrets.openai_key = "fake-openai-key"
st_mock.secrets.anthropic_key = "fake-anthropic-key"
st_mock.secrets.replicate_key = "fake-replicate-key"
st_mock.secrets.minimax_key = "fake-minimax-key"

# Set up llama_index mock classes
MockOpenAI = MagicMock()
MockAnthropic = MagicMock()
MockReplicate = MagicMock()
MockLLMBase = MagicMock()
MockBaseModel = MagicMock()

sys.modules["llama_index.llms"].OpenAI = MockOpenAI
sys.modules["llama_index.llms"].Anthropic = MockAnthropic
sys.modules["llama_index.llms"].Replicate = MockReplicate
sys.modules["llama_index.llms.base"].LLM = MockLLMBase

# Mock pydantic classes that core.utils uses
from pydantic import BaseModel, Field

# Now import core.utils - need to force re-import
if "core.utils" in sys.modules:
    del sys.modules["core.utils"]
if "core" in sys.modules:
    del sys.modules["core"]
if "core.builder_config" in sys.modules:
    del sys.modules["core.builder_config"]

# Mock builder_config
builder_config_mock = MagicMock()
builder_config_mock.BUILDER_LLM = MagicMock()
sys.modules["core.builder_config"] = builder_config_mock

import core.utils
from core.utils import _resolve_llm, RAGParams


class TestResolveLLMMiniMax:
    """Unit tests for _resolve_llm with MiniMax provider."""

    def setup_method(self):
        """Reset mocks before each test."""
        MockOpenAI.reset_mock()
        MockAnthropic.reset_mock()
        MockReplicate.reset_mock()
        st_mock.secrets.minimax_key = "fake-minimax-key"

    def test_minimax_prefix_creates_openai_with_correct_params(self):
        """Test that 'minimax:MiniMax-M2.7' creates OpenAI with MiniMax API base."""
        MockOpenAI.return_value = MagicMock()

        result = _resolve_llm("minimax:MiniMax-M2.7")

        MockOpenAI.assert_called_once_with(
            model="MiniMax-M2.7",
            api_key="fake-minimax-key",
            api_base="https://api.minimax.io/v1",
            temperature=1.0,
        )
        assert result == MockOpenAI.return_value

    def test_minimax_highspeed_model(self):
        """Test that 'minimax:MiniMax-M2.7-highspeed' works correctly."""
        MockOpenAI.return_value = MagicMock()

        result = _resolve_llm("minimax:MiniMax-M2.7-highspeed")

        MockOpenAI.assert_called_once_with(
            model="MiniMax-M2.7-highspeed",
            api_key="fake-minimax-key",
            api_base="https://api.minimax.io/v1",
            temperature=1.0,
        )
        assert result == MockOpenAI.return_value

    def test_minimax_uses_minimax_key_from_secrets(self):
        """Test that MiniMax provider reads API key from st.secrets.minimax_key."""
        st_mock.secrets.minimax_key = "test-api-key-12345"
        MockOpenAI.return_value = MagicMock()

        _resolve_llm("minimax:MiniMax-M2.7")

        call_kwargs = MockOpenAI.call_args[1]
        assert call_kwargs["api_key"] == "test-api-key-12345"

    def test_minimax_temperature_default_is_1(self):
        """Test that MiniMax provider sets temperature to 1.0 by default."""
        MockOpenAI.return_value = MagicMock()

        _resolve_llm("minimax:MiniMax-M2.7")

        call_kwargs = MockOpenAI.call_args[1]
        assert call_kwargs["temperature"] == 1.0

    def test_minimax_api_base_is_overseas(self):
        """Test that MiniMax uses overseas API base URL."""
        MockOpenAI.return_value = MagicMock()

        _resolve_llm("minimax:MiniMax-M2.7")

        call_kwargs = MockOpenAI.call_args[1]
        assert call_kwargs["api_base"] == "https://api.minimax.io/v1"

    def test_minimax_does_not_set_openai_env_var(self):
        """Test that MiniMax provider doesn't pollute OPENAI_API_KEY env var."""
        # Clear any existing key
        prev = os.environ.pop("OPENAI_API_KEY", None)
        MockOpenAI.return_value = MagicMock()

        _resolve_llm("minimax:MiniMax-M2.7")

        assert os.environ.get("OPENAI_API_KEY") is None or os.environ.get("OPENAI_API_KEY") == prev
        if prev:
            os.environ["OPENAI_API_KEY"] = prev

    def test_openai_provider_still_works(self):
        """Regression test: OpenAI provider is unaffected by MiniMax addition."""
        MockOpenAI.return_value = MagicMock()

        _resolve_llm("openai:gpt-4")

        MockOpenAI.assert_called_once_with(model="gpt-4")
        assert os.environ.get("OPENAI_API_KEY") == "fake-openai-key"

    def test_unknown_provider_raises_error(self):
        """Test that unknown provider prefix raises ValueError."""
        with pytest.raises(ValueError, match="not recognized"):
            _resolve_llm("unknown:some-model")


class TestResolveLLMExistingProviders:
    """Regression tests for existing providers."""

    def setup_method(self):
        MockOpenAI.reset_mock()
        MockAnthropic.reset_mock()
        MockReplicate.reset_mock()

    def test_anthropic_provider(self):
        """Regression test: Anthropic provider still works."""
        MockAnthropic.return_value = MagicMock()

        _resolve_llm("anthropic:claude-2")

        MockAnthropic.assert_called_once_with(model="claude-2")

    def test_replicate_provider(self):
        """Regression test: Replicate provider still works."""
        MockReplicate.return_value = MagicMock()

        _resolve_llm("replicate:some-model")

        MockReplicate.assert_called_once_with(model="some-model")

    def test_default_openai_no_prefix(self):
        """Test that no prefix defaults to OpenAI."""
        MockOpenAI.return_value = MagicMock()

        _resolve_llm("gpt-4")

        MockOpenAI.assert_called_once_with(model="gpt-4")


class TestRAGParamsWithMiniMax:
    """Test RAGParams with MiniMax model strings."""

    def test_rag_params_accepts_minimax_llm_string(self):
        """Test that RAGParams accepts a minimax: prefixed LLM string."""
        params = RAGParams(llm="minimax:MiniMax-M2.7")
        assert params.llm == "minimax:MiniMax-M2.7"

    def test_rag_params_accepts_minimax_highspeed(self):
        """Test that RAGParams accepts MiniMax-M2.7-highspeed."""
        params = RAGParams(llm="minimax:MiniMax-M2.7-highspeed")
        assert params.llm == "minimax:MiniMax-M2.7-highspeed"

    def test_rag_params_default_llm_unchanged(self):
        """Test that default LLM is still gpt-4-1106-preview."""
        params = RAGParams()
        assert params.llm == "gpt-4-1106-preview"
