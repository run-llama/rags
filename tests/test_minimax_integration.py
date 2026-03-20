"""Integration tests for MiniMax LLM provider via OpenAI-compatible API.

These tests make real API calls to MiniMax and require MINIMAX_API_KEY.
Run with: MINIMAX_API_KEY=<key> pytest tests/test_minimax_integration.py -v
"""

import os

import pytest

# Skip all tests if MINIMAX_API_KEY is not set
pytestmark = pytest.mark.skipif(
    not os.environ.get("MINIMAX_API_KEY"),
    reason="MINIMAX_API_KEY not set",
)


class TestMiniMaxAPIIntegration:
    """Integration tests that call the real MiniMax API."""

    def test_minimax_chat_completion(self):
        """Test basic chat completion via MiniMax OpenAI-compatible API."""
        from openai import OpenAI

        client = OpenAI(
            api_key=os.environ["MINIMAX_API_KEY"],
            base_url="https://api.minimax.io/v1",
        )

        response = client.chat.completions.create(
            model="MiniMax-M2.7",
            messages=[{"role": "user", "content": "Say hello in one word."}],
            temperature=1.0,
            max_tokens=10,
        )

        assert response.choices
        assert len(response.choices[0].message.content.strip()) > 0

    def test_minimax_highspeed_model(self):
        """Test MiniMax-M2.7-highspeed model works."""
        from openai import OpenAI

        client = OpenAI(
            api_key=os.environ["MINIMAX_API_KEY"],
            base_url="https://api.minimax.io/v1",
        )

        response = client.chat.completions.create(
            model="MiniMax-M2.7-highspeed",
            messages=[{"role": "user", "content": "Say hello in one word."}],
            temperature=1.0,
            max_tokens=100,
        )

        assert response.choices
        assert len(response.choices[0].message.content.strip()) > 0

    def test_minimax_streaming(self):
        """Test streaming works with MiniMax API."""
        from openai import OpenAI

        client = OpenAI(
            api_key=os.environ["MINIMAX_API_KEY"],
            base_url="https://api.minimax.io/v1",
        )

        stream = client.chat.completions.create(
            model="MiniMax-M2.7",
            messages=[{"role": "user", "content": "Say hi."}],
            temperature=1.0,
            max_tokens=10,
            stream=True,
        )

        chunks = list(stream)
        assert len(chunks) > 0
