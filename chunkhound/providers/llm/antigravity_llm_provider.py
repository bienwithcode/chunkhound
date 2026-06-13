import asyncio
from typing import Any, TypeVar
from loguru import logger
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

from chunkhound.core.config.llm_config import DEFAULT_LLM_TIMEOUT
from chunkhound.core.utils import estimate_tokens_llm
from chunkhound.interfaces.llm_provider import LLMProvider, LLMResponse

try:
    from google.antigravity import Agent, LocalAgentConfig
    ANTIGRAVITY_AVAILABLE = True
except ImportError:
    Agent = None  # type: ignore
    LocalAgentConfig = None  # type: ignore
    ANTIGRAVITY_AVAILABLE = False
    logger.warning("google-antigravity not available - install with: uv add google-antigravity")

class AntigravityLLMProvider(LLMProvider):
    """Google Antigravity Python SDK LLM provider."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "",
        timeout: int = DEFAULT_LLM_TIMEOUT,
        max_retries: int = 3,
    ):
        """Initialize Antigravity LLM provider.

        Args:
            api_key: Optional Gemini API key. If omitted, falls back to environment variables.
            model: Optional model name. If omitted, letting the SDK resolve default behavior.
            timeout: Request timeout in seconds.
            max_retries: Number of retry attempts.
        """
        if not ANTIGRAVITY_AVAILABLE:
            raise ImportError(
                "google-antigravity not available - install with: uv add google-antigravity"
            )

        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._max_retries = max_retries

        # Usage tracking
        self._requests_made = 0
        self._tokens_used = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0

    @property
    def name(self) -> str:
        """Provider name."""
        return "antigravity-sdk"

    @property
    def model(self) -> str:
        """Model name."""
        return self._model

    @property
    def timeout(self) -> int:
        """Request timeout in seconds."""
        return self._timeout

    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        max_completion_tokens: int = 4096,
        timeout: int | None = None,
    ) -> LLMResponse:
        self._requests_made += 1
        config_kwargs = {}
        if self._api_key:
            config_kwargs["api_key"] = self._api_key
        if self._model:
            config_kwargs["model"] = self._model
        config_kwargs["system_instructions"] = system

        try:
            config = LocalAgentConfig(**config_kwargs)
            async with Agent(config) as agent:
                response = await agent.chat(prompt)
                
                # Retrieve thoughts and print them to stdout
                if hasattr(response, "thoughts") and response.thoughts:
                    async for thought in response.thoughts:
                        print(thought, end="", flush=True)
                
                # Retrieve text content
                content = ""
                if hasattr(response, "text"):
                    if callable(response.text):
                        content = await response.text()
                    else:
                        content = response.text

                # Retrieve usage token counts
                prompt_tokens = 0
                completion_tokens = 0
                total_tokens = 0
                if hasattr(agent, "conversation") and agent.conversation:
                    if hasattr(agent.conversation, "total_usage") and agent.conversation.total_usage:
                        usage = agent.conversation.total_usage
                        prompt_tokens = getattr(usage, "prompt_token_count", 0)
                        completion_tokens = getattr(usage, "candidates_token_count", 0)
                        total_tokens = getattr(usage, "total_token_count", prompt_tokens + completion_tokens)

                self._prompt_tokens += prompt_tokens
                self._completion_tokens += completion_tokens
                self._tokens_used += total_tokens

                return LLMResponse(
                    content=content,
                    tokens_used=total_tokens,
                    model=self._model,
                    finish_reason="stop",
                )
        except Exception as e:
            logger.error(f"Antigravity SDK completion failed: {e}")
            raise RuntimeError(str(e)) from e

    async def complete_structured(
        self,
        prompt: str,
        json_schema: dict[str, Any],
        system: str | None = None,
        max_completion_tokens: int = 4096,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError(
            "antigravity-sdk provider does not support dict-based structured outputs. "
            "Use complete_structured_typed with a Pydantic model instead."
        )

    async def complete_structured_typed(
        self,
        prompt: str,
        response_model: type[T],
        system: str | None = None,
        max_completion_tokens: int = 4096,
    ) -> T:
        self._requests_made += 1
        config_kwargs = {}
        if self._api_key:
            config_kwargs["api_key"] = self._api_key
        if self._model:
            config_kwargs["model"] = self._model
        config_kwargs["system_instructions"] = system
        config_kwargs["response_schema"] = response_model

        try:
            config = LocalAgentConfig(**config_kwargs)
            async with Agent(config) as agent:
                response = await agent.chat(prompt)
                
                # Retrieve thoughts and print them to stdout
                if hasattr(response, "thoughts") and response.thoughts:
                    async for thought in response.thoughts:
                        print(thought, end="", flush=True)
                
                # Retrieve text content
                content = ""
                if hasattr(response, "text"):
                    if callable(response.text):
                        content = await response.text()
                    else:
                        content = response.text

                # Retrieve usage token counts
                prompt_tokens = 0
                completion_tokens = 0
                total_tokens = 0
                if hasattr(agent, "conversation") and agent.conversation:
                    if hasattr(agent.conversation, "total_usage") and agent.conversation.total_usage:
                        usage = agent.conversation.total_usage
                        prompt_tokens = getattr(usage, "prompt_token_count", 0)
                        completion_tokens = getattr(usage, "candidates_token_count", 0)
                        total_tokens = getattr(usage, "total_token_count", prompt_tokens + completion_tokens)

                self._prompt_tokens += prompt_tokens
                self._completion_tokens += completion_tokens
                self._tokens_used += total_tokens

                return response_model.model_validate_json(content)
        except Exception as e:
            logger.error(f"Antigravity SDK structured completion failed: {e}")
            raise RuntimeError(str(e)) from e

    async def batch_complete(
        self,
        prompts: list[str],
        system: str | None = None,
        max_completion_tokens: int = 4096,
    ) -> list[LLMResponse]:
        tasks = [
            self.complete(prompt, system, max_completion_tokens) for prompt in prompts
        ]
        return await asyncio.gather(*tasks)

    def estimate_tokens(self, text: str) -> int:
        return estimate_tokens_llm(text)

    async def health_check(self) -> dict[str, Any]:
        try:
            response = await self.complete("Say 'OK'", max_completion_tokens=10)
            return {
                "status": "healthy",
                "provider": "antigravity-sdk",
                "model": self._model,
                "test_response": response.content[:50],
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": "antigravity-sdk",
                "model": self._model,
                "error": str(e),
            }

    def get_usage_stats(self) -> dict[str, Any]:
        return {
            "requests_made": self._requests_made,
            "total_tokens": self._tokens_used,
            "prompt_tokens": self._prompt_tokens,
            "completion_tokens": self._completion_tokens,
        }
