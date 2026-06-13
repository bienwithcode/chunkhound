from __future__ import annotations
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from chunkhound.core.config.llm_config import DEFAULT_LLM_TIMEOUT
from chunkhound.providers.llm.antigravity_llm_provider import AntigravityLLMProvider

@pytest.fixture
def mock_agent_class():
    with patch("chunkhound.providers.llm.antigravity_llm_provider.Agent") as mock_agent:
        yield mock_agent

@pytest.fixture
def mock_config_class():
    with patch("chunkhound.providers.llm.antigravity_llm_provider.LocalAgentConfig") as mock_config:
        yield mock_config

class TestAntigravitySDKConstruction:
    def test_construction_with_params(self, mock_agent_class, mock_config_class):
        """Verify constructor registers key parameters correctly."""
        provider = AntigravityLLMProvider(
            api_key="test-key-123",
            model="gemini-3.5-flash",
            timeout=60,
            max_retries=2
        )
        assert provider.name == "antigravity-sdk"
        assert provider.model == "gemini-3.5-flash"
        assert provider.timeout == 60

    def test_construction_defaults(self, mock_agent_class, mock_config_class):
        """Verify default parameters are used if omitted."""
        provider = AntigravityLLMProvider()
        assert provider.name == "antigravity-sdk"
        assert provider.model == ""
        assert provider.timeout == DEFAULT_LLM_TIMEOUT

    def test_import_error_when_unavailable(self, mock_agent_class, mock_config_class):
        """Verify ImportError is raised on initialization if library is missing."""
        with patch("chunkhound.providers.llm.antigravity_llm_provider.ANTIGRAVITY_AVAILABLE", False):
            with pytest.raises(ImportError, match="google-antigravity not available"):
                AntigravityLLMProvider()

class TestAntigravitySDKComplete:
    @pytest.mark.asyncio
    async def test_complete_success(self, mock_agent_class, mock_config_class, capsys):
        """Verify complete() executes successfully, streams thoughts, and records tokens."""
        # Setup mock Agent instance and response
        mock_agent = MagicMock()
        mock_agent_class.return_value.__aenter__.return_value = mock_agent

        mock_response = MagicMock()
        async def mock_text():
            return "final answer text"
        mock_response.text = mock_text

        async def mock_thoughts():
            yield "thinking "
            yield "deeply"
        mock_response.thoughts = mock_thoughts()

        mock_agent.chat = AsyncMock(return_value=mock_response)

        # Setup mock usage metadata
        mock_usage = MagicMock()
        mock_usage.prompt_token_count = 10
        mock_usage.candidates_token_count = 20
        mock_usage.thoughts_token_count = 5
        mock_usage.total_token_count = 35
        mock_agent.conversation.total_usage = mock_usage

        provider = AntigravityLLMProvider(
            api_key="test-key",
            model="gemini-3.5-flash"
        )
        response = await provider.complete(
            prompt="what is the capital of France?",
            system="be concise"
        )

        # Assertions
        assert response.content == "final answer text"
        assert response.tokens_used == 35
        assert response.model == "gemini-3.5-flash"
        assert response.finish_reason == "stop"

        # Check mock config creation calls
        mock_config_class.assert_called_once_with(
            api_key="test-key",
            model="gemini-3.5-flash",
            system_instructions="be concise"
        )

        # Check stdout contains captured thoughts
        captured = capsys.readouterr()
        assert "thinking deeply" in captured.out

    @pytest.mark.asyncio
    async def test_complete_omitted_model_and_key(self, mock_agent_class, mock_config_class):
        """Verify empty model and key are omitted from config creation."""
        mock_agent = MagicMock()
        mock_agent_class.return_value.__aenter__.return_value = mock_agent
        mock_response = MagicMock()
        async def mock_text():
            return "ok"
        mock_response.text = mock_text
        mock_response.thoughts = None
        mock_agent.chat = AsyncMock(return_value=mock_response)
        
        provider = AntigravityLLMProvider()
        await provider.complete(prompt="hello")

        # Verify config is created with defaults (empty/None omitted)
        mock_config_class.assert_called_once_with(
            system_instructions=None
        )

    @pytest.mark.asyncio
    async def test_complete_error_wrapping(self, mock_agent_class, mock_config_class):
        """Verify exceptions raised by the SDK are wrapped in a RuntimeError."""
        mock_agent = MagicMock()
        mock_agent_class.return_value.__aenter__.return_value = mock_agent
        mock_agent.chat = AsyncMock(side_effect=Exception("SDK network failure"))

        provider = AntigravityLLMProvider()
        with pytest.raises(RuntimeError, match="SDK network failure"):
            await provider.complete(prompt="hello")


class TestAntigravitySDKStructured:
    @pytest.mark.asyncio
    async def test_complete_structured_typed_success(self, mock_agent_class, mock_config_class):
        """Verify complete_structured_typed configures response_schema with the Pydantic class."""
        from pydantic import BaseModel
        
        class MockSchema(BaseModel):
            result: str
            confidence: float

        mock_agent = MagicMock()
        mock_agent_class.return_value.__aenter__.return_value = mock_agent

        mock_response = MagicMock()
        async def mock_text():
            return '{"result": "success", "confidence": 0.99}'
        mock_response.text = mock_text
        mock_response.thoughts = None
        mock_agent.chat = AsyncMock(return_value=mock_response)

        # Setup mock usage metadata
        mock_usage = MagicMock()
        mock_usage.prompt_token_count = 10
        mock_usage.candidates_token_count = 20
        mock_usage.total_token_count = 30
        mock_agent.conversation.total_usage = mock_usage

        provider = AntigravityLLMProvider(
            api_key="test-key",
            model="gemini-3.5-flash"
        )
        response_obj = await provider.complete_structured_typed(
            prompt="parse this info",
            response_model=MockSchema,
            system="be precise"
        )

        assert isinstance(response_obj, MockSchema)
        assert response_obj.result == "success"
        assert response_obj.confidence == 0.99

        # Verify configuration was initialized with response_schema
        mock_config_class.assert_called_once_with(
            api_key="test-key",
            model="gemini-3.5-flash",
            system_instructions="be precise",
            response_schema=MockSchema
        )

    @pytest.mark.asyncio
    async def test_complete_structured_not_implemented(self, mock_agent_class, mock_config_class):
        """Verify dict-based complete_structured raises NotImplementedError."""
        provider = AntigravityLLMProvider()
        with pytest.raises(NotImplementedError, match="does not support dict-based structured outputs"):
            await provider.complete_structured(
                prompt="hello",
                json_schema={"type": "object"}
            )


class TestAntigravitySDKBatchComplete:
    @pytest.mark.asyncio
    async def test_batch_complete_success(self, mock_agent_class, mock_config_class):
        """Verify batch_complete runs completions concurrently and returns results."""
        provider = AntigravityLLMProvider(model="gemini-3.5-flash")
        
        from chunkhound.interfaces.llm_provider import LLMResponse
        with patch.object(provider, "complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = LLMResponse(
                content="mocked response",
                tokens_used=10,
                model="gemini-3.5-flash"
            )
            
            prompts = ["prompt 1", "prompt 2", "prompt 3"]
            results = await provider.batch_complete(prompts, system="system instruction")
            
            assert len(results) == 3
            assert all(r.content == "mocked response" for r in results)
            assert mock_complete.call_count == 3
            mock_complete.assert_any_call("prompt 1", "system instruction", 4096)


class TestAntigravityCLIConstruction:
    def test_construction_with_params(self):
        """Verify CLI constructor registers key parameters correctly."""
        from chunkhound.providers.llm.antigravity_cli_provider import AntigravityCLIProvider
        provider = AntigravityCLIProvider(
            api_key="test-key-123",
            model="gemini-3.5-flash",
            timeout=60,
            max_retries=2
        )
        assert provider.name == "antigravity-cli"
        assert provider.model == "gemini-3.5-flash"
        assert provider.timeout == 60

    def test_construction_defaults(self):
        """Verify default parameters are used if omitted."""
        from chunkhound.providers.llm.antigravity_cli_provider import AntigravityCLIProvider
        provider = AntigravityCLIProvider()
        assert provider.name == "antigravity-cli"
        assert provider.model == "default"
        assert provider.timeout == DEFAULT_LLM_TIMEOUT


class TestAntigravityCLIRunCommand:
    @pytest.mark.asyncio
    @patch("asyncio.create_subprocess_exec")
    async def test_run_cli_command_success(self, mock_create_exec):
        """Verify _run_cli_command constructs arguments correctly and executes subprocess."""
        from chunkhound.providers.llm.antigravity_cli_provider import AntigravityCLIProvider
        import subprocess

        # Mock process return value
        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"output text from cli\n", b"")
        mock_process.returncode = 0
        mock_create_exec.return_value = mock_process

        provider = AntigravityCLIProvider(model="gemini-3.5-flash")
        result = await provider._run_cli_command(
            prompt="hello cli",
            system="be quick"
        )

        assert result == "output text from cli"

        # Assert arguments passed to create_subprocess_exec
        mock_create_exec.assert_called_once()
        args, kwargs = mock_create_exec.call_args
        
        # Verify executable and flags
        assert args[0] == "agy"
        assert "--print" in args
        assert "--model" in args
        assert "gemini-3.5-flash" in args
        assert "--dangerously-skip-permissions" in args
        
        # Verify merged prompt is passed as argument
        expected_prompt = provider._merge_prompts("hello cli", "be quick")
        assert expected_prompt in args

        # Verify execution context (temp dir)
        assert kwargs["cwd"] is not None
        import tempfile
        assert kwargs["cwd"] == tempfile.gettempdir()
        assert kwargs["stdin"] == subprocess.PIPE
        assert kwargs["stdout"] == subprocess.PIPE
        assert kwargs["stderr"] == subprocess.PIPE

    @pytest.mark.asyncio
    @patch("asyncio.create_subprocess_exec")
    async def test_run_cli_command_omit_model(self, mock_create_exec):
        """Verify model flag is omitted if model is default, empty or None."""
        from chunkhound.providers.llm.antigravity_cli_provider import AntigravityCLIProvider

        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"ok", b"")
        mock_process.returncode = 0
        mock_create_exec.return_value = mock_process

        # Case 1: default model
        provider = AntigravityCLIProvider()
        await provider._run_cli_command(prompt="hello")
        args, _ = mock_create_exec.call_args
        assert "--model" not in args

        # Case 2: empty model
        mock_create_exec.reset_mock()
        provider = AntigravityCLIProvider(model="")
        await provider._run_cli_command(prompt="hello")
        args, _ = mock_create_exec.call_args
        assert "--model" not in args

