import asyncio
import os
import subprocess
import tempfile

from loguru import logger

from chunkhound.providers.llm.base_cli_provider import BaseCLIProvider
from chunkhound.utils.text_sanitization import sanitize_error_text

class AntigravityCLIProvider(BaseCLIProvider):
    """Antigravity Go-based CLI LLM provider."""

    async def _run_cli_command(
        self,
        prompt: str,
        system: str | None = None,
        max_completion_tokens: int | None = None,
        timeout: int | None = None,
    ) -> str:
        # Build CLI command
        cmd = ["agy", "--print"]
        if self._model and self._model != "default" and self._model != "":
            cmd.extend(["--model", self._model])
        cmd.append("--dangerously-skip-permissions")

        merged_prompt = self._merge_prompts(prompt, system)
        cmd.append(merged_prompt)

        # Use provided timeout or default
        request_timeout = timeout if timeout is not None else self._timeout

        # Run command with retry logic
        last_error = None
        for attempt in range(self._max_retries):
            process = None
            try:
                # Create subprocess with neutral CWD to prevent workspace scanning
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=tempfile.gettempdir(),
                )

                # Wrap communicate() with timeout
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=request_timeout,
                )

                if process.returncode != 0:
                    raw_err = (stderr or stdout or b"").decode("utf-8", errors="ignore")
                    error_msg = (
                        sanitize_error_text(raw_err.strip())
                        or f"Exit code {process.returncode}"
                    )
                    last_error = RuntimeError(
                        f"CLI command failed (exit {process.returncode}): {error_msg}"
                    )
                    if attempt < self._max_retries - 1:
                        logger.warning(
                            f"CLI attempt {attempt + 1} failed, retrying: {error_msg}"
                        )
                        continue
                    raise last_error

                return stdout.decode("utf-8").strip()

            except asyncio.TimeoutError as e:
                # Kill the subprocess if it's still running
                if process and process.returncode is None:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
                    await process.wait()

                last_error = RuntimeError(
                    f"CLI command timed out after {request_timeout}s"
                )
                if attempt < self._max_retries - 1:
                    logger.warning(f"CLI attempt {attempt + 1} timed out, retrying")
                    continue
                raise last_error from e

            except Exception as e:
                if isinstance(e, RuntimeError):
                    raise
                # Kill the subprocess if it's still running on unexpected errors
                if process and process.returncode is None:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
                    await process.wait()

                last_error = RuntimeError(f"CLI command failed: {e}")
                if attempt < self._max_retries - 1:
                    logger.warning(f"CLI attempt {attempt + 1} failed: {e}")
                    continue
                raise last_error from e

        # Should not reach here, but just in case
        raise last_error or RuntimeError("CLI command failed after retries")

    def _get_provider_name(self) -> str:
        return "antigravity-cli"
