"""
Google BaseChatCompletionClient implementation.

This module provides integration with Google's Gemini API using the official
`google-genai>=2.32.0` client library.
"""

import base64
import json
import logging
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel

try:
    from google import genai
    from google.genai import types as GoogleTypes
    from google.genai.errors import APIError as GoogleAPIError

    _GOOGLE_AVAILABLE = True
except ImportError:  # optional extra - fail when the client is used, not on import
    _GOOGLE_AVAILABLE = False
    genai = GoogleTypes = GoogleAPIError = None  # type: ignore[assignment,misc]

from .._component_config import Component
from ..messages import AssistantMessage, Message, ToolCallRequest
from ..types import ChatCompletionChunk, ChatCompletionResult, Usage
from ._base import (
    AuthenticationError,
    BaseChatCompletionClient,
    BaseChatCompletionError,
    InvalidRequestError,
    RateLimitError,
)

logger = logging.getLogger(__name__)


class GoogleChatCompletionClientConfig(BaseModel):
    """Configuration for GoogleChatCompletionClient serialization."""

    model: str = "gemini-3.8-flash"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    enterprise: bool = False
    project: Optional[str] = None
    location: Optional[str] = None
    config: Dict[str, Any] = {}


class GoogleChatCompletionClient(
    Component[GoogleChatCompletionClientConfig], BaseChatCompletionClient
):
    """
    Google implementation of BaseChatCompletionClient.

    Supports Gemini models with function calling, multimodal input, and structured
    output capabilities through the official Google Gen AI SDK.
    """

    component_config_schema = GoogleChatCompletionClientConfig
    component_type = "model_client"
    component_provider_override = "nanoagents.llm.GoogleChatCompletionClient"

    def __init__(
        self,
        model: str = "gemini-3.8-flash",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        enterprise: bool = False,
        project: Optional[str] = None,
        location: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Initialize Google client.

        Args:
            model: Google model name (e.g., "gemini-3.8-flash", "gemini-2.5-pro")
            api_key: Google AI API key (uses GOOGLE_API_KEY or GEMINI_API_KEY if omitted)
            base_url: Optional custom base URL for an enterprise API gateway
            enterprise: Whether to use the Gemini Enterprise Agent Platform API
            project: Google Cloud project ID when enterprise is enabled
            location: Google Cloud location when enterprise is enabled
            **kwargs: Additional Google Gen AI client configuration

        Raises:
            ImportError: If the optional `google-genai` extra is not installed.
        """
        if not _GOOGLE_AVAILABLE:
            raise ImportError(
                "GoogleChatCompletionClient requires the Google Gen AI extra. "
                'Install it with: pip install "nanoagents[google]"'
            )
        if base_url and not enterprise:
            raise InvalidRequestError("A custom Google base URL requires enterprise=True.")

        super().__init__(model, api_key, **kwargs)

        client_kwargs = dict(kwargs)
        client_kwargs["enterprise"] = enterprise
        if api_key is not None:
            client_kwargs["api_key"] = api_key
        if project is not None:
            client_kwargs["project"] = project
        if location is not None:
            client_kwargs["location"] = location
        if base_url:
            # Google documents custom base URLs through HttpOptions. The custom
            # base URL flow is currently supported for the enterprise API.
            client_kwargs["http_options"] = GoogleTypes.HttpOptions(base_url=base_url)

        self.client = genai.Client(**client_kwargs)
        self.base_url = base_url
        self.enterprise = enterprise
        self.project = project
        self.location = location

    async def create(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        output_format: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> ChatCompletionResult:
        """
        Make a single Google API call.

        Args:
            messages: List of messages to send
            tools: Optional function definitions for tool calling
            output_format: Optional Pydantic model for structured output
            **kwargs: Additional Google generation parameters (temperature, max_output_tokens, etc.)

        Returns:
            Standardized chat completion result
        """
        try:
            start_time = time.time()
            contents, system_instruction = self._convert_messages_to_google_format(messages)
            request_config = self._build_generation_config(
                tools=tools,
                output_format=output_format,
                system_instruction=system_instruction,
                kwargs=kwargs,
            )

            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=contents,
                config=request_config,
            )
            duration_ms = int((time.time() - start_time) * 1000)

            candidate = self._get_candidate(response)
            assistant_content = ""
            tool_calls: List[ToolCallRequest] = []

            if candidate and candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if getattr(part, "text", None):
                        assistant_content += part.text
                    function_call = getattr(part, "function_call", None)
                    if function_call and function_call.name:
                        call_id = getattr(function_call, "id", None) or (f"call_{len(tool_calls)}")
                        arguments = getattr(function_call, "args", None) or {}
                        tool_calls.append(
                            ToolCallRequest(
                                tool_name=function_call.name,
                                parameters=dict(arguments),
                                call_id=call_id,
                            )
                        )

            structured_output = None
            if output_format:
                if not assistant_content:
                    raise BaseChatCompletionError(
                        "Google structured output was requested, but the response "
                        "did not contain JSON text."
                    )
                try:
                    structured_output = output_format.model_validate_json(assistant_content)
                except Exception as e:
                    raise BaseChatCompletionError(
                        f"Failed to parse Google structured output: {e}"
                    ) from e

            assistant_message = AssistantMessage(
                content=assistant_content,
                source="llm",  # Temporary source, will be overwritten by agent
                tool_calls=tool_calls if tool_calls else None,
                structured_content=structured_output,
            )

            usage_metadata = getattr(response, "usage_metadata", None)
            input_tokens = getattr(usage_metadata, "prompt_token_count", 0) or 0
            output_tokens = getattr(usage_metadata, "candidates_token_count", 0) or 0
            usage = Usage(
                duration_ms=duration_ms,
                llm_calls=1,
                tokens_input=input_tokens,
                tokens_output=output_tokens,
                tool_calls=len(tool_calls),
                cost_estimate=self._estimate_cost(input_tokens, output_tokens),
            )

            return ChatCompletionResult(
                message=assistant_message,
                usage=usage,
                model=getattr(response, "model_version", None) or self.model,
                finish_reason=self._get_finish_reason(candidate),
                structured_output=structured_output,
            )

        except BaseChatCompletionError:
            raise
        except Exception as e:
            self._raise_provider_error(e)

    async def create_stream(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        output_format: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        """
        Make a streaming Google API call.

        Args:
            messages: List of messages to send
            tools: Optional function definitions for tool calling
            output_format: Optional Pydantic model for structured output
            **kwargs: Additional Google generation parameters

        Yields:
            ChatCompletionChunk objects with incremental response data
        """
        try:
            contents, system_instruction = self._convert_messages_to_google_format(messages)
            request_config = self._build_generation_config(
                tools=tools,
                output_format=output_format,
                system_instruction=system_instruction,
                kwargs=kwargs,
            )

            accumulated_content = ""
            # Function-call arguments can arrive in partial chunks. Buffer them so
            # the framework receives valid, complete JSON rather than malformed fragments.
            tool_call_chunks: Dict[int, Dict[str, Any]] = {}
            final_response = None

            stream = await self.client.aio.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=request_config,
            )
            async for chunk in stream:
                final_response = chunk
                candidates = getattr(chunk, "candidates", None) or []
                if not candidates:
                    continue
                candidate = candidates[0]
                candidate_content = getattr(candidate, "content", None)
                parts = getattr(candidate_content, "parts", None) or []

                for part_index, part in enumerate(parts):
                    text_delta = getattr(part, "text", None)
                    if text_delta:
                        accumulated_content += text_delta
                        yield ChatCompletionChunk(
                            content=text_delta,
                            is_complete=False,
                            tool_call_chunk=None,
                        )

                    function_call = getattr(part, "function_call", None)
                    if function_call and function_call.name:
                        block_index = getattr(part, "_index", part_index)
                        call_id = getattr(function_call, "id", None) or f"call_{block_index}"
                        call_chunk = tool_call_chunks.setdefault(
                            block_index,
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {"name": function_call.name, "arguments": ""},
                            },
                        )
                        call_chunk["function"]["name"] = function_call.name
                        arguments = getattr(function_call, "args", None) or {}
                        if isinstance(arguments, dict):
                            call_chunk["function"]["arguments"] = json.dumps(arguments)

            if output_format:
                if not accumulated_content:
                    raise BaseChatCompletionError(
                        "Google structured output was requested, but the stream "
                        "did not contain JSON text."
                    )
                try:
                    output_format.model_validate_json(accumulated_content)
                except Exception as e:
                    raise BaseChatCompletionError(
                        f"Failed to parse Google structured output from stream: {e}"
                    ) from e

            for tool_call_chunk in tool_call_chunks.values():
                yield ChatCompletionChunk(
                    content="",
                    is_complete=False,
                    tool_call_chunk=tool_call_chunk,
                )

            usage_metadata = getattr(final_response, "usage_metadata", None)
            input_tokens = getattr(usage_metadata, "prompt_token_count", 0) or 0
            output_tokens = getattr(usage_metadata, "candidates_token_count", 0) or 0
            usage = Usage(
                duration_ms=0,  # Duration tracked at agent level
                llm_calls=1,
                tokens_input=input_tokens,
                tokens_output=output_tokens,
                tool_calls=len(tool_call_chunks),
                cost_estimate=self._estimate_cost(input_tokens, output_tokens),
            )
            yield ChatCompletionChunk(
                content="",
                is_complete=True,
                tool_call_chunk=None,
                usage=usage,
            )

        except BaseChatCompletionError:
            raise
        except Exception as e:
            self._raise_provider_error(e)

    def _convert_messages_to_google_format(
        self, messages: List[Message]
    ) -> Tuple[List[Any], Optional[str]]:
        """
        Convert internal Message objects to Google Gen AI API format.

        Args:
            messages: List of internal Message objects

        Returns:
            A tuple containing API contents and the system instruction, if present
        """
        from ..messages import MultiModalMessage, SystemMessage, ToolMessage

        contents = []
        system_messages: List[str] = []

        for msg in messages:
            if isinstance(msg, SystemMessage) or msg.role == "system":
                if msg.content:
                    system_messages.append(msg.content)
                continue

            if isinstance(msg, MultiModalMessage):
                parts = []
                if msg.content and msg.content.strip():
                    parts.append(GoogleTypes.Part.from_text(text=msg.content))
                if msg.is_image() or msg.is_audio() or msg.is_video():
                    if msg.data is not None:
                        if isinstance(msg.data, str):
                            try:
                                media_bytes = base64.b64decode(msg.to_base64(), validate=True)
                            except Exception as e:
                                raise InvalidRequestError(f"Invalid base64 media data: {e}") from e
                        else:
                            media_bytes = msg.data
                        parts.append(
                            GoogleTypes.Part.from_bytes(
                                data=media_bytes,
                                mime_type=msg.mime_type,
                            )
                        )
                    elif msg.media_url:
                        if msg.media_url.startswith("gs://"):
                            parts.append(
                                GoogleTypes.Part.from_uri(
                                    file_uri=msg.media_url,
                                    mime_type=msg.mime_type,
                                )
                            )
                        else:
                            raise InvalidRequestError(
                                "Google Gen AI media URLs must be supported URI sources, "
                                "such as gs://. Download HTTP(S) media and pass it as data."
                            )
                if parts:
                    role = "model" if msg.role == "assistant" else "user"
                    contents.append(GoogleTypes.Content(role=role, parts=parts))
                continue

            if isinstance(msg, AssistantMessage):
                parts = []
                if msg.content:
                    parts.append(GoogleTypes.Part.from_text(text=msg.content))
                for tool_call in msg.tool_calls or []:
                    parts.append(
                        GoogleTypes.Part(
                            function_call=GoogleTypes.FunctionCall(
                                id=tool_call.call_id,
                                name=tool_call.tool_name,
                                args=tool_call.parameters,
                            )
                        )
                    )
                contents.append(GoogleTypes.Content(role="model", parts=parts))
                continue

            if isinstance(msg, ToolMessage):
                response_payload: Dict[str, Any] = {
                    "result": msg.content,
                    "success": msg.success,
                }
                if msg.error:
                    response_payload["error"] = msg.error
                if msg.metadata:
                    response_payload["metadata"] = msg.metadata
                function_response = GoogleTypes.FunctionResponse(
                    id=msg.tool_call_id,
                    name=msg.tool_name,
                    response=response_payload,
                )
                contents.append(
                    GoogleTypes.Content(
                        role="tool",
                        parts=[GoogleTypes.Part(function_response=function_response)],
                    )
                )
                continue

            contents.append(
                GoogleTypes.Content(
                    role="user" if msg.role == "user" else "model",
                    parts=[GoogleTypes.Part.from_text(text=msg.content)],
                )
            )

        system_instruction = "\n\n".join(system_messages) if system_messages else None
        return contents, system_instruction

    def _convert_tools_to_google_format(self, tools: List[Dict[str, Any]]) -> List[Any]:
        """
        Convert OpenAI-style tool definitions to Google Gen AI format.

        Args:
            tools: List of OpenAI-format tool definitions

        Returns:
            List of Google tool definitions
        """
        function_declarations = []

        for tool in tools:
            if tool.get("type") == "function":
                function = tool.get("function", {})
                name = function.get("name")
                if not name:
                    raise InvalidRequestError("Google tool definitions require a function name.")
                declaration: Dict[str, Any] = {
                    "name": name,
                    "description": function.get("description", ""),
                    "parameters_json_schema": function.get("parameters", {"type": "object"}),
                }
                function_declarations.append(GoogleTypes.FunctionDeclaration(**declaration))

        if not function_declarations:
            return []
        return [GoogleTypes.Tool(function_declarations=function_declarations)]

    def _build_generation_config(
        self,
        tools: Optional[List[Dict[str, Any]]],
        output_format: Optional[Type[BaseModel]],
        system_instruction: Optional[str],
        kwargs: Dict[str, Any],
    ) -> Any:
        """Build a Google generation configuration from framework arguments."""
        config: Dict[str, Any] = dict(kwargs)
        if system_instruction:
            config["system_instruction"] = system_instruction
        if tools:
            config["tools"] = self._convert_tools_to_google_format(tools)
            # Tool calls are handled by nanoagents, not automatically executed by the SDK.
            config["automatic_function_calling"] = GoogleTypes.AutomaticFunctionCallingConfig(
                disable=True
            )
        if output_format:
            config["response_mime_type"] = "application/json"
            config["response_schema"] = output_format
        return GoogleTypes.GenerateContentConfig(**config)

    def _get_candidate(self, response: Any) -> Any:
        """Return the first candidate or raise a useful error when none is available."""
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            return candidates[0]
        prompt_feedback = getattr(response, "prompt_feedback", None)
        block_reason = getattr(prompt_feedback, "block_reason", None)
        detail = f" Prompt feedback: {block_reason}." if block_reason else ""
        raise InvalidRequestError(
            "Google returned no response candidate. The prompt may have been blocked." + detail
        )

    @staticmethod
    def _get_finish_reason(candidate: Any) -> str:
        """Convert Google's finish reason to the framework's string representation."""
        finish_reason = getattr(candidate, "finish_reason", None)
        if finish_reason is None:
            return "stop"
        value = getattr(finish_reason, "name", None) or str(finish_reason)
        normalized = value.lower()
        if normalized in {"stop", "finishreason.stop"}:
            return "stop"
        if normalized in {"max_tokens", "max_tokens_reached", "finishreason.max_tokens"}:
            return "length"
        if normalized in {"safety", "recitation", "blocklist", "prohibited_content"}:
            return "content_filter"
        return normalized

    def _raise_provider_error(self, error: Exception) -> None:
        """Map Google SDK exceptions to the framework's standardized errors."""
        status_code = getattr(error, "code", None) or getattr(error, "status_code", None)
        try:
            status_code = int(status_code) if status_code is not None else None
        except (TypeError, ValueError):
            status_code = None

        message = str(error)
        if status_code in {401, 403}:
            raise AuthenticationError(
                f"Google authentication failed: {message}", status_code=status_code
            ) from error
        if status_code == 429:
            raise RateLimitError(
                f"Google rate limit exceeded: {message}", status_code=status_code
            ) from error
        if status_code is not None and 400 <= status_code < 500:
            raise InvalidRequestError(
                f"Google API request failed: {message}", status_code=status_code
            ) from error
        if _GOOGLE_AVAILABLE and GoogleAPIError and isinstance(error, GoogleAPIError):
            raise BaseChatCompletionError(
                f"Google API error: {message}", status_code=status_code
            ) from error
        raise BaseChatCompletionError(
            f"Unexpected Google error: {message}", status_code=status_code
        ) from error

    def _estimate_cost(self, input_tokens: int, output_tokens: int) -> Optional[float]:
        """
        Estimate the cost of the API call based on token usage.

        Returns None when no verified price entry is available for the model.
        Pricing depends on model, service tier, context size, and modality. Keep
        this estimate conservative rather than assigning an unsupported rate.
        """
        # Pricing is intentionally not guessed. Add model-specific rates only after
        # verifying them against the official pricing page for the configured API.
        logger.debug(
            "Cost estimate unavailable for Google model %r; input_tokens=%d, output_tokens=%d",
            self.model,
            input_tokens,
            output_tokens,
        )
        return None

    def _to_config(self) -> GoogleChatCompletionClientConfig:
        """Convert client to configuration for serialization."""
        return GoogleChatCompletionClientConfig(
            model=self.model,
            api_key=self.api_key,
            base_url=getattr(self, "base_url", None),
            enterprise=self.enterprise,
            project=self.project,
            location=self.location,
            config=self.config,
        )

    @classmethod
    def _from_config(cls, config: GoogleChatCompletionClientConfig) -> "GoogleChatCompletionClient":
        """Create client from configuration.

        Args:
            config: Client configuration
        """
        return cls(
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
            enterprise=config.enterprise,
            project=config.project,
            location=config.location,
            **config.config,
        )
