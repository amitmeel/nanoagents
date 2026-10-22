"""
Anthropic BaseChatCompletionClient implementation.

This module provides integration with Anthropic's Claude API using the official
anthropic>=2.1.0 client library.
"""

import json
import logging
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel

try:
    from anthropic import APIError, AsyncAnthropic, BadRequestError, transform_schema
    from anthropic import AuthenticationError as AnthropicAuthError
    from anthropic import RateLimitError as AnthropicRateLimitError
    from anthropic.types import Message as AnthropicMessage
    from anthropic.types import ToolUseBlock

    _ANTHROPIC_AVAILABLE = True
except ImportError:  # optional extra - fail when the client is used, not on import
    _ANTHROPIC_AVAILABLE = False
    APIError = AsyncAnthropic = BadRequestError = None  # type: ignore[assignment,misc]
    transform_schema = None  # type: ignore[assignment]
    AnthropicAuthError = AnthropicRateLimitError = None  # type: ignore[assignment,misc]
    AnthropicMessage = ToolUseBlock = None  # type: ignore[assignment,misc]

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


class AnthropicChatCompletionClientConfig(BaseModel):
    """Configuration for AnthropicChatCompletionClient serialization."""

    model: str = "claude-sonnet-4-5"  # Sonnet supports structured outputs
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    config: Dict[str, Any] = {}


class AnthropicChatCompletionClient(
    Component[AnthropicChatCompletionClientConfig], BaseChatCompletionClient
):
    """
    Anthropic implementation of BaseChatCompletionClient.

    Supports Claude models with function calling and structured output capabilities.
    """

    component_config_schema = AnthropicChatCompletionClientConfig
    component_type = "model_client"
    component_provider_override = "nanoagents.llm.AnthropicChatCompletionClient"

    def __init__(
        self,
        model: str = "claude-sonnet-4-5",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Initialize Anthropic client.

        Args:
            model: Anthropic model name (e.g., "claude-sonnet-4-5", "claude-opus-4-1")
            api_key: Anthropic API key (will use ANTHROPIC_API_KEY env var if not provided)
            base_url: Custom base URL for API calls
            **kwargs: Additional Anthropic client configuration

        Raises:
            ImportError: If the optional `anthropic` extra is not installed.
        """
        if not _ANTHROPIC_AVAILABLE:
            raise ImportError(
                "AnthropicChatCompletionClient requires the anthropic extra. "
                'Install it with: pip install "nanoagents[anthropic]"'
            )
        super().__init__(model, api_key, **kwargs)

        self.client = AsyncAnthropic(api_key=api_key, base_url=base_url, **kwargs)

    async def create(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        output_format: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> ChatCompletionResult:
        """
        Make a single Anthropic API call.

        Args:
            messages: List of messages to send
            tools: Optional function definitions for tool calling
            output_format: Optional Pydantic model for structured output
            **kwargs: Additional Anthropic parameters (max_tokens, etc.)

        Returns:
            Standardized chat completion result

        Raises:
            InvalidRequestError: If the request is invalid or uses unsupported parameters.
            BaseChatCompletionError: If the API call fails.
        """
        try:
            start_time = time.time()

            # Sampling parameters were removed in Anthropic API version 2026-10-21.
            # Reject them explicitly instead of silently changing the caller's requested behavior.
            self._validate_request_kwargs(kwargs)

            # Convert messages to Anthropic format
            api_messages = self._convert_messages_to_anthropic_format(messages)

            # Extract system messages because Anthropic uses a separate system parameter.
            system_messages = [msg for msg in api_messages if msg["role"] == "system"]
            api_messages = [msg for msg in api_messages if msg["role"] != "system"]
            system_message = self._combine_system_messages(system_messages)

            # Prepare request parameters
            request_params: Dict[str, Any] = {
                "model": self.model,
                "messages": api_messages,
                "max_tokens": kwargs.get("max_tokens", 4096),  # Anthropic requires max_tokens
            }

            # Add system message if present
            if system_message:
                request_params["system"] = system_message

            # Add tools if provided
            if tools:
                request_params["tools"] = self._convert_tools_to_anthropic_format(tools)

            # Add structured output if requested
            if output_format:
                schema = self._make_schema_compatible(output_format.model_json_schema())
                request_params["output_config"] = {
                    "format": {
                        "type": "json_schema",
                        "schema": schema,
                    }
                }

            # Forward supported additional request parameters without changing their values.
            supported_params = {
                "metadata",
                "service_tier",
                "stop_sequences",
                "thinking",
                "tool_choice",
                "output_config",
                "speed",
                "container",
                "context_management",
                "mcp_servers",
                "inference_geo",
                "user_profile_id",
                "workspace_id",
            }
            for key in supported_params:
                if key in kwargs and key not in request_params:
                    request_params[key] = kwargs[key]

            # Effort is part of output_config in the current Messages API.
            if "effort" in kwargs:
                request_params.setdefault("output_config", {})["effort"] = kwargs["effort"]

            response: AnthropicMessage = await self.client.messages.create(**request_params)

            duration_ms = int((time.time() - start_time) * 1000)

            # Extract content and tool calls
            assistant_content = ""
            tool_calls = []

            for block in response.content:
                if getattr(block, "type", None) == "text":
                    assistant_content += block.text
                elif isinstance(block, ToolUseBlock):
                    tool_calls.append(
                        ToolCallRequest(
                            tool_name=block.name,
                            parameters=block.input if isinstance(block.input, dict) else {},
                            call_id=block.id,
                        )
                    )

            assistant_message = AssistantMessage(
                content=assistant_content,
                source="llm",  # Temporary source, will be overwritten by agent
                tool_calls=tool_calls if tool_calls else None,
            )

            # Parse structured output if requested
            structured_output = None
            if output_format:
                try:
                    structured_output = output_format.model_validate_json(assistant_content)
                except Exception as e:
                    raise BaseChatCompletionError(
                        f"Anthropic returned structured output that could not be parsed "
                        f"as {output_format.__name__}: {e}"
                    ) from e

            # Create usage statistics
            usage = Usage(
                duration_ms=duration_ms,
                llm_calls=1,
                tokens_input=response.usage.input_tokens,
                tokens_output=response.usage.output_tokens,
                tool_calls=len(tool_calls),
                cost_estimate=self._estimate_cost(
                    response.usage.input_tokens, response.usage.output_tokens
                ),
            )

            return ChatCompletionResult(
                message=assistant_message,
                usage=usage,
                model=response.model,
                finish_reason=response.stop_reason or "stop",
                structured_output=structured_output,
            )

        except AnthropicAuthError as e:
            raise AuthenticationError(f"Anthropic authentication failed: {str(e)}") from e
        except AnthropicRateLimitError as e:
            raise RateLimitError(f"Anthropic rate limit exceeded: {str(e)}") from e
        except BadRequestError as e:
            raise InvalidRequestError(f"Anthropic request is invalid: {str(e)}") from e
        except APIError as e:
            raise BaseChatCompletionError(f"Anthropic API error: {str(e)}") from e
        except (AuthenticationError, RateLimitError, InvalidRequestError, BaseChatCompletionError):
            raise
        except Exception as e:
            raise BaseChatCompletionError(f"Unexpected error: {str(e)}") from e

    async def create_stream(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        output_format: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        """
        Make a streaming Anthropic API call.

        Args:
            messages: List of messages to send
            tools: Optional function definitions for tool calling
            output_format: Optional Pydantic model for structured output
            **kwargs: Additional Anthropic parameters (max_tokens, etc.)

        Yields:
            ChatCompletionChunk objects with incremental response data

        Raises:
            InvalidRequestError: If the request is invalid or uses unsupported parameters.
            BaseChatCompletionError: If the API call fails.
        """
        try:
            self._validate_request_kwargs(kwargs)

            # Convert messages to Anthropic format
            api_messages = self._convert_messages_to_anthropic_format(messages)

            # Extract system messages because Anthropic uses a separate system parameter.
            system_messages = [msg for msg in api_messages if msg["role"] == "system"]
            api_messages = [msg for msg in api_messages if msg["role"] != "system"]
            system_message = self._combine_system_messages(system_messages)

            # Prepare request parameters
            request_params: Dict[str, Any] = {
                "model": self.model,
                "messages": api_messages,
                "max_tokens": kwargs.get("max_tokens", 4096),
            }

            if system_message:
                request_params["system"] = system_message

            if tools:
                request_params["tools"] = self._convert_tools_to_anthropic_format(tools)

            # Structured outputs are supported while streaming through output_config.format.
            if output_format:
                schema = self._make_schema_compatible(output_format.model_json_schema())
                request_params["output_config"] = {
                    "format": {
                        "type": "json_schema",
                        "schema": schema,
                    }
                }

            supported_params = {
                "metadata",
                "service_tier",
                "stop_sequences",
                "thinking",
                "tool_choice",
                "output_config",
                "speed",
                "container",
                "context_management",
                "mcp_servers",
                "inference_geo",
                "user_profile_id",
                "workspace_id",
            }
            for key in supported_params:
                if key in kwargs and key not in request_params:
                    request_params[key] = kwargs[key]

            # Effort is part of output_config in the current Messages API.
            if "effort" in kwargs:
                request_params.setdefault("output_config", {})["effort"] = kwargs["effort"]

            async with self.client.messages.stream(**request_params) as stream:
                tool_call_chunks: Dict[int, Dict[str, Any]] = {}

                async for event in stream:
                    if event.type == "content_block_start":
                        # Initialize tool call chunks with Anthropic's actual tool call ID.
                        if getattr(event.content_block, "type", None) == "tool_use":
                            block_index = event.index
                            tool_call_chunks[block_index] = {
                                "id": event.content_block.id,
                                "function": {
                                    "name": event.content_block.name,
                                    "arguments": "",
                                },
                            }
                    elif event.type == "content_block_delta":
                        if getattr(event.delta, "type", None) == "text_delta":
                            yield ChatCompletionChunk(
                                content=event.delta.text,
                                is_complete=False,
                                tool_call_chunk=None,
                            )
                        elif getattr(event.delta, "type", None) == "input_json_delta":
                            block_index = event.index
                            if block_index not in tool_call_chunks:
                                # A valid tool input delta normally follows content_block_start.
                                # Keep a defensive fallback for incomplete or unexpected streams.
                                tool_call_chunks[block_index] = {
                                    "id": f"call_{block_index}",
                                    "function": {"name": "", "arguments": ""},
                                }
                            tool_call_chunks[block_index]["function"]["arguments"] += (
                                event.delta.partial_json
                            )
                            yield ChatCompletionChunk(
                                content="",
                                is_complete=False,
                                tool_call_chunk=tool_call_chunks[block_index],
                            )

                # Get final message for usage statistics and final structured-output validation.
                final_message = await stream.get_final_message()

                if output_format:
                    text_content = "".join(
                        block.text
                        for block in final_message.content
                        if getattr(block, "type", None) == "text"
                    )
                    try:
                        output_format.model_validate_json(text_content)
                    except Exception as e:
                        raise BaseChatCompletionError(
                            f"Anthropic returned structured output that could not be parsed "
                            f"as {output_format.__name__}: {e}"
                        ) from e

                usage_data = Usage(
                    duration_ms=0,  # Duration tracked at agent level
                    llm_calls=1,
                    tokens_input=final_message.usage.input_tokens,
                    tokens_output=final_message.usage.output_tokens,
                    tool_calls=len(tool_call_chunks),
                    cost_estimate=self._estimate_cost(
                        final_message.usage.input_tokens, final_message.usage.output_tokens
                    ),
                )

                yield ChatCompletionChunk(
                    content="",
                    is_complete=True,
                    tool_call_chunk=None,
                    usage=usage_data,
                )

        except AnthropicAuthError as e:
            raise AuthenticationError(f"Anthropic authentication failed: {str(e)}") from e
        except AnthropicRateLimitError as e:
            raise RateLimitError(f"Anthropic rate limit exceeded: {str(e)}") from e
        except BadRequestError as e:
            raise InvalidRequestError(f"Anthropic request is invalid: {str(e)}") from e
        except APIError as e:
            raise BaseChatCompletionError(f"Anthropic API error: {str(e)}") from e
        except (AuthenticationError, RateLimitError, InvalidRequestError, BaseChatCompletionError):
            raise
        except Exception as e:
            raise BaseChatCompletionError(f"Unexpected error: {str(e)}") from e

    def _validate_request_kwargs(self, kwargs: Dict[str, Any]) -> None:
        """Validate parameters that are no longer supported by the current Messages API."""
        unsupported = {"temperature", "top_p", "top_k"}.intersection(kwargs)
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise InvalidRequestError(
                f"The Anthropic Messages API version used by anthropic>=2.0.0 no longer "
                f"supports {names}. Remove these parameters from the request."
            )

    def _combine_system_messages(self, system_messages: List[Dict[str, Any]]) -> Optional[str]:
        """Combine system messages into the single system parameter supported by Anthropic."""
        if not system_messages:
            return None
        contents = []
        for message in system_messages:
            content = message.get("content", "")
            if isinstance(content, str):
                contents.append(content)
            elif isinstance(content, list):
                contents.extend(
                    block["text"]
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                )
        return "\n\n".join(contents) or None

    def _convert_messages_to_anthropic_format(
        self, messages: List[Message]
    ) -> List[Dict[str, Any]]:
        """
        Convert internal Message objects to Anthropic API format.

        Args:
            messages: List of internal Message objects

        Returns:
            List of messages in Anthropic API format
        """
        from ..messages import MultiModalMessage, ToolMessage

        api_messages: List[Dict[str, Any]] = []

        for msg in messages:
            if msg.role == "system":
                # System messages are extracted into Anthropic's separate system parameter.
                api_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, MultiModalMessage):
                # Handle multimodal messages
                content_parts = []

                if msg.content and msg.content.strip():
                    content_parts.append({"type": "text", "text": msg.content})

                if msg.is_image() and (msg.data is not None or msg.media_url):
                    if msg.data is not None:
                        # Use base64 data
                        base64_data = msg.to_base64()
                        content_parts.append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": msg.mime_type,
                                    "data": base64_data,
                                },
                            }
                        )
                    elif msg.media_url:
                        # Use a URL source for images hosted online.
                        content_parts.append(
                            {
                                "type": "image",
                                "source": {"type": "url", "url": msg.media_url},
                            }
                        )

                api_messages.append(
                    {
                        "role": msg.role,
                        "content": content_parts if content_parts else msg.content,
                    }
                )
            elif isinstance(msg, AssistantMessage) and msg.tool_calls:
                # Handle assistant messages with tool calls
                content_blocks = []

                if msg.content:
                    content_blocks.append({"type": "text", "text": msg.content})

                for tc in msg.tool_calls:
                    content_blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.call_id,
                            "name": tc.tool_name,
                            "input": tc.parameters,
                        }
                    )

                api_messages.append({"role": "assistant", "content": content_blocks})
            elif isinstance(msg, ToolMessage):
                # Anthropic expects tool results in a user message, associated with tool_use_id.
                tool_result: Dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": msg.tool_call_id,
                    "content": msg.content,
                }
                if not msg.success:
                    tool_result["is_error"] = True
                api_messages.append({"role": "user", "content": [tool_result]})
            else:
                api_messages.append({"role": msg.role, "content": msg.content})

        return api_messages

    def _convert_tools_to_anthropic_format(
        self, tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Convert OpenAI-style tool definitions to Anthropic format.

        Args:
            tools: List of OpenAI-format tool definitions

        Returns:
            List of Anthropic-format tool definitions
        """
        anthropic_tools = []

        for tool in tools:
            if tool.get("type") == "function":
                function = tool.get("function", {})
                anthropic_tools.append(
                    {
                        "name": function.get("name"),
                        "description": function.get("description", ""),
                        "input_schema": self._make_schema_compatible(
                            function.get("parameters", {"type": "object", "properties": {}})
                        ),
                    }
                )

        return anthropic_tools

    def _make_schema_compatible(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Modify schema to be compatible with Anthropic Structured Outputs requirements.

        Args:
            schema: The JSON schema to make compatible

        Returns:
            Modified schema with Anthropic compatibility fixes
        """
        # The SDK transformer handles unsupported JSON Schema constraints and nested objects.
        # Keep the local fallback for environments where the optional SDK is not installed.
        if transform_schema is not None:
            return transform_schema(schema)

        # Make a deep copy to avoid modifying nested structures in the original schema.
        compatible_schema = json.loads(json.dumps(schema))

        def update_schema(current_schema: Dict[str, Any]) -> None:
            if current_schema.get("type") == "object":
                current_schema["additionalProperties"] = False
                for prop_schema in current_schema.get("properties", {}).values():
                    if isinstance(prop_schema, dict):
                        update_schema(prop_schema)

            if current_schema.get("type") == "array":
                items = current_schema.get("items")
                if isinstance(items, dict):
                    update_schema(items)

            for key in ("anyOf", "oneOf", "allOf"):
                variants = current_schema.get(key)
                if isinstance(variants, list):
                    for variant in variants:
                        if isinstance(variant, dict):
                            update_schema(variant)

            definitions = current_schema.get("$defs", current_schema.get("definitions", {}))
            if isinstance(definitions, dict):
                for definition in definitions.values():
                    if isinstance(definition, dict):
                        update_schema(definition)

        update_schema(compatible_schema)
        return compatible_schema

    def _estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Estimate the cost of the API call based on token usage.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Estimated cost in USD
        """
        # Base per-token rates, from the published per-MTok prices. Verify these rates
        # against https://platform.claude.com/docs/en/about-claude/pricing before release.
        # Ordered most-specific prefix first because model prefixes can overlap.
        # This is a base-rate estimate only. It does not model prompt-caching multipliers,
        # the Batch API discount, fast mode, or the data-residency multiplier.
        M = 1_000_000
        pricing: List[Tuple[str, Dict[str, float]]] = [
            ("claude-fable-5", {"input": 10 / M, "output": 50 / M}),
            ("claude-opus-5", {"input": 5 / M, "output": 25 / M}),
            ("claude-opus-4-8", {"input": 5 / M, "output": 25 / M}),
            ("claude-opus-4-7", {"input": 5 / M, "output": 25 / M}),
            ("claude-opus-4-6", {"input": 5 / M, "output": 25 / M}),
            ("claude-opus-4-5", {"input": 5 / M, "output": 25 / M}),
            ("claude-opus-4", {"input": 15 / M, "output": 75 / M}),
            ("claude-sonnet-5", {"input": 2 / M, "output": 10 / M}),
            ("claude-sonnet-4", {"input": 3 / M, "output": 15 / M}),
            ("claude-haiku-4-5", {"input": 1 / M, "output": 5 / M}),
            # Legacy IDs, retained for older configs and saved runs.
            ("claude-3-5-sonnet", {"input": 3 / M, "output": 15 / M}),
            ("claude-3-5-haiku", {"input": 0.80 / M, "output": 4 / M}),
            ("claude-3-opus", {"input": 15 / M, "output": 75 / M}),
            ("claude-3-sonnet", {"input": 3 / M, "output": 15 / M}),
            ("claude-3-haiku", {"input": 0.25 / M, "output": 1.25 / M}),
        ]

        model_pricing = None
        for model_prefix, prices in pricing:
            if self.model.startswith(model_prefix):
                model_pricing = prices
                break

        if model_pricing is None:
            # Fall back to mid-tier Sonnet rates, but say so. Silently pricing an unknown
            # model produces a plausible number that is wrong, which is worse than an estimate
            # that is explicitly marked as potentially inaccurate.
            logger.warning(
                "No pricing entry for model %r; estimating at Sonnet 4.5 rates "
                "($3/$15 per MTok). Reported cost may be inaccurate.",
                self.model,
            )
            model_pricing = {"input": 3 / M, "output": 15 / M}

        input_cost = input_tokens * model_pricing["input"]
        output_cost = output_tokens * model_pricing["output"]

        return input_cost + output_cost

    def _to_config(self) -> AnthropicChatCompletionClientConfig:
        """Convert client to configuration for serialization."""
        base_url = getattr(self.client, "base_url", None)

        return AnthropicChatCompletionClientConfig(
            model=self.model,
            api_key=self.api_key,
            base_url=str(base_url) if base_url else None,
            config=self.config,
        )

    @classmethod
    def _from_config(
        cls, config: AnthropicChatCompletionClientConfig
    ) -> "AnthropicChatCompletionClient":
        """Create client from configuration.

        Args:
            config: Client configuration
        """
        return cls(
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
            **config.config,
        )
