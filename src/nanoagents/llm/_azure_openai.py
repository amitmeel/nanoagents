"""
Azure OpenAI BaseChatCompletionClient implementation.

This module provides integration with Azure OpenAI API using the official
openai Python client library.
"""

import json
import time
from copy import deepcopy
from typing import Any, AsyncGenerator, Dict, List, Optional, Type

from pydantic import BaseModel, Field, ValidationError

try:
    from openai import (
        APIError,
        AsyncAzureOpenAI,
        BadRequestError,
        ContentFilterFinishReasonError,
        LengthFinishReasonError,
    )
    from openai import AuthenticationError as OpenAIAuthError
    from openai import RateLimitError as OpenAIRateLimitError
    from openai.types.chat import ChatCompletion
    from openai.types.chat.chat_completion import Choice
    from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
    from openai.types.completion_usage import CompletionUsage
except ImportError as e:
    raise ImportError(
        'OpenAI library not installed. Please install with: pip install "openai>=1.100.2"'
    ) from e

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


class AzureOpenAIChatCompletionClientConfig(BaseModel):
    """Configuration for AzureOpenAIChatCompletionClient serialization."""

    model: str = "gpt-4.1-mini"
    azure_endpoint: Optional[str] = None
    api_key: Optional[str] = None
    api_version: str = "2024-10-21"
    azure_deployment: Optional[str] = None
    temperature: Optional[float] = None
    config: Dict[str, Any] = Field(default_factory=dict)


class AzureOpenAIChatCompletionClient(
    Component[AzureOpenAIChatCompletionClientConfig], BaseChatCompletionClient
):
    """
    Azure OpenAI implementation of BaseChatCompletionClient.

    Supports Azure OpenAI deployments with function calling capabilities.
    """

    component_config_schema = AzureOpenAIChatCompletionClientConfig
    component_type = "model_client"
    component_provider_override = "nanoagents.llm.AzureOpenAIChatCompletionClient"

    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        azure_endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        api_version: str = "2024-10-21",
        azure_deployment: Optional[str] = None,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ):
        """
        Initialize Azure Azure OpenAI client.

        Args:
            model: Azure OpenAI model name. For structured output, use a deployment that supports Structured Outputs.
                   For vision (MultiModalMessage with images), use a deployment that supports image inputs.
            azure_endpoint: Azure OpenAI endpoint URL (e.g., "https://your-resource.openai.azure.com/")
            api_key: Azure OpenAI API key (will use AZURE_OPENAI_API_KEY env var if not provided)
            api_version: Azure OpenAI API version
            azure_deployment: Deployment name in Azure (defaults to model)
            temperature: Default temperature for completions (passed on each request)
            **kwargs: Additional Azure OpenAI client configuration
        """
        super().__init__(model, api_key, **kwargs)

        # Validate required parameters
        if not azure_endpoint:
            raise ValueError("azure_endpoint is required for Azure OpenAI client")

        # Use deployment name if provided, otherwise use model name
        self.azure_deployment = azure_deployment or model
        self.azure_endpoint = azure_endpoint
        self.api_version = api_version
        self.temperature = temperature

        self.client = AsyncAzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version,
            **kwargs,
        )

    async def create(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        output_format: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> ChatCompletionResult:
        """
        Make a single OpenAI API call.

        Args:
            messages: List of messages to send
            tools: Optional function definitions for tool calling
            output_format: Optional Pydantic model for structured output
            **kwargs: Additional OpenAI parameters (temperature, max_completion_tokens, etc.)

        Returns:
            Standardized chat completion result
        """
        try:
            start_time = time.perf_counter()

            # Convert messages to Azure OpenAI format
            api_messages = self._convert_messages_to_api_format(messages)

            # Prepare request parameters
            request_params = {
                "model": self.azure_deployment,
                "messages": api_messages,
                **kwargs,
            }

            # Add default temperature if set and not overridden by kwargs
            if self.temperature is not None and "temperature" not in request_params:
                request_params["temperature"] = self.temperature

            # Add tools if provided
            if tools:
                request_params["tools"] = tools
                request_params["tool_choice"] = "auto"

            # Add structured output if requested
            # Prefer the SDK parsing helper for Pydantic structured output. The helper
            # owns schema conversion and response parsing. It requires strict function
            # tools, however, so retain the existing response_format path for callers
            # that combine output_format with non-strict tools.
            use_sdk_parse = output_format is not None and (
                not tools or self._tools_support_strict_parsing(tools)
            )
            structured_output = None

            if use_sdk_parse:
                if not callable(getattr(self.client.chat.completions, "parse", None)):
                    raise BaseChatCompletionError(
                        "The installed OpenAI SDK does not support chat.completions.parse; "
                        "upgrade the SDK to use Pydantic structured output"
                    )

                try:
                    # Make API call
                    response = await self.client.chat.completions.parse(
                        model=self.azure_deployment,
                        messages=api_messages,
                        response_format=output_format,
                        **{
                            key: value
                            for key, value in request_params.items()
                            if key not in {"model", "messages", "response_format", "stream"}
                        },
                    )
                except (ValidationError, json.JSONDecodeError) as e:
                    raise BaseChatCompletionError(
                        f"Azure OpenAI response could not be parsed as {output_format.__name__}: {e}"
                    ) from e
            else:
                if output_format is not None:
                    # Compatibility path: the SDK parse helper rejects non-strict tools.
                    # Keep the existing ability to combine ordinary function tools with
                    # structured output, and validate the resulting content ourselves.
                    schema = self._make_schema_compatible(output_format.model_json_schema())
                    request_params["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": schema.get("title", output_format.__name__),
                            "description": schema.get(
                                "description",
                                f"Structured output for {output_format.__name__}",
                            ),
                            "strict": True,
                            "schema": schema,
                        },
                    }

                # Make API call
                response = await self.client.chat.completions.create(**request_params)

            duration_ms = int((time.perf_counter() - start_time) * 1000)

            if not response.choices:
                raise BaseChatCompletionError("Azure OpenAI returned a completion without choices")

            # This result type represents a single assistant response.
            choice: Choice = response.choices[0]
            assistant_content = choice.message.content or ""

            # Handle refusals explicitly when structured output was requested. A refusal
            # is not a valid instance of the requested Pydantic model.
            if output_format is not None and choice.message.refusal:
                raise BaseChatCompletionError(
                    f"Azure OpenAI refused to produce {output_format.__name__}: {choice.message.refusal}"
                )

            # Handle tool calls if present
            tool_calls = []
            if choice.message.tool_calls:
                for tc in choice.message.tool_calls:
                    # Only handle function tool calls, skip other tool call types
                    if tc.type == "function":
                        # Type narrowing: tc is now ChatCompletionMessageFunctionToolCall
                        function_call = tc.function
                        try:
                            parameters = json.loads(function_call.arguments or "{}")
                        except json.JSONDecodeError as e:
                            raise BaseChatCompletionError(
                                f"Azure OpenAI returned invalid JSON for tool call "
                                f"{function_call.name}: {e}"
                            ) from e
                        if not isinstance(parameters, dict):
                            raise BaseChatCompletionError(
                                f"Azure OpenAI returned non-object arguments for tool call "
                                f"{function_call.name}"
                            )
                        tool_calls.append(
                            ToolCallRequest(
                                tool_name=function_call.name,
                                parameters=parameters,
                                call_id=tc.id,
                            )
                        )

            assistant_message = AssistantMessage(
                content=choice.message.refusal or assistant_content,
                source="llm",  # Temporary source, will be overwritten by agent
                tool_calls=tool_calls if tool_calls else None,
            )

            # Read the validated Pydantic model returned by the SDK parser, or validate
            # the content on the compatibility path used with non-strict tools.
            if output_format is not None:
                if use_sdk_parse:
                    structured_output = getattr(choice.message, "parsed", None)
                else:
                    try:
                        structured_output = output_format.model_validate_json(assistant_content)
                    except (ValidationError, ValueError, json.JSONDecodeError) as e:
                        raise BaseChatCompletionError(
                            f"Azure OpenAI response could not be parsed as {output_format.__name__}: {e}"
                        ) from e

                if structured_output is None:
                    raise BaseChatCompletionError(
                        f"Azure OpenAI did not return a validated structured output for "
                        f"{output_format.__name__}"
                    )

            # Create usage statistics
            usage_data = response.usage
            usage = Usage(
                duration_ms=duration_ms,
                llm_calls=1,
                tokens_input=usage_data.prompt_tokens if usage_data else 0,
                tokens_output=usage_data.completion_tokens if usage_data else 0,
                tool_calls=len(tool_calls),
                cost_estimate=self._estimate_cost(usage_data) if usage_data else None,
            )

            return ChatCompletionResult(
                message=assistant_message,
                usage=usage,
                model=response.model,
                finish_reason=choice.finish_reason or "stop",
                structured_output=structured_output,
            )

        except OpenAIAuthError as e:
            raise AuthenticationError(f"Azure OpenAI authentication failed: {str(e)}") from e
        except OpenAIRateLimitError as e:
            raise RateLimitError(f"Azure OpenAI rate limit exceeded: {str(e)}") from e
        except BadRequestError as e:
            raise InvalidRequestError(f"Azure OpenAI request was invalid: {str(e)}") from e
        except (LengthFinishReasonError, ContentFilterFinishReasonError) as e:
            raise BaseChatCompletionError(
                f"Azure OpenAI could not complete the requested response: {str(e)}"
            ) from e
        except APIError as e:
            raise BaseChatCompletionError(f"Azure OpenAI API error: {str(e)}") from e
        except (AuthenticationError, RateLimitError, InvalidRequestError, BaseChatCompletionError):
            raise
        except Exception as e:
            raise BaseChatCompletionError(f"Unexpected error: {str(e)}") from e

    async def create_stream(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        output_format: Optional[Type[BaseModel]] = None,
        stream_options: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        """
        Make a streaming OpenAI API call.

        Args:
            messages: List of messages to send
            tools: Optional function definitions for tool calling
            output_format: Optional Pydantic model for structured output (not supported by this stream result type)
            stream_options: Stream options (defaults to {"include_usage": True} to enable token tracking)
            **kwargs: Additional OpenAI parameters

        Yields:
            ChatCompletionChunk objects with incremental response data (final chunk includes usage if stream_options.include_usage=true)
        """
        try:
            # ChatCompletionChunk has no field for a parsed Pydantic result. Reject structured
            # output rather than emitting partial JSON as if it were a validated result.
            if output_format is not None:
                raise InvalidRequestError(
                    "Structured output is not supported by create_stream because "
                    "ChatCompletionChunk cannot carry a validated structured result; use create"
                )

            # Convert messages to Azure OpenAI format
            api_messages = self._convert_messages_to_api_format(messages)

            # Prepare request parameters
            request_params = {
                "model": self.azure_deployment,
                "messages": api_messages,
                "stream": True,
                **kwargs,
            }

            # Add default temperature if set and not overridden by kwargs
            if self.temperature is not None and "temperature" not in request_params:
                request_params["temperature"] = self.temperature

            # Add stream options, default to including usage for token tracking
            if stream_options is None:
                stream_options = {"include_usage": True}
            if stream_options:
                request_params["stream_options"] = stream_options

            # Add tools if provided
            if tools:
                request_params["tools"] = tools
                request_params["tool_choice"] = "auto"

            # Make streaming API call
            stream = await self.client.chat.completions.create(**request_params)

            tool_call_chunks: Dict[Any, Dict[str, Any]] = {}
            completion_emitted = False
            finish_reason = None

            # Close the HTTP stream even if the consumer cancels iteration or raises.
            async with stream:
                async for chunk in stream:
                    # Handle usage-only chunk (comes after the finish_reason chunk)
                    if chunk.usage is not None and not chunk.choices:
                        if finish_reason == "length":
                            raise BaseChatCompletionError(
                                "Azure OpenAI streaming response was truncated because the token limit was reached"
                            )
                        if finish_reason == "content_filter":
                            raise BaseChatCompletionError(
                                "Azure OpenAI streaming response was stopped by the content filter"
                            )
                        if finish_reason is None:
                            raise BaseChatCompletionError(
                                "Azure OpenAI streaming response ended without a terminal finish_reason"
                            )
                        # This is the final usage-only chunk
                        usage_data = Usage(
                            duration_ms=0,  # Duration tracked at agent level
                            llm_calls=1,
                            tokens_input=chunk.usage.prompt_tokens,
                            tokens_output=chunk.usage.completion_tokens,
                            tool_calls=0,  # Tool calls tracked at agent level
                            cost_estimate=self._estimate_cost(chunk.usage),
                        )
                        yield ChatCompletionChunk(
                            content="",
                            is_complete=True,
                            tool_call_chunk=None,
                            usage=usage_data,
                        )
                        completion_emitted = True
                        break

                    if not chunk.choices:
                        continue

                    chunk_choice: ChunkChoice = chunk.choices[0]
                    delta = chunk_choice.delta

                    if chunk_choice.finish_reason:
                        finish_reason = chunk_choice.finish_reason

                    # Handle content chunks
                    if delta.content:
                        yield ChatCompletionChunk(
                            content=delta.content,
                            is_complete=False,
                            tool_call_chunk=None,
                        )

                    # Handle tool call chunks
                    if delta.tool_calls:
                        for tc_delta in delta.tool_calls:
                            # Use index as the primary key, fallback to id
                            index = getattr(tc_delta, "index", None)
                            call_id = tc_delta.id

                            # Use index as the tracking key when available
                            tracking_key = index if index is not None else call_id
                            if tracking_key is None:
                                continue

                            # Initialize tool call entry if not exists
                            if tracking_key not in tool_call_chunks:
                                tool_call_chunks[tracking_key] = {
                                    "id": call_id,
                                    "function": {"name": "", "arguments": ""},
                                }

                            # Update the call_id if we have one (first chunk for this index)
                            if call_id:
                                tool_call_chunks[tracking_key]["id"] = call_id

                            # Update function data
                            if tc_delta.function:
                                if tc_delta.function.name:
                                    tool_call_chunks[tracking_key]["function"]["name"] += (
                                        tc_delta.function.name
                                    )
                                if tc_delta.function.arguments:
                                    tool_call_chunks[tracking_key]["function"]["arguments"] += (
                                        tc_delta.function.arguments
                                    )

                            # Yield a snapshot so later chunks do not mutate previously yielded data
                            yield ChatCompletionChunk(
                                content="",
                                is_complete=False,
                                tool_call_chunk=deepcopy(tool_call_chunks[tracking_key]),
                            )

                    # Check if stream is complete (has finish_reason)
                    # NOTE: Usage will come in a SEPARATE chunk after this one
                    # if chunk_choice.finish_reason:
                    #     # Don't yield completion yet - wait for usage chunk
                    #     continue

            # A stream without a terminal finish reason may have ended prematurely.
            if finish_reason is None:
                raise BaseChatCompletionError(
                    "Azure OpenAI streaming response ended without a terminal finish_reason"
                )

            # A truncated or filtered stream must not look like a successful completion.
            if finish_reason == "length":
                raise BaseChatCompletionError(
                    "Azure OpenAI streaming response was truncated because the token limit was reached"
                )
            if finish_reason == "content_filter":
                raise BaseChatCompletionError(
                    "Azure OpenAI streaming response was stopped by the content filter"
                )

            if not completion_emitted:
                yield ChatCompletionChunk(
                    content="",
                    is_complete=True,
                    tool_call_chunk=None,
                    usage=None,
                )

        except OpenAIAuthError as e:
            raise AuthenticationError(f"Azure OpenAI authentication failed: {str(e)}") from e
        except OpenAIRateLimitError as e:
            raise RateLimitError(f"Azure OpenAI rate limit exceeded: {str(e)}") from e
        except BadRequestError as e:
            raise InvalidRequestError(f"Azure OpenAI request was invalid: {str(e)}") from e
        except APIError as e:
            raise BaseChatCompletionError(f"Azure OpenAI API error: {str(e)}") from e
        except (AuthenticationError, RateLimitError, InvalidRequestError, BaseChatCompletionError):
            raise
        except Exception as e:
            raise BaseChatCompletionError(f"Unexpected error: {str(e)}") from e

    def _estimate_cost(self, usage: CompletionUsage) -> float:
        """
        Estimate the cost of the API call based on token usage.

        Note: These are approximate rates and may not reflect current pricing.

        Args:
            usage: OpenAI usage object

        Returns:
            Estimated cost in USD
        """
        # Preserve the original pricing behavior for backward compatibility.
        pricing = {
            "gpt-4": {"input": 0.03 / 1000, "output": 0.06 / 1000},
            "gpt-4-turbo": {"input": 0.01 / 1000, "output": 0.03 / 1000},
            "gpt-3.5-turbo": {"input": 0.0005 / 1000, "output": 0.0015 / 1000},
        }

        model_name = f"{self.model} {self.azure_deployment}".lower()
        model_key = next((key for key in pricing if key in model_name), None)
        model_pricing = pricing.get(model_key, pricing["gpt-4"])
        input_cost = usage.prompt_tokens * model_pricing["input"]
        output_cost = usage.completion_tokens * model_pricing["output"]
        return input_cost + output_cost

    def _make_schema_compatible(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Modify a Pydantic JSON schema for Azure OpenAI Structured Outputs strict mode.

        Handles: $defs/$ref, nested objects, array items, and Optional fields.
        Azure OpenAI strict mode requires all properties in 'required' and
        additionalProperties=false on every object.

        Note: This compatibility helper is retained only for the non-strict-tools path,
        where chat.completions.parse() cannot be used without changing caller-provided tools.
        """
        compatible_schema = deepcopy(schema)

        # Process $defs (Pydantic nested models generate these)
        if "$defs" in compatible_schema:
            compatible_schema["$defs"] = {
                name: self._make_schema_compatible(definition)
                for name, definition in compatible_schema["$defs"].items()
            }

        if compatible_schema.get("type") == "object":
            compatible_schema["additionalProperties"] = False
            properties = compatible_schema.get("properties", {})

            # Azure OpenAI requires ALL properties in 'required'
            compatible_schema["required"] = list(properties.keys())
            for property_name, property_schema in properties.items():
                if isinstance(property_schema, dict):
                    compatible_schema["properties"][property_name] = self._make_schema_compatible(
                        property_schema
                    )

        if compatible_schema.get("type") == "array":
            items = compatible_schema.get("items")
            if isinstance(items, dict):
                compatible_schema["items"] = self._make_schema_compatible(items)

        return compatible_schema

    @staticmethod
    def _tools_support_strict_parsing(tools: List[Dict[str, Any]]) -> bool:
        """Return whether all supplied function tools are declared in strict mode."""
        for tool in tools:
            if tool.get("type") != "function":
                return False
            function = tool.get("function")
            if not isinstance(function, dict) or function.get("strict") is not True:
                return False
        return True

    async def close(self) -> None:
        """Close the underlying asynchronous Azure OpenAI client."""
        await self.client.close()

    def _to_config(self) -> AzureOpenAIChatCompletionClientConfig:
        """Convert client to configuration for serialization."""
        return AzureOpenAIChatCompletionClientConfig(
            model=self.model,
            azure_endpoint=self.azure_endpoint,
            api_key=self.api_key,
            api_version=self.api_version,
            azure_deployment=self.azure_deployment,
            temperature=self.temperature,
            config=self.config,
        )

    @classmethod
    def _from_config(
        cls, config: AzureOpenAIChatCompletionClientConfig
    ) -> "AzureOpenAIChatCompletionClient":
        """Create client from configuration.

        Args:
            config: Client configuration
        """
        return cls(
            model=config.model,
            azure_endpoint=config.azure_endpoint,
            api_key=config.api_key,
            api_version=config.api_version,
            azure_deployment=config.azure_deployment,
            temperature=config.temperature,
            **config.config,
        )
