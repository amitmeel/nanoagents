"""
Core data types and models for nanoagents framework using Pydantic.

This module defines all the structured types used throughout the framework
for type safety and data validation.
"""

from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union

from pydantic import BaseModel, ConfigDict, Field

from .messages import Message

if TYPE_CHECKING:
    from .context import AgentContext, ToolApprovalRequest


class Usage(BaseModel):
    """Strucutred execution statistics and resource consumption."""

    duration_ms: int = Field(..., description="Total execution time in milliseconds")
    llm_calls: int = Field(default=0, description="Number of LLM API calls made")
    tokens_input: int = Field(default=0, description="Total input tokens consumed")
    tokens_output: int = Field(default=0, description="Total output tokens generated")
    tool_calls: int = Field(default=0, description="Number of tool executions")
    memory_operations: int = Field(
        default=0, description="Number of memory read/write operations"
    )
    cost_estimate: Optional[float] = Field(
        default=None, description="Estimated cost in USD"
    )

    def __add__(self, other: "Usage") -> "Usage":
        """Aggregate usage statistics from multiple sources."""
        return Usage(
            duration_ms=max(
                self.duration_ms, other.duration_ms
            ),  # Max for parellel execution
            llm_calls=self.llm_calls + other.llm_calls,
            tokens_input=self.tokens_input + other.tokens_input,
            tool_calls=self.tool_calls + other.tool_calls,
            memory_operations=self.memory_operations + other.memory_operations,
            cost_estimate=(self.cost_estimate or 0) + (other.cost_estimate or 0)
            or None,
        )

    model_config = ConfigDict(frozen=True)


class ToolResult(BaseModel):
    """Standardized tool execution result."""

    success: bool = Field(..., description="Whether tool execution succeeded")
    result: str = Field(..., description="The actual result data")
    error: Optional[str] = Field(
        default=None, description="Error message if tool execution failed"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Execution time, etc."
    )

    model_config = ConfigDict(frozen=True)


class AgentResponse(BaseModel):
    """Final result from Agent.run() containing context with all state and messages."""

    context: Optional["AgentContext"] = Field(
        default=None, description="Complete context with messages and state"
    )
    source: str = Field(..., description="Source agent that generated this response")
    usage: Usage = Field(
        ..., description="Execution statistics and resource consumtion"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now, description="When the Message was created"
    )
    finish_reason: str = Field(
        ...,
        description=(
            "Why the agent stopped: stop, approval_needed, max_iterations, ",
            "error, cancelled",
        ),
    )

    # Allow modification for context updates
    model_config = ConfigDict(frozen=False)

    @property
    def messages(self) -> List[Message]:
        """Access messages through context."""
        return self.context.messages if self.context else []

    @property
    def needs_approval(self) -> bool:
        """Check if response is waiting for approvals."""
        return self.context.waiting_for_approval if self.context else False

    @property
    def approval_requests(self) -> List["ToolApprovalRequest"]:
        """Get pending approval requests"""
        return self.context.pending_approval_requests if self.context else []

    @property
    def final_content(self) -> str:
        """Get the content of last message, truncated for display"""
        if self.messages:
            content = self.messages[-1].content
            return content[:50] + "..." if len(content) > 50 else content
        return "No messages"

    def __str__(self) -> str:
        """Returns a user friendly string representation with messages and usage."""
        # Concat all message str representations
        message_str = "\n".join(str(msg) for msg in self.messages)

        # Format duration
        duration_s = self.usage.duration_ms / 1000

        # Format tokens
        tokens_in = (
            f"{self.usage.tokens_input / 1000:.1f}k"
            if self.usage.tokens_input >= 1000
            else str(self.usage.tokens_input)
        )
        tokens_out = (
            f"{self.usage.tokens_output / 1000:.1f}k"
            if self.usage.tokens_output >= 1000
            else str(self.usage.tokens_output)
        )

        # Format cost if available
        cost_str = (
            f", cost: ${self.usage.cost_estimate:.4f}"
            if self.usage.cost_estimate
            else ""
        )

        # Add approval status if needed
        if self.needs_approval:
            approval_str = f" | ⚠️ {len(self.approval_requests)} approvals needed"
        else:
            approval_str = f" | finish: {self.finish_reason}"

        usage_line = (
            f"[usage] duration: {duration_s:.1f}s, "
            f"tokens: in: {tokens_in}, out: {tokens_out}"
            f"{cost_str}{approval_str}"
        )
        return f"{message_str}\n\n{usage_line}"

    def __repr__(self) -> str:
        """Returns an unambiguous, developer friendly representation."""
        approval_info = (
            f", approvals_needed={len(self.approval_requests)}"
            if self.needs_approval
            else ""
        )
        return (
            f"AgentResponse("
            f"source='{self.source}', "
            f"messages={len(self.messages)}, "
            f"finish_reason='{self.finish_reason}', "
            f"usage={self.usage}"
            f"{approval_info}"
            ")"
        )

class ChatCompletionResult(BaseModel):
    """Standardized LLM response from BaseChatCompletionClient."""

    message: "AssistantMessage" = Field(..., description="The LLM response")
    usage: Usage = Field(..., description="Token COnsumption and timing metrics")
    model: str = Field(..., description="Actual Model used for the request")
    finish_reason: str = Field(
        ..., description="Completion status: stop, tool_calls, length, error"
    )
    structured_output: Optional[BaseModel] = Field(
        default=None,
        description="Parsed structured output when output_format is specified"
    )

    model_config = ConfigDict(frozen=True)


class ChatCompletionChunk(BaseModel):
    """Streaming chunk response from BaseChatCompletionClient."""

    content: str = Field(..., description="Partial text content from stream")
    is_complete: bool = Field(..., decription="whether this is the final chunk")
    tool_call_chunk: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Partial tool call data"
    )
    usage: Optional["Usage"] = Field(
        default=None,
        description=(
            "Token usage statistics ("
            "only present in final chunk when stream_options.include_usage=true)"
        )
    )

    model_config =ConfigDict(frozen=True)


# Base event class for Streaming
class BaseEvent(BaseModel):
    """Abstract base class for all agent events"""

    timestamp: datetime = Field(
        default_factory=datetime.now, description="When the event occured"
    )
    source: str = Field(
        ..., description="Source of the event (agent name, system, orchestrator, etc.)"
    )
    event_type: str = Field(..., description="Type of event")

    model_config = ConfigDict(frozen=True)


    def __str__(self) -> str:
        """Returns a user friendly string representation."""
        time_str = self.timestamp.strftime("%H:%M:%S")
        return f"[{self.source}] {time_str} | {self.event_type}"

    def __repr__(self) -> str:
        """Returns a developer friendly representation"""
        class_name = self.__class__.__name__
        return (
            f"{class_name}("
            f"event_type='{self.event_type}',"
            f"source='{self.source}',"
            f"timestamp='{self.timestamp}')"
        )
            

# Execution events
class TaskStartEvent(BaseEvent):
    """Event emitted when task processing begins."""

    event_type: str = Field(default="task_start", description="Event type identifier")
    task: str = Field(..., description="the task being started")


class TaskCompleteEvent(BaseEvent):
    """Event emitted when task processing ends."""

    event_type: str = Field(default="task_complete", description="Event type identifier")
    result: str = Field(..., description="The final task result")


class ModelCallEvent(BaseEvent):
    """Event emitted when LLM API call is initiated."""

    event_type: str = Field(default="model_call", description="Event type identifier")
    input_messages: Sequence[Message] = Field(
        ..., description="Messages sent to the model"
    )
    model: str = Field(..., description="Model being called")


class ModelResponseEvent(BaseEvent):
    """Event emitted when LLM response is received."""

    event_type: str = Field(default="model_response", description="Event type identifier")
    response: str = Field(..., description="The model's response")
    has_tool_calls: bool = Field(
        default=False, description="Whether response contains tool calls"
    )


class ModelStreamChunkEvent(BaseEvent):
    """Event emitted for each streaming chunk from LLM."""

    event_type: str = Field(
        default="model_stream_chunk", description="Event type identifier"
    )
    chunk: str = Field(..., description="Incremental text chunk")
    is_final: bool = Field(default=False, description="Whether this is the final chunk")


# Tool events
class ToolCallEvent(BaseEvent):
    """Event emitted when tool execution begins."""

    event_type: str = Field(default="tool_call", description="Event type identifier")
    tool_name: str = Field(..., description="Name of the tool being called")
    parameters: Dict[str, Any] = Field(..., description="Arguments passed to the tool")
    call_id: str = Field(..., description="Unique identifier for this tool call")

    def __str__(self) -> str:
        """Return a user friendly string representation with tool details"""
        time_str = self.timestamp.strftime("%H:%M:%S")
        params_str = ", ".join([f"{k}={v}" for k,v in self.parameters.items()])
        return f"[{self.source}] {time_str}| tool_call: {self.tool_name}({params_str})"


class ToolCallResponseEvent(BaseEvent):
    """Event emitted when tool execution completes."""

    event_type: str = Field(default="tool_call_response", description="Event type identifier")
    call_id: str = Field(..., description="Unique dentifier for this tool call")
    result: Optional[ToolResult] = Field(default=None, description="Tool execution result")

    def __str__(self) -> str:
        """Returns a user friendly string representation with result information."""
        time_str = self.timestamp.strftime("%H:%M:%S")
        if self.result:
            status = "✓" if self.result.success else "✗"
            result_preview = (
                str(self.result.result)[:50] + "..."
                if len(self.result.result) > 50 
                else str(self.result.result)
            )
            return (
                f"[{self.source}] {time_str} | tool_response: {status} {result_preview}"
            )
        else:
            return f"[{self.source}] {time_str} | tool_response: (no result)"


class ToolApprovalEvent(BaseEvent):
    """Event emitted when tool execution requires approval."""

    event_type: str = Field(
        default="tool_approval", description="Event type identifier"
    )
    approval_request: "ToolApprovalRequest" = Field(
        ..., description="The approval request details"
    )

    def __str__(self) -> str:
        """Retruns a user friendly string representation"""
        time_str = self.timestamp.strftime("%H:%M:%S")
        return f"[{self.source}] {time_str} | ⚠️ approval needed: {self.approval_request.tool_name}"


class ToolValidationEvent(BaseEvent):
    """Event emitted after tool parameters validation."""

    event_type: str = Field(
        default="tool_validation", description="Event type identifier"
    )
    tool_name: str = Field(..., description="Name of the tool being validated")
    is_valid: bool = Field(..., description="Whether parameters are valid")
    errors: Optional[List[str]] = Field(default=None, description="Validation error messages")


#Memory Events
class MemoryUpdateEvent(BaseEvent):
    """Event emitted when memory state changes."""

    event_type: str = Field(
        default="memory_update", description="Event type identifier"
    )
    operation: str = Field(
        ..., description="Type of memory operation: add, update or delete"
    )
    content_summary: str = Field(..., description="Summary of what was stored/updated")


class MemoryRetrievalEvent(BaseEvent):
    """Event emitted when memory content is accessed."""

    event_type: str = Field(
        default="memory_retrieval", description="Event type identifier"
    )
    query: str = Field(..., description="Query used to retrieve memories")
    results_count: int = Field(..., description="Number of memories retrieved")


# Error Events
class ErrorEvent(BaseEvent):
    """Event emitted for recoverable errors that terminate execution."""

    event_type: str = Field(
        default="error", description="Event type identifier"
    )
    error_message: str = Field(..., description="Description of the error")
    error_type: str = Field(..., description="Type/Category of the error")
    is_recoverable: bool = Field(
        default=False, description="Whether error can be reocvered from"
    )


class FatalErrorEvent(BaseEvent):
    """Event emitted for unrecoverable errors that terminate execution."""

    event_type: str = Field(
        default="error", description="Event type identifier"
    )
    error_message: str = Field(..., description="Description of the error")
    error_type: str = Field(..., description="Type/Category of the error")
    is_recoverable: bool = Field(
        default=False, description="Always False for fatal errors"
    )



# Union Type for all events
AgentEvent = Union[
    TaskStartEvent,
    TaskCompleteEvent,
    ModelCallEvent,
    ModelResponseEvent,
    ModelStreamChunkEvent,
    ToolCallEvent,
    ToolCallResponseEvent,
    ToolValidationEvent,
    MemoryUpdateEvent,
    MemoryRetrievalEvent,
    ErrorEvent,
    FatalErrorEvent

] 


# Fix forward references
from .context import AgentContext, ToolApprovalRequest
from .messages import AssistantMessage