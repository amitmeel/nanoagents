"""
Core data types and models for nanoagents framework using Pydantic.

This module defines all the structured types used throughout the framework
for type safety and data validation."""

from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING, Union

from pydantic import BaseModel, Field

from .messages import Message, ConfigDict


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
    cost_estimate: Optional[float] = Field(default=None, description="Estimated cost in USD")


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
    error: Optional[str] = Field(default=None, description="Error message if tool execution failed")
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
    usage: Usage = Field(..., description="Execution statistics and resource consumtion")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="When the Message was created"
    )
    finish_reason: str = Field(
        ..., description="Why the agent stopped: stop, approval_needed, max_iterations, error, cancelled"
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
            return content[:50] + '...' if len(content)>50 else content
        return "No messages"
    
    def __str__(self) -> str:
        """Returns a user friendly string representation with messages and usage."""
        # Concat all message str representations
        message_str = "\n".join(str(msg) for msg in self.messages)

        # Format duration
        duration_s = self.usage.duration_ms / 1000

        # Format tokens
        tokens_in =  (
            f"{self.usage.tokens_input/1000:.1f}k"
            if self.usage.tokens_input >= 1000
            else str(self.usage.tokens_input)
        )
        tokens_out =  (
            f"{self.usage.tokens_output/1000:.1f}k"
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

        usage_line = f"[usage] duration: {duration_s:.1f}s, tokens: in: {tokens_in}, out:{tokens_out}{cost_str}{approval_str}"

        return f"{message_str}\n\n{usage_line}"
    
    def __repr__(self) -> str:
        """Returns an unambiguous, developer friendly representation."""
        approval_info = f", approvals_needed={len(self.approval_requests)}" if self.needs_approval else ""
        return f"AgentResponse(source='{self.source}', messages={len(self.messages)}, finish_reason='{self.finish_reason}', usage={self.usage}{approval_info})"

    



