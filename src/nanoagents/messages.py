"""Core messages types for agent communication using pydantic models.

This module defines the structured message types that agents used to communicate
with each other and with LLM, following the OpenAI API format."""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union, TYPE_CHECKING

from pydantic import BaseModel, Field, model_validator, ConfigDict

class BaseMessage(BaseModel):
    content: str = Field(..., description="The message content")
    source: str = Field(..., description="Source of the message (agent name, system, user, etc.)")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the message was created")

    model_config = ConfigDict(frozen=True)

    def __str__(self) -> str:
        """Returns a user friendly string representation."""
        time_str = self.timestamp.strftime("%H:%M:%S")
        return f"[{self.source}] {time_str} | {self.content}"
    
    def __repr__(self) -> str:
        """Returns a developer friendly representation."""
        class_name = self.__class__.__name__
        return f"{class_name}(source='{self.source}', content='{self.content[:50]}...', timestamp='{self.timestamp}')"


class SystemMessage(BaseMessage):
    """System message containing the instructions/role definition for the agent."""
    role: Literal["system"] = Field(default="system", description="Message role")


class UserMessage(BaseMessage):
    """User message containing input from human or external system."""
    role: Literal["user"] = Field(default="user", description="Message role")
    name: Optional[str] = Field(default=None, description="Optional name of the user")


class ToolCallRequest(BaseModel):
    """Structured representation of an LLM's tool call request."""
    tool_name: str = Field(..., description="Name of th tool to call")
    parameters: Dict[str, Any] = Field(..., description="Arguments for the tool")
    call_id: str = Field(..., description="Unique identifier for this tool call")

    model_config = ConfigDict(frozen=True)


class AssistantMessage(BaseMessage):
    """Assistant message containing response from the agent/LLM."""

    role: Literal["assistant"] = Field(default="assistant", description="Message role")
    tool_calls: Optional[List[ToolCallRequest]] = Field(
        default=None, description="Tool calls made by the assistant."
        )
    structured_content: Optional[BaseModel] = Field(
        default=None, description="Strucutred data when output_format is used."
    )
    usage: Optional["Usage"] = Field(default=None, description="Token usage for this LLM call.")

    def __str__(self) -> str:
        """Returns a user friendly string representation."""
        time_str = self.timestamp.strftime("%H:%M:%S")

        if self.tool_calls:
            "Show tool calls information"
            tool_info = ", ".join(
                [
                    f"{tc.tool_name}({', '.join(f'{k}={v}' for k,v in tc.parameters.items())})"
                    for tc in self.tool_calls
                ]
            )
            if self.content and self.content.strip():
                return(
                    f"[{self.source}] {time_str} | {self.content} [tools: {tool_info}]"
                )
            else:
                return f"[{self.source}] {time_str} | [calling tools: {tool_info}]"
        else:
            f"[{self.source}] {time_str} | {self.content}"


class ToolMessage(BaseMessage):
    """Tool message containing result from tool execution."""

    role: Literal["tool"] = Field(default="tool", description="Message role")
    tool_call_id: str = Field(
        ..., description="ID of the tool call this is responding to"
    )
    tool_name: str = Field(..., description="Name of the tool that was executed")
    success: bool = Field(..., description="Whether tool execution succeeded")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Tool specific metadata (e.g. sub-agent usage)"
    )
        
