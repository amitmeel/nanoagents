"""
Agent context management for nanoagents.

This module provides the AgentContext class that replaces the simple message history
list with a more structured and extensible context object.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .messages import AssistantMessage, Message, ToolCallRequest, UserMessage


class ToolApprovalResponse(BaseModel):
    """Response to a tool approval request."""

    request_id: str = Field(..., description="Approval reqest id")
    tool_call_id: str = Field(..., description="Tool call id which needed the approval")
    approved: bool = Field(..., description="Whether execution was approved")
    reason: Optional[str] = Field(default=None, description="Reason for approval/rejection")


class ToolApprovalRequest(BaseModel):
    """Request for user approval of a tool call."""

    request_id: str = Field(..., description="Unique approval request id")
    tool_call_id: str = Field(..., description="Tool call ID needing the approval")
    tool_name: str = Field(..., description="Name of the tool")
    parameters: Dict[str, Any] = Field(..., "Tool Parameters")
    original_tool_call: ToolCallRequest = Field(..., description="Original tool call object")

    def create_response(self, approved: bool, reason: Optional[str]=None) -> ToolApprovalResponse:
        """Create an approval response."""
        return ToolApprovalResponse(
            request_id=self.request_id,
            tool_call_id=self.tool_call_id,
            approved=approved,
            reason=reason
        )


class AgentContext(BaseModel):
    """
    Unified context object for agents.

    This replaces the simple message history list with a structured context 
    that can carry messages, metadata, shared state, and environment info.
    """

    messages: List[Message] = Field(
        default_factory=list, description="Conversation history"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="State shared across agents in orchestration"
    )
    environment: Dict[str, Any] = Field(
        default_factory=dict,  description="Environment variables and configuration" 
    )
    session_id: Optional[str] = Field(
        default=None, description="Unique session identifier"
    )
    created_at: datetime = Field(
        default_factory=datetime.now, description="context creation timestamp"
    )

    #Approval related fields
    pending_approval_requests: List[ToolApprovalRequest] = Field(
        default_factory=list, description="Pending tool approval requests"
    )
    approval_responses: Dict[str, ToolApprovalResponse] = Field(
        default_factory=dict, description="Approval responses by tool call ID"
    )
    pending_tool_calls: Dict[str, ToolCallRequest] = Field(
        default_factory=dict, description="Tool calls waiting for approval"
    )

    def add_message(self, message: Message) -> None:
        """Add a message to the conversation history."""
        self.messages.append(message)

    def get_last_user_message(self) -> Optional[UserMessage]:
        """Get the most recent user message."""
        for message in reversed(self.messages):
            if isinstance(message, UserMessage):
                return message
        return None

    def get_last_assistant_message(self) -> Optional[UserMessage]:
        """Get the most recent user message."""
        for message in reversed(self.messages):
            if isinstance(message, AssistantMessage):
                return message
        return None

    def clear_messages(self) -> None:
        """clear message history while preserving metadata and state."""
        self.messages.clear()

    def reset(self) -> None:
        """Complete reset of context including messages, state, and metadata."""
        self.messages.clear()
        self.shared_state.clear()
        self.metadata.clear()
        # keep environment and session_id as they are typically persistent

    @property
    def message_count(self) -> int:
        """Get the number of messages in the context."""
        return len(self.messages)

    @property
    def is_empty(self) -> bool:
        """Check if context has no messages"""
        return len(self.messages) == 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionaries for serialization."""
        return self.model_dump()

    @classmethod
    def from_message(cls, messages: List[Message]) -> "AgentContext":
        """Create context from a list of messages """
        return cls(messages=messages)

    @property
    def waiting_for_approval(self) -> bool:
        """Check if there are pending approval request."""
        return len(self.pending_approval_requests) > 0

    def add_approval_request(
            self, tool_call: ToolCallRequest, tool_name: str
    ) -> ToolApprovalRequest:
        """Create and add an approval request for a tool call."""
        request = ToolApprovalRequest(
            request_id=f"approval_{tool_call.call_id}",
            tool_call_id=tool_call.call_id,
            tool_name=tool_name,
            parameters=tool_call.parameters,
            original_tool_call=tool_call, 
        )
        self.pending_approval_requests.append(request)
        self.pending_tool_calls[tool_call.call_id] = tool_call
        return request

    def add_approval_response(self, response: ToolApprovalResponse) -> None:
        """Process an approval response."""
        self.approval_responses[response.tool_call_id] = response

        # Remove from pending
        self.pending_approval_requests = [
            req
            for req in self.pending_approval_requests
            if req.tool_call_id != response.tool_call_id
        ]

    def get_approval_response(self, tool_call_id: str) -> Optional[ToolApprovalResponse]:
        """Get the approval response for a specific tool call."""
        return self.approval_responses.get(tool_call_id)

    def get_approved_tool_calls(self) -> List[ToolCallRequest]:
        """Get tool calls that have been approved and clear them."""
        approved = []
        processed_ids = []

        for call_id, response in self.approval_responses.items():
            if response.approved and call_id in self.pending_tool_calls:
                approved.append(self.pending_tool_calls[call_id])
                processed_ids.append(call_id)

        # Clear processed approvals
        for call_id in processed_ids:
            del self.approval_responses[call_id]
            del self.pending_tool_calls[call_id]

        return approved

    def get_rejected_tool_calls(self) -> List[tuple[str, ToolCallRequest]]:
        """Get tool calls that were rejected and clear them."""
        rejected = []
        processed_ids = []

        for call_id, response in self.approval_responses.items():
            if not response.approved and call_id in self.pending_tool_calls:
                rejected.append((call_id, self.pending_tool_calls[call_id]))
                processed_ids.append(call_id)

        # Clear processed rejections
        for call_id in processed_ids:
            del self.approval_responses[call_id]
            del self.pending_tool_calls[call_id]

        return rejected

    def __str__(self) -> str:
        """String representation of the context."""
        approval_info = f", {len(self.pending_approval_requests)} pending approvals" if self.waiting_for_approval else ""
        return f"AgentContext(messages={self.message_count}, session={self.session_id}{approval_info})"
