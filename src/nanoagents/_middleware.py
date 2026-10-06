"""
Middleware system for nanoagents

This module provides the midddleware infrastructure for intercepting and processing
agent operaiton like model calls, tool calls, and memory access.

Middleware uses async generaors to support:
- Event emission (Observability and reporting)
- Approval requests (pausing execution)
- Error recovery
"""


import asyncio
import logging
import re
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from .context import AgentContext
from .messages import Message

if TYPE_CHECKING:
    from .types import AgentEvent


class MiddlewareContext(BaseModel):
    """Context passed through middleware chain."""

operation: str = Field(
    description="Operation Type: model_call, tool_call, memory_access"
)
agent_name: str = Field(description="Name of the agent executing the operation")
data: Any = Field(
    description="Operation-Specific data (messages, tool request, etc)"
)
metadata: Dict[str, Any] = Field(
    default_factory=dict, description="Additional metadata for middleware use"
)


class BaseMiddleware(ABC):
    """Abstract base class for middleware components.
    
    Middleware intercepts agent operation and can:
    - Emit events for observability (logs, metrics, approval requests)
    - Transform requests / responses
    - Pause execution
    - Handle erros with recovery

    All middleware methods are async generators (AsyncGenerator[YIELD_TYPE, SEND_TYPE])
    that yield events and/or results.
    """

    @abstractmethod
    async def process_request(
        self, context: AgentContext
    ) -> AsyncGenerator[Union[MiddlewareContext, "AgentEvent"], None]:
        """
        Process before the operation executes

        Yields:
            - AgentEvent: Events for observability (logs, approval requests etc)
            - MiddlewareContext: Final yield MUST be context (modified or original)

        To pause execution for approval:
            ```python
            yiled ToolApprovalEvent(approval_request=req)
            return # Don't yield context - agent will pause
            ```

        Example:
            ```python
            async def process_request(self, context):
                # Emit log event
                yield LogEvent(message="Processing requests")

                # check if approval is needed
                if needs_approval:
                    yield ToolApprovalEvent(approval_request=req)
                    return # Pause - don't yield context
                
                # continue execution
                yield context
            ```
        
        Raises:
            Exception: to abort the execution (caught and handled by proces_error)
        """
        yield context

    @abstractmethod
    async def process_response(
        self, context: AgentContext, result: Any
    ) -> AsyncGenerator[Union[Any, "AgentEvent"], None]:
        """
        Process after the operation complete successfully.

        Args:
            context: The middleware context
            result: The operation result

        Yields:
            - AgentEvent: Events for observability
            - Any: Final yield MUST be the result (modified or original)

        Example:
            ```python
            async def process_response(self, context, result):
                # Emit metric event
                yield MetricEvent(duration=elapsed)

                # Modify result
                result.metadata["processed"]=True
                yield result
            ```
        """
        yield result

