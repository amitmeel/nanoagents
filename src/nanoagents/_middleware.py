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

    @abstractmethod
    async def process_error(
        self, context: MiddlewareContext, error: Exception
    ) -> AsyncGenerator[Union[Any, "AgentEvent"], None]:
        """
        Handle errors from the operation.

        Args:
            context: The middleware context
            error: The exception that occurred

        Yields:
            - AgentEvent: Events for observability
            - Any: Recovery value to use instead of raising error

        To propagate the error (no recovery):
            ```python
            async def process_error(self, context, error):
                yield ErrorEvent(error=str(error))
                raise error
            ```

        To recover from error:
            ```python
            async def process_error(self, context, error):
                yield ErrorEvent(error=str(error))
                yield default_value  # Use this instead of raising
            ```
        """
        if False:  # Type checker hint - never executed
            # It makes the function a valid async generator from the
            # type checker's point of view, without actually yielding anything at runtime.
            yield
        raise error

class MiddlewareChain:
    """Executes a chain of middleware as async generator pipelines."""

    def __init__(self, middlewares: Optional[List[BaseMiddleware]]=None):
        """Initialized the middleware chain.
        
        Args:
            middlewares: List of middlewares to execute in order
        """
        self.middlewares = middlewares or []

    def add(self, middleware: BaseMiddleware) -> None:
        """Add a middleware to the chain."""
        self.middlewares.append(middleware)

    def remove(self, middleware: BaseMiddleware) -> None:
        """Remove a middleware from the chain."""
        if middleware in self.middlewares:
            self.middlewares.remove(middleware)

    async def execute_stream(
        self,
        operaion: str,
        agent_name: str,
        agent_context: AgentContext,
        data: Any,
        func: Callable,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[Union[Any, "AgentEvent"], None]:
        """Execute the middleware chain with streaming support.
        
        This method process requests thourgh middleware, execute the operation,
        and process reponses, all while yielding events for observability.

        Args:
            operation: Type of operation being performed
            agent_name: Name of the Agent
            agent_context: Agent's context
            data: Operation specific data
            func: The actual operation to execute
            metadata: Optional metadata to pass to middleware

        Yields:
            - AgentEvent: Events from middleware (logs, metrics, approval requests)
            - Any: Final result from the operation

        The execution flow:
        1. Pre-process through middleware (forward order) - yields events
        2. Execute th operation
        3. Post-process through middleware(reveresed oder) - yiled events
        4. Yield final result
        """

        # create middleware context
        ctx = MiddlewareContext(
            operation=operaion,
            agent_name=agent_name,
            agent_context=agent_context,
            data=data,
            metadata=metadata or {},
        )

        # Execution Flow
        
        # PHASE 1: Pre-process thorugh all midlewares (forward order)
        for middleware in self.middlewares:
            try:
                final_ctx=None
                async for item in middleware.process_request(ctx):
                    # Events imported here to avoid circular dependency
                    from .types import (
                        ErrorEvent,
                        FatalErrorEvent,
                        MemoryRetrievalEvent,
                        MemoryUpdateEvent,
                        ModelCallEvent,
                        ModelResponseEvent,
                        TaskCompleteEvent,
                        TaskStartEvent,
                        ToolApprovalEvent,
                        ToolCallEvent,
                        ToolCallResponseEvent,
                        ToolValidationEvent,
                    )

                    if isinstance(item, MiddlewareContext):
                        # This is the final context
                        final_ctx = item
                    elif isinstance(
                        item,
                        (
                        ErrorEvent,
                        FatalErrorEvent,
                        MemoryRetrievalEvent,
                        MemoryUpdateEvent,
                        ModelCallEvent,
                        ModelResponseEvent,
                        TaskCompleteEvent,
                        TaskStartEvent,
                        ToolApprovalEvent,
                        ToolCallEvent,
                        ToolCallResponseEvent,
                        ToolValidationEvent,
                        )
                    ):
                        # Middleware emitted an event
                        yield item

                        if isinstance(item, ToolApprovalEvent):
                            return # Stop - wait for approval
                
                if final_ctx is None:
                    # Middleware didn't yield context - it must be paused
                    return

                ctx = final_ctx
            except Exception as e:
                # Middleware raised exception - try error handlers (reverse order)
                recovered = False
                for error_mw in reversed(self.middlewares):
                    try:
                        async for item in error_mw.process_error(ctx, e):
                            from .types import (
                                ErrorEvent,
                                FatalErrorEvent,
                                MemoryRetrievalEvent,
                                MemoryUpdateEvent,
                                ModelCallEvent,
                                ModelResponseEvent,
                                TaskCompleteEvent,
                                TaskStartEvent,
                                ToolApprovalEvent,
                                ToolCallEvent,
                                ToolCallResponseEvent,
                                ToolValidationEvent,
                            )

                            if isinstance(
                                item,
                                (
                                    ErrorEvent,
                                    FatalErrorEvent,
                                    MemoryRetrievalEvent,
                                    MemoryUpdateEvent,
                                    ModelCallEvent,
                                    ModelResponseEvent,
                                    TaskCompleteEvent,
                                    TaskStartEvent,
                                    ToolApprovalEvent,
                                    ToolCallEvent,
                                    ToolCallResponseEvent,
                                    ToolValidationEvent,
                                )
                            ):
                                yield item
                            else:
                                # Middleware recovered - yield recovery value and stop
                                yield item
                                recovered=True
                                return
                    except Exception:
                        continue
                
                if not recovered:
                    raise e

        # PHASE 2: Execute actual operation
        # result is guranteed to be assigned before PHASE 3 due to exception handling
        result : Any
        try:
            result = await func(self.data)
        except Exception as e:
            # Error handling through middleware (reverse order)
            recovered = False
            for middleware in reversed(self.middlewares):
                try:
                    async for item in middleware.process_error(ctx, e):
                        from .types import (
                            ErrorEvent,
                            FatalErrorEvent,
                            MemoryRetrievalEvent,
                            MemoryUpdateEvent,
                            ModelCallEvent,
                            ModelResponseEvent,
                            ModelStreamChunkEvent,
                            TaskCompleteEvent,
                            TaskStartEvent,
                            ToolApprovalEvent,
                            ToolCallEvent,
                            ToolCallResponseEvent,
                            ToolValidationEvent,
                        )

                        if isinstance(
                            item,
                            (
                                TaskStartEvent,
                                TaskCompleteEvent,
                                ModelCallEvent,
                                ModelResponseEvent,
                                ModelStreamChunkEvent,
                                ToolCallEvent,
                                ToolCallResponseEvent,
                                ToolApprovalEvent,
                                ToolValidationEvent,
                                MemoryUpdateEvent,
                                MemoryRetrievalEvent,
                                ErrorEvent,
                                FatalErrorEvent,
                            ),
                        ):
                            yield item
                        else:
                            # Middleware recovered - yield recovery value
                            yield item
                            recovered = True
                            return
                except Exception:
                    continue

            if not recovered:
                raise e
            # If we reach here, middleware recovered but didn't return (shouldn't happen)
            # This path is unreachable in practice, but we assign result for type checker
            raise RuntimeError("Middleware recovery logic error")  # pragma: no cover
 

        # PHASE 3: Post-process through middleware (reverse order)
        for middleware in reversed(self.middlewares):
            try:
                final_result = None
                async for item in middleware.process_response(ctx, result):
                    # Import event types to check
                    from .types import (
                        ErrorEvent,
                        FatalErrorEvent,
                        MemoryRetrievalEvent,
                        MemoryUpdateEvent,
                        ModelCallEvent,
                        ModelResponseEvent,
                        ModelStreamChunkEvent,
                        TaskCompleteEvent,
                        TaskStartEvent,
                        ToolApprovalEvent,
                        ToolCallEvent,
                        ToolCallResponseEvent,
                        ToolValidationEvent,
                    )

                    # Check if item is an event type (avoid isinstance with Union)
                    if isinstance(
                        item,
                        (
                            TaskStartEvent,
                            TaskCompleteEvent,
                            ModelCallEvent,
                            ModelResponseEvent,
                            ModelStreamChunkEvent,
                            ToolCallEvent,
                            ToolCallResponseEvent,
                            ToolApprovalEvent,
                            ToolValidationEvent,
                            MemoryUpdateEvent,
                            MemoryRetrievalEvent,
                            ErrorEvent,
                            FatalErrorEvent,
                        ),
                    ):
                        yield item
                    else:
                        final_result = item

                result = final_result if final_result is not None else result

            except Exception as e:
                # Error in response processing
                for error_mw in reversed(self.middlewares):
                    try:
                        async for item in error_mw.process_error(ctx, e):
                            # Import event types
                            from .types import (
                                ErrorEvent,
                                FatalErrorEvent,
                                MemoryRetrievalEvent,
                                MemoryUpdateEvent,
                                ModelCallEvent,
                                ModelResponseEvent,
                                ModelStreamChunkEvent,
                                TaskCompleteEvent,
                                TaskStartEvent,
                                ToolApprovalEvent,
                                ToolCallEvent,
                                ToolCallResponseEvent,
                                ToolValidationEvent,
                            )

                            if isinstance(
                                item,
                                (
                                    TaskStartEvent,
                                    TaskCompleteEvent,
                                    ModelCallEvent,
                                    ModelResponseEvent,
                                    ModelStreamChunkEvent,
                                    ToolCallEvent,
                                    ToolCallResponseEvent,
                                    ToolApprovalEvent,
                                    ToolValidationEvent,
                                    MemoryUpdateEvent,
                                    MemoryRetrievalEvent,
                                    ErrorEvent,
                                    FatalErrorEvent,
                                ),
                            ):
                                yield item
                            else:
                                yield item
                                return
                    except Exception:
                        continue
                raise e

        # Yield final result
        yield result

