"""
Base tool classes and interfaces for the nanoagents framework.

This module defines the abstract base classes and core functionality
for tools that agents can use to interact with the world.
"""

import inspect
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Union,
    get_type_hints
)

from pydantic import BaseModel

from .._component_config import ComponentBase
from ..types import ToolResult


if TYPE_CHECKING:
    from .._cancellation_token import CancellationToken
    from ..messages import Message
    from ..types import AgentEvent


class ApprovalMode(Enum):
    """Tool approval requirements."""

    NEVER = "never_require"
    ALWAYS = "always_require"


class BaseTool(ComponentBase[BaseModel], ABC):
    """
    Abstract base class that all tools must implement.
    
    Defines the interface for tools that agents can use to
    beyond text generation (e.g., web search, file operations, API calls).
    """

    def __init__(
            self,
            name: str,
            description: str,
            version: str = "1.0.0",
            approval_mode: ApprovalMode = ApprovalMode.NEVER,
    ):
        """
        Initialize the base tool.
        
        Args:
            name: Unique tool identifier
            description: What the tool does (for LLM understanding)
            version: Tool version following semantic versioning (default: "1.0.0")
            approval_mode: Whether approval is required before execution
        """
        self.name = name
        self.description = description
        self.version = version
        self.approval_mode = approval_mode

    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """
        JSON schema defining expected inputs for this tool.
        
        Returns:
        JSON schema dictionary describing tool parameters
        """
        pass

    @abstractmethod
    def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """
        Execute the tool with the given paramters.
        
        Args:
            paramters: Tool input parameters
            
        Returns:
            ToolResult containing execution outcom
        """
        pass

    async def execute_stream(
            self,
            parameters: Dict[str, Any],
            cancellation_token: Optional["CancellationToken"]=None,
    ) -> AsyncGenerator[Union["Message", "AgentEvent", ToolResult], None]:
        """
        Execute the tool with streaming output support.
        
        Tools can override this to provide streaming capabilities.
        Default implementation wraps execute().
        
        Args:
            paramters: Tool input paramter
            cancellation_token: Optional token for cancelling the task
            
        Yields:
            Messages, events during execution, and final ToolResult
        """
        # Default implementation: just call execute and yield result
        result = await self.execute(parameters)
        yield result

    def support_streaming(self) -> bool:
        """
        Check if this tool support streaming execution.
        
        Returns:
            True if tool has overrideen execute_stream, False otherwise
        """
        # check if execute_stream was overrridden
        return type(self).execute_stream is not BaseTool.execute_stream

    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """
        Valdate that the provided paramters matches the tool's schema.
        
        Args:
            params: Parameters to validate
        
        Returns:
            True if parameters are valid, False otherwise
        """
        try:
            # Basic validation - subclasses can override for more sophisticated checks
            schema = self.parameters
            required_fields = schema.get("required", [])

            # check required fields are present
            for field in required_fields:
                if field not in params:
                    return False

            # Check parameter types if specified in schema
            properties = schema.get("properties", {})
            for param_name, param_value in params.items():
                if param_name in properties:
                    expected_type = properties[param_name].get("type")
                    if expected_type and not self._check_type(
                        param_value, expected_type
                    ):
                        return False

            return True
        except Exception:
            return False

    def _check_type(self, value:Any, expected_type: str) -> bool:
        """Check if vlaue atches expected JSON schema type."""
        type_mapping = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict
        }

        expected_python_type = type_mapping.get(expected_type)
        if expected_python_type is None:
            return True # Unknown types, assumes valid

        return isinstance(value, expected_python_type)

    def to_llm_format(self) -> Dict[str, Any]:
        """
        Convert tool to OpenAI function calling format.

        Returns:
            Dictionary in OpenAI function format
        """
        # Include version in tool name for compatibility tracking (Anthropic pattern)
        # Format: toolname_v1.0.0 for versioned tools
        versioned_name = (
            f"{self.name}_v{self.version}" if self.version != "1.0.0" else self.name
        )

        return {
            "type": "function",
            "function": {
                "name": versioned_name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', description='{self.description}')"



