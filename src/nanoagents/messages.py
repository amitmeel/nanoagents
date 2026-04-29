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
