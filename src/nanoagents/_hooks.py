"""
Determinastic loop hooks for nanoagents.

Hooks provide determinastic control points around the LLM controlled tool loop.
Unlike middleware (which intercepts individual operations like model_call, tool_call),
hooks operate at the loop level - befoe the first LLM call and when the loop is about to exit.

Two hook points:
    - start hooks: Run before the first LLM call. Injection Instructions.
    - end hooks: Run when agent would stop (no tool call). Can resume the loop.

The key property is that hooks are DETERMINESTIC python code -no LLM involved in hook logic.
The LLM only see the result (injected UserMessage).

Example:
    from nanoagents import Agent
    from nanoagents._hooks import PlanningHook, CompletionCheckHook
    from nanoagents.tools import create_coding_tools, create_todo_tools

    agent = Agent(
        name="Planner",
        instructions="You are a software engineer",
        model_client=client,
        tools=[*create_coding_tools(),*create_todo_tools()],
        start_hooks=[PlanningHook()],
        end_hooks=[CompletionChckHook(max_restarts=2)],
        max_iteration=20
    )
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .context import AgentContext
from .messages import Message

# ==================================================================================
# Loop Context
# ==================================================================================

@dataclass
class LoopContext:
    """Shared state passed to hooks during a single agent run.
    
    Provides hooks with access to the agent context, current messages,
    and mutable metadata for inter-hook communication.
    
    Attributes:
        agent_context: The agent's working context(message, metadata)
        llm_messages: current message list being sent to the LLM
        agent_name: Name of the agent running
        iteration: Current loop iteration (0-indexed)
        restart_count: How many times end hooks have resumed the loop
        metadata: mutable dict for hooks to share state
        model_client: The agent's model client (for hooks that needs LLM calls)
        """
    agent_context: AgentContext
    llm_messages: List[Message]
    agent_name: str
    iteration: int = 0
    restart_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    model_client: Any = None


# ==============================================================================
# Termination conditions
# ==============================================================================

class TerminationCondition(ABC):
    """Controls when end hooks should stop restarting the agent loop.
    
    Composable with | (OR) and & (AND) operators:
        # Stop after 3 restarts OR when all tood is done
        condition = MaxRestartsTermination(3) | AllTodosComplete()
    """

    @abstractmethod
    def should_terminate(self, context: LoopContext) -> bool:
        """Return True to allow the agent loop to stop (no more restarts)."""
        ...

    def reset(self):
        """Reset state for a new agent run."""
        pass

    def __or__(self, other: "TerminationCondition") -> "CompositeTermination":
        """Combine with OR - terminate if EITHER condition is met."""
        return CompositeTermination([self, other], mode="any")

    def __or__(self, other: "TerminationCondition") -> "CompositeTermination":
        """Combine with AND - terminate if BOTH conditions are met."""
        return CompositeTermination([self, other], mode="all")


class MaxRestartsTermination(TerminationCondition):
    """Terminate after amaximum number of loop restarts.

    Args:
        max_restarts: Maximum restarts before allowing stop. Default 2.

    Example:
        hook = CompletionCheckHook(termination=MaxRestartsTermination(3))
    """

    def __init__(self, max_restarts: int = 2):
        self.max_restarts = max_restarts

    def should_terminate(self, context: LoopContext) -> bool:
        return context.restart_count >= self.max_restarts

    def __repr__(self) -> str:
        return f"MaxRestartsTermination(max_restarts={self.max_restarts})"

    
class CompositeTermination(TerminationCondition):
    """Combines multiple termination conditions with AND/OR logic.
    
    Created automatically via | and & operators on TerminationCondition.

    Args:
        conditions: List of conditions to combine
        mode: "any" (OR - terminate if any is True) or "all" (AND - all must be True)
    """

    def __init__(self, conditions: List[TerminationCondition], mode: str = "any"):
        if mode not in ("any", "all"):
            raise ValueError("Mode must be 'any' or 'all'")
        self.conditions = conditions
        self.mode = mode

    def should_terminate(self, context: LoopContext) -> bool:
        results = [c.should_terminate for c in self.conditions]
        return any(results) if self.mode=="any" else all(results)

    def reset(self) -> None:
        for c in self.conditions:
            c.reset()

    def __or__(self, other: TerminationCondition) -> "CompositeTermination":
        if self.mode == "any":
            # Flatten: (A | B) | C = A | B | C
            if isinstance(other, CompositeTermination) and other.mode=="any":
                return CompositeTermination(
                    self.conditions + other.conditions, mode="Any"
                )
            return CompositeTermination(
                self.conditions + [other.conditions], mode="any"
            )
        # self is AND mode - treat as single unit (A & B) | C
        return CompositeTermination([self, other], mode="any")

    def __and__(self, other: TerminationCondition) -> "CompositeTermination":
        if self.mode == "all":
            # Flatten: (A & B) & C = A & B & C
            if isinstance(other, CompositeTermination) and other.mode == "all":
                return CompositeTermination(
                    self.conditions + other.conditions, mode="all"
                )
            return CompositeTermination(self.conditions + [other], mode="all")
        # self is OR mode - treat as a single unit: (A | B) & C
        return CompositeTermination([self, other], mode="all")

    def __repr__(self) -> str:
        op = " | " if self.mode == "any" else " & "
        return f"({op.join(repr(c) for c in self.conditions)})"


# ===================================================================================
# Base Hook classes
# ===================================================================================

class BaseStartHook(ABC):
    """Base class for hooks that run before th first LLM call.
    
    Return a string to inject as UserMessage, or None to do nothing.
    Multiple start hooks run in order - all injections are applied.
    
    Example:
        class MyStartHook(BaseStartHook):
            async def on_start(self, context):
                return "Always think step by step before answering.
    """

    @abstractmethod
    async def on_start(self, context: LoopContext) -> Optional[str]:
        """Called before the first llm call.
        
        Args:
            context: Loop context with agent state
        
        Returns:
            Text to inject as UserMessage or None for no injection.
        """
        ...


class BaseEndHook(ABC):
    """Base class for hooks that run when the agent would stop.
    
    Return a string to inject as UserMessage and RESUME the loop,
    or None to allow the agent to stop.
    
    The first hook to return a non-None value wins - subsequent hooks
    are not called for that exit attempt.
    
    Example:
        class MyEndHook(BaseEndHook):
            async def on_end(self, context):
                if some_condition_not_met():
                    return "You are not done yet, Continue working."
                return None # Allow stop
    """

    @abstractmethod
    async def on_end(self, context: LoopContext) -> Optional[str]:
        """Called when agent would stop (no tool calls returned.

        Args:
            context: Loop context with agent stae and restart_count

        Returns:
            Text to inject as UserMessage to resume the loop or None to stop.
        """
        ...


# ============================================================================
# Concrete Implementations
# ============================================================================


class PlanningHook(BaseStartHook):
    """Start hook that injects a planning instructions before the first LLM call.
    
    Instructs the agent to analyze the task and create a structured todo
    list using todo_write tool before starting any work.

    Requires todo tools (todo_write, todo_read) in the agent's tool list.

    Args:
        instruction: Custom instruction text. if None, use a sensible default.

    Example:
        agent = Agent(
            ...,
            tools=[*create_coding_tools(), *create_todo_tools()],
            start_hooks=[PlanningHook()],
        )
    """

    def __init__(self, instruction: Optional[str] = None):
        self.instruction = instruction or (
            "Before starting any work, you MUST:\n"
            "1. Analyze the task and break it into clear, actionable steps\n"
            "2. Use the todo_write tool to create a structured task list\n"
            "3. Each todo needs: content (what to do), status ('pending'), "
            "activeForm (present tense)\n"
            "4. Mark the first task as 'in_progress' before starting it\n"
            "5. Only ONE task should be 'in_progress' at a time\n\n"
            "As you work, update the todo list: mark tasks 'completed' "
            "when done, set the next task to 'in_progress'."
        )

    async def on_start(self, context: LoopContext) -> Optional[str]:
        return self.instruction

    def __repr__(self) -> str:
        return "PlanningHook()"


class CompletionCheckHook(BaseEndHook):
    """End hook that checks todo list completion before allowing the agent to stop.
    
    When the agent would stop (no tool calls), this hook:
    1. Load the current todo list from file
    2. Check if all tools are marked as 'completed'
    3. If incomplete items remain AND termination allow: resumes the loop
    4. If any complete OR termination says stop: allow the agent to stop
    
    Args:
        termination: Termination condition. Default to MaxRestartsTermination(2).
        max_restarts: Shorthand for MaxRestartsTermination(N). Ignore if 
        terminaiton is provided.

    Example:
        agent = Agent(
            ...,
            end_hooks=[CompletionCheckHook(max_restarts=3)]
        )    
    """

    def __init__(
            self,
            termination: Optional[TerminationCondition] = None,
            max_restarts: int = 2
    ):
        if termination is not None:
            self.termination = termination
        else:
            self.termination = MaxRestartsTermination(max_restarts)

    async def on_end(self, context: LoopContext) -> Optional[str]:
        # Check termination condition first
        if self.termination.should_terminate(context):
            return None # Allow stop

        # Load todo from file
        from .tools._context_tools import _load_todos

        todos = _load_todos()

        if not todos:
            return None  # No todos = nothing to check

        # check for incomplete items
        incomplete = [t for t in todos if t.get("status") != "completed"]

        if not incomplete:
            return None # All done, allow stop

        # Build resume message
        total = len(todos)
        completed = total - len(incomplete)

        items = "\n".join(
            f"  - [{t.get('status', 'pending')}] {t.get('content', '')}"
            for t in incomplete
        )

        return (
            f"You have {len(incomplete)} incomplete tasks "
            f"({completed}/{total} completed):\n{items}\n\n"
            f"Continue working on the next pending task. "
            f"Update the todo list as you make progress. "
            f"Do not ask for user input - proceed autonomously."
        )

    def __repr__(self) -> str:
        return f"CompletionCheckHook(termination={self.termination!r})"


class LLMCompletionHook(BaseEndHook):
    """End hook that uses an LLM to judge whether the task is complete.
    
    When the agent would stop (no tool calls), this hook:
    1. Extracts the original task from the conversation
    2. Builds a summary of the full conversation (tool calls + results + responses)
    3. Asks a judge model: "Is this task complete?"
    4. If not complete: injects a message telling the agent to continue
    5. If complete or termination limit reached: allows stop

    This is generic - no hardcoded logic about files, todos, or specific
    task types. The LLM judges completion based on the task description
    and conversation history.

    Args:
        model_client: LLM client for the completion check. If None,
            uses the agent's own model_client from LoopContext.
        termination: Termination condition. Defaults to MaxRestartsTermination(2).
        max_restarts: Shorthand for MaxRestartsTermination(N). Ignored if
            termination is provided.

    Example:
        # Uses agent's own model client:
        agent = Agent(
            ...,
            end_hooks=[LLMCompletionCheckHook(max_restarts=3)],
        )

        # Or with a separate (cheaper) judge model:
        judge = OpenAIChatCompletionClient(model="gpt-4.1-mini")
        agent = Agent(
            ...,
            end_hooks=[LLMCompletionCheckHook(
                model_client=judge, max_restarts=3,
            )],
        )
    """
    def __init__(
        self,
        model_client: Any = None,
        termination: Optional[TerminationCondition] = None,
        max_restarts: int = 2,
    ):
        self.model_client = model_client
        if termination is not None:
            self.termination = termination
        else:
            self.termination = MaxRestartsTermination(max_restarts)

    def _build_conversation_summary(
        self, messages: List[Message], max_chars: int = 6000
    ) -> str:
        """Build a condensed summary of the full conversation.

        Includes tool calls with parameter AND result summaries,
        plus any text responses from the agent.
        """
        lines: List[str] = []
        total_chars = 0

        for msg in messages:
            role = getattr(msg, "role", "")
            content = getattr(msg, "content", "")

            if role == "user" and getattr(msg, "source", "") != "hook":
                # Skip the original task — it's shown separately
                continue

            if role == "user" and getattr(msg, "source", "") == "hook":
                line = f"[HOOK] {content[:200]}"
            elif role == "assistant":
                tool_calls = getattr(msg, "tool_calls", None)
                if tool_calls:
                    for tc in tool_calls:
                        params = str(tc.parameters)[:100]
                        line = f"[CALL] {tc.tool_name}({params})"
                        lines.append(line)
                        total_chars += len(line)
                    if content and content.strip():
                        line = f"[TEXT] {content[:300]}"
                    else:
                        continue
                else:
                    # Final response text
                    line = f"[RESPONSE] {content[:500]}"
            elif role == "tool":
                # Show tool result size and preview
                size = len(content)
                preview = content[:150].replace("\n", " ")
                tool_name = getattr(msg, "tool_name", "?")
                line = f"[RESULT] {tool_name} ({size} chars): {preview}"
            else:
                continue

            if total_chars + len(line) > max_chars:
                lines.append(f"... ({len(messages) - len(lines)} more messages truncated)")
                break

            lines.append(line)
            total_chars += len(line)

        return "\n".join(lines)

    async def on_end(self, context: LoopContext) -> Optional[str]:
        if self.termination.should_terminate(context):
            return None

        # Resolve model client: explicit > from context
        client = self.model_client or context.model_client
        if client is None:
            return None  # No client available, allow stop

        # Extract original task (first user message)
        task = ""
        for msg in context.llm_messages:
            if hasattr(msg, "role") and msg.role == "user":
                task = msg.content
                break

        if not task:
            return None

        # Build rich conversation summary
        summary = self._build_conversation_summary(context.llm_messages)

        # Ask the judge model
        from .messages import SystemMessage as SM
        from .messages import UserMessage as UM

        judge_messages = [
            SM(
                content=(
                    "You are a strict task completion judge. Given a task "
                    "and a log of what an agent has done, determine if the "
                    "task is COMPLETE.\n\n"
                    "The log shows every tool call the agent made, what "
                    "results it got, and what text it produced.\n\n"
                    "Reply with exactly one of:\n"
                    "- COMPLETE: <reason>\n"
                    "- INCOMPLETE: <what specific work remains>\n\n"
                    "Be strict. Judge based on what the agent ACTUALLY DID "
                    "(the tool calls and results), not what it CLAIMS to "
                    "have done in its response text. If the task requires "
                    "reading files and the agent only read a few, that is "
                    "INCOMPLETE even if the agent wrote a confident review."
                ),
                source="system",
            ),
            UM(
                content=(
                    f"## Original Task\n{task}\n\n"
                    f"## Agent Activity Log\n{summary}\n\n"
                    f"Is the task COMPLETE or INCOMPLETE?"
                ),
                source="judge",
            ),
        ]

        try:
            result = await client.create(judge_messages)
            response_text = result.message.content.strip()
            response_upper = response_text.upper()

            if response_upper.startswith("COMPLETE"):
                return None  # Task is done, allow stop

            # Extract reason if provided
            if ":" in response_text:
                reason = response_text.split(":", 1)[1].strip()
            else:
                reason = "The task is not yet complete."

            return (
                f"You are not done yet. {reason}\n\n"
                f"Continue working on the task. Do not stop until "
                f"the task is fully complete. Do not ask for user input."
            )
        except Exception:
            # If judge call fails, allow stop rather than blocking
            return None

    def __repr__(self) -> str:
        return (
            f"LLMCompletionCheckHook("
            f"termination={self.termination!r})"
        )

        