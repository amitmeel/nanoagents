"""
Component configuration system for nanoagents.

This module provides a flexible component serialization and loading system that allows
components to be saved as JSON configuation and loaded back as instance. This is essential
for workflow persistance, UI integration, and system interoperability.

Originally based on Autogen's component system, adapted for nanoagents.
"""

from __future__ import annotations

import importlib
import warnings
from typing import Any, ClassVar, Dict, Generic, Literal, Type, Union, cast, overload

from pydantic import BaseModel
from typing_extensions import Self, TypeGuard, TypeVar

ComponentType = Union[
    Literal[
        "model", "agent", "tool", "termination", "orchestrator", "step", "workflow"
    ], str
]
ConfigT = TypeVar("ConfigT", bound=BaseModel)
FromConfigT = TypeVar("FromConfigT", bound=BaseModel, contravariant=True)
ToConfigT = TypeVar("ToConfigT", bound=BaseModel, covariant=True)

T = TypeVar("T", bound=BaseModel, covariant=True)


class ComponentModel(BaseModel):
    """
    Model class for a seriazable component.

    Contains all information required to instantiate a component, including:
    - Provider information for class loading
    - Type and version metadata
    - Configuration data for instantiation

    This enables components to be saved as JSON configuration and loaded
    back as instances, supporting workflow persistance and UI integration.
    """

    provider: str
    """
    Fully qualified class name for component instantiation.
    Format: "module.path.ClassName
    Example: "nanoagents.agent.Agent
    """

    component_type: ComponentType | None = None
    """
    Logical type of the component (e.g., 'agent', 'workflow', 'step').
    If missing, assumes the default type of the provider.
    """

    version: int | None = None
    """
    Version of the component specific schema.
    If missing, assumes current library version ( dangerous for production).
    Should always be specified for persistent configuration.
    """

    component_version: int | None = None
    """
    Version of the specific component implementation.
    Used for schema migration when component structure changes.
    """

    description: str | None = None
    """
    Human readable description of the component.
    Auto-generated from class docstring if not provided.
    """

    label: str | None = None
    """
    Display name of the components in UIs.
    Defaults to the class name if not provided.
    """

    config: dict[str, Any]
    """
    Schema validated configuration data for component instatiation.
    Passed to the component class's _from_config() method.
    """


def _type_to_provider_str(t: type) -> str:
    return f"{t.__module__}.{t.__qualname__}"


WELL_KNOWN_PROVIDERS = {
    # nanoagents model clients
    "openai_chat_completion_client": "nanoagents.llm.OpenAIChatCompletionClient",
    "OpenAIChatCompletionClient": "nanoagents.llm.OpenAIChatCompletionClient",
    "model_client": "nanoagents.llm.OpenAIChatCompletionClient", # Default model client
    # nanoagents Agents
    "agent": "nanoagents.agents.Agent",
    "Agent": "nanoagents.agents.Agent",
    # nanoagents memory
    "list_memory": "nanoagents.memory.ListMemory",
    "ListMemory": "nanoagents.memory.ListMemory",
    "file_memory": "nanoagents.memory.FileMemory",
    "FileMemory": "nanoagents.memory.FileMemory",
    "Memory": "nanoagents.memory.ListMemory", # Default memory
    # nanoagents termination
    "max_message_termination": "nanoagents.termination.MaxMessageTermination",
    "MaxMessageTermination": "nanoagents.termination.MaxMessageTermination",
    "text_mention_termination": "nanoagents.termination.TextMentionTermination",
    "TextMentionTermination": "nanoagents.termination.TextMentionTermination",
    "composite_termination": "nanoagents.termination.CompositeTermination",
    "CompositeTermination": "nanoagents.termination.CompositeTermination",
    "termination": "nanoagents.termination.MaxMessageTermination",  # Default termination
    # nanoagents Orchestrators
    "round_robin_orchestrator": "nanoagents.orchestration.RoundRobinOrchestrator",
    "RoundRobinOrchestrator": "nanoagents.orchestration.RoundRobinOrchestrator",
    "ai_orchestrator": "nanoagents.orchestration.AIOrchestrator",
    "AIOrchestrator": "nanoagents.orchestration.AIOrchestrator",
    "orchestrator": "nanoagents.orchestration.RoundRobinOrchestrator",  # Default orchestrator
    # Workflow Components
    "workflow": "nanoagents.workflow.Workflow",
    "Workflow": "nanoagents.workflow.Workflow",
    "nanoagent_step": "nanoagents.workflow.NanoAgentStep",
    "NanoAgentStep": "nanoagents.workflow.NanoAgentStep",
}


class ComponentFromConfig(Generic[FromConfigT]):
    """
    Mixin class for components that can be loaded from configuration.

    Implement this interface to enable your component to be instantiated 
    from a serialized configuration object.
    """

    @classmethod
    def _from_config(cls, config: FromConfigT) -> Self:
        """
        Create a new instance of the component from a configuration object.

        Args:
            config: Validated configration object (pydantic model)
        Returns:
            A new instance of the component initialized from the config
        Raises:
            NotImplementedError: If the component doesn't support config loading
        """
        raise NotImplementedError("This component does not support loading from config")

    @classmethod
    def _from_config_past_version(cls, config: Dict[str, Any], version: int) -> Self:
        """Create a new instance of the component from a previous version of the configuration object.

        This is only called when the version of the configuration object is less than the current version, since in this case the schema is not known.

        Args:
            config (Dict[str, Any]): The configuration object.
            version (int): The version of the configuration object.

        Returns:
            Self: The new instance of the component.

        :meta public:
        """
        raise NotImplementedError(
            "This component does not support loading from past versions"
        )


class ComponentToConfig(Generic[ToConfigT]):
    """
    The two methods a class must implement to be a component.

    Args:
        Protocol (ConfigT): Type which derives from :py:class: `pydantic.BaseModel`.
    """

    component_type: ClassVar[ComponentType]
    """The logical type of the component."""
    component_version: ClassVar[int] = 1
    """The version of the component, if schema incompatibilities are introduced this should be updated."""
    component_provider_override: ClassVar[str|None] = None
    """Override the provider string for the component. This should be used to prevent internal module names being a part of the module name."""
    component_description: ClassVar[str|None] = None
    """A description of the component. If not provided, the docstring of the class will be used."""
    component_label: ClassVar[str|None] = None
    """A human readable label for the component. If not provided, the component class name will be used."""

    def _to_config(self) -> ToConfigT:
        """Dump the configuration that would be require to create a new instance of a component matching the configuration of this instance.
        
        Returns:
            T: The configuration of the component.
            
        :meta public:
        """
        raise NotImplementedError("This component does not suppport dumping to config")

    def dump_component(self) -> ComponentModel:
        """Dump the component to a model that can be loaded back in.
        
        Raises:
            TypeError: If he component is a local class.
        
        Returns:
            ComponentModel: The model representing the component
        """
        if self.component_provider_override is not None:
            provider = self.component_provider_override
        else:
            provider = _type_to_provider_str(self.__class__)
            # warn if internal module name is used
            if '_.' in provider:
                warnings.warn(
                    (
                    "Internal module name used in prpovider string. This is not recommended and may cause "
                    "issues in the future. Silence this warning by setting component_provider_override to this value."
                    ),
                    stacklevel=2
                )

        if "<locals>" in provider:
            raise TypeError("Cannot dump componet with local class")

        if not hasattr(self, "componet_type"):
            raise AttributeError("component_type not defined")

        description = self.component_description
        if description is None and self.__class__.__doc__:
            # use docstring as description
            docstring = self.__class__.__doc__.strip()
            for marker in ["\n\nArgs:", "\n\nParameters:", "\n\nAttributes:", "\n\n"]:
                docstring = docstring.split(marker)[0]
            description = docstring.strip()

        obj_config = self._to_config().model_dump(exclude_none=True)
        model = ComponentModel(
            provider=provider,
            component_type=self.component_type,
            version=self.component_version,
            component_version=self.component_version,
            description=description,
            label=self.component_label or self.__class__.__name__,
            config=obj_config,
        )
        return model


ExpectedType = TypeVar("ExpectedType")


class ComponentLoader:
    @overload
    @classmethod
    def load_component(
        cls, model: ComponentModel | Dict[str, Any], expected: None = None
    ) -> Self:
        ...

    @overload
    @classmethod
    def load_component(
        cls, model: ComponentModel | Dict[str, Any], expected: Type[ExpectedType]
    ) -> ExpectedType:
        ...

    @classmethod
    def load_component(
        cls,
        model: ComponentModel | Dict[str, Any],
        expected: Type[ExpectedType] | None = None,
    ) -> Self | ExpectedType:
        """Load a component from a model. Intended to be used with the return type of :py:meth:`nanoagents.ComponentConfig.dump_component`.

        Example:

            .. code-block:: python

                from nanoagents import ComponentModel
                from nanoagents.models import ChatCompletionClient

                component: ComponentModel = ...  # type: ignore

                model_client = ChatCompletionClient.load_component(component)

        Args:
            model (ComponentModel): The model to load the component from.

        Returns:
            Self: The loaded component.

        Args:
            model (ComponentModel): _description_
            expected (Type[ExpectedType] | None, optional): Explicit type only if used directly on ComponentLoader. Defaults to None.

        Raises:
            ValueError: If the provider string is invalid.
            TypeError: Provider is not a subclass of ComponentConfigImpl, or the expected type does not match.

        Returns:
            Self | ExpectedType: The loaded component.
        """

        # Use global and add further type checks

        if isinstance(model, dict):
            loaded_model = ComponentModel(**model)
        else:
            loaded_model = model

        # First, do a look up in well known providers
        if  loaded_model.provider in WELL_KNOWN_PROVIDERS:
            loaded_model.provider = WELL_KNOWN_PROVIDERS[loaded_model.provider]

        output = loaded_model.provider.rsplit(".", maxsplit=1)
        if len(output) != 2:
            raise ValueError("Invalid")

        module_path, class_name = output
        module = importlib.import_module(module_path)
        component_class = module.__getattribute__(class_name)

        if not is_component_class(component_class):
            raise TypeError("Invalid component class")

        # We need to check the schema is valid
        if not hasattr(component_class, "component_config_schema"):
            raise AttributeError("component_config_schema not defined")

        if not hasattr(component_class, "component_type"):
            raise AttributeError("component_type not defined")

        loaded_config_version = (
            loaded_model.component_version or component_class.component_version
        )
        if loaded_config_version < component_class.component_version:
            try:
                instance = component_class._from_config_past_version(loaded_model.config, loaded_config_version)  # type: ignore
            except NotImplementedError as e:
                raise NotImplementedError(
                    f"Tried to load component {component_class} which is on version {component_class.component_version} with a config on version {loaded_config_version} but _from_config_past_version is not implemented"
                ) from e
        else:
            schema = component_class.component_config_schema  # type: ignore
            validated_config = schema.model_validate(loaded_model.config)

            # We're allowed to use the private method here
            instance = component_class._from_config(validated_config)  # type: ignore

        if expected is None and not isinstance(instance, cls):
            raise TypeError("Expected type does not match")
        elif expected is None:
            return cast(Self, instance)
        elif not isinstance(instance, expected):
            raise TypeError("Expected type does not match")
        else:
            return cast(ExpectedType, instance)


class ComponentSchemaType(Generic[ConfigT]):
    # Ideally would be ClassVar[Type[ConfigT]], but this is disallowed https://github.com/python/typing/discussions/1424 (despite being valid in this context)
    component_config_schema: Type[ConfigT]
    """The Pydantic model class which represents the configuration of the component."""

    required_class_vars = ["component_config_schema", "component_type"]

    def __init_subclass__(cls, **kwargs: Any):
        super().__init_subclass__(**kwargs)

        if cls.__name__ != "Component" and not cls.__name__ == "_ConcreteComponent":
            # TODO: validate provider is loadable
            for var in cls.required_class_vars:
                if not hasattr(cls, var):
                    warnings.warn(
                        f"Class variable '{var}' must be defined in {cls.__name__} to be a valid component",
                        stacklevel=2,
                    )


class ComponentBase(ComponentToConfig[ConfigT], ComponentLoader, Generic[ConfigT]):
    ...


class Component(
    ComponentFromConfig[ConfigT],
    ComponentSchemaType[ConfigT],
    Generic[ConfigT],
):
    """
    Base class for serializable nanoagents components.

    To create a serializable component:
    1. Inherit from this class and ComponentBase
    2. Define component_config_schema (Pydantic model for config)
    3. Define component_type (logical type string)
    4. Implement _to_config() and _from_config() methods

    Example:

    .. code-block:: python

        from pydantic import BaseModel
        from nanoagents import Component, ComponentBase


        class MyConfig(BaseModel):
            name: str
            value: int


        class MyComponent(Component[MyConfig], ComponentBase[MyConfig]):
            component_type = "custom"
            component_config_schema = MyConfig

            def __init__(self, name: str, value: int):
                self.name = name
                self.value = value

            def _to_config(self) -> MyConfig:
                return MyConfig(name=self.name, value=self.value)

            @classmethod
            def _from_config(cls, config: MyConfig) -> MyComponent:
                return cls(name=config.name, value=config.value)

    This enables the component to be serialized via dump_component()
    and loaded back via load_component().
    """

    def __init_subclass__(cls, **kwargs: Any):
        super().__init_subclass__(**kwargs)

        if not is_component_class(cls):
            warnings.warn(
                f"Component class '{cls.__name__}' must subclass the following: ComponentFromConfig, ComponentToConfig, ComponentSchemaType, ComponentLoader, individually or with ComponentBase and Component. Look at the component config documentation or how OpenAIChatCompletionClient does it.",
                stacklevel=2,
            )


# Should never be used directly, only for type checking
class _ConcreteComponent(
    ComponentFromConfig[ConfigT],
    ComponentSchemaType[ConfigT],
    ComponentToConfig[ConfigT],
    ComponentLoader,
    Generic[ConfigT],
):
    ...


def is_component_instance(cls: Any) -> TypeGuard[_ConcreteComponent[BaseModel]]:
    return (
        isinstance(cls, ComponentFromConfig)
        and isinstance(cls, ComponentToConfig)
        and isinstance(cls, ComponentSchemaType)
        and isinstance(cls, ComponentLoader)
    )


def is_component_class(cls: type) -> TypeGuard[Type[_ConcreteComponent[BaseModel]]]:
    return (
        issubclass(cls, ComponentFromConfig)
        and issubclass(cls, ComponentToConfig)
        and issubclass(cls, ComponentSchemaType)
        and issubclass(cls, ComponentLoader)
    )




