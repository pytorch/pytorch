from copy import deepcopy
from functools import wraps
from typing import Any, Callable, Mapping, Optional, Union


class Registry(dict):
    """
    `Registry` for components.

    Examples:
        >>> registry = Registry()
        >>> @registry.register(object)
        ... @registry.register("Module")
        ... class Module:
        ...     def __init__(self, a, b):
        ...         self.a = a
        ...         self.b = b
        >>> module = registry.register(Module, "module")
        >>> registry
        {'Module': <class 'torch.utils.registry.Module'>, <class 'object'>: <class 'torch.utils.registry.Module'>, 'module': <class 'torch.utils.registry.Module'>}
        >>> registry.lookup(object)
        <class 'torch.utils.registry.Module'>
        >>> config = {"module": {"name": "Module", "a": 1, "b": 2}}
        >>> module = registry.build(config["module"])
        >>> type(module)
        <class 'torch.utils.registry.Module'>
        >>> module.a
        1
        >>> module.b
        2
    """

    def register(self, component: Callable, name: Optional[Any] = None) -> Callable:
        r"""
        Register a new component.

        Args:
            component: The component to register.
            name: The name of the component.

        Returns:
            component: The registered component.

        Examples:
            >>> registry = Registry()
            >>> @registry.register(object)
            ... @registry.register("Module")
            ... class Module:
            ...     def __init__(self, a, b):
            ...         self.a = a
            ...         self.b = b
            >>> module = registry.register(Module, "module")
            >>> registry
            {'Module': <class 'torch.utils.registry.Module'>, <class 'object'>: <class 'torch.utils.registry.Module'>, 'module': <class 'torch.utils.registry.Module'>}
        """

        # Registry.register()
        if name is not None:
            self[name] = component

        # @Registry.register()
        @wraps(self.register)
        def register(component, name=None):
            if name is None:
                name = component
            self[name] = component
            return component

        return lambda x: register(x, component)

    def lookup(self, name: str) -> Any:
        r"""
        Lookup for a component.

        Args:
            name:

        Returns:
            (Any): The component.

        Raises:
            KeyError: If the component is not registered.

        Examples:
            >>> registry = Registry()
            >>> @registry.register("Module")
            ... class Module:
            ...     def __init__(self, a, b):
            ...         self.a = a
            ...         self.b = b
            >>> registry.lookup("Module")
            <class 'torch.utils.registry.Module'>
        """

        return self[name]

    def build(self, name: Union[str, Mapping], *args, **kwargs) -> Any:
        r"""
        Build a component.

        Args:
            name (str | Mapping):
                If its a `Mapping`, it must contain `"name"` as a member, the rest will be treated as `**kwargs`.
                Note that values in `kwargs` will override values in `name` if its a `Mapping`.
            *args: The arguments to pass to the component.
            **kwargs: The keyword arguments to pass to the component.

        Returns:
            (Any):

        Raises:
            KeyError: If the component is not registered.

        Examples:
            >>> registry = Registry()
            >>> @registry.register("Module")
            ... class Module:
            ...     def __init__(self, a, b):
            ...         self.a = a
            ...         self.b = b
            >>> config = {"module": {"name": "Module", "a": 1, "b": 2}}
            >>> # registry.register(Module)
            >>> module = registry.build(**config["module"])
            >>> type(module)
            <class 'torch.utils.registry.Module'>
            >>> module.a
            1
            >>> module.b
            2
            >>> module = registry.build(config["module"], a=2)
            >>> module.a
            2
        """

        if isinstance(name, Mapping):
            name = deepcopy(name)
            name, kwargs = name.pop("name"), dict(name, **kwargs)  # type: ignore
        return self[name](*args, **kwargs)  # type: ignore

    def __wrapped__(self, *args, **kwargs):
        pass


GlobalRegistry = Registry()
