"""Mutation tracking and dynamic module detection system for Dynamo.

This module provides mechanisms to track and respond to mutations in PyTorch modules
and detect dynamically created or modified modules.

Key components:
- MutationTracker: Tracks mutations to objects and invalidates associated cached code
- GenerationTracker: Tracks module creation timing to identify dynamic instances
- Patching system for nn.Module to detect mutations and dynamic creation

The system ensures that Dynamo's optimizations remain valid by detecting and responding
to runtime changes in module state and structure.
"""

import functools
import weakref
from collections.abc import MutableMapping
from typing import Any

import torch.nn
from torch.nn import Module

from . import config
from .utils import ExactWeakKeyDictionary, nn_module_has_global_hooks


unpatched_nn_module_init = torch.nn.Module.__init__


class MutationTracker:
    db: ExactWeakKeyDictionary = ExactWeakKeyDictionary()

    def __init__(self) -> None:
        self.mutation_count: int = 0
        self.watchers: list[weakref.ReferenceType[Any]] = []

    def on_mutation(self, name: str) -> None:
        self.mutation_count += 1
        tmp = self.watchers
        self.watchers = []
        for ref in tmp:
            guarded = ref()
            if guarded is not None:
                guarded.invalidate(ref)

    def track(self, guarded_code: Any) -> None:
        self.watchers.append(weakref.ref(guarded_code))


def watch(obj: Any, guarded_code: Any) -> None:
    """invalidate guarded_code when obj is mutated"""
    ensure_patched(type(obj))

    if obj not in MutationTracker.db:
        MutationTracker.db[obj] = MutationTracker()
    tracker = MutationTracker.db[obj]
    tracker.track(guarded_code)


def ensure_patched(cls: Any) -> None:
    if getattr(cls, "___needs_mutation_patch", True):
        cls.___needs_mutation_patch = False
        original_setattr = cls.__setattr__

        @functools.wraps(original_setattr)
        def custom_setattr(self: Any, key: str, value: Any) -> None:
            try:
                MutationTracker.db[self].on_mutation(key)
            except KeyError:
                pass
            return original_setattr(self, key, value)

        cls.__setattr__ = custom_setattr


class GenerationTracker:
    generation: int = 0
    dynamic_classes: ExactWeakKeyDictionary = ExactWeakKeyDictionary()
    generation_values: ExactWeakKeyDictionary = ExactWeakKeyDictionary()

    @classmethod
    def tag(cls, obj: Any) -> None:
        cls.generation_values[obj] = cls.generation

    @staticmethod
    def mark_class_dynamic(cls: type[torch.nn.Module]) -> None:
        assert issubclass(cls, torch.nn.Module)
        GenerationTracker.dynamic_classes[cls] = True

    @classmethod
    def get_generation_value(cls, obj: Any) -> int:
        if obj not in cls.generation_values:
            return -1
        return cls.generation_values[obj]

    @classmethod
    def check(cls, obj: Any) -> bool:
        return (
            obj in cls.generation_values
            and cls.generation_values[obj] == cls.generation
        )

    @classmethod
    def clear(cls) -> None:
        cls.generation = 0
        cls.dynamic_classes = ExactWeakKeyDictionary()
        cls.generation_values = ExactWeakKeyDictionary()


def is_dynamic_nn_module(obj: Any, is_export: bool) -> bool:
    """Check for nn.Modules() created dynamically or mutated"""
    if isinstance(obj, torch.nn.Module) and (
        "forward" in obj.__dict__ or isinstance(obj, (dict, MutableMapping))
    ):
        # A monkey patched `.forward` indicates something wacky is going on
        # Similarly a nn module also subclassed as a dict is unusual.
        return True
    if hasattr(obj, "torchdynamo_force_dynamic"):
        return obj.torchdynamo_force_dynamic
    if (
        isinstance(obj, torch.nn.Module)
        and config.inline_inbuilt_nn_modules
        and (not is_export or config.install_free_tensors)
    ):
        return True

    if isinstance(obj, torch.nn.Module) and nn_module_has_global_hooks():
        return True
    dyn = GenerationTracker.dynamic_classes.get(type(obj)) or GenerationTracker.check(
        obj
    )
    return dyn


def install_generation_tagging_init() -> None:
    """
    Monkey patch torch.nn.Module.__init__ and torch.nn.Module.__setstate__
    so we can detect nn.Module instances created dynamically inside forward methods.
    """

    if getattr(Module, "___needs_generation_tag_patch", True):
        init = Module.__init__

        def patched_init(self: Module, *args: Any, **kwargs: Any) -> None:
            init(self, *args, **kwargs)
            GenerationTracker.tag(self)

        Module.__init__ = patched_init  # type: ignore[method-assign]

        setstate = Module.__setstate__

        def patched_setstate(self: Module, state: Any) -> None:
            setstate(self, state)
            GenerationTracker.tag(self)

        Module.__setstate__ = patched_setstate  # type: ignore[method-assign]

        Module.___needs_generation_tag_patch = False  # type: ignore[attr-defined]

    GenerationTracker.generation += 1
