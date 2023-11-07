import functools
import weakref

import torch.nn
from torch.nn import Module

from .utils import ExactWeakKeyDictionary, is_lazy_module


class MutationTracker:
    db = ExactWeakKeyDictionary()

    def __init__(self):
        self.mutation_count = 0
        self.watchers = []

    def on_mutation(self, name):
        self.mutation_count += 1
        tmp = self.watchers
        self.watchers = []
        for ref in tmp:
            guarded = ref()
            if guarded is not None:
                guarded.invalidate(ref)

    def track(self, guarded_code):
        self.watchers.append(weakref.ref(guarded_code))


def watch(obj, guarded_code):
    """invalidate guarded_code when obj is mutated"""
    ensure_patched(type(obj))

    if obj not in MutationTracker.db:
        MutationTracker.db[obj] = MutationTracker()
    tracker = MutationTracker.db[obj]
    tracker.track(guarded_code)


def ensure_patched(cls):
    if getattr(cls, "___needs_mutation_patch", True):
        cls.___needs_mutation_patch = False
        original_setattr = cls.__setattr__

        @functools.wraps(original_setattr)
        def custom_setattr(self, key, value):
            try:
                MutationTracker.db[self].on_mutation(key)
            except KeyError:
                pass
            return original_setattr(self, key, value)

        cls.__setattr__ = custom_setattr


class GenerationTracker:
    generation = 0
    dynamic_classes = ExactWeakKeyDictionary()
    generation_values = ExactWeakKeyDictionary()

    @classmethod
    def tag(cls, obj):
        cls.generation_values[obj] = cls.generation

    @staticmethod
    def mark_class_dynamic(cls):
        assert issubclass(cls, torch.nn.Module)
        GenerationTracker.dynamic_classes[cls] = True

    @classmethod
    def get_generation_value(cls, obj):
        if obj not in cls.generation_values:
            return -1
        return cls.generation_values[obj]

    @classmethod
    def check(cls, obj):
        return (
            obj in cls.generation_values
            and cls.generation_values[obj] == cls.generation
        )


def is_dynamic_nn_module(obj):
    """Check for nn.Modules() created dynamically or mutated"""
    if isinstance(obj, torch.nn.Module) and "forward" in obj.__dict__:
        # A monkey patched `.forward` indicates something wacky is going on
        return True
    if hasattr(obj, "torchdynamo_force_dynamic"):
        return obj.torchdynamo_force_dynamic
    if is_lazy_module(obj):
        return False
    dyn = GenerationTracker.dynamic_classes.get(type(obj)) or GenerationTracker.check(
        obj
    )
    return dyn


def install_generation_tagging_init():
    """
    Monkey patch torch.nn.Module.__init__ and torch.nn.Module.__setstate__
    so we can detect nn.Module instances created dynamically inside forward methods.
    """

    if getattr(Module, "___needs_generation_tag_patch", True):
        init = Module.__init__

        def patched_init(self, *args, **kwargs):
            init(self, *args, **kwargs)
            GenerationTracker.tag(self)

        Module.__init__ = patched_init

        setstate = Module.__setstate__

        def patched_setstate(self, state):
            setstate(self, state)
            GenerationTracker.tag(self)

        Module.__setstate__ = patched_setstate

        Module.___needs_generation_tag_patch = False

    GenerationTracker.generation += 1
