"""JIT-related state.

This module stores various pieces of Python-global state relating to the JIT.

This is not intended to be imported directly; please the exposed
functionalities in `torch.jit`.
"""
import os
import weakref

import torch


class EnabledProxy:
    """Stores whether the JIT is enabled or not.

    This is just a wrapper for a bool, so that we get reference semantics
    """

    def __init__(self):
        self.enabled = self.parse_env(
            "PYTORCH_JIT", True, "> Using PyTorch JIT", "> PyTorch JIT DISABLED"
        )

    def parse_env(self, name, default, true_message, false_message):
        value = os.environ.get(name)
        if value is None:
            return default
        if value.lower() in {"1", "true", "yes"}:
            return True
        elif value.lower() in {"0", "false", "no"}:
            return False
        if value == "1v":
            print(true_message)
            return True
        elif value == "0v":
            print(false_message)
            return False
        raise ValueError(f"Unknown setting of {name}. Try using 0 or 1.")

    def __bool__(self):
        return self.enabled


_enabled = EnabledProxy()


def disable():
    _enabled.enabled = False


def enable():
    _enabled.enabled = True


# The Python CompilationUnit. All functions and modules defined in Python will
# live in here. It's defined in Python because doing in cpp creates static
# destruction order issues.
_python_cu = torch._C.CompilationUnit()


# python class => ScriptClass mapping
_script_classes = {}
_name_to_pyclass = {}


def _add_script_class(python_class, script_class):
    _script_classes[python_class] = script_class
    _name_to_pyclass[script_class.qualified_name()] = python_class


def _get_script_class(python_class):
    override = getattr(python_class, "_jit_override_qualname", None)
    if override is not None:
        python_class = _get_python_class(override)
    return _script_classes.get(python_class, None)


def _get_python_class(qualified_name):
    return _name_to_pyclass.get(qualified_name, None)


def _clear_class_state():
    _script_classes.clear()
    _name_to_pyclass.clear()


# Caching: we currently cache compilation of free functions and overloaded functions.
# To cache free functions we hold a weak ref to the function object and
# map to the compiled fn's qualified name.
# To cache overloaded functions we hold a weak ref to the function obj and
# map to all of its overloaded compiled fns.
# In the future we could consider caching more types of objects so that
# aliasing is preserved across separate compilations of the same object.

_jit_caching_layer: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_jit_function_overload_caching: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _try_get_jit_cached_overloads(key):
    qual_names = _jit_function_overload_caching.get(key, None)
    if qual_names:
        return [_python_cu.find_function(qual_name) for qual_name in qual_names]
    else:
        return None


def _set_jit_overload_cache(key, compiled_fns):
    _jit_function_overload_caching[key] = [fn.qualified_name for fn in compiled_fns]


def _try_get_jit_cached_function(key):
    if getattr(key, "__disable_jit_function_caching__", False) is True:
        return None
    qual_name = _jit_caching_layer.get(key, None)
    if qual_name:
        return _python_cu.find_function(qual_name)
    else:
        return None


def _set_jit_function_cache(key, value):
    # only free functions currently supported
    assert isinstance(value, torch.jit.ScriptFunction)
    _jit_caching_layer[key] = value.qualified_name
