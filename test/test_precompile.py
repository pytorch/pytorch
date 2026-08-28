# Owner(s): ["oncall: pt2"]
import base64
import bisect
import contextvars
import copy
import cProfile
import dataclasses
import decimal
import functools
import hashlib
import heapq
import importlib
import inspect
import io
import math
import operator
import os
import pickle
import subprocess
import sys
import tempfile
import textwrap
import types
import typing
import unittest
import weakref
from collections import defaultdict, deque

import torch
import torch.utils._pytree as _pytree
from torch._dynamo import graph_break as _precompile_dynamo_break_here
from torch._dynamo.decorators import mark_dynamic, mark_unbacked
from torch._dynamo.exc import PackageError
from torch._precompile import PrecompileError
from torch.testing import make_tensor
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCUDA,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfTorchDynamo,
    TestCase,
)


if torch.distributed.is_available():
    from torch.distributed._functional_collectives import AsyncCollectiveTensor


# A module-level (global) model + a function referencing it, to exercise the
# constant-tensor guard against a baked global.
_GLOBAL_TENSOR = torch.randn(3)
_DYNAMO_TENSOR_DEFAULT = torch.randn(3)
_DYNAMO_CONTAINER_IDENTITY = [1]


def _precompile_dynamo_dynamic(x):
    return x.sin() + x.shape[0]


def _precompile_dynamo_torch_sin(x):
    return torch.sin(x)


def _precompile_dynamo_global_tensor(x):
    return x + _GLOBAL_TENSOR


class _PrecompileDynamoTensorClassState:
    value = _DYNAMO_TENSOR_DEFAULT

    @classmethod
    def apply(cls, x):
        return x + cls.value

    @classmethod
    def pure(cls, x):
        return x + 1


def _precompile_dynamo_class_tensor_state(x):
    return _PrecompileDynamoTensorClassState.apply(x)


def _precompile_dynamo_unused_class_tensor_state(x):
    return _PrecompileDynamoTensorClassState.pure(x)


class _PrecompileDynamoTensorDefaultMethod:
    def apply(self, x, bias=_DYNAMO_TENSOR_DEFAULT):
        return x + bias


def _precompile_dynamo_input_method_tensor_default(x, obj):
    return obj.apply(x)


def _precompile_dynamo_imported_module_identity(x):
    import _precompile_identity_module

    return x + 1 if x is _precompile_identity_module.TOKEN else x - 1


def _precompile_dynamo_imported_callable_identity(x):
    import _precompile_identity_module

    return x + 1 if x is _precompile_identity_module.CALLABLE() else x - 1


def _precompile_dynamo_imported_nested_callable_identity(x):
    import _precompile_identity_module

    return x + 1 if x is _precompile_identity_module.SUBMODULE.CALLABLE() else x - 1


def _precompile_dynamo_imported_dynamic_identity(x):
    import _precompile_identity_module

    return x + 1 if x is _precompile_identity_module.DYNAMIC else x - 1


def _precompile_dynamo_imported_module_alias_identity(x):
    import torch as local_torch

    alias = local_torch
    return x + 1 if x is alias._precompile_identity_token else x - 1


def _precompile_dynamo_dotted_import_identity(x):
    import _precompile_identity_package.sub.mod as identity_module

    return x + 1 if x is identity_module.TOKEN else x - 1


_DYNAMO_UNRELATED_IMPORT_RECEIVER = types.SimpleNamespace(SECRET=3)


def _precompile_dynamo_imported_unrelated_attribute(x):
    import _precompile_identity_module

    return (
        x + _DYNAMO_UNRELATED_IMPORT_RECEIVER.SECRET + _precompile_identity_module.USED
    )


def _precompile_dynamo_import_shadowed_in_nested_scope(x):
    import _precompile_identity_module

    def helper(_precompile_identity_module, value):
        return value + _precompile_identity_module.SECRET

    return (
        helper(_DYNAMO_UNRELATED_IMPORT_RECEIVER, x)
        + _precompile_identity_module.USED * 0
    )


def _precompile_dynamo_local_import_tensor(x):
    import _precompile_local_tensor_module

    return x + _precompile_local_tensor_module.TENSOR


def _precompile_dynamo_local_import_tensor_default(x):
    from _precompile_local_tensor_module import helper

    return helper(x)


def _precompile_dynamo_library_object_identity(x):
    return torch._precompile_reviewlib.helper(torch._precompile_reviewlib.HOLDER, x)


def _precompile_dynamo_library_nested_helper_identity(x):
    return torch._precompile_reviewlib.outer(x)


def _precompile_dynamo_library_container_helper_identity(x):
    return torch._precompile_reviewlib.container_outer(x)


def _precompile_dynamo_library_partial_helper_identity(x):
    return torch._precompile_reviewlib.partial_outer(x)


def _precompile_dynamo_library_callable_helper_identity(x):
    return torch._precompile_reviewlib.callable_outer(x)


def _precompile_dynamo_library_bound_partial_identity(x):
    return torch._precompile_reviewlib.bound_partial_outer(x)


def _precompile_dynamo_library_keyword_partial_identity(x):
    return torch._precompile_reviewlib.keyword_partial_outer(x)


def _precompile_dynamo_library_property_callable_identity(x):
    return torch._precompile_reviewlib.property_callable_outer(x)


def _precompile_dynamo_library_factory_identity(x):
    return torch._precompile_reviewlib.factory_outer(x)


def _precompile_dynamo_library_module_value_identity(x):
    return torch._precompile_reviewlib.module_outer(x)


def _precompile_dynamo_library_dynamic_module_identity(x):
    return torch._precompile_reviewlib.dynamic_module_outer(x)


def _precompile_dynamo_library_direct_method_identity(x):
    return torch._precompile_reviewlib.direct_method_outer(x)


def _precompile_dynamo_library_direct_property_identity(x):
    return torch._precompile_reviewlib.direct_property_outer(x)


def _precompile_dynamo_library_late_import_identity(x):
    return torch._precompile_latelib.outer(x)


def _precompile_dynamo_metadata_helper(x):
    return x + 1


def _precompile_dynamo_calls_metadata_helper(x):
    return _precompile_dynamo_metadata_helper(x)


def _precompile_dynamo_construct_relu(x):
    return torch.nn.ReLU()(x)


def _precompile_dynamo_mse_loss(x, target):
    return torch.nn.functional.mse_loss(x, target)


_DYNAMO_EXTERNAL_ALIAS_TOKEN = None


def _precompile_dynamo_external_alias_template(x):
    return x + 1 if x is _DYNAMO_EXTERNAL_ALIAS_TOKEN else x - 1


_DYNAMO_EXTERNAL_ALIAS_MODULE = types.ModuleType("_dynamo_external_alias")
_DYNAMO_EXTERNAL_ALIAS_MODULE._DYNAMO_EXTERNAL_ALIAS_TOKEN = _DYNAMO_TENSOR_DEFAULT
_DYNAMO_EXTERNAL_ALIAS_MODULE.helper = types.FunctionType(
    _precompile_dynamo_external_alias_template.__code__,
    _DYNAMO_EXTERNAL_ALIAS_MODULE.__dict__,
    "helper",
)


def _precompile_dynamo_external_wrapper_template(x):
    return _DYNAMO_EXTERNAL_ALIAS_MODULE.helper(x)


_DYNAMO_EXTERNAL_WRAPPER_MODULE = types.ModuleType("_dynamo_external_wrapper")
_DYNAMO_EXTERNAL_WRAPPER_MODULE._DYNAMO_EXTERNAL_ALIAS_MODULE = (
    _DYNAMO_EXTERNAL_ALIAS_MODULE
)
_DYNAMO_EXTERNAL_WRAPPER_MODULE.wrapper = types.FunctionType(
    _precompile_dynamo_external_wrapper_template.__code__,
    _DYNAMO_EXTERNAL_WRAPPER_MODULE.__dict__,
    "wrapper",
)


def _precompile_dynamo_external_module_identity(x):
    return _DYNAMO_EXTERNAL_WRAPPER_MODULE.wrapper(x)


_precompile_box = []


def _precompile_dynamo_torch_helper_template(x):
    return x + 1 if x is _precompile_box[0] else x - 1


def _precompile_dynamo_torch_helper_identity(x):
    return torch._precompile_helper(x)


def _precompile_dynamo_torch_module_helper_template(module, x):
    return module.secret(x)


def _precompile_dynamo_torch_module_getattr_template(module, x):
    name = "secret"
    return getattr(module, name)(x)


def _precompile_dynamo_torch_module_helper_identity(module, x):
    return torch._precompile_module_helper(module, x)


def _precompile_dynamo_torch_callable_template(self, x):
    return x + 1 if x is _precompile_box[0] else x - 1


def _precompile_dynamo_torch_callable_identity(x):
    return torch._precompile_callable(x)


def _precompile_dynamo_torch_getitem_template(self, x):
    return x + 1 if x is _precompile_box[0] else x - 1


def _precompile_dynamo_torch_getitem_identity(x):
    return torch._precompile_getitem[x]


def _precompile_dynamo_torch_class_getitem_template(cls, x):
    return x + 1 if x is _precompile_box[0] else x - 1


def _precompile_dynamo_torch_class_getitem_identity(x):
    return torch._precompile_generic[x]


def _precompile_dynamo_torch_metaclass_getitem_template(cls, x):
    return x + 1 if x is _precompile_box[0] else x - 1


def _precompile_dynamo_torch_metaclass_getitem_identity(x):
    return torch._precompile_meta_generic[x]


class _PrecompileDynamoLibraryIndexerModule(torch.nn.Module):
    def __init__(self, helper):
        super().__init__()
        self.helper = helper

    def forward(self, x):
        return self.helper[x]


def _precompile_dynamo_function_metadata_helper():
    return None


_DYNAMO_FUNCTION_GLOBAL_ALIAS = _DYNAMO_TENSOR_DEFAULT
_precompile_dynamo_function_metadata_helper.__annotations__["token"] = (
    _DYNAMO_TENSOR_DEFAULT
)
_DYNAMO_FUNCTION_METADATA_HELPERS = [_precompile_dynamo_function_metadata_helper]


def _precompile_dynamo_function_globals_identity(x):
    token = _precompile_dynamo_function_metadata_helper.__globals__[
        "_DYNAMO_FUNCTION_GLOBAL_ALIAS"
    ]
    return x + 1 if x is token else x - 1


def _precompile_dynamo_function_annotations_identity(x):
    token = _precompile_dynamo_function_metadata_helper.__annotations__["token"]
    return x + 1 if x is token else x - 1


def _precompile_dynamo_function_container_globals_identity(x):
    token = _DYNAMO_FUNCTION_METADATA_HELPERS[0].__globals__[
        "_DYNAMO_FUNCTION_GLOBAL_ALIAS"
    ]
    return x + 1 if x is token else x - 1


def _precompile_dynamo_function_container_annotations_identity(x):
    token = _DYNAMO_FUNCTION_METADATA_HELPERS[0].__annotations__["token"]
    return x + 1 if x is token else x - 1


def _precompile_dynamo_external_module_method_template(module, x):
    return module.secret(x)


_DYNAMO_EXTERNAL_MODULE_METHOD = types.ModuleType("_dynamo_external_module_method")
_DYNAMO_EXTERNAL_MODULE_METHOD.invoke = types.FunctionType(
    _precompile_dynamo_external_module_method_template.__code__,
    _DYNAMO_EXTERNAL_MODULE_METHOD.__dict__,
    "invoke",
)


def _precompile_dynamo_call_external_module_method(module, x):
    return _DYNAMO_EXTERNAL_MODULE_METHOD.invoke(module, x)


def _precompile_dynamo_tensor_default(x, bias=_DYNAMO_TENSOR_DEFAULT):
    return x + bias


class _PrecompileDynamoTensorDefault:
    def __init__(self):
        self.tensor = torch.randn(3)


_DYNAMO_OBJECT_DEFAULT = _PrecompileDynamoTensorDefault()


class _PrecompileDynamoTupleDefault(tuple):
    __slots__ = ()

    @property
    def tensor(self):
        return _DYNAMO_TENSOR_DEFAULT


_DYNAMO_TUPLE_DEFAULT = _PrecompileDynamoTupleDefault((1,))


def _precompile_dynamo_object_default(x, state=_DYNAMO_OBJECT_DEFAULT):
    return x + state.tensor


def _precompile_dynamo_tuple_subclass_default(x, state=_DYNAMO_TUPLE_DEFAULT):
    return x + state.tensor


def _precompile_dynamo_container_identity(container, x):
    return x + 1 if container is _DYNAMO_CONTAINER_IDENTITY else x - 1


def _precompile_dynamo_identity_helper():
    pass


_precompile_dynamo_identity_helper.value = _DYNAMO_TENSOR_DEFAULT
_DYNAMO_IDENTITY_HELPERS = [_precompile_dynamo_identity_helper]


def _precompile_dynamo_helper_attribute_identity(x):
    return x + 1 if x is _precompile_dynamo_identity_helper.value else x - 1


def _precompile_dynamo_helper_container_attribute_identity(x):
    return x + 1 if x is _DYNAMO_IDENTITY_HELPERS[0].value else x - 1


class _PrecompileDynamoCustomDescriptor:
    def __get__(self, instance, owner):
        return instance.unused


class _PrecompileDynamoIdentityDescriptor:
    used = 2
    unused = _DYNAMO_TENSOR_DEFAULT
    custom = _PrecompileDynamoCustomDescriptor()

    @property
    def value(self):
        attribute = "unused"
        return getattr(self, attribute)


_DYNAMO_IDENTITY_DESCRIPTOR = _PrecompileDynamoIdentityDescriptor()


class _PrecompileDynamoSlottedIdentity:
    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value


_DYNAMO_SLOTTED_IDENTITY = _PrecompileDynamoSlottedIdentity(_DYNAMO_TENSOR_DEFAULT)


class _PrecompileDynamoModuleAttributeIdentity(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value


def _precompile_dynamo_module_attribute_identity(module, x):
    return x + 1 if module.value is _DYNAMO_TENSOR_DEFAULT else x - 1


class _PrecompileDynamoModuleForwardGlobalIdentity(torch.nn.Module):
    def forward(self, x):
        return self.helper(x)

    def helper(self, x):
        return x + 1 if x is _DYNAMO_TENSOR_DEFAULT else x - 1


class _PrecompileDynamoModuleMethodGlobalIdentity(torch.nn.Module):
    def forward(self, x):
        return x.sin()

    def helper(self, x):
        return x + 1 if x is _DYNAMO_TENSOR_DEFAULT else x - 1

    def secret(self, x):
        return x + 1 if x is _DYNAMO_TENSOR_DEFAULT else x - 1


class _PrecompileDynamoModuleGetitemGlobalIdentity(torch.nn.Module):
    def forward(self, x):
        return x.sin()

    def __getitem__(self, x):
        return x + 1 if x is _DYNAMO_TENSOR_DEFAULT else x - 1


def _precompile_dynamo_call_identity_module(module, x):
    return module(x)


def _precompile_dynamo_call_identity_module_helper(module, x):
    return module.helper(x)


def _precompile_dynamo_call_identity_module_helper_nested(module, x):
    def call(module, value):
        return module.helper(value)

    return call(module, x)


def _precompile_dynamo_call_identity_module_methodcaller(module, x):
    return operator.methodcaller("helper", x)(module)


def _precompile_dynamo_getitem_identity_module(module, x):
    return module[x]


def _precompile_dynamo_stdlib_module_dynamic_identity(x):
    return x + 1 if x is math.__getattribute__("_precompile_token") else x - 1


def _precompile_dynamo_stdlib_module_attrgetter_identity(x):
    token = operator.attrgetter("_precompile_token")(math)
    return x + 1 if x is token else x - 1


def _precompile_dynamo_stdlib_module_getattr_identity(x):
    name = "_precompile_token"
    token = getattr(math, name)
    return x + 1 if x is token else x - 1


_DYNAMO_STDLIB_MODULE_GETTERS = [math.__getattribute__]


def _precompile_dynamo_stdlib_module_bound_getattr_identity(x):
    token = _DYNAMO_STDLIB_MODULE_GETTERS[0]("_precompile_token")
    return x + 1 if x is token else x - 1


def _precompile_dynamo_call_nested_identity_module(modules, x):
    return modules[0](x)


class _PrecompileDynamoDynamicIdentity:
    def __init__(self, value):
        self._value = value

    def __getattr__(self, name):
        if name == "value":
            return self._value
        raise AttributeError(name)


_DYNAMO_DYNAMIC_IDENTITY = _PrecompileDynamoDynamicIdentity(_DYNAMO_TENSOR_DEFAULT)


class _PrecompileDynamoGetattributeIdentity:
    value = 0

    def __init__(self, value):
        self._value = value

    def __getattribute__(self, name):
        if name == "value":
            return object.__getattribute__(self, "_value")
        return object.__getattribute__(self, name)


_DYNAMO_GETATTRIBUTE_IDENTITY = _PrecompileDynamoGetattributeIdentity(
    _DYNAMO_TENSOR_DEFAULT
)


def _make_precompile_dynamo_dynamic_module():
    module = types.ModuleType("_precompile_dynamo_dynamic_module")
    module.TOKEN = _DYNAMO_TENSOR_DEFAULT

    def module_getattr(name):
        if name == "value":
            return module.__dict__["TOKEN"]
        raise AttributeError(name)

    module.__getattr__ = module_getattr
    return module


_DYNAMO_DYNAMIC_MODULE = _make_precompile_dynamo_dynamic_module()
_DYNAMO_GETATTR = getattr


def _precompile_dynamo_getattr_identity(x):
    return x + 1 if x is _DYNAMO_GETATTR(_DYNAMO_DYNAMIC_MODULE, "TOKEN") else x - 1


_DYNAMO_EXTERNAL_GETATTR_MODULE = types.ModuleType("_dynamo_external_getattr")
_DYNAMO_EXTERNAL_GETATTR_MODULE._DYNAMO_GETATTR = getattr
_DYNAMO_EXTERNAL_GETATTR_MODULE._DYNAMO_DYNAMIC_MODULE = _DYNAMO_DYNAMIC_MODULE
_DYNAMO_EXTERNAL_GETATTR_MODULE.helper = types.FunctionType(
    _precompile_dynamo_getattr_identity.__code__,
    _DYNAMO_EXTERNAL_GETATTR_MODULE.__dict__,
    "helper",
)


def _precompile_dynamo_external_getattr_identity(x):
    return _DYNAMO_EXTERNAL_GETATTR_MODULE.helper(x)


class _PrecompileDynamoCallableIdentity:
    def __call__(self):
        return _DYNAMO_TENSOR_DEFAULT


_DYNAMO_CALLABLE_IDENTITY = _PrecompileDynamoCallableIdentity()


class _PrecompileDynamoDequeIdentity(deque):
    pass


_DYNAMO_DEQUE_IDENTITY = _PrecompileDynamoDequeIdentity([_DYNAMO_TENSOR_DEFAULT])
_DYNAMO_DEQUE_IDENTITY.note = "unrelated instance state"
_DYNAMO_CONTEXT_IDENTITY = contextvars.ContextVar("_DYNAMO_CONTEXT_IDENTITY")
_DYNAMO_CONTEXT_IDENTITY.set(_DYNAMO_TENSOR_DEFAULT)


class _PrecompileDynamoWeakProxyIdentity:
    __slots__ = ("value", "__weakref__")

    def __init__(self, value):
        self.value = value


_DYNAMO_WEAK_PROXY_HOLDER = _PrecompileDynamoWeakProxyIdentity(_DYNAMO_TENSOR_DEFAULT)
_DYNAMO_WEAK_PROXY_IDENTITY = weakref.proxy(_DYNAMO_WEAK_PROXY_HOLDER)


class _PrecompileDynamoTypeIdentity:
    def __new__(cls):
        return _DYNAMO_TENSOR_DEFAULT


class _PrecompileDynamoNestedDescriptor:
    unused = _DYNAMO_TENSOR_DEFAULT

    @property
    def value(self):
        def inner():
            return self.unused

        return inner()


_DYNAMO_NESTED_DESCRIPTOR = _PrecompileDynamoNestedDescriptor()
_DYNAMO_INPUT_WEAKREF = weakref.ref(_DYNAMO_TENSOR_DEFAULT)


def _precompile_dynamo_descriptor_identity(x):
    return x + 1 if x is _DYNAMO_IDENTITY_DESCRIPTOR.value else x - 1


def _precompile_dynamo_custom_descriptor_identity(x):
    return x + 1 if x is _DYNAMO_IDENTITY_DESCRIPTOR.custom else x - 1


def _precompile_dynamo_slotted_identity(x):
    return x + 1 if x is _DYNAMO_SLOTTED_IDENTITY.value else x - 1


def _precompile_dynamo_dynamic_identity(x):
    return x + 1 if x is _DYNAMO_DYNAMIC_IDENTITY.value else x - 1


def _precompile_dynamo_getattribute_identity(x):
    return x + 1 if x is _DYNAMO_GETATTRIBUTE_IDENTITY.value else x - 1


def _precompile_dynamo_module_getattr_identity(x):
    return x + 1 if x is _DYNAMO_DYNAMIC_MODULE.value else x - 1


def _precompile_dynamo_callable_identity(x):
    return x + 1 if x is _DYNAMO_CALLABLE_IDENTITY() else x - 1


def _precompile_dynamo_deque_identity(x):
    return x + 1 if x is _DYNAMO_DEQUE_IDENTITY[0] else x - 1


def _precompile_dynamo_context_identity(x):
    return x + 1 if x is _DYNAMO_CONTEXT_IDENTITY.get() else x - 1


def _precompile_dynamo_weak_proxy_identity(x):
    return x + 1 if x is _DYNAMO_WEAK_PROXY_IDENTITY.value else x - 1


def _precompile_dynamo_type_identity(x):
    return x + 1 if x is _PrecompileDynamoTypeIdentity() else x - 1


def _precompile_dynamo_nested_descriptor_identity(x):
    return x + 1 if x is _DYNAMO_NESTED_DESCRIPTOR.value else x - 1


def _precompile_dynamo_weakref_identity(x):
    return x + 1 if x is _DYNAMO_INPUT_WEAKREF() else x - 1


def _precompile_dynamo_unrelated_attribute(token, x):
    return x + token.sum() * 0 + _DYNAMO_IDENTITY_DESCRIPTOR.used


def _precompile_dynamo_varargs(*xs):
    return xs[0] + xs[1]


def _precompile_dynamo_varkw(x, /, **kwargs):
    return x + kwargs["x"]


def _precompile_dynamo_scalar(x, scale):
    return x + scale


def _precompile_dynamo_scalar_branch(x, scale):
    if scale == 2:
        return x.sin()
    return x.cos()


def _precompile_dynamo_many_variants(x, mode):
    if mode == "m0":
        return x + 0
    if mode == "m1":
        return x + 1
    if mode == "m2":
        return x + 2
    if mode == "m3":
        return x + 3
    if mode == "m4":
        return x + 4
    if mode == "m5":
        return x + 5
    if mode == "m6":
        return x + 6
    if mode == "m7":
        return x + 7
    if mode == "m8":
        return x + 8
    return x + 9


def _precompile_dynamo_callable(x, op):
    return op(x)


_PRECOMPILE_DYNAMO_SCALAR_IDENTITY = int("1000001")
_PRECOMPILE_DYNAMO_SCALAR_VALUE = 1
_PRECOMPILE_DYNAMO_ENV_IDENTITY_A = 1
_PRECOMPILE_DYNAMO_ENV_IDENTITY_B = 1
_PRECOMPILE_DYNAMO_NAN_IDENTITY = float("nan")
_PRECOMPILE_DYNAMO_INDEX = 0
_PRECOMPILE_DYNAMO_KEY = "value"
_PRECOMPILE_DYNAMO_OBJECT_IDENTITY = object()


def _precompile_dynamo_scalar_identity(x, token):
    if token is _PRECOMPILE_DYNAMO_SCALAR_IDENTITY:
        return x.sin()
    return x.cos()


def _precompile_dynamo_code_literal_identity(x, token):
    literal = 1000003
    if token is literal:
        return x.sin()
    return x.cos()


def _precompile_dynamo_input_identity(x, left, right):
    if left is right:
        return x.sin()
    return x.cos()


def _precompile_dynamo_object_identity_predicate(value):
    return value is _PRECOMPILE_DYNAMO_OBJECT_IDENTITY


def _precompile_dynamo_map_identity(x, values):
    if any(map(_precompile_dynamo_object_identity_predicate, values)):
        return x.sin()
    return x.cos()


def _precompile_dynamo_filter_identity(x, values):
    if any(filter(_precompile_dynamo_object_identity_predicate, values)):
        return x.sin()
    return x.cos()


def _precompile_dynamo_reduce_predicate(found, value):
    return found or _precompile_dynamo_object_identity_predicate(value)


def _precompile_dynamo_reduce_identity(x, values):
    if functools.reduce(
        _precompile_dynamo_reduce_predicate,
        values,
        False,
    ):
        return x.sin()
    return x.cos()


def _precompile_dynamo_default_identity(x, token=_PRECOMPILE_DYNAMO_SCALAR_IDENTITY):
    if token is _PRECOMPILE_DYNAMO_SCALAR_IDENTITY:
        return x.sin()
    return x.cos()


def _precompile_dynamo_environment_identity_helper(token):
    torch._dynamo.graph_break()
    return token is _PRECOMPILE_DYNAMO_SCALAR_IDENTITY


def _precompile_dynamo_environment_only_identity(x):
    if _precompile_dynamo_environment_identity_helper(
        _PRECOMPILE_DYNAMO_SCALAR_IDENTITY
    ):
        return x.sin()
    return x.cos()


def _precompile_dynamo_tensor_equals_nan(x):
    return x == _PRECOMPILE_DYNAMO_NAN_IDENTITY


def _precompile_dynamo_opaque_return_identity(x, token):
    return x.sin() if token is decimal.getcontext() else x.cos()


def _precompile_dynamo_singleton_identity(x, token):
    return x.sin() if token is None else x.cos()


def _precompile_dynamo_ellipsis_default(x, token=...):
    return x.sin() if token is ... else x.cos()


class _PrecompileDynamoDefaultIdentityMethod:
    def apply(self, token=_PRECOMPILE_DYNAMO_SCALAR_IDENTITY):
        return token is _PRECOMPILE_DYNAMO_SCALAR_IDENTITY


def _precompile_dynamo_calls_default_identity_method(x, obj):
    return x.sin() if obj.apply() else x.cos()


def _precompile_dynamo_calls_environment_identity_method(x, obj):
    return x.sin() if obj.apply(_PRECOMPILE_DYNAMO_SCALAR_IDENTITY) else x.cos()


def _precompile_dynamo_global_index(x):
    return x[_PRECOMPILE_DYNAMO_INDEX].sin()


def _precompile_dynamo_global_key(x, values):
    return x + values[_PRECOMPILE_DYNAMO_KEY]


def _precompile_dynamo_nested_scalar_identity(x, tokens):
    if tokens[0] is _PRECOMPILE_DYNAMO_SCALAR_IDENTITY:
        return x.sin()
    return x.cos()


def _precompile_dynamo_get_scalar_identity():
    return _PRECOMPILE_DYNAMO_SCALAR_IDENTITY


def _precompile_dynamo_helper_scalar_identity(x, token):
    if token is _precompile_dynamo_get_scalar_identity():
        return x.sin()
    return x.cos()


def _precompile_dynamo_is_same(a, b):
    return a is b


def _precompile_dynamo_keyword_is_same(ignore, a, b):
    return a is b


def _precompile_dynamo_is_same_with_ignored(token, ignored):
    return token is _PRECOMPILE_DYNAMO_SCALAR_IDENTITY


def _make_precompile_dynamo_closure_scalar_identity():
    captured = _PRECOMPILE_DYNAMO_SCALAR_IDENTITY

    def same(token):
        return token is captured

    return same


_PRECOMPILE_DYNAMO_CLOSURE_SCALAR_IDENTITY = (
    _make_precompile_dynamo_closure_scalar_identity()
)


def _precompile_dynamo_called_scalar_identity(x, token):
    if _precompile_dynamo_is_same(token, _PRECOMPILE_DYNAMO_SCALAR_IDENTITY):
        return x.sin()
    return x.cos()


def _precompile_dynamo_closure_scalar_identity(x, token):
    if _PRECOMPILE_DYNAMO_CLOSURE_SCALAR_IDENTITY(token):
        return x.sin()
    return x.cos()


def _precompile_dynamo_conditional_call_arg_scalar_identity(x, token, cond):
    if _precompile_dynamo_is_same_with_ignored(token, 1 if cond else 2):
        return x.sin()
    return x.cos()


def _precompile_dynamo_starargs_scalar_identity(x, token):
    if _precompile_dynamo_is_same(*(token, _PRECOMPILE_DYNAMO_SCALAR_IDENTITY)):
        return x.sin()
    return x.cos()


def _precompile_dynamo_operator_scalar_identity(x, token):
    if operator.is_(token, _PRECOMPILE_DYNAMO_SCALAR_IDENTITY):
        return x.sin()
    return x.cos()


def _precompile_dynamo_operator_starargs_scalar_identity(x, token):
    if operator.is_(*(token, _PRECOMPILE_DYNAMO_SCALAR_IDENTITY)):
        return x.sin()
    return x.cos()


_PRECOMPILE_DYNAMO_PARTIAL_SCALAR_IDENTITY = functools.partial(
    _precompile_dynamo_is_same, b=_PRECOMPILE_DYNAMO_SCALAR_IDENTITY
)
_PRECOMPILE_DYNAMO_KEYWORD_PARTIAL_SCALAR_IDENTITY = functools.partial(
    _precompile_dynamo_keyword_is_same,
    0,
    b=_PRECOMPILE_DYNAMO_SCALAR_IDENTITY,
)


def _precompile_dynamo_partial_scalar_identity(x, token):
    if _PRECOMPILE_DYNAMO_PARTIAL_SCALAR_IDENTITY(token):
        return x.sin()
    return x.cos()


def _precompile_dynamo_keyword_partial_scalar_identity(x, token):
    if _PRECOMPILE_DYNAMO_KEYWORD_PARTIAL_SCALAR_IDENTITY(a=token):
        return x.sin()
    return x.cos()


def _precompile_dynamo_keyword_scalar_identity(x, token):
    if _precompile_dynamo_keyword_is_same(
        a=token, ignore=0, b=_PRECOMPILE_DYNAMO_SCALAR_IDENTITY
    ):
        return x.sin()
    return x.cos()


_PRECOMPILE_DYNAMO_SCALAR_IDENTITY_HELPERS = [_precompile_dynamo_is_same]


def _precompile_dynamo_container_helper_scalar_identity(x, token):
    if _PRECOMPILE_DYNAMO_SCALAR_IDENTITY_HELPERS[0](
        token, _PRECOMPILE_DYNAMO_SCALAR_IDENTITY
    ):
        return x.sin()
    return x.cos()


def _precompile_dynamo_list_extend_scalar_identity(x, token):
    source = [token]
    values = [*source]
    value = values[0]
    if value is _PRECOMPILE_DYNAMO_SCALAR_IDENTITY:
        return x.sin()
    return x.cos()


def _precompile_dynamo_dict_update_scalar_identity(x, token):
    source = {"token": token}
    value = {**source}["token"]
    if value is _PRECOMPILE_DYNAMO_SCALAR_IDENTITY:
        return x.sin()
    return x.cos()


def _precompile_dynamo_walrus_scalar_identity(x, token):
    if (value := token) is _PRECOMPILE_DYNAMO_SCALAR_IDENTITY:  # noqa: F841
        return x.sin()
    return x.cos()


def _precompile_dynamo_swap_scalar_identity(x, token):
    left, right = token, 0
    left, right = right, left
    if right is _PRECOMPILE_DYNAMO_SCALAR_IDENTITY:
        return x.sin()
    return x.cos()


def _precompile_dynamo_local_import_scalar_identity(x, token):
    import operator as local_operator

    if local_operator.is_(token, _PRECOMPILE_DYNAMO_SCALAR_IDENTITY):
        return x.sin()
    return x.cos()


class _PrecompileDynamoScalarIdentityHelper:
    token = _PRECOMPILE_DYNAMO_SCALAR_IDENTITY

    def same(self, token):
        return token is self.token

    @staticmethod
    def static_same(token, expected):
        return token is expected

    @classmethod
    def class_same(cls, token):
        return token is cls.token

    def __call__(self, token):
        return token is self.token


_PRECOMPILE_DYNAMO_SCALAR_IDENTITY_HELPER = _PrecompileDynamoScalarIdentityHelper()


class _PrecompileDynamoInitScalarIdentity:
    def __init__(self, token):
        self.same = token is _PRECOMPILE_DYNAMO_SCALAR_IDENTITY


class _PrecompileDynamoScalarIdentityMeta(type):
    def __call__(cls, token):
        return token is _PRECOMPILE_DYNAMO_SCALAR_IDENTITY


class _PrecompileDynamoMetaclassScalarIdentity(
    metaclass=_PrecompileDynamoScalarIdentityMeta
):
    pass


class _PrecompileDynamoBaseScalarIdentityHelper:
    def same(self, token):
        return token is _PRECOMPILE_DYNAMO_SCALAR_IDENTITY


class _PrecompileDynamoSuperScalarIdentityHelper(
    _PrecompileDynamoBaseScalarIdentityHelper
):
    def same(self, token):
        return super().same(token)


_PRECOMPILE_DYNAMO_SUPER_SCALAR_IDENTITY_HELPER = (
    _PrecompileDynamoSuperScalarIdentityHelper()
)


class _PrecompileDynamoBaseApply:
    def apply(self, x):
        return x.sin()


class _PrecompileDynamoSuperApply(_PrecompileDynamoBaseApply):
    def apply(self, x):
        return super().apply(x)


_PRECOMPILE_DYNAMO_SUPER_APPLY = _PrecompileDynamoSuperApply()


def _precompile_dynamo_method_scalar_identity(x, token):
    if _PRECOMPILE_DYNAMO_SCALAR_IDENTITY_HELPER.same(token):
        return x.sin()
    return x.cos()


def _precompile_dynamo_staticmethod_scalar_identity(x, token):
    if _PrecompileDynamoScalarIdentityHelper.static_same(
        token, _PRECOMPILE_DYNAMO_SCALAR_IDENTITY
    ):
        return x.sin()
    return x.cos()


def _precompile_dynamo_classmethod_scalar_identity(x, token):
    if _PrecompileDynamoScalarIdentityHelper.class_same(token):
        return x.sin()
    return x.cos()


def _precompile_dynamo_callable_scalar_identity(x, token):
    if _PRECOMPILE_DYNAMO_SCALAR_IDENTITY_HELPER(token):
        return x.sin()
    return x.cos()


def _precompile_dynamo_init_scalar_identity(x, token):
    if _PrecompileDynamoInitScalarIdentity(token).same:
        return x.sin()
    return x.cos()


def _precompile_dynamo_metaclass_scalar_identity(x, token):
    if _PrecompileDynamoMetaclassScalarIdentity(token):
        return x.sin()
    return x.cos()


def _precompile_dynamo_super_scalar_identity(x, token):
    if _PRECOMPILE_DYNAMO_SUPER_SCALAR_IDENTITY_HELPER.same(token):
        return x.sin()
    return x.cos()


def _precompile_dynamo_calls_pure_super(x):
    return _PRECOMPILE_DYNAMO_SUPER_APPLY.apply(x)


def _precompile_dynamo_conditional_scalar_identity(x, token, cond):
    value = token if cond else 0
    if value is _PRECOMPILE_DYNAMO_SCALAR_IDENTITY:
        return x.sin()
    return x.cos()


def _precompile_dynamo_varargs_scalar_identity(x, *tokens):
    if tokens[0] is _PRECOMPILE_DYNAMO_SCALAR_IDENTITY:
        return x.sin()
    return x.cos()


def _precompile_dynamo_kwargs_scalar_identity(x, **tokens):
    if tokens["token"] is _PRECOMPILE_DYNAMO_SCALAR_IDENTITY:
        return x.sin()
    return x.cos()


def _precompile_dynamo_exception_scalar_identity(x, token):
    try:
        raise ValueError
    except ValueError:
        if token is _PRECOMPILE_DYNAMO_SCALAR_IDENTITY:
            return x.sin()
        return x.cos()


def _precompile_dynamo_contains_scalar_identity(x, values):
    if _PRECOMPILE_DYNAMO_NAN_IDENTITY in values:
        return x.sin()
    return x.cos()


def _precompile_dynamo_count_scalar_identity(x, values):
    if values.count(_PRECOMPILE_DYNAMO_NAN_IDENTITY):
        return x.sin()
    return x.cos()


def _precompile_dynamo_operator_contains_scalar_identity(x, values):
    if operator.contains(values, _PRECOMPILE_DYNAMO_NAN_IDENTITY):
        return x.sin()
    return x.cos()


def _precompile_dynamo_container_equality_scalar_identity(x, values):
    if values == [_PRECOMPILE_DYNAMO_NAN_IDENTITY]:
        return x.sin()
    return x.cos()


def _precompile_dynamo_input_container_equality(x, left, right):
    return x.sin() if left == right else x.cos()


def _precompile_dynamo_input_container_membership(x, values, item):
    return x.sin() if item in values else x.cos()


def _precompile_dynamo_input_container_contains(x, values, item):
    return x.sin() if values.__contains__(item) else x.cos()


def _precompile_dynamo_input_container_count(x, values, item):
    return x.sin() if values.count(item) else x.cos()


def _precompile_dynamo_input_sequence_index(x, values, index):
    return x + values[index]


def _precompile_dynamo_operator_input_sequence_index(x, values, index):
    return x + operator.getitem(values, index)


_PRECOMPILE_DYNAMO_NAN_VALUES = [float("nan")]


def _precompile_dynamo_operator_environment_sequence_index(x, index):
    return x + operator.getitem(_PRECOMPILE_DYNAMO_NAN_VALUES, index)


def _precompile_dynamo_scalar_value(x, scale):
    return x * scale + _PRECOMPILE_DYNAMO_SCALAR_VALUE


def _precompile_dynamo_environment_scalar_identity(x, scale):
    offset = (
        1
        if _PRECOMPILE_DYNAMO_ENV_IDENTITY_A is _PRECOMPILE_DYNAMO_ENV_IDENTITY_B
        else 2
    )
    return x * scale + offset


def _precompile_dynamo_aliasing(a, b):
    a.add_(1)
    return a * b


def _precompile_dynamo_rebinds_storage(a, b):
    a.set_(b)
    return a + 1


class _PrecompileDynamoTensorBox:
    def __init__(self, a, b):
        self.a = a
        self.b = b


class _PrecompileDynamoTensorList(list):
    def __init__(self, hidden):
        super().__init__()
        self.hidden = hidden


def _precompile_dynamo_box_aliasing(box):
    box.a.add_(1)
    return box.a * box.b


def _precompile_dynamo_mapping_aliasing(values):
    values["a"].add_(1)
    return values["a"] * values["b"]


def _precompile_dynamo_list_subclass_aliasing(values, other):
    values.hidden.add_(1)
    return values.hidden * other


class _PrecompileDynamoInputGetitem:
    def __getitem__(self, x):
        return x + 1 if x is _DYNAMO_TENSOR_DEFAULT else x - 1


class _PrecompileDynamoInputObjectBox:
    def __init__(self):
        self.inner = _PrecompileDynamoInputGetitem()


def _precompile_dynamo_input_getitem(obj, x):
    return obj[x]


def _precompile_dynamo_nested_input_getitem(box, x):
    return box.inner[x]


_DYNAMO_INPUT_OBJECT_GLOBAL = [_DYNAMO_TENSOR_DEFAULT]


class _PrecompileDynamoInputStateGlobalAlias:
    def __init__(self, token):
        self.token = token

    def __getitem__(self, x):
        return x + 1 if self.token is _DYNAMO_INPUT_OBJECT_GLOBAL[0] else x - 1


class _PrecompileDynamoInputList(list):
    def __getitem__(self, index):
        return _DYNAMO_TENSOR_DEFAULT


def _precompile_dynamo_dict_order(x, values):
    for value in values.values():
        x = x * value + 1
    return x


class _PrecompileDynamoInputAttribute:
    def __init__(self):
        self.flag = True


def _precompile_dynamo_input_attribute(x, state):
    return x + (1 if hasattr(state, "flag") else 2)


class _PrecompileDynamoCallableAttribute(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.op = torch.sin

    def forward(self, x):
        return self.op(x)


def _precompile_eager_cond(x):
    return torch.cond(x.sum() > 0, lambda t: t.sin(), lambda t: t.cos(), (x,))


def _precompile_eager_while_loop(x):
    return torch.while_loop(
        lambda i, t: i < 3,
        lambda i, t: (i + 1, t + 1.0),
        (torch.tensor(0), x),
    )[1]


def _precompile_eager_checkpoint(x):
    return torch.utils.checkpoint.checkpoint(
        lambda t: t.sin().cos(), x, use_reentrant=False
    )


def _precompile_eager_vmap(x):
    return torch.vmap(lambda t: t * 2.0)(x)


def _precompile_eager_autocast(x):
    with torch.autocast("cpu", dtype=torch.bfloat16):
        return x @ x


def _precompile_eager_no_grad(x):
    y = x * 2.0
    with torch.no_grad():
        return y.sin()


_PRECOMPILE_EAGER_ROUND_TRIP = {
    "autocast": _precompile_eager_autocast,
    "checkpoint": _precompile_eager_checkpoint,
    "cond": _precompile_eager_cond,
    "no_grad": _precompile_eager_no_grad,
    "vmap": _precompile_eager_vmap,
    "while_loop": _precompile_eager_while_loop,
}


def _precompile_eager_graph_break(key, x):
    y = x * 2.0
    torch._dynamo.graph_break()
    return _PRECOMPILE_EAGER_ROUND_TRIP[key](y)


class _PrecompileWeakValue:
    pass


def _precompile_dynamo_weakref_input(x, values):
    ref = values.data["value"]
    return x + (1 if ref.__callback__ is not None else 2)


_PRECOMPILE_DYNAMO_GENERATOR = torch.Generator()


def _precompile_dynamo_generator_environment(x):
    return x + (1 if _PRECOMPILE_DYNAMO_GENERATOR.device.type == "cpu" else 2)


def _precompile_dynamo_generator_input(x, generator):
    return x + (1 if generator.device.type == "cpu" else 2)


def _precompile_dynamo_graph_break(x):
    y = x + 1
    torch._dynamo.graph_break()
    y = y * 2
    torch._dynamo.graph_break()
    return y.sin()


def _precompile_dynamo_branching_graph_break(x, flag):
    y = x.sin()
    torch._dynamo.graph_break()
    return y.cos() if flag else y * 2


def _precompile_dynamo_unreachable_helper(x):
    y = x * 3
    torch._dynamo.graph_break()
    return y.sum()


def _precompile_dynamo_unreachable_caller(x):
    return _precompile_dynamo_unreachable_helper(x * 2)


def _precompile_dynamo_unreachable_branch(x, mode):
    if mode == 0:
        y = x * 2
    elif mode == 1:
        y = x * 3
    else:
        y = x * 4
    torch._dynamo.graph_break()
    return y.sum()


def _precompile_dynamo_unreachable_branch_caller(x, mode):
    return _precompile_dynamo_unreachable_branch(x, mode)


class _PrecompileDynamoBreakingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 3)

    def forward(self, x):
        y = self.linear(x).relu()
        torch._dynamo.graph_break()
        return y.sin()


class _PrecompileDynamoDisabledMethodHelper:
    @torch._dynamo.disable
    def call(self, x):
        return x + 1


_PRECOMPILE_DYNAMO_DISABLED_METHOD_HELPER = _PrecompileDynamoDisabledMethodHelper()


class _PrecompileDynamoDisabledMethodModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x):
        return _PRECOMPILE_DYNAMO_DISABLED_METHOD_HELPER.call(self.linear(x)).sum()


class _PrecompileDynamoDataDependentModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x):
        value = self.linear(x)
        scale = value.abs().max().item()
        return (value * scale).sum()


class _PrecompileDynamoBreakInLoopModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x):
        for _ in range(3):
            x = self.linear(x)
            x = _precompile_dynamo_disabled(x)
        return x.sum()


_PRECOMPILE_DYNAMO_GLOBAL_SCALE = 3.0


class _PrecompileDynamoIdentityToken:
    pass


_PRECOMPILE_DYNAMO_IDENTITY_TOKEN = _PrecompileDynamoIdentityToken()


def _precompile_dynamo_input_global_identity(x, token):
    if token is _PRECOMPILE_DYNAMO_IDENTITY_TOKEN:
        return x.sin()
    return x.cos()


class _PrecompileDynamoTiedWeights(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(4, 4, bias=False)
        self.b = torch.nn.Linear(4, 4, bias=False)
        self.b.weight = self.a.weight

    def forward(self, x):
        return self.b(self.a(x))


class _PrecompileDynamoUnguardedAttribute(torch.nn.Module):
    def __init__(self, extra):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)
        self.extra = extra

    def forward(self, x):
        return self.linear(x).relu().sum()


class _PrecompileDynamoReadsTensorAttribute(torch.nn.Module):
    def forward(self, x):
        companion = getattr(x, "_cpu_copy", None)
        return x * 2 if companion is None else x * 2 + companion.to(x.device)


def _precompile_dynamo_tensor_attribute_break(module, x):
    torch._dynamo.graph_break()
    return module(x).sum()


def _precompile_dynamo_reads_tensor_flag(x):
    return x * getattr(x, "my_flag", 1)


class _PrecompileDynamoStepCounter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)
        self.step = 0

    def forward(self, x):
        self.step += 1
        return self.linear(x) * self.step


class _PrecompileDynamoPipeline:
    def __init__(self, model):
        self.model = model
        self.iterator = (index for index in range(3))


class _PrecompileLockHolder:
    pass


def _precompile_dynamo_pipeline(pipeline, x):
    return pipeline.model(x).relu().sum()


class _PrecompilePlusOneMode(torch.overrides.TorchFunctionMode):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is torch.add and not isinstance(args[1], torch.Tensor):
            return func(args[0], args[1] + 1, **kwargs)
        return func(*args, **kwargs)


def _precompile_add_one(x):
    return torch.add(x, 1.0)


class _PrecompileDynamoFoldsGlobal(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 3)

    def forward(self, x):
        return self.linear(x), _PRECOMPILE_DYNAMO_GLOBAL_SCALE


class _PrecompileDynamoPlainMatmul(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(8, 8))

    def forward(self, x):
        return (x @ self.weight).relu()


class _PrecompileDynamoCustomOpMatmul(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(8, 8))

    def forward(self, x):
        return torch.ops.precompile_parity.fused_matmul(x, self.weight).relu()


class _PrecompileDynamoGradState(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(4))
        self.register_buffer("buffer", torch.randn(4, requires_grad=True))

    def step(self, x):
        ((self.weight + self.buffer) * x).sum().backward()


def _precompile_dynamo_call_module(module, x):
    return module(x)


def _precompile_dynamo_grad_state(module, x):
    module.step(x)


def _precompile_dynamo_late_varying(fixed, varying):
    prefix = fixed.sin()
    torch._dynamo.graph_break()
    return prefix.sum() + varying.cos().sum()


def _precompile_dynamo_wrong_call_module(module, x):
    return module(x) + 100


def _precompile_dynamo_backward(module, x):
    module(x).sum().backward()


class _PrecompileDynamoIndependentOutputs(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.left = torch.nn.Linear(4, 3)
        self.right = torch.nn.Linear(4, 3)

    def forward(self, x):
        return self.left(x).sin(), self.right(x).cos()


@torch._dynamo.disable
def _precompile_dynamo_backward_one(first, second, use_first):
    (first if use_first else second).sum().backward()


def _precompile_dynamo_undefined_tangent_step(module, x, use_first):
    first, second = module(x)
    _precompile_dynamo_backward_one(first, second, use_first)


def _precompile_dynamo_optional_tangent_step(x, y, use_first):
    first, second = torch.ops.precompile_optional_tangents.split.default(x, y)
    _precompile_dynamo_backward_one(first, second, use_first)


def _precompile_dynamo_async_collective_tangent_step(x, y):
    first = x.sin()
    second = AsyncCollectiveTensor(y.cos())
    _precompile_dynamo_backward_one(first, second, True)


@torch._dynamo.disable
def _precompile_dynamo_backward_aliased_outputs(first, second, detached, independent):
    if not first.requires_grad or not second.requires_grad:
        raise AssertionError("differentiable view outputs must retain autograd history")
    if detached.requires_grad or detached.grad_fn is not None:
        raise AssertionError("the detached output must remain non-differentiable")
    (first.sum() + second.sum() + independent.sum()).backward()


def _precompile_dynamo_dealias_marked_returns_step(x):
    value = x.sin()
    _precompile_dynamo_backward_aliased_outputs(
        value[0:4], value[4:8], value.detach(), x * 3
    )


@torch._dynamo.disable
def _precompile_mutation_replay_backward(output):
    output.backward()


def _precompile_mutation_replay_step(weight, value):
    torch.ops.precompile_mutation_replay.opaque_add_(weight, value)
    _precompile_mutation_replay_backward((weight * value).sum())


def _precompile_dynamo_autograd_grad(module, x, target):
    loss = torch.nn.functional.mse_loss(module(x), target)
    return torch.autograd.grad(loss, tuple(module.parameters()))


def _precompile_dynamo_aliased_graph_break(x):
    y = x + 1
    _precompile_dynamo_break_here()
    return y * 2


_PRECOMPILE_DYNAMO_RESUME_VALUE = 7


def _precompile_dynamo_empty_resume(x):
    y = x + 1
    torch._dynamo.graph_break()
    return y, _PRECOMPILE_DYNAMO_RESUME_VALUE


_PRECOMPILE_DYNAMO_UNPORTABLE_GLOBAL = object()


def _precompile_dynamo_unportable_resume(x):
    y = x + 1
    torch._dynamo.graph_break()
    return y, _PRECOMPILE_DYNAMO_UNPORTABLE_GLOBAL


_PRECOMPILE_DYNAMO_DISABLED_SCALE = 0.5


@torch._dynamo.disable
def _precompile_dynamo_disabled(x):
    return torch.cos(x) * _PRECOMPILE_DYNAMO_DISABLED_SCALE


def _precompile_dynamo_with_disabled(x):
    y = x.sin() + x.shape[0]
    y = _precompile_dynamo_disabled(y)
    return y * x.shape[0]


_PRECOMPILE_DYNAMO_TEMPLATE_SCALE = 1.0


def _precompile_dynamo_disabled_template(x):
    return x.sin() * _PRECOMPILE_DYNAMO_TEMPLATE_SCALE


def _make_precompile_dynamo_disabled(scale):
    function = types.FunctionType(
        _precompile_dynamo_disabled_template.__code__,
        {
            "__name__": __name__,
            "torch": torch,
            "_PRECOMPILE_DYNAMO_TEMPLATE_SCALE": scale,
        },
    )
    return torch._dynamo.disable(function)


_PRECOMPILE_DYNAMO_DISABLED_A = _make_precompile_dynamo_disabled(2.0)
_PRECOMPILE_DYNAMO_DISABLED_B = _make_precompile_dynamo_disabled(3.0)


def _precompile_dynamo_with_two_disabled(x):
    return _PRECOMPILE_DYNAMO_DISABLED_A(x) + _PRECOMPILE_DYNAMO_DISABLED_B(x)


def _precompile_dynamo_add_ten(function):
    @functools.wraps(function)
    def wrapper(x):
        return function(x) + 10

    return wrapper


@torch._dynamo.disable
@_precompile_dynamo_add_ten
def _precompile_dynamo_decorated_disabled(x):
    return x * 2


def _precompile_dynamo_with_decorated_disabled(x):
    return _precompile_dynamo_decorated_disabled(x) + 1


@torch._dynamo.disable
def _precompile_dynamo_forward_helper(x):
    return x.cos()


forward = _precompile_dynamo_forward_helper


def _precompile_dynamo_with_forward_global(x):
    return forward(x.sin()) * 2


def _precompile_dynamo_cellvar(x, scale):
    def inner():
        return scale

    torch._dynamo.graph_break()
    return x + inner()


_PRECOMPILE_DYNAMO_MUTATED_GLOBAL = 0
_PRECOMPILE_DYNAMO_INPUT_GLOBAL = None


class _PrecompileDynamoMutationHolder:
    state = [0]
    value = 0


def _precompile_dynamo_mutates_global_attribute(x):
    _PrecompileDynamoMutationHolder.state[0] += 1
    return x + _PrecompileDynamoMutationHolder.state[0]


def _precompile_dynamo_operator_mutates_global_attribute(x):
    operator.setitem(
        _PrecompileDynamoMutationHolder.state,
        0,
        _PrecompileDynamoMutationHolder.state[0] + 1,
    )
    return x + _PrecompileDynamoMutationHolder.state[0]


def _precompile_dynamo_operator_iadd_global_attribute(x):
    operator.iadd(_PrecompileDynamoMutationHolder.state, [1])
    return x + len(_PrecompileDynamoMutationHolder.state)


def _precompile_dynamo_setattr_global_attribute(x):
    setattr(  # noqa: B010
        _PrecompileDynamoMutationHolder,
        "value",
        _PrecompileDynamoMutationHolder.value + 1,
    )
    return x + _PrecompileDynamoMutationHolder.value


_PRECOMPILE_DYNAMO_DEFAULT_MUTATION_STATE = []


def _precompile_dynamo_default_mutation_helper(
    x, state=_PRECOMPILE_DYNAMO_DEFAULT_MUTATION_STATE
):
    state.append(1)
    return x + len(state)


def _precompile_dynamo_calls_default_mutation_helper(x):
    return _precompile_dynamo_default_mutation_helper(x)


def _precompile_dynamo_descriptor_append_default_mutation_helper(
    x, state=_PRECOMPILE_DYNAMO_DEFAULT_MUTATION_STATE
):
    list.append(state, 1)
    return x + len(state)


def _precompile_dynamo_calls_descriptor_append_default_mutation_helper(x):
    return _precompile_dynamo_descriptor_append_default_mutation_helper(x)


def _precompile_dynamo_descriptor_iadd_default_mutation_helper(
    x, state=_PRECOMPILE_DYNAMO_DEFAULT_MUTATION_STATE
):
    list.__iadd__(state, [1])
    return x + len(state)


def _precompile_dynamo_calls_descriptor_iadd_default_mutation_helper(x):
    return _precompile_dynamo_descriptor_iadd_default_mutation_helper(x)


_PRECOMPILE_DYNAMO_PARTIAL_APPEND = functools.partial(
    list.append, _PRECOMPILE_DYNAMO_DEFAULT_MUTATION_STATE
)


def _precompile_dynamo_partial_descriptor_default_mutation_helper(
    x, append=_PRECOMPILE_DYNAMO_PARTIAL_APPEND
):
    append(1)
    return x + len(_PRECOMPILE_DYNAMO_DEFAULT_MUTATION_STATE)


def _precompile_dynamo_calls_partial_descriptor_default_mutation_helper(x):
    return _precompile_dynamo_partial_descriptor_default_mutation_helper(x)


_PRECOMPILE_DYNAMO_DEQUE_MUTATION_STATE = deque()


def _precompile_dynamo_deque_appendleft_default_mutation_helper(
    x, state=_PRECOMPILE_DYNAMO_DEQUE_MUTATION_STATE
):
    state.appendleft(1)
    return x + len(state)


def _precompile_dynamo_calls_deque_appendleft_default_mutation_helper(x):
    return _precompile_dynamo_deque_appendleft_default_mutation_helper(x)


def _precompile_dynamo_deque_rotate_default_mutation_helper(
    x, state=_PRECOMPILE_DYNAMO_DEQUE_MUTATION_STATE
):
    state.rotate(1)
    return x + len(state)


def _precompile_dynamo_calls_deque_rotate_default_mutation_helper(x):
    return _precompile_dynamo_deque_rotate_default_mutation_helper(x)


class _PrecompileDynamoMutatingGetitem:
    def __init__(self):
        self.value = 0

    def __getitem__(self, key):
        self.value += 1
        return self.value


_PRECOMPILE_DYNAMO_MUTATING_GETITEM = _PrecompileDynamoMutatingGetitem()


def _precompile_dynamo_mutating_getitem(x):
    return x + _PRECOMPILE_DYNAMO_MUTATING_GETITEM[0]


_PRECOMPILE_DYNAMO_DEFAULTDICT = defaultdict(int)


def _precompile_dynamo_defaultdict_getitem(x):
    return x + _PRECOMPILE_DYNAMO_DEFAULTDICT[len(_PRECOMPILE_DYNAMO_DEFAULTDICT)]


class _PrecompileDynamoMutatingProtocol:
    def __init__(self):
        self.value = 0

    @property
    def property(self):
        self.value += 1
        return self.value

    def __iter__(self):
        self.value += 1
        return iter((self.value,))

    def __bool__(self):
        self.value += 1
        return True

    def __add__(self, other):
        self.value += 1
        return self.value

    def __lt__(self, other):
        self.value += 1
        return True

    def __index__(self):
        self.value += 1
        return 0

    def __len__(self):
        self.value += 1
        return 1

    def __format__(self, format_spec):
        self.value += 1
        return format(self.value, format_spec)

    def __enter__(self):
        self.value += 1
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.value += 1
        return False

    def touch(self):
        self.value += 1


_PRECOMPILE_DYNAMO_MUTATING_PROTOCOL = _PrecompileDynamoMutatingProtocol()
_PRECOMPILE_DYNAMO_INDEX_VALUES = [7]


class _PrecompileDynamoMutatingEqual:
    def __init__(self):
        self.value = 0

    def __eq__(self, other):
        self.value += 1
        return False


_PRECOMPILE_DYNAMO_MUTATING_EQUAL = _PrecompileDynamoMutatingEqual()
_PRECOMPILE_DYNAMO_PROTOCOL_VALUES = [_PRECOMPILE_DYNAMO_MUTATING_EQUAL]


class _PrecompileDynamoMutatingRadd:
    def __init__(self):
        self.value = 0

    def __radd__(self, other):
        self.value += 1
        return other


_PRECOMPILE_DYNAMO_MUTATING_RADD = _PrecompileDynamoMutatingRadd()
_PRECOMPILE_DYNAMO_SUM_VALUES = [_PRECOMPILE_DYNAMO_MUTATING_RADD]
_PRECOMPILE_DYNAMO_NESTED_VALUES = (_PRECOMPILE_DYNAMO_MUTATING_PROTOCOL,)
_PRECOMPILE_DYNAMO_NESTED_MAPPING = {"value": _PRECOMPILE_DYNAMO_MUTATING_PROTOCOL}


class _PrecompileDynamoMutatingMetaclass(type):
    def __bool__(cls):
        cls.value += 1
        return True

    def __iter__(cls):
        cls.value += 1
        return iter((cls.value,))

    def __hash__(cls):
        cls.value += 1
        return id(cls)


class _PrecompileDynamoMutatingClass(metaclass=_PrecompileDynamoMutatingMetaclass):
    value = 0

    @classmethod
    def touch(cls):
        cls.value += 1


class _PrecompileDynamoReturningMetaclass(type):
    @property
    def target(cls):
        return _PrecompileDynamoMutatingClass


class _PrecompileDynamoReturningClass(metaclass=_PrecompileDynamoReturningMetaclass):
    pass


_PRECOMPILE_DYNAMO_BISECT_VALUES = [1]
_PRECOMPILE_DYNAMO_CLASS_VALUES = [_PrecompileDynamoMutatingClass]
_PRECOMPILE_DYNAMO_CONTEXT_CLASS = contextvars.ContextVar(
    "_PRECOMPILE_DYNAMO_CONTEXT_CLASS", default=_PrecompileDynamoMutatingClass
)


def _precompile_dynamo_stateful_helper():
    pass


_precompile_dynamo_stateful_helper.state = []
_PRECOMPILE_DYNAMO_FUNCTION_VALUES = (_precompile_dynamo_stateful_helper,)


def _precompile_dynamo_identity_callback(value):
    return value


def _precompile_dynamo_mutating_property(x):
    return x + _PRECOMPILE_DYNAMO_MUTATING_PROTOCOL.property


def _precompile_dynamo_mutating_iter(x):
    for value in _PRECOMPILE_DYNAMO_MUTATING_PROTOCOL:
        return x + value
    return x


def _precompile_dynamo_mutating_bool(x):
    return (
        x + _PRECOMPILE_DYNAMO_MUTATING_PROTOCOL.value
        if _PRECOMPILE_DYNAMO_MUTATING_PROTOCOL
        else x
    )


def _precompile_dynamo_mutating_add(x):
    return x + (_PRECOMPILE_DYNAMO_MUTATING_PROTOCOL + 0)


def _precompile_dynamo_mutating_compare(x):
    return x + int(_PRECOMPILE_DYNAMO_MUTATING_PROTOCOL < 1)


def _precompile_dynamo_mutating_unpack(x):
    (unused,) = _PRECOMPILE_DYNAMO_MUTATING_PROTOCOL
    return x + unused * 0


def _precompile_dynamo_mutating_format(x):
    _ = f"{_PRECOMPILE_DYNAMO_MUTATING_PROTOCOL}"
    return x + 1


def _precompile_dynamo_mutating_context(x):
    with _PRECOMPILE_DYNAMO_MUTATING_PROTOCOL:
        pass
    return x + 1


def _precompile_dynamo_mutating_len(x):
    _ = len(_PRECOMPILE_DYNAMO_MUTATING_PROTOCOL)
    return x + 1


def _precompile_dynamo_mutating_index(x):
    _ = _PRECOMPILE_DYNAMO_INDEX_VALUES[_PRECOMPILE_DYNAMO_MUTATING_PROTOCOL]
    return x + 1


def _precompile_dynamo_mutating_contains(x):
    _ = 0 in _PRECOMPILE_DYNAMO_PROTOCOL_VALUES
    return x + 1


def _precompile_dynamo_mutating_sum(x):
    _ = sum(_PRECOMPILE_DYNAMO_SUM_VALUES)
    return x + 1


def _precompile_dynamo_mutating_nested_property(x):
    _ = _PRECOMPILE_DYNAMO_NESTED_VALUES[0].property
    return x + 1


def _precompile_dynamo_mutating_nested_method(x):
    _PRECOMPILE_DYNAMO_NESTED_MAPPING.get("value").touch()
    return x + 1


def _precompile_dynamo_mutating_iterated_method(x):
    for value in _PRECOMPILE_DYNAMO_NESTED_VALUES:
        value.touch()
    return x + 1


def _precompile_dynamo_mutating_short_circuit(x):
    _ = _PrecompileDynamoMutatingClass and 1
    return x + 1


def _precompile_dynamo_mutating_star_unpack(x):
    _ = [*_PrecompileDynamoMutatingClass]
    return x + 1


def _precompile_dynamo_mutating_hash(x):
    _ = {_PrecompileDynamoMutatingClass}
    return x + 1


def _precompile_dynamo_mutating_bisect(x):
    bisect.insort(_PRECOMPILE_DYNAMO_BISECT_VALUES, 2)
    return x + len(_PRECOMPILE_DYNAMO_BISECT_VALUES)


def _precompile_dynamo_mutating_map_result(x):
    next(
        map(_precompile_dynamo_identity_callback, _PRECOMPILE_DYNAMO_CLASS_VALUES)
    ).touch()
    return x + 1


def _precompile_dynamo_mutating_copy_result(x):
    _PRECOMPILE_DYNAMO_CLASS_VALUES.copy()[0].touch()
    return x + 1


def _precompile_dynamo_mutating_iterator_result(x):
    next(_PRECOMPILE_DYNAMO_CLASS_VALUES.__iter__()).touch()
    return x + 1


def _precompile_dynamo_mutating_descriptor_result(x):
    _PrecompileDynamoReturningClass.target.touch()
    return x + 1


def _precompile_dynamo_mutating_contextvar_result(x):
    _PRECOMPILE_DYNAMO_CONTEXT_CLASS.get().touch()
    return x + 1


def _precompile_dynamo_mutating_unpack_result(x):
    (helper,) = _PRECOMPILE_DYNAMO_FUNCTION_VALUES
    helper.state.append(1)
    return x + len(helper.state)


def _precompile_dynamo_stateful_generator():
    yield _precompile_dynamo_stateful_helper


def _precompile_dynamo_mutating_yield_result(x):
    next(_precompile_dynamo_stateful_generator()).state.append(1)
    return x + len(_precompile_dynamo_stateful_helper.state)


def _precompile_dynamo_mutating_slice_result(x):
    value = slice(_precompile_dynamo_stateful_helper, None).start
    value.state.append(1)
    return x + len(value.state)


def _precompile_dynamo_mutating_local_function_result(x):
    def get_helper():
        return _precompile_dynamo_stateful_helper

    value = get_helper()
    value.state.append(1)
    return x + len(value.state)


def _precompile_dynamo_torch_add(x):
    return torch.add(x, 1)


class _PrecompileDynamoPureAdd:
    def add(self, x):
        return x + 1


_PRECOMPILE_DYNAMO_PURE_ADD = _PrecompileDynamoPureAdd()


def _precompile_dynamo_pure_add(x):
    return _PRECOMPILE_DYNAMO_PURE_ADD.add(x)


_PRECOMPILE_DYNAMO_HEAP = [1]


def _precompile_dynamo_heapq_mutation(x):
    heapq.heappush(_PRECOMPILE_DYNAMO_HEAP, 2)
    return x + len(_PRECOMPILE_DYNAMO_HEAP)


def _make_precompile_dynamo_nonlocal_mutation():
    count = 0

    def helper(x):
        nonlocal count
        count += 1
        return x + count

    return helper


_PRECOMPILE_DYNAMO_NONLOCAL_MUTATION = _make_precompile_dynamo_nonlocal_mutation()


def _precompile_dynamo_calls_nonlocal_mutation(x):
    return _PRECOMPILE_DYNAMO_NONLOCAL_MUTATION(x)


def _precompile_dynamo_recursive_identity_helper(token, depth):
    if depth:
        return _precompile_dynamo_recursive_identity_helper(token, depth - 1)
    return token is _PRECOMPILE_DYNAMO_OBJECT_IDENTITY


def _precompile_dynamo_recursive_identity(x, token, depth):
    return (
        x.sin()
        if _precompile_dynamo_recursive_identity_helper(token, depth)
        else x.cos()
    )


def _precompile_dynamo_recursive_pure_helper(x, depth):
    if depth:
        return _precompile_dynamo_recursive_pure_helper(x + 1, depth - 1)
    return x


def _precompile_dynamo_recursive_pure(x, depth):
    return _precompile_dynamo_recursive_pure_helper(x, depth)


def _precompile_dynamo_stores_input_global(x):
    global _PRECOMPILE_DYNAMO_INPUT_GLOBAL
    _PRECOMPILE_DYNAMO_INPUT_GLOBAL = x
    return x + 1


def _precompile_dynamo_helper_stores_input_global(x):
    global _PRECOMPILE_DYNAMO_INPUT_GLOBAL
    _PRECOMPILE_DYNAMO_INPUT_GLOBAL = x
    return x


def _precompile_dynamo_calls_global_mutating_helper(x):
    return _precompile_dynamo_helper_stores_input_global(x) + 1


_PRECOMPILE_DYNAMO_MUTATING_HELPERS = [_precompile_dynamo_helper_stores_input_global]


def _precompile_dynamo_calls_container_mutating_helper(x):
    return _PRECOMPILE_DYNAMO_MUTATING_HELPERS[0](x) + 1


def _precompile_dynamo_mutation_carrier():
    pass


_precompile_dynamo_mutation_carrier.helper = (
    _precompile_dynamo_helper_stores_input_global
)


def _precompile_dynamo_calls_function_metadata_mutating_helper(x):
    return _precompile_dynamo_mutation_carrier.__dict__["helper"](x) + 1


def _precompile_dynamo_calls_function_globals_mutating_helper(x):
    return _precompile_dynamo_mutation_carrier.__globals__[
        "_precompile_dynamo_helper_stores_input_global"
    ](x)


def _make_precompile_dynamo_closure_carrier():
    helper = _precompile_dynamo_helper_stores_input_global

    def carrier():
        return helper

    return carrier


_PRECOMPILE_DYNAMO_CLOSURE_CARRIER = _make_precompile_dynamo_closure_carrier()


def _precompile_dynamo_calls_function_closure_mutating_helper(x):
    return _PRECOMPILE_DYNAMO_CLOSURE_CARRIER.__closure__[0].cell_contents(x)


class _PrecompileDynamoGlobalMutatingMethod:
    def apply(self, x):
        global _PRECOMPILE_DYNAMO_INPUT_GLOBAL
        _PRECOMPILE_DYNAMO_INPUT_GLOBAL = x
        return x + 1

    def __getitem__(self, x):
        global _PRECOMPILE_DYNAMO_INPUT_GLOBAL
        _PRECOMPILE_DYNAMO_INPUT_GLOBAL = x
        return x + 1


def _precompile_dynamo_calls_input_mutating_method(obj, x):
    return obj.apply(x)


def _precompile_dynamo_calls_input_mutating_getitem(obj, x):
    return obj[x]


class _PrecompileDynamoInputScalarIdentity:
    def __init__(self, token):
        self.token = token

    def same(self):
        return self.token is _PRECOMPILE_DYNAMO_SCALAR_IDENTITY

    def __bool__(self):
        return self.token is _PRECOMPILE_DYNAMO_SCALAR_IDENTITY


def _precompile_dynamo_calls_input_scalar_identity(x, obj):
    if obj.same():
        return x.sin()
    return x.cos()


def _precompile_dynamo_calls_input_bool_identity(x, obj):
    if obj:
        return x.sin()
    return x.cos()


class _PrecompileDynamoGlobalMutatingFactory:
    def __new__(cls, x):
        global _PRECOMPILE_DYNAMO_INPUT_GLOBAL
        _PRECOMPILE_DYNAMO_INPUT_GLOBAL = x
        return x + 1


def _precompile_dynamo_calls_mutating_factory(x):
    return _PrecompileDynamoGlobalMutatingFactory(x)


class _PrecompileDynamoNonMutatingFactory:
    def __new__(cls, x):
        return x + 2


def _precompile_dynamo_calls_conditional_mutating_factory(x, mutate):
    factory = (
        _PrecompileDynamoGlobalMutatingFactory
        if mutate
        else _PrecompileDynamoNonMutatingFactory
    )
    return factory(x)


def _precompile_dynamo_calls_input_factory(factory, x):
    return factory(x)


@torch._dynamo.disable
def _precompile_dynamo_mutates_global(x):
    def mutate():
        global _PRECOMPILE_DYNAMO_MUTATED_GLOBAL
        _PRECOMPILE_DYNAMO_MUTATED_GLOBAL += 1

    mutate()
    return x


def _precompile_dynamo_with_mutated_global(x):
    return _precompile_dynamo_mutates_global(x) + 1


@torch._dynamo.disable
def _precompile_dynamo_dynamic_global(x):
    return x * globals()["_PRECOMPILE_DYNAMO_MUTATED_GLOBAL"]


def _precompile_dynamo_with_dynamic_global(x):
    return _precompile_dynamo_dynamic_global(x) + 1


def _precompile_dynamo_with_dead_disabled_branch(x, use_good):
    if use_good:
        return _precompile_dynamo_disabled(x) + 1
    return _precompile_dynamo_mutates_global(x)


@torch._dynamo.disable
def _precompile_dynamo_nested_disabled(x):
    def inner(y):
        return y + 1

    return inner(x)


def _precompile_dynamo_with_nested_disabled(x):
    return _precompile_dynamo_nested_disabled(x) * 2


@torch._dynamo.disable
def __compiled_fn_user(x):
    return x + 2


def _precompile_dynamo_with_compiled_prefix_helper(x):
    return __compiled_fn_user(x) * 3


_PRECOMPILE_DYNAMO_EPHEMERAL_MODULE = types.ModuleType("precompile_ephemeral")


def _precompile_dynamo_identity(x):
    return x


_PRECOMPILE_DYNAMO_EPHEMERAL_MODULE.identity = _precompile_dynamo_identity


@torch._dynamo.disable
def _precompile_dynamo_uses_ephemeral_module(x):
    return _PRECOMPILE_DYNAMO_EPHEMERAL_MODULE.identity(x)


def _precompile_dynamo_with_ephemeral_module(x):
    return _precompile_dynamo_uses_ephemeral_module(x) + 1


# A custom pytree node whose context (a set) is not JSON-dumpable and which has no
# to_dumpable_context serializer, so treespec_dumps raises TypeError (distinct from the
# unregistered-namedtuple NotImplementedError path). Registered once at module load and
# used by test_unserializable_context_in_spec_still_compiles.
class _UnserializableCtxInput:
    def __init__(self, a, b):
        self.a = a
        self.b = b


_pytree.register_pytree_node(
    _UnserializableCtxInput,
    lambda n: ([n.a, n.b], {"ctx"}),
    lambda children, _ctx: _UnserializableCtxInput(children[0], children[1]),
    serialized_type_name="test_precompile._UnserializableCtxInput",
)


def _strip_artifact(cache: bytes) -> bytes:
    """Return the cache envelope with its compiled artifact removed, forcing load()
    onto the inlined (no-cache) path that JIT-compiles from python_code. Many tests
    reload the same artifact both cache-primed and stripped to check they agree."""
    blob = torch.load(io.BytesIO(cache), weights_only=True)
    blob["artifact"] = None
    buf = io.BytesIO()
    torch.save(blob, buf)
    return buf.getvalue()


def _dynamo_serialized_guard_summary(
    code: str,
) -> list[tuple[list[str], list[str], list[str], bool]]:
    from torch._dynamo.package import load_guards_state
    from torch._precompile import _parse_dynamo_state

    state = _parse_dynamo_state(code)
    summary = []
    for code_state in state.codes:
        for variant in code_state.variants:
            guards_state = load_guards_state(variant.guards_state)
            summary.append(
                (
                    [
                        guard.create_fn_name()
                        for guard in guards_state.output_graph.guards
                    ],
                    [
                        type(guard).__name__
                        for guard in guards_state.output_graph.aotautograd_guards
                    ],
                    sorted(
                        source.name
                        for source in guards_state.output_graph.guard_on_key_order
                    ),
                    guards_state.shape_code_parts is not None,
                )
            )
    return summary


def _dynamo_frame_variant_counts(code: str) -> list[tuple[str, int]]:
    from torch._dynamo.package import SerializedCode
    from torch._precompile import _parse_dynamo_state

    state = _parse_dynamo_state(code)
    return [
        (
            SerializedCode.to_code_object(code_state.code).co_name,
            len(code_state.variants),
        )
        for code_state in state.codes
    ]


def _default_and_inlined_loaders(code: str, cache: bytes, backend: str):
    """Yield (label, loaded_fn) for the load paths a backend exposes: the default
    (cache-primed) path always, plus -- on inductor only -- the inlined path that
    strips the artifact to force JIT from python_code. The eager backend has a single
    driver, so it yields the default path alone."""
    yield "default", torch.compiler.precompile.load(code, cache)
    if backend == "inductor":
        yield "inlined", torch.compiler.precompile.load(code, _strip_artifact(cache))


# precompile drives make_fx internally, which cannot symbolically trace a
# dynamo-optimized function; the whole suite is therefore incompatible with
# PYTORCH_TEST_WITH_DYNAMO (dynamo_wrapped CI), so skip it there.
@skipIfTorchDynamo("precompile's make_fx capture is incompatible with dynamo wrapping")
@instantiate_parametrized_tests
class TestPrecompile(TestCase):
    def test_decompositions_kwarg(self):
        # The decompositions table is threaded into make_fx during capture; a
        # custom decomposition is invoked and the result still matches eager.
        called = []

        def my_relu_decomp(x):
            called.append(True)
            return (x > 0) * x

        decomps = {torch.ops.aten.relu.default: my_relu_decomp}
        m = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU()).eval()
        x = torch.randn(5, 4)
        code, cache = torch.compiler.precompile(
            lambda model, x: model(x), example_inputs=[(m, x)], decompositions=decomps
        )
        self.assertTrue(called)  # the table was used during capture

        f_c = torch.compiler.precompile.load(code, cache)
        self.assertEqual(f_c(m, x), m(x))

    def test_constant_tensor_is_rejected(self):
        captured = torch.randn(3)
        with self.assertRaisesRegex(PrecompileError, "hard-coded"):
            torch.compiler.precompile(
                lambda x: x + captured, example_inputs=[(torch.randn(3),)]
            )

    def test_global_tensor_rejected_unlike_make_fx(self):
        # Vanilla make_fx silently bakes a referenced global tensor into the
        # GraphModule as a get_attr constant; precompile must instead error.
        from torch.fx.experimental.proxy_tensor import make_fx

        def f(x):
            return x + _GLOBAL_TENSOR

        gm = make_fx(f)(torch.randn(3))
        baked = [
            n.target
            for n in gm.graph.nodes
            if n.op == "get_attr"
            and isinstance(getattr(gm, n.target, None), torch.Tensor)
        ]
        self.assertTrue(baked, "expected vanilla make_fx to bake a tensor constant")

        with self.assertRaisesRegex(PrecompileError, "hard-coded"):
            torch.compiler.precompile(f, example_inputs=[(torch.randn(3),)])

    def test_unregistered_module_tensor_attr_is_rejected(self):
        # A plain tensor attribute (not a registered parameter/buffer) is not
        # lifted, so referencing it would bake it in -- this must error.
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(4, 4))
                self.scale = torch.randn(4)  # plain attr, NOT a buffer/parameter

            def forward(self, x):
                return (x @ self.weight) * self.scale

        m = M().eval()
        with self.assertRaisesRegex(PrecompileError, "hard-coded"):
            torch.compiler.precompile(
                lambda model, x: model(x), example_inputs=[(m, torch.randn(2, 4))]
            )

    def test_export_and_reload_roundtrip(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = torch.nn.Linear(4, 3)
                self.register_buffer("b2", torch.randn(3))

            def forward(self, x):
                return torch.relu(self.lin(x)) + self.b2

        m = M().eval()
        x = torch.randn(5, 4)
        code, cache = torch.compiler.precompile(
            lambda model, x: model(x), example_inputs=[(m, x)]
        )

        self.assertIn("Inductor output code", code)
        self.assertIn("def forward(", code)
        self.assertIn("PARAM_NAMES = ['lin.weight', 'lin.bias']", code)

        f_c = torch.compiler.precompile.load(code, cache)
        self.assertEqual(f_c(m, x), m(x))

    def test_self_contained_exec_needs_no_cache(self):
        # python_code runs standalone with NO cache: exec it and call forward().
        # The default eager backend has no kernels; the captured graph is
        # interpreted directly from the inlined source and the cache is always
        # empty (artifact=None), so python_code is fully self-contained.
        m = torch.nn.Sequential(torch.nn.Linear(4, 3)).eval()
        x = torch.randn(5, 4)
        code, _cache = torch.compiler.precompile(
            lambda model, x: model(x), example_inputs=[(m, x)]
        )

        ns = {"__name__": "_artifact"}
        exec(compile(code, "<artifact>", "exec"), ns)
        self.assertEqual(ns["forward"](m, x), m(x))

    @unittest.skipUnless(
        torch.cuda.is_available(), "needs CUDA + Triton for the kernel cache"
    )
    @torch._inductor.config.patch({"compile_threads": 1})
    def test_cache_reload_without_eager_static_launcher_rehydration(self):
        # A cold load should use JIT instead of eagerly rehydrating the static launcher.
        import torch._inductor.config as ind_config

        if ind_config.force_disable_caches or not ind_config.fx_graph_cache:
            self.skipTest("requires inductor FxGraphCache enabled")
        if not ind_config.use_static_cuda_launcher:
            self.skipTest("requires the static CUDA launcher")
        from torch._dynamo.utils import counters
        from torch._inductor.utils import fresh_cache

        m = (
            torch.nn.Sequential(
                torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 4)
            )
            .eval()
            .cuda()
        )
        x = torch.randn(3, 8, device="cuda")
        code, cache = torch.compiler.precompile(
            lambda model, x: model(x), example_inputs=[(m, x)]
        )
        self.assertIsInstance(cache, bytes)

        with fresh_cache():
            counters.clear()
            f_c = torch.compiler.precompile.load(code, cache)
            self.assertEqual(f_c(m, x), m(x))
            self.assertEqual(
                counters["inductor"]["triton_bundler_load_static_autotuner"], 0
            )

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA for Triton autotuning")
    def test_cache_bundles_autotune_artifacts(self):
        from torch._inductor.utils import fresh_cache

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.l1 = torch.nn.Linear(512, 512)
                self.l2 = torch.nn.Linear(512, 512)

            def forward(self, x):
                return torch.softmax(self.l2(torch.relu(self.l1(x))), dim=-1)

        m = M().cuda().eval()
        x = torch.randn(128, 512, device="cuda")
        code, cache = torch.compiler.precompile(
            lambda model, x: model(x), example_inputs=[(m, x)]
        )
        with fresh_cache():
            f_c = torch.compiler.precompile.load(code, cache)
            self.assertEqual(f_c(m, x), m(x))

    def test_dtensor_subclass(self):
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_gloo_available():
            self.skipTest("gloo not available")

        from torch.distributed.tensor import DeviceMesh, distribute_tensor, Replicate
        from torch.testing._internal.common_utils import find_free_port

        # Use a free port (a hardcoded one flakes on shared CI) and restore the
        # env afterwards so we do not leak MASTER_ADDR/MASTER_PORT to later tests.
        saved_env = {k: os.environ.get(k) for k in ("MASTER_ADDR", "MASTER_PORT")}
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        dist.init_process_group("gloo", rank=0, world_size=1)
        try:
            mesh = DeviceMesh("cpu", list(range(1)))
            m = torch.nn.Linear(4, 3).eval()
            for name, p in list(m.named_parameters()):
                setattr(
                    m,
                    name,
                    torch.nn.Parameter(
                        distribute_tensor(p.detach(), mesh, [Replicate()])
                    ),
                )
            x = distribute_tensor(torch.randn(5, 4), mesh, [Replicate()])
            ref = m(x)

            code, cache = torch.compiler.precompile(
                lambda model, x: model(x), example_inputs=[(m, x)]
            )
            # Subclass handling is via our own protocol-based driver, not embedded
            # AOTAutograd wrapper source.
            self.assertIn("__tensor_unflatten__", code)
            self.assertNotIn("subclass_wrapper", code)

            # load() takes the bundled-artifact path (real AOTAutograd runtime).
            f_c = torch.compiler.precompile.load(code, cache)
            self.assertEqual(f_c(m, x).to_local(), ref.to_local())

            # Also exercise the standalone driver (the generated python, no cache):
            # subclass inputs/outputs handled by the inlined recipes via
            # __tensor_flatten__/__tensor_unflatten__.
            ns = {"__name__": "_dt"}
            exec(compile(code, "<dt>", "exec"), ns)
            self.assertEqual(ns["forward"](m, x).to_local(), ref.to_local())
        finally:
            dist.destroy_process_group()
            for k, v in saved_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_cache_holds_only_artifact(self):
        # The cache is purely an acceleration: the only COMPILED blob it carries is the
        # ``artifact`` (no weights, no calling-convention metadata -- that lives in
        # python_code, the single source of truth, and load() parses it back from
        # there). The envelope additionally carries a lightweight format/version/backend
        # integrity tag (plain str/int), which load() verifies.
        m = torch.nn.Sequential(torch.nn.Linear(4, 3)).eval()
        x = torch.randn(5, 4)
        code, cache = torch.compiler.precompile(
            lambda model, x: model(x), example_inputs=[(m, x)]
        )

        from torch._precompile import _CACHE_FORMAT, _CACHE_VERSION

        blob = torch.load(io.BytesIO(cache), weights_only=False)
        # The artifact is the only compiled blob; the rest is the integrity tag (the
        # format/version/backend tag plus a code_hash binding the cache to its python_code).
        self.assertEqual(
            set(blob), {"artifact", "format", "version", "backend", "code_hash"}
        )
        self.assertEqual(blob["format"], _CACHE_FORMAT)
        self.assertEqual(blob["version"], _CACHE_VERSION)
        self.assertEqual(blob["backend"], "inductor")
        self.assertIsInstance(blob["artifact"], bytes)
        # The calling convention is recoverable from python_code alone.
        from torch._precompile import _parse_artifact_metadata

        meta = _parse_artifact_metadata(code)
        self.assertEqual(meta["BACKEND"], "inductor")
        self.assertEqual(meta["MODULE_POSITIONS"], [0])

        # load() works using metadata from python_code + artifact from the cache.
        f_c = torch.compiler.precompile.load(code, cache)
        self.assertEqual(f_c(m, x), m(x))

    def test_inlined_fallback_when_artifact_absent(self):
        # When the cache holds no serialized artifact, load() falls back to
        # executing the inlined python (recompiling kernels). Force that branch by
        # stripping the artifact and check it still matches eager; this also
        # exercises the self-contained inlined path (JIT from inlined source).
        m = torch.nn.Sequential(torch.nn.Linear(4, 3)).eval()
        x = torch.randn(5, 4)
        code, cache = torch.compiler.precompile(
            lambda model, x: model(x), example_inputs=[(m, x)]
        )

        blob = torch.load(io.BytesIO(cache), weights_only=False)
        self.assertIsNotNone(blob["artifact"])

        f_c = torch.compiler.precompile.load(code, _strip_artifact(cache))
        self.assertEqual(f_c(m, x), m(x))

    def test_cache_envelope_is_weights_only_safe(self):
        # The cache is a plain {"artifact": bytes, "format"/"version"/"backend": ...}
        # envelope of only str/int/bytes: it loads with the safe unpickler
        # (weights_only=True). The executable part is the inner artifact bytes, fed to
        # load_cache_artifacts inside load() to prime the inductor cache -- that (plus the
        # subsequent exec of python_code) is the code-execution step, not this outer load.
        # The integrity tag is present and correct (and itself weights_only-safe).
        from torch._precompile import _CACHE_FORMAT, _CACHE_VERSION

        m = torch.nn.Sequential(torch.nn.Linear(4, 3)).eval()
        x = torch.randn(5, 4)
        _code, cache = torch.compiler.precompile(
            lambda model, x: model(x), example_inputs=[(m, x)]
        )
        blob = torch.load(io.BytesIO(cache), weights_only=True)  # must not raise
        self.assertEqual(
            set(blob), {"artifact", "format", "version", "backend", "code_hash"}
        )
        self.assertEqual(blob["format"], _CACHE_FORMAT)
        self.assertEqual(blob["version"], _CACHE_VERSION)
        self.assertEqual(blob["backend"], "inductor")
        # code_hash is a plain str (sha256 hexdigest), so the envelope stays
        # weights_only-safe even with this added key.
        self.assertIsInstance(blob["code_hash"], str)

    def test_wrong_param_count_model_rejected(self):
        # Invariant 2: a runtime model whose param/buffer count differs from the
        # traced model is rejected with a clear error rather than an opaque inner
        # failure. This exercises the default eager load path, which execs
        # python_code (the eager cache carries no artifact).
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        code, cache = torch.compiler.precompile(
            lambda model, x: model(x), example_inputs=[(m, x)]
        )
        f_c = torch.compiler.precompile.load(code, cache)

        bigger = torch.nn.Sequential(
            torch.nn.Linear(4, 4), torch.nn.Linear(4, 3)
        ).eval()
        with self.assertRaisesRegex(PrecompileError, "structurally identical"):
            f_c(bigger, x)

    def test_wrong_param_count_rejected_inlined(self):
        # The same guard fires on the inlined (no-cache) path with the same exception
        # type as the cached path (PrecompileError): strip the artifact so load()
        # execs python_code, then call with a structurally different model.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        code, cache = torch.compiler.precompile(
            lambda model, x: model(x), example_inputs=[(m, x)]
        )
        f_c = torch.compiler.precompile.load(code, _strip_artifact(cache))

        bigger = torch.nn.Sequential(
            torch.nn.Linear(4, 4), torch.nn.Linear(4, 3)
        ).eval()
        with self.assertRaisesRegex(PrecompileError, "structurally identical"):
            f_c(bigger, x)

    def test_runtime_input_structure_mismatch_rejected(self):
        # Invariant 3: a runtime input whose pytree structure differs from the traced
        # example (here a list where a bare tensor was traced) is rejected via the
        # IN_SPEC check, rather than silently flattening to the wrong leaves.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        code, cache = torch.compiler.precompile(
            lambda model, x: model(x), example_inputs=[(m, x)]
        )
        f_c = torch.compiler.precompile.load(code, cache)
        with self.assertRaisesRegex(PrecompileError, "different structure"):
            f_c(m, [x, x])

    def test_unserializable_in_spec_still_compiles(self):
        # A runtime input whose pytree TreeSpec is not JSON-serializable (an unregistered
        # collections.namedtuple) must still compile/run on the default eager backend:
        # IN_SPEC degrades to None and the structure check is skipped rather than
        # hard-failing.
        import collections

        P = collections.namedtuple("P", ["x", "y"])
        m = torch.nn.Linear(4, 3).eval()
        inp = P(torch.randn(5, 4), torch.randn(5, 4))
        code, cache = torch.compiler.precompile(
            lambda model, p: model(p.x + p.y), example_inputs=[(m, inp)]
        )
        self.assertIn("IN_SPEC = None", code)
        f_c = torch.compiler.precompile.load(code, cache)
        self.assertEqual(f_c(m, inp), m(inp.x + inp.y))

    def test_unserializable_context_in_spec_still_compiles(self):
        # A registered pytree node whose context is not JSON-dumpable makes
        # treespec_dumps raise TypeError (not NotImplementedError); IN_SPEC must still
        # degrade to None rather than crashing precompile.
        m = torch.nn.Linear(4, 3).eval()
        inp = _UnserializableCtxInput(torch.randn(5, 4), torch.randn(5, 4))
        code, cache = torch.compiler.precompile(
            lambda model, h: model(h.a + h.b), example_inputs=[(m, inp)]
        )
        self.assertIn("IN_SPEC = None", code)
        f_c = torch.compiler.precompile.load(code, cache)
        self.assertEqual(f_c(m, inp), m(inp.a + inp.b))

    def test_unserializable_out_spec_hard_fails(self):
        # OUT_SPEC is load-bearing (the driver rebuilds fn's output via tree_unflatten),
        # so unlike IN_SPEC it CANNOT degrade to None. An fn that RETURNS an unregistered
        # collections.namedtuple has a non-JSON-serializable output TreeSpec and must
        # raise a clear PrecompileError rather than leaking a raw pytree error.
        import collections

        Out = collections.namedtuple("Out", ["a", "b"])
        with self.assertRaisesRegex(
            PrecompileError, "cannot serialize the output structure"
        ):
            torch.compiler.precompile(
                lambda x: Out(x + 1, x + 2), example_inputs=[(torch.randn(4),)]
            )

    def test_input_leaf_count_mismatch_rejected_when_spec_unserializable(self):
        # When IN_SPEC degrades to None the structural in_spec check is skipped; a runtime
        # input flattening to a DIFFERENT leaf count must still raise a clean
        # PrecompileError (not a raw zip/unpack error) on the live and eager-inlined paths.
        m = torch.nn.Linear(4, 3).eval()
        inp = _UnserializableCtxInput(torch.randn(5, 4), torch.randn(5, 4))
        for backend in ("inductor", "eager"):
            code, cache = torch.compiler.precompile(
                lambda model, h: model(h.a + h.b),
                example_inputs=[(m, inp)],
                backend=backend,
            )
            self.assertIn("IN_SPEC = None", code)
            f = torch.compiler.precompile.load(code, cache)
            with self.assertRaisesRegex(PrecompileError, "flattened to"):
                f(m, torch.randn(5, 4))  # one leaf vs the traced two

    def test_user_input_error_precedes_structural_error(self):
        # All three load paths run the user-input checks BEFORE the structural model-name
        # check, so a call violating BOTH (wrong dtype and a different model) reports the
        # user-input (dtype) error, keeping the first-reported error consistent.
        m = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 3)).eval()
        x = torch.randn(5, 4)

        class B(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.l0 = torch.nn.Linear(4, 4)
                self.l1 = torch.nn.Linear(4, 3)

            def forward(self, t):
                return self.l1(self.l0(t))

        code, cache = torch.compiler.precompile(
            lambda mm, t: mm(t), example_inputs=[(m, x)]
        )
        f_c = torch.compiler.precompile.load(code, cache)
        f_i = torch.compiler.precompile.load(code, _strip_artifact(cache))
        code_e, cache_e = torch.compiler.precompile(
            lambda mm, t: mm(t), example_inputs=[(m, x)], backend="eager"
        )
        f_e = torch.compiler.precompile.load(code_e, cache_e)
        for f in (f_c, f_i, f_e):
            with self.assertRaisesRegex(PrecompileError, "dtype"):
                f(
                    B(), x.double()
                )  # wrong model AND wrong dtype -> dtype reported first

    def test_unserializable_out_spec_rejected(self):
        # OUT_SPEC is load-bearing (the driver rebuilds fn's output via tree_unflatten),
        # so unlike IN_SPEC it cannot degrade to None: a fn returning an unregistered
        # namedtuple must fail with a clear PrecompileError, not a raw pytree error, on
        # both backends. A registered namedtuple output round-trips fine.
        import collections

        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        NT = collections.namedtuple("NT", ["p", "q"])
        for backend in ("inductor", "eager"):
            with self.assertRaisesRegex(PrecompileError, "output structure"):
                torch.compiler.precompile(
                    lambda model, xx: NT(model(xx), model(xx) + 1),
                    example_inputs=[(m, x)],
                    backend=backend,
                )
        # A registered namedtuple output serializes and round-trips on both backends.
        # Registration mutates the process-global pytree registry, so deregister it on
        # cleanup rather than leaking the node into later tests.
        RNT = collections.namedtuple("RNT", ["p", "q"])
        _pytree._register_namedtuple(RNT, serialized_type_name="test_precompile.RNT")
        self.addCleanup(_pytree._deregister_pytree_node, RNT)
        ref = (m(x), m(x) + 1)
        for backend in ("inductor", "eager"):
            code, cache = torch.compiler.precompile(
                lambda model, xx: RNT(model(xx), model(xx) + 1),
                example_inputs=[(m, x)],
                backend=backend,
            )
            out = torch.compiler.precompile.load(code, cache)(m, x)
            self.assertEqual((out.p, out.q), ref)

    def test_cached_and_inlined_paths_agree(self):
        # Both load paths exec the SAME inlined driver in python_code; the only difference
        # is whether the cache primed the kernels first (warm) or not (cold JIT). They must
        # produce identical results -- cross-check via identical scattered grads from a
        # cache-primed load and a cache-stripped (artifact=None) load of the SAME artifact,
        # with multiple modules AND a tied weight across two of them (the case where an
        # ordering divergence in the embedded _extract_param_buffers would show).
        torch.manual_seed(0)
        a = torch.nn.Linear(4, 4, bias=False)
        b = torch.nn.Linear(4, 4, bias=False)
        b.weight = a.weight  # tie across two distinct module args
        c = torch.nn.Linear(4, 3)
        loss_fn = torch.nn.MSELoss()
        x = torch.randn(2, 4)
        target = torch.randn(2, 3)

        def step(ma, mb, mc, x, target):
            loss_fn(mc(mb(torch.relu(ma(x)))), target).backward()

        code, cache = torch.compiler.precompile(
            step, example_inputs=[(a, b, c, x, target)]
        )

        def grads(ms):
            return [p.grad for m in ms for p in m.parameters()]

        # deepcopy the three together so the a/b weight tie is preserved.
        ca, cb, cc = copy.deepcopy((a, b, c))
        torch.compiler.precompile.load(code, cache)(
            ca, cb, cc, x, target
        )  # cached path

        ia, ib, ic = copy.deepcopy((a, b, c))
        torch.compiler.precompile.load(code, _strip_artifact(cache))(
            ia, ib, ic, x, target
        )  # inlined

        for cg, ig in zip(grads((ca, cb, cc)), grads((ia, ib, ic))):
            self.assertEqual(cg, ig)

    def test_eager_param_ordering_agrees_with_inductor(self):
        # Both backends now emit the same _extract_param_buffers (from
        # torch._precompile_driver), which must stay in sync with
        # torch._precompile._intern_param_buffers. The test above cross-checks only the
        # cached vs inductor-inlined paths; cross-check the EAGER backend too, on the same
        # multi-module + tied-weight + backward step, so an ordering divergence in the
        # shared driver shows as a scattered-grad mismatch against the inductor cached path.
        torch.manual_seed(0)
        a = torch.nn.Linear(4, 4, bias=False)
        b = torch.nn.Linear(4, 4, bias=False)
        b.weight = a.weight  # tie across two distinct module args
        c = torch.nn.Linear(4, 3)
        loss_fn = torch.nn.MSELoss()
        x = torch.randn(2, 4)
        target = torch.randn(2, 3)

        def step(ma, mb, mc, x, target):
            loss_fn(mc(mb(torch.relu(ma(x)))), target).backward()

        def grads(ms):
            return [p.grad for m in ms for p in m.parameters()]

        # deepcopy the three together so the a/b weight tie is preserved.
        icode, icache = torch.compiler.precompile(
            step, example_inputs=[(a, b, c, x, target)]
        )
        ia, ib, ic = copy.deepcopy((a, b, c))
        torch.compiler.precompile.load(icode, icache)(
            ia, ib, ic, x, target
        )  # inductor cached path

        ecode, ecache = torch.compiler.precompile(
            step, example_inputs=[(a, b, c, x, target)], backend="eager"
        )
        ea, eb, ec = copy.deepcopy((a, b, c))
        torch.compiler.precompile.load(ecode, ecache)(
            ea, eb, ec, x, target
        )  # eager path

        ind_grads = grads((ia, ib, ic))
        eager_grads = grads((ea, eb, ec))
        self.assertEqual(len(ind_grads), len(eager_grads))
        for ig, eg in zip(ind_grads, eager_grads):
            self.assertEqual(ig, eg)

    def test_non_module_at_module_position_rejected(self):
        # Passing a non-nn.Module where the traced fn took a module yields a clear
        # PrecompileError citing invariant 2, not a bare AttributeError.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        code, cache = torch.compiler.precompile(
            lambda model, x: model(x), example_inputs=[(m, x)]
        )
        f_c = torch.compiler.precompile.load(code, cache)
        with self.assertRaisesRegex(PrecompileError, "must be the nn.Module"):
            f_c(x, x)  # tensor at the module slot

    def test_wrong_arg_count_rejected(self):
        # A runtime call with the wrong number of positional args raises a clear
        # PrecompileError (invariant 2) -- not a raw IndexError -- on all three load
        # paths, including when a module is at a non-zero position (where args[i] would
        # otherwise index past the short args tuple).
        m = torch.nn.Linear(4, 3)
        x = torch.randn(2, 4)
        # Module at position 1 (so a missing trailing arg would index past args).
        code, cache = torch.compiler.precompile(
            lambda xx, model: model(xx), example_inputs=[(x, m)]
        )
        inlined_cache = _strip_artifact(cache)  # force the inlined path
        ecode, ecache = torch.compiler.precompile(
            lambda xx, model: model(xx), example_inputs=[(x, m)], backend="eager"
        )
        loaders = {
            "cached": torch.compiler.precompile.load(code, cache),
            "inlined": torch.compiler.precompile.load(code, inlined_cache),
            "eager": torch.compiler.precompile.load(ecode, ecache),
        }
        for label, f_c in loaders.items():
            with self.subTest(path=label):
                with self.assertRaisesRegex(PrecompileError, "expected 2 positional"):
                    f_c(x)  # too few (omits the module arg)
                with self.assertRaisesRegex(PrecompileError, "expected 2 positional"):
                    f_c(x, m, x)  # too many
                self.assertEqual(f_c(x, m), m(x))  # correct arity still works

    def test_buffer_requiring_grad_rejected(self):
        # A registered buffer with requires_grad=True that receives a gradient is not
        # harvested (only params are), so precompile rejects it rather than silently
        # dropping the grad.
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("b", torch.randn(4, requires_grad=True))

            def forward(self, x):
                return (x * self.b).sum()

        m = M()
        x = torch.randn(4)
        with self.assertRaisesRegex(PrecompileError, "buffer received a gradient"):
            torch.compiler.precompile(
                lambda model, x: model(x).backward(), example_inputs=[(m, x)]
            )

    def test_user_input_requiring_grad_rejected(self):
        # Sibling of the buffer guard: a requires_grad USER INPUT (not a param) that
        # receives a gradient during the traced backward is not harvested (only params
        # are), so precompile rejects it rather than silently dropping the grad.
        x = torch.randn(4, requires_grad=True)
        with self.assertRaisesRegex(PrecompileError, "user input received a gradient"):
            torch.compiler.precompile(
                lambda t: (t * t).sum().backward(), example_inputs=[(x,)]
            )

    def test_control_flow_subgraph_rejected(self):
        # torch.cond captures as a HOP with get_attr subgraph submodules, which the
        # standalone artifact cannot inline; reject it at capture with a clear message.
        def f(x):
            return torch.cond(x.sum() > 0, lambda t: t + 1, lambda t: t - 1, (x,))

        with self.assertRaisesRegex(PrecompileError, "control-flow subgraph"):
            torch.compiler.precompile(f, example_inputs=[(torch.randn(4),)])

    def test_load_falls_back_when_cache_unreconstructable(self):
        # The cache is only an acceleration; python_code always runs standalone. A
        # corrupt / stale cache must degrade to the inlined JIT path, not crash.
        m = torch.nn.Sequential(torch.nn.Linear(4, 3)).eval()
        x = torch.randn(5, 4)
        code, cache = torch.compiler.precompile(
            lambda model, x: model(x), example_inputs=[(m, x)]
        )
        blob = torch.load(io.BytesIO(cache), weights_only=True)
        self.assertIsNotNone(blob["artifact"])
        blob["artifact"] = b"corrupt-not-a-real-artifact"
        buf = io.BytesIO()
        torch.save(blob, buf)

        f_c = torch.compiler.precompile.load(code, buf.getvalue())  # must not raise
        self.assertEqual(f_c(m, x), m(x))

    def test_load_falls_back_on_corrupt_cache_envelope(self):
        # Not just a bad inner artifact -- a corrupt/truncated cache ENVELOPE (not even
        # a valid torch.save blob) must also degrade to the inlined python_code path,
        # since the cache is purely an acceleration.
        m = torch.nn.Sequential(torch.nn.Linear(4, 3)).eval()
        x = torch.randn(5, 4)
        code, _cache = torch.compiler.precompile(
            lambda model, x: model(x), example_inputs=[(m, x)]
        )
        f_c = torch.compiler.precompile.load(
            code, b"not-a-torch-save-blob"
        )  # must not raise
        self.assertEqual(f_c(m, x), m(x))

    def test_load_invalid_python_code_rejected(self):
        # load() surfaces a clear PrecompileError (not a raw SyntaxError) when
        # python_code is not valid Python.
        buf = io.BytesIO()
        torch.save({"artifact": None}, buf)
        with self.assertRaisesRegex(PrecompileError, "not valid Python"):
            torch.compiler.precompile.load("def (:::", buf.getvalue())

    def test_untrusted_input_warning_fires_per_load(self):
        # The trust warning is emitted PER load (not warning_once) via log.warning on the
        # torch._precompile logger: load() always execs python_code (through
        # _make_inlined_forward), which warns before the exec, whether or not the cache
        # primed the kernels first. Calling load() TWICE must fire the untrusted-input
        # warning on BOTH calls, locking in per-load behavior rather than once-per-process.
        m = torch.nn.Sequential(torch.nn.Linear(4, 3)).eval()
        x = torch.randn(5, 4)
        # Cached path (inductor): the exec of python_code warns about untrusted input.
        code, cache = torch.compiler.precompile(
            lambda model, t: model(t), example_inputs=[(m, x)]
        )
        for _ in range(2):
            with self.assertLogs("torch._precompile", level="WARNING") as cm:
                torch.compiler.precompile.load(code, cache)
            self.assertTrue(
                any("untrusted" in line.lower() for line in cm.output),
                f"cached load did not warn about untrusted input: {cm.output}",
            )
        # Eager backend (empty cache, nothing to prime): load() still EXECs python_code
        # via _make_inlined_forward, which warns about exec'ing untrusted code every load.
        ecode, ecache = torch.compiler.precompile(
            lambda model, t: model(t), example_inputs=[(m, x)], backend="eager"
        )
        for _ in range(2):
            with self.assertLogs("torch._precompile", level="WARNING") as cm:
                torch.compiler.precompile.load(ecode, ecache)
            self.assertTrue(
                any("untrusted" in line.lower() for line in cm.output),
                f"inlined load did not warn about untrusted input: {cm.output}",
            )
            self.assertTrue(
                any("EXEC" in line for line in cm.output),
                f"inlined load did not warn about exec'ing python_code: {cm.output}",
            )

    def test_no_compute_graph_rejected_inductor(self):
        # The inductor backend produces no runnable module for a graph with no compute
        # to lower -- one that returns inputs or Python constants unchanged (a constant,
        # a bare passthrough, or an alias like .detach()). Reject with a clear
        # PrecompileError rather than a raw "found 0 runnable modules" RuntimeError. The
        # eager backend handles these (the contract is otherwise identical).
        x = torch.randn(4)
        for fn in (lambda xx: 7, lambda xx: xx, lambda xx: xx.detach()):
            with self.assertRaisesRegex(PrecompileError, "no compute"):
                torch.compiler.precompile(fn, example_inputs=[(x,)])
        # The eager backend handles a passthrough and a constant fn.
        code, cache = torch.compiler.precompile(
            lambda xx: xx, example_inputs=[(x,)], backend="eager"
        )
        self.assertEqual(torch.compiler.precompile.load(code, cache)(x), x)
        code, cache = torch.compiler.precompile(
            lambda xx: 7, example_inputs=[(x,)], backend="eager"
        )
        self.assertEqual(torch.compiler.precompile.load(code, cache)(x), 7)

    def test_same_count_different_structure_rejected(self):
        # Invariant 2: the structural check now compares the baked PARAM_NAMES /
        # BUFFER_NAMES against the runtime model's extracted param/buffer names, so a
        # same-count-but-different-structure (here, differently-NAMED submodules) model
        # is REJECTED rather than silently running the traced graph with the wrong
        # weights. Both the cached and the inlined (artifact-stripped) load paths fire.
        a = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)).eval()
        x = torch.randn(2, 4)
        code, cache = torch.compiler.precompile(
            lambda m, x: m(x), example_inputs=[(a, x)]
        )
        # The traced names come from the Sequential (``0.weight``, ``1.weight`` ...).
        self.assertIn(
            "PARAM_NAMES = ['0.weight', '0.bias', '1.weight', '1.bias']", code
        )

        class B(torch.nn.Module):  # same 4 params (same count/shapes), different names
            def __init__(self):
                super().__init__()
                self.l0 = torch.nn.Linear(4, 4)
                self.l1 = torch.nn.Linear(4, 4)

            def forward(self, x):
                return self.l0(x) + self.l1(x)

        b = B().eval()
        loaders = {
            "cached": torch.compiler.precompile.load(code, cache),
            "inlined": torch.compiler.precompile.load(code, _strip_artifact(cache)),
        }
        for label, f_c in loaders.items():
            with self.subTest(path=label):
                with self.assertRaisesRegex(
                    PrecompileError, "do not match the traced model"
                ):
                    f_c(b, x)

    def test_same_count_different_structure_rejected_eager(self):
        # The eager driver's _check_structure rejects a same-param-COUNT but
        # different-NAME model (here differently-named submodules) rather than
        # silently running the traced graph with the wrong weights (invariant 2).
        # What's distinct from test_wrong_param_count_model_rejected above is the
        # INPUT -- same count / different name, not a count mismatch.
        a = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)).eval()
        x = torch.randn(2, 4)
        code, cache = torch.compiler.precompile(
            lambda m, x: m(x), example_inputs=[(a, x)], backend="eager"
        )
        self.assertIn(
            "PARAM_NAMES = ['0.weight', '0.bias', '1.weight', '1.bias']", code
        )

        class B(torch.nn.Module):  # same 4 params (same count/shapes), different names
            def __init__(self):
                super().__init__()
                self.l0 = torch.nn.Linear(4, 4)
                self.l1 = torch.nn.Linear(4, 4)

            def forward(self, x):
                return self.l0(x) + self.l1(x)

        b = B().eval()
        f_c = torch.compiler.precompile.load(code, cache)
        with self.assertRaisesRegex(PrecompileError, "do not match the traced model"):
            f_c(b, x)

    # Input mutation, output aliasing, tensor subclasses, and functionalized RNG are
    # SUPPORTED: the inductor backend lowers through aot_autograd.compile_to_python,
    # which composes AOTAutograd's own codegen'd prelude/epilogue into the artifact.
    # Only effectful ops are rejected up front (see test_effectful_op_unsupported).

    def test_effectful_op_unsupported(self):
        # Effectful custom ops are rejected up front by _assert_supported, which
        # detects the with_effects HOP in the captured graph -- the effect cannot
        # be lowered to standalone source, so capture fails cleanly.
        from torch._higher_order_ops.effects import _EffectType, _register_effectful_op
        from torch.library import _scoped_library

        with _scoped_library("mlprecompile", "FRAGMENT") as lib:
            lib.define("eff(Tensor x) -> Tensor")
            lib.impl("eff", lambda x: x + 1.0, "CompositeExplicitAutograd")
            lib.impl("eff", lambda x: torch.empty_like(x), "Meta")
            op = torch.ops.mlprecompile.eff.default
            _register_effectful_op(op, _EffectType.ORDERED)
            try:
                with self.assertRaisesRegex(
                    PrecompileError, "effectful op.*not supported yet"
                ):
                    torch.compiler.precompile(
                        lambda a: torch.ops.mlprecompile.eff(a),
                        example_inputs=[(torch.randn(4),)],
                    )
            finally:
                _register_effectful_op(op, None)

    def test_public_api_surface(self):
        # precompile is a public API under the compiler namespace
        # (torch.compiler.precompile), with a load method and a public error type;
        # it is deliberately NOT a top-level torch.* verb.
        self.assertIn("precompile", torch.compiler.__all__)
        self.assertNotIn("precompile", torch.__all__)
        # __all__ membership and the attribute itself are independent, so lock in
        # removal of the top-level entry point too (re-adding the re-export without
        # touching __all__ would silently resurrect torch.precompile).
        self.assertFalse(hasattr(torch, "precompile"))
        self.assertTrue(callable(torch.compiler.precompile))
        self.assertTrue(callable(torch.compiler.precompile.load))
        self.assertIs(torch.compiler.precompile.PrecompileError, PrecompileError)
        # The public location: test_public_bindings.test_correct_module_names also
        # enforces this for every torch.compiler.__all__ member.
        self.assertEqual(torch.compiler.precompile.__module__, "torch.compiler")
        for name in (
            "ExampleInput",
            "GuardFact",
            "FrameInvariants",
            "PrecompiledCallable",
            "PrecompileSummary",
        ):
            self.assertIn(name, torch.compiler.__all__)
            self.assertEqual(getattr(torch.compiler, name).__module__, "torch.compiler")

    def test_precompile_public_signature_and_type_hints(self):
        signature = inspect.signature(torch.compiler.precompile)
        self.assertEqual(
            signature.parameters["example_inputs"].kind,
            inspect.Parameter.KEYWORD_ONLY,
        )
        self.assertEqual(
            typing.get_type_hints(torch.compiler.precompile.__call__)["return"],
            tuple[str, bytes],
        )
        self.assertIs(
            typing.get_type_hints(torch.compiler.precompile.load)["return"],
            torch.compiler.PrecompiledCallable,
        )
        self.assertEqual(torch.compiler.precompile.load.__module__, "torch.compiler")
        self.assertEqual(torch.compiler.precompile.load.__qualname__, "precompile.load")

    def test_precompile_documents_dynamo_capture_options(self):
        documentation = inspect.getdoc(torch.compiler.precompile)
        for option in (
            "guard_filter_fn",
            "recompile_limit",
            "dynamic",
            "invariants",
            "require_complete",
            "require_no_risky_drops",
            "require_no_dropped_guards",
        ):
            self.assertIn(option, documentation)

    def test_backend_invalid_raises(self):
        a, b = torch.randn(4, 4), torch.randn(4, 4)
        with self.assertRaisesRegex(
            ValueError, "backend must be 'inductor' or 'eager'"
        ):
            torch.compiler.precompile(
                lambda x, y: x + y, example_inputs=[(a, b)], backend="nope"
            )

    def test_tracer_default_and_explicit_make_fx(self):
        # tracer defaults to "make_fx"; passing it explicitly is equivalent and works.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        for kwargs in ({}, {"tracer": "make_fx"}):
            code, cache = torch.compiler.precompile(
                lambda model, xx: model(xx), example_inputs=[(m, x)], **kwargs
            )
            self.assertEqual(torch.compiler.precompile.load(code, cache)(m, x), m(x))

    @parametrize("num_examples", [0, 2])
    def test_make_fx_requires_one_example_input(self, num_examples):
        x = torch.randn(4)
        message = "requires example_inputs" if num_examples == 0 else "exactly one"
        with self.assertRaisesRegex((AssertionError, ValueError), message):
            torch.compiler.precompile(
                lambda t: t + 1,
                example_inputs=[(x,)] * num_examples,
                backend="eager",
            )

    def test_positional_example_inputs_remain_supported(self):
        x = torch.randn(4)
        y = torch.randn(4)
        code, cache = torch.compiler.precompile(
            lambda left, right: left + right, x, y, backend="eager"
        )
        self.assertEqual(torch.compiler.precompile.load(code, cache)(x, y), x + y)

        with self.assertRaisesRegex(TypeError, "both positional examples and"):
            torch.compiler.precompile(
                lambda t: t + 1,
                x,
                example_inputs=[(x,)],
                backend="eager",
            )

    @parametrize(
        "option",
        (
            {"recompile_limit": 8},
            {"dynamic": False},
            {"guard_filter_fn": lambda guards: [True] * len(guards)},
            {"invariants": "unused.txt"},
            {"require_complete": False},
            {"require_no_risky_drops": False},
            {"require_no_dropped_guards": True},
        ),
    )
    def test_precompile_dynamo_options_require_dynamo(self, option):
        with self.assertRaisesRegex(ValueError, "apply only to tracer='dynamo'"):
            torch.compiler.precompile(
                lambda x: x + 1,
                example_inputs=[(torch.randn(4),)],
                backend="eager",
                **option,
            )

    def test_zero_argument_call_remains_supported(self):
        code, cache = torch.compiler.precompile(
            lambda: torch.ones(4) + 1, backend="eager"
        )
        self.assertEqual(
            torch.compiler.precompile.load(code, cache)(), torch.full((4,), 2.0)
        )

    @parametrize("bad", (torch.ones(1), torch.nn.Linear(1, 1), "input"))
    def test_precompile_rejects_bare_example_container(self, bad):
        with self.assertRaisesRegex(TypeError, "sequence of calls"):
            torch.compiler.precompile(lambda x: x, example_inputs=bad, backend="eager")

    def test_example_input_supports_make_fx_positional_args_only(self):
        x = torch.randn(4)
        example = torch.compiler.ExampleInput(args=(x,))
        code, cache = torch.compiler.precompile(
            lambda t: t + 1, example_inputs=[example], backend="eager"
        )
        self.assertEqual(torch.compiler.precompile.load(code, cache)(x), x + 1)
        with self.assertRaisesRegex(NotImplementedError, "keyword example inputs"):
            torch.compiler.precompile(
                lambda t, *, scale: t * scale,
                example_inputs=[
                    torch.compiler.ExampleInput(args=(x,), kwargs={"scale": 2})
                ],
                backend="eager",
            )

    @parametrize("backend", ("eager", "inductor"))
    def test_make_fx_artifact_reproduces_capture_autocast(self, backend):
        def fn(a, b):
            return a @ b

        a = torch.randn(8, 8)
        b = torch.randn(8, 8)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            expected = fn(a, b)
            code, cache = torch.compiler.precompile(
                fn, example_inputs=[(a, b)], backend=backend
            )
        loaded = torch.compiler.precompile.load(code, cache)
        plain = loaded(a, b)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            under_autocast = loaded(a, b)
        self.assertEqual(plain.dtype, torch.bfloat16)
        self.assertEqual(plain, expected)
        self.assertEqual(under_autocast, expected)

    @parametrize("backend", ("eager", "inductor"))
    @parametrize("decompositions", (None, {}))
    def test_make_fx_torch_function_mode_applies_once(self, backend, decompositions):
        x = torch.zeros(3)
        with _PrecompilePlusOneMode():
            expected = _precompile_add_one(x).clone()
            code, cache = torch.compiler.precompile(
                _precompile_add_one,
                example_inputs=[(x,)],
                backend=backend,
                decompositions=decompositions,
            )
        self.assertEqual(torch.compiler.precompile.load(code, cache)(x), expected)

    @torch._dynamo.config.patch(
        automatic_dynamic_shapes=True, assume_static_by_default=True
    )
    def test_tracer_dynamo_recompiles_to_dynamic_graph(self):
        examples = [(torch.randn(size, 4),) for size in (2, 3, 5)]
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_dynamic,
            example_inputs=examples,
            tracer="dynamo",
        )

        self.assertIn('TRACER = "dynamo"', code)
        self.assertIn("FRAME_COUNT = 1", code)
        self.assertIn("VARIANT_COUNT = 2", code)
        self.assertIn("GRAPH_COUNT = 2", code)
        self.assertIn("DYNAMIC_GRAPH_COUNT = 1", code)
        self.assertIn("Inductor output code", code)
        self.assertIn("Guard trees and Dynamo/disabled-function bytecode", code)
        self.assertIn("# Backend graph 0:", code)
        self.assertIn("_DYNAMO_BACKENDS[", code)
        self.assertNotIn("_DYNAMO_BACKEND_SOURCES", code)
        self.assertIn(
            "# Generated by torch._functorch.aot_autograd.compile_to_python", code
        )
        blob = torch.load(io.BytesIO(cache), weights_only=True)
        self.assertIsInstance(blob["artifact"], list)
        self.assertTrue(
            all(item is None or isinstance(item, bytes) for item in blob["artifact"])
        )
        guard_summaries = _dynamo_serialized_guard_summary(code)
        self.assertEqual(len(guard_summaries), 2)
        for guard_types, _, _, has_shape_guards in guard_summaries:
            self.assertIn("TENSOR_MATCH", guard_types)
            self.assertNotIn("GLOBAL_STATE", guard_types)
            self.assertTrue(has_shape_guards)

        for _, loaded in _default_and_inlined_loaders(code, cache, "inductor"):
            self.assertEqual(loaded.capture_summary.variant_examples[0], (0, 1))
            for size in (2, 7):
                x = torch.randn(size, 4)
                self.assertEqual(loaded(x), _precompile_dynamo_dynamic(x))
            with self.assertRaisesRegex(PrecompileError, "no captured Dynamo variant"):
                loaded(torch.randn(1, 4))

    def test_tracer_dynamo_external_python_function(self):
        x = torch.randn(4)
        target = torch.randn(4)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_mse_loss,
            example_inputs=[(x, target)],
            tracer="dynamo",
            backend="eager",
        )
        loaded = torch.compiler.precompile.load(code, cache)
        self.assertEqual(loaded(x, target), _precompile_dynamo_mse_loss(x, target))

    def test_tracer_dynamo_rejects_non_function_callable(self):
        with self.assertRaisesRegex(NotImplementedError, "requires a Python function"):
            torch.compiler.precompile(
                functools.partial(_precompile_dynamo_dynamic),
                example_inputs=[(torch.randn(4),)],
                tracer="dynamo",
                backend="eager",
            )

    def test_tracer_dynamo_rejects_closure(self):
        offset = 1

        def fn(x):
            return x + offset

        with self.assertRaisesRegex(NotImplementedError, "closure cells"):
            torch.compiler.precompile(
                fn,
                example_inputs=[(torch.randn(4),)],
                tracer="dynamo",
                backend="eager",
            )

    def test_tracer_dynamo_rejects_decompositions(self):
        with self.assertRaisesRegex(NotImplementedError, "decompositions"):
            torch.compiler.precompile(
                _precompile_dynamo_dynamic,
                example_inputs=[(torch.randn(4),)],
                tracer="dynamo",
                backend="eager",
                decompositions={},
            )

    def test_tracer_dynamo_library_class_construction(self):
        x = torch.randn(4)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_construct_relu,
            example_inputs=[(x,)],
            tracer="dynamo",
            backend="eager",
        )
        loaded = torch.compiler.precompile.load(code, cache)
        self.assertEqual(loaded(x), _precompile_dynamo_construct_relu(x))

    def test_tracer_dynamo_rejects_imported_helper_input_alias(self):
        with self.assertRaisesRegex(PrecompileError, "input-derived"):
            torch.compiler.precompile(
                _precompile_dynamo_external_module_identity,
                example_inputs=[(_DYNAMO_TENSOR_DEFAULT,)],
                tracer="dynamo",
                backend="eager",
            )

    def test_tracer_dynamo_rejects_library_helper_container_input_alias(self):
        torch._precompile_box = [_DYNAMO_TENSOR_DEFAULT]
        torch._precompile_helper = types.FunctionType(
            _precompile_dynamo_torch_helper_template.__code__,
            torch.__dict__,
            "_precompile_helper",
        )
        try:
            with self.assertRaisesRegex(PrecompileError, "input-derived"):
                torch.compiler.precompile(
                    _precompile_dynamo_torch_helper_identity,
                    example_inputs=[(_DYNAMO_TENSOR_DEFAULT,)],
                    tracer="dynamo",
                    backend="eager",
                )
        finally:
            del torch._precompile_helper
            del torch._precompile_box

    def test_tracer_dynamo_rejects_library_callable_container_input_alias(self):
        torch._precompile_box = [_DYNAMO_TENSOR_DEFAULT]
        call = types.FunctionType(
            _precompile_dynamo_torch_callable_template.__code__,
            torch.__dict__,
            "__call__",
        )
        callable_type = type(
            "_PrecompileCallable",
            (),
            {"__module__": "torch", "__call__": call},
        )
        torch._precompile_callable = callable_type()
        try:
            with self.assertRaisesRegex(PrecompileError, "input-derived"):
                torch.compiler.precompile(
                    _precompile_dynamo_torch_callable_identity,
                    example_inputs=[(_DYNAMO_TENSOR_DEFAULT,)],
                    tracer="dynamo",
                    backend="eager",
                )
        finally:
            del torch._precompile_callable
            del torch._precompile_box

    def test_tracer_dynamo_rejects_library_getitem_container_input_alias(self):
        torch._precompile_box = [_DYNAMO_TENSOR_DEFAULT]
        getitem = types.FunctionType(
            _precompile_dynamo_torch_getitem_template.__code__,
            torch.__dict__,
            "__getitem__",
        )
        getitem_type = type(
            "_PrecompileGetitem",
            (),
            {"__module__": "torch", "__getitem__": getitem},
        )
        torch._precompile_getitem = getitem_type()
        try:
            with self.assertRaisesRegex(PrecompileError, "input-derived"):
                torch.compiler.precompile(
                    _precompile_dynamo_torch_getitem_identity,
                    example_inputs=[(_DYNAMO_TENSOR_DEFAULT,)],
                    tracer="dynamo",
                    backend="eager",
                )
        finally:
            del torch._precompile_getitem
            del torch._precompile_box

    def test_tracer_dynamo_rejects_library_class_getitem_input_alias(self):
        torch._precompile_box = [_DYNAMO_TENSOR_DEFAULT]
        class_getitem = types.FunctionType(
            _precompile_dynamo_torch_class_getitem_template.__code__,
            torch.__dict__,
            "__class_getitem__",
        )
        torch._precompile_generic = type(
            "_PrecompileGeneric",
            (),
            {
                "__module__": "torch",
                "__class_getitem__": classmethod(class_getitem),
            },
        )
        try:
            with self.assertRaisesRegex(PrecompileError, "input-derived"):
                torch.compiler.precompile(
                    _precompile_dynamo_torch_class_getitem_identity,
                    example_inputs=[(_DYNAMO_TENSOR_DEFAULT,)],
                    tracer="dynamo",
                    backend="eager",
                )
        finally:
            del torch._precompile_generic
            del torch._precompile_box

    def test_tracer_dynamo_rejects_library_metaclass_getitem_input_alias(self):
        torch._precompile_box = [_DYNAMO_TENSOR_DEFAULT]
        getitem = types.FunctionType(
            _precompile_dynamo_torch_metaclass_getitem_template.__code__,
            torch.__dict__,
            "__getitem__",
        )
        metaclass = type(
            "_PrecompileMeta",
            (type,),
            {"__module__": "torch", "__getitem__": getitem},
        )
        torch._precompile_meta_generic = metaclass(
            "_PrecompileMetaGeneric", (), {"__module__": "torch"}
        )
        try:
            with self.assertRaisesRegex(PrecompileError, "input-derived"):
                torch.compiler.precompile(
                    _precompile_dynamo_torch_metaclass_getitem_identity,
                    example_inputs=[(_DYNAMO_TENSOR_DEFAULT,)],
                    tracer="dynamo",
                    backend="eager",
                )
        finally:
            del torch._precompile_meta_generic
            del torch._precompile_box

    def test_tracer_dynamo_rejects_module_state_library_behavior_alias(self):
        torch._precompile_box = [_DYNAMO_TENSOR_DEFAULT]
        getitem = types.FunctionType(
            _precompile_dynamo_torch_getitem_template.__code__,
            torch.__dict__,
            "__getitem__",
        )
        helper_type = type(
            "_PrecompileGetitem",
            (),
            {"__module__": "torch", "__getitem__": getitem},
        )
        module = _PrecompileDynamoLibraryIndexerModule(helper_type())
        try:
            with self.assertRaisesRegex(PrecompileError, "input callable"):
                torch.compiler.precompile(
                    _precompile_dynamo_call_identity_module,
                    example_inputs=[(module, _DYNAMO_TENSOR_DEFAULT)],
                    tracer="dynamo",
                    backend="eager",
                )
        finally:
            del torch._precompile_box

    @parametrize(
        "fn",
        (
            _precompile_dynamo_function_globals_identity,
            _precompile_dynamo_function_annotations_identity,
            _precompile_dynamo_function_container_globals_identity,
            _precompile_dynamo_function_container_annotations_identity,
        ),
    )
    def test_tracer_dynamo_rejects_function_metadata_input_alias(self, fn):
        with self.assertRaisesRegex(PrecompileError, "input-derived"):
            torch.compiler.precompile(
                fn,
                example_inputs=[(_DYNAMO_TENSOR_DEFAULT,)],
                tracer="dynamo",
                backend="eager",
            )

    @parametrize(
        "helper",
        (
            _precompile_dynamo_torch_module_helper_template,
            _precompile_dynamo_torch_module_getattr_template,
        ),
    )
    def test_tracer_dynamo_rejects_library_helper_module_dispatch(self, helper):
        module = _PrecompileDynamoModuleMethodGlobalIdentity()
        torch._precompile_module_helper = types.FunctionType(
            helper.__code__, torch.__dict__, "_precompile_module_helper"
        )
        try:
            with self.assertRaisesRegex(
                (PrecompileError, NotImplementedError),
                "input callable|dynamic global access",
            ):
                torch.compiler.precompile(
                    _precompile_dynamo_torch_module_helper_identity,
                    example_inputs=[(module, _DYNAMO_TENSOR_DEFAULT)],
                    tracer="dynamo",
                    backend="eager",
                )
        finally:
            del torch._precompile_module_helper

    def test_tracer_dynamo_filters_torch_global_guards_before_serializing(self):
        x = torch.randn(4)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_torch_sin,
            example_inputs=[(x,)],
            tracer="dynamo",
            backend="eager",
        )
        loaded = torch.compiler.precompile.load(code, cache)
        self.assertEqual(loaded(x), torch.sin(x))

    @parametrize("construct", sorted(_PRECOMPILE_EAGER_ROUND_TRIP))
    @parametrize("graph_break", (False, True))
    def test_tracer_dynamo_eager_higher_order_graph(self, construct, graph_break):
        entry = (
            _precompile_eager_graph_break
            if graph_break
            else _PRECOMPILE_EAGER_ROUND_TRIP[construct]
        )
        args = (construct,) if graph_break else ()
        x = torch.randn(4, 4)
        with torch.no_grad():
            expected = torch.compile(entry, backend="eager")(*args, x)
        torch._dynamo.reset()
        code, cache = torch.compiler.precompile(
            entry,
            example_inputs=[(*args, x)],
            tracer="dynamo",
            backend="eager",
            dynamic=False,
        )
        loaded = torch.compiler.precompile.load(code, cache)
        try:
            self.assertEqual(loaded.capture_summary.risky_dropped_guards, ())
            with torch.no_grad():
                self.assertEqual(loaded(*args, x), expected)
        finally:
            if hasattr(loaded, "unload"):
                loaded.unload()

    @parametrize("graph_break", (False, True))
    def test_tracer_dynamo_eager_preserves_no_grad_region(self, graph_break):
        entry = (
            _precompile_eager_graph_break if graph_break else _precompile_eager_no_grad
        )
        args = ("no_grad",) if graph_break else ()
        x = torch.randn(4, requires_grad=True)
        with torch.enable_grad():
            expected = torch.compile(entry, backend="eager")(*args, x)
            torch._dynamo.reset()
            code, cache = torch.compiler.precompile(
                entry,
                example_inputs=[(*args, x)],
                tracer="dynamo",
                backend="eager",
                dynamic=False,
                training=True,
            )
        loaded = torch.compiler.precompile.load(code, cache)
        with torch.enable_grad():
            actual = loaded(*args, x)
        self.assertFalse(actual.requires_grad)
        self.assertEqual(actual, expected)

    def test_tracer_dynamo_eager_load_preserves_ambient_grad_mode(self):
        x = torch.randn(4)
        with torch.no_grad():
            code, cache = torch.compiler.precompile(
                _precompile_eager_no_grad,
                example_inputs=[(x,)],
                tracer="dynamo",
                backend="eager",
                dynamic=False,
            )
            self.assertFalse(torch.is_grad_enabled())
            torch.compiler.precompile.load(code, cache)
            self.assertFalse(torch.is_grad_enabled())

    def test_tracer_dynamo_eager_higher_order_source_runs_in_fresh_process(self):
        from unittest import mock

        x = torch.ones(4)
        code, cache = torch.compiler.precompile(
            _precompile_eager_cond,
            example_inputs=[(x,)],
            tracer="dynamo",
            backend="eager",
            dynamic=False,
        )
        self.assertIn("def _graph_forward", code)
        self.assertIn("def _eager_subgraph_0", code)
        self.assertIn("_EAGER_GRAPH_BODY", code)
        with mock.patch.object(
            torch.fx.Tracer,
            "trace",
            side_effect=AssertionError("load must not symbolically retrace"),
        ):
            loaded = torch.compiler.precompile.load(code, cache)
        self.assertEqual(loaded(x), _precompile_eager_cond(x))

        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as artifact:
            artifact.write(code)
            artifact_path = artifact.name
        try:
            subprocess.check_call(
                [
                    sys.executable,
                    "-c",
                    textwrap.dedent(
                        """
                        import runpy
                        import sys
                        import torch

                        forward = runpy.run_path(sys.argv[1])["forward"]
                        for x in (torch.ones(4), -torch.ones(4)):
                            expected = x.sin() if x.sum() > 0 else x.cos()
                            torch.testing.assert_close(forward(x), expected)
                        """
                    ),
                    artifact_path,
                ]
            )
        finally:
            os.unlink(artifact_path)

    @parametrize("graph_break", (False, True))
    def test_tracer_dynamo_eager_checkpoint_training(self, graph_break):
        entry = (
            _precompile_eager_graph_break
            if graph_break
            else _precompile_eager_checkpoint
        )
        args = ("checkpoint",) if graph_break else ()
        x = torch.randn(4, requires_grad=True)
        reference_input = x.detach().clone().requires_grad_()
        expected = entry(*args, reference_input)
        expected.sum().backward()
        code, cache = torch.compiler.precompile(
            entry,
            example_inputs=[(*args, x)],
            tracer="dynamo",
            backend="eager",
            dynamic=False,
            training=True,
        )
        actual_input = x.detach().clone().requires_grad_()
        loaded = torch.compiler.precompile.load(code, cache)
        try:
            actual = loaded(*args, actual_input)
            actual.sum().backward()
            self.assertEqual(actual, expected)
            self.assertEqual(actual_input.grad, reference_input.grad)
        finally:
            if hasattr(loaded, "unload"):
                loaded.unload()

    @torch._dynamo.config.patch(
        automatic_dynamic_shapes=True, assume_static_by_default=True
    )
    def test_tracer_dynamo_eager_higher_order_dynamic_shapes(self):
        code, cache = torch.compiler.precompile(
            _precompile_eager_cond,
            example_inputs=[(torch.ones(2),), (torch.ones(3),)],
            tracer="dynamo",
            backend="eager",
        )
        loaded = torch.compiler.precompile.load(code, cache)
        for x in (torch.ones(5), -torch.ones(7)):
            self.assertEqual(loaded(x), _precompile_eager_cond(x))

    @torch._dynamo.config.patch(
        automatic_dynamic_shapes=True, assume_static_by_default=True
    )
    def test_tracer_dynamo_dynamic_graph_keeps_tensor_contract(self):
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_dynamic,
            example_inputs=[(torch.randn(2, 4),), (torch.randn(3, 4),)],
            tracer="dynamo",
        )
        loaded = torch.compiler.precompile.load(code, cache)
        with self.assertRaisesRegex(PrecompileError, "dtype"):
            loaded(torch.randn(7, 4, dtype=torch.float64))

    def test_tracer_dynamo_keeps_invariant_input_attribute_guard(self):
        x = torch.randn(4)
        state = _PrecompileDynamoInputAttribute()
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_input_attribute,
            example_inputs=[(x, state)],
            tracer="dynamo",
            backend="eager",
        )
        loaded = torch.compiler.precompile.load(code, cache)
        self.assertEqual(loaded(x, state), _precompile_dynamo_input_attribute(x, state))
        del state.flag
        with self.assertRaisesRegex(PrecompileError, "no captured Dynamo variant"):
            loaded(x, state)

    def test_tracer_dynamo_does_not_treat_module_callable_as_environment(self):
        model = _PrecompileDynamoCallableAttribute()
        with self.assertRaisesRegex(PrecompileError, "can affect dispatch"):
            torch.compiler.precompile(
                _precompile_dynamo_call_module,
                example_inputs=[(model, torch.randn(4))],
                tracer="dynamo",
                backend="eager",
            )

    @torch._dynamo.config.patch(
        automatic_dynamic_shapes=True, assume_static_by_default=True
    )
    def test_tracer_dynamo_automatic_dynamic_is_per_frame(self):
        fixed = torch.randn(4)
        examples = [(fixed, torch.randn(size)) for size in (3, 5)]
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_late_varying,
            example_inputs=examples,
            tracer="dynamo",
            backend="eager",
        )

        counts = _dynamo_frame_variant_counts(code)
        entry = [
            count for name, count in counts if name == "_precompile_dynamo_late_varying"
        ]
        resumes = [
            count for name, count in counts if name.startswith("torch_dynamo_resume_in")
        ]
        self.assertEqual(entry, [1])
        self.assertEqual(resumes, [2])

        loaded = torch.compiler.precompile.load(code, cache)
        for size in (3, 5, 7):
            varying = torch.randn(size)
            self.assertEqual(
                loaded(fixed, varying),
                _precompile_dynamo_late_varying(fixed, varying),
            )

    def test_tracer_dynamo_executes_each_example_once(self):
        examples = [(torch.zeros(4),), (torch.zeros(8),)]
        torch.compiler.precompile(
            lambda x: x.add_(1),
            example_inputs=examples,
            tracer="dynamo",
            backend="eager",
        )
        for (example,) in examples:
            self.assertEqual(example, torch.ones_like(example))

    def test_tracer_dynamo_executes_each_backward_once(self):
        model = torch.nn.Linear(4, 3)
        calls: list[torch.Tensor] = []
        handle = model.weight.register_hook(
            lambda grad: calls.append(grad.detach().clone())
        )
        examples = [(model, torch.randn(size, 4)) for size in (2, 3)]
        try:
            torch.compiler.precompile(
                _precompile_dynamo_backward,
                example_inputs=examples,
                tracer="dynamo",
                backend="eager",
                training=True,
            )
        finally:
            handle.remove()
        self.assertEqual(len(calls), len(examples))
        self.assertIsNone(model.weight.grad)

    def test_tracer_dynamo_preserves_all_example_grads(self):
        model = _PrecompileDynamoGradState()
        x = torch.randn(4)
        model.step(x)
        before = [
            (tensor, tensor.grad, tensor.grad.detach().clone())
            for tensor in (model.weight, model.buffer)
        ]
        torch.compiler.precompile(
            _precompile_dynamo_grad_state,
            example_inputs=[(model, x)],
            tracer="dynamo",
            backend="eager",
            training=True,
            dynamic=False,
        )

        for tensor, grad, value in before:
            self.assertIs(tensor.grad, grad)
            self.assertEqual(tensor.grad, value)

    def test_tracer_dynamo_rejects_bound_method(self):
        model = _PrecompileDynamoGradState()
        with self.assertRaisesRegex(NotImplementedError, "bound methods"):
            torch.compiler.precompile(
                model.step,
                example_inputs=[(torch.randn(4),)],
                tracer="dynamo",
                backend="eager",
            )

    def test_tracer_dynamo_guards_mutating_module_at_capture_state(self):
        torch.manual_seed(0)
        model = _PrecompileDynamoStepCounter()
        examples = [(model, torch.randn(size, 8)) for size in (2, 3, 4)]
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_call_module,
            example_inputs=examples,
            tracer="dynamo",
            backend="eager",
        )
        self.assertEqual(model.step, len(examples))

        torch.manual_seed(0)
        runtime = _PrecompileDynamoStepCounter()
        torch.manual_seed(0)
        reference = _PrecompileDynamoStepCounter()
        loaded = torch.compiler.precompile.load(code, cache)
        for _, x in examples:
            self.assertEqual(loaded(runtime, x), reference(x))

    @parametrize("graph_break", (False, True))
    def test_tracer_dynamo_guard_through_tensor_attribute(self, graph_break):
        model = _PrecompileDynamoReadsTensorAttribute()
        x = torch.randn(8)
        x._cpu_copy = torch.randn(8)
        x.unused_generator = torch.Generator()
        fn = (
            _precompile_dynamo_tensor_attribute_break
            if graph_break
            else _precompile_dynamo_call_module
        )
        code, cache = torch.compiler.precompile(
            fn,
            example_inputs=[(model, x)],
            tracer="dynamo",
            backend="eager",
        )
        loaded = torch.compiler.precompile.load(code, cache)
        self.assertEqual(loaded(model, x), fn(model, x))
        x._cpu_copy = torch.randn(8)
        self.assertEqual(loaded(model, x), fn(model, x))

    def test_tracer_dynamo_self_referential_tensor_attribute(self):
        x = torch.randn(4)
        x.my_flag = x
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_reads_tensor_flag,
            example_inputs=[(x,)],
            tracer="dynamo",
            backend="eager",
        )
        loaded = torch.compiler.precompile.load(code, cache)
        self.assertEqual(loaded(x), _precompile_dynamo_reads_tensor_flag(x))

    def test_tracer_dynamo_rejects_unrebuildable_input_guard(self):
        from unittest import mock

        from torch._dynamo.guards import GuardsStatePickler

        carry = GuardsStatePickler._carried_tensor_attributes

        def omit_cpu_copy(self, tensor):
            state = carry(self, tensor)
            if state is not None:
                state.pop("_cpu_copy", None)
            return state or None

        model = _PrecompileDynamoReadsTensorAttribute()
        x = torch.randn(8)
        x._cpu_copy = torch.randn(8)
        with (
            mock.patch.object(
                GuardsStatePickler, "_carried_tensor_attributes", omit_cpu_copy
            ),
            self.assertRaisesRegex(PrecompileError, "input-derived guard"),
        ):
            torch.compiler.precompile(
                _precompile_dynamo_call_module,
                example_inputs=[(model, x)],
                tracer="dynamo",
                backend="eager",
            )

    def test_tracer_dynamo_rejects_drifted_input_guard(self):
        from unittest import mock

        from torch._dynamo.guards import GuardsStatePickler

        restore = GuardsStatePickler._restore_tensor_attributes.__func__

        def _restore_tensor_attributes(cls, tensor, state):
            state = dict(state)
            if "_cpu_copy" in state:
                state["_cpu_copy"] = torch.randn(2)
            restore(cls, tensor, state)

        model = _PrecompileDynamoReadsTensorAttribute()
        x = torch.randn(8)
        x._cpu_copy = torch.randn(8)
        with (
            mock.patch.object(
                GuardsStatePickler,
                "_restore_tensor_attributes",
                classmethod(_restore_tensor_attributes),
            ),
            self.assertRaisesRegex(PrecompileError, "input-derived guard.*changed"),
        ):
            torch.compiler.precompile(
                _precompile_dynamo_call_module,
                example_inputs=[(model, x)],
                tracer="dynamo",
                backend="eager",
            )

    @parametrize("change", ("add", "remove"))
    def test_tracer_dynamo_rejects_guard_leaf_drift(self, change):
        from unittest import mock

        from torch._dynamo.guards import GuardManagerWrapper

        fingerprint = GuardManagerWrapper.leaf_fingerprint
        calls = 0

        def drift_after_capture(self):
            nonlocal calls
            calls += 1
            result = fingerprint(self)
            if calls > 1:
                if change == "add":
                    return result | {("x", "TEST_GUARD", "changed after serialization")}
                return result - {sorted(result)[0]}
            return result

        with (
            mock.patch.object(
                GuardManagerWrapper, "leaf_fingerprint", drift_after_capture
            ),
            self.assertRaisesRegex(PrecompileError, "changed input-derived checks"),
        ):
            torch.compiler.precompile(
                _precompile_dynamo_dynamic,
                example_inputs=[(torch.randn(4),)],
                tracer="dynamo",
                backend="eager",
            )

    def test_tracer_dynamo_wrapper_subclass_requires_grad(self):
        from torch.testing._internal.two_tensor import TwoTensor

        x = TwoTensor(torch.randn(3), torch.randn(3)).requires_grad_(True)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_dynamic,
            example_inputs=[(x,)],
            tracer="dynamo",
            backend="eager",
            training=True,
        )
        loaded = torch.compiler.precompile.load(code, cache)
        self.assertEqual(loaded(x), _precompile_dynamo_dynamic(x))

    @parametrize(
        "marking",
        ("unbacked", "unbacked_bounds", "unbacked_shape_id", "static", "dynamic"),
    )
    def test_tracer_dynamo_marked_artifact_serves_capture_tensor(self, marking):
        model = torch.nn.Linear(4, 3).eval()
        x = torch.randn(8, 4)
        {
            "unbacked": lambda t: mark_unbacked(t, 0),
            "unbacked_bounds": lambda t: mark_unbacked(t, 0, min=4, max=16),
            "unbacked_shape_id": lambda t: mark_unbacked(t, 0, shape_id="batch"),
            "static": lambda t: torch._dynamo.decorators.mark_static(t, 0),
            "dynamic": lambda t: mark_dynamic(t, 0),
        }[marking](x)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_call_module,
            example_inputs=[(model, x)],
            tracer="dynamo",
            backend="eager",
            training=True,
        )
        loaded = torch.compiler.precompile.load(code, cache)
        self.assertEqual(loaded(model, x), model(x))

    def test_tracer_dynamo_captures_every_example_past_default_limit(self):
        x = torch.randn(4)
        examples = [(x, f"m{i}") for i in range(10)]
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_many_variants,
            example_inputs=examples,
            tracer="dynamo",
            backend="eager",
        )
        loaded = torch.compiler.precompile.load(code, cache)
        for example in examples:
            self.assertEqual(
                loaded(*example), _precompile_dynamo_many_variants(*example)
            )

    def test_tracer_dynamo_capture_preserves_existing_compile_entries(self):
        torch._dynamo.reset()
        counter = torch._dynamo.testing.CompileCounter()
        compiled = torch.compile(
            _precompile_dynamo_dynamic, backend=counter, dynamic=False
        )
        x = torch.randn(4)
        self.assertEqual(compiled(x), _precompile_dynamo_dynamic(x))
        self.assertEqual(counter.frame_count, 1)

        torch.compiler.precompile(
            _precompile_dynamo_scalar,
            example_inputs=[(x, 2)],
            tracer="dynamo",
            backend="eager",
        )
        self.assertEqual(compiled(x), _precompile_dynamo_dynamic(x))
        self.assertEqual(counter.frame_count, 1)

    def test_tracer_dynamo_capture_isolated_from_same_function_cache(self):
        torch._dynamo.reset()
        counter = torch._dynamo.testing.CompileCounter()
        compiled = torch.compile(
            _precompile_dynamo_dynamic, backend=counter, dynamic=False
        )
        x = torch.randn(4)
        self.assertEqual(compiled(x), _precompile_dynamo_dynamic(x))

        code, cache = torch.compiler.precompile(
            _precompile_dynamo_dynamic,
            example_inputs=[(x,)],
            tracer="dynamo",
            backend="eager",
            dynamic=False,
        )
        self.assertIn("GRAPH_COUNT = 1", code)
        self.assertEqual(
            torch.compiler.precompile.load(code, cache)(x),
            _precompile_dynamo_dynamic(x),
        )
        self.assertEqual(counter.frame_count, 1)

    @torch._dynamo.config.patch(
        automatic_dynamic_shapes=True, assume_static_by_default=True
    )
    def test_tracer_dynamo_capture_does_not_leak_auto_dynamic_state(self):
        dynamic_code, _ = torch.compiler.precompile(
            _precompile_dynamo_dynamic,
            example_inputs=[(torch.randn(2, 4),), (torch.randn(3, 4),)],
            tracer="dynamo",
            backend="eager",
        )
        self.assertIn("DYNAMIC_GRAPH_COUNT = 1", dynamic_code)

        static_code, static_cache = torch.compiler.precompile(
            _precompile_dynamo_dynamic,
            example_inputs=[(torch.randn(5, 4),)],
            tracer="dynamo",
            backend="eager",
        )
        self.assertIn("DYNAMIC_GRAPH_COUNT = 0", static_code)
        x = torch.randn(5, 4)
        self.assertEqual(
            torch.compiler.precompile.load(static_code, static_cache)(x),
            _precompile_dynamo_dynamic(x),
        )

    def test_tracer_dynamo_failed_capture_cleans_up_state(self):
        with self.assertRaisesRegex(PrecompileError, "recompile_limit=1"):
            torch.compiler.precompile(
                _precompile_dynamo_scalar_branch,
                example_inputs=[(torch.randn(4), 2), (torch.randn(4), 3)],
                tracer="dynamo",
                backend="eager",
                dynamic=False,
                recompile_limit=1,
            )

        x = torch.randn(4)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_dynamic,
            example_inputs=[(x,)],
            tracer="dynamo",
            backend="eager",
        )
        self.assertEqual(
            torch.compiler.precompile.load(code, cache)(x),
            _precompile_dynamo_dynamic(x),
        )

    def test_tracer_dynamo_rejects_inference_examples(self):
        with torch.inference_mode():
            inference_tensor = torch.randn(4)
            inference_module = torch.nn.Linear(4, 3)
        with self.assertRaisesRegex(PrecompileError, "inference tensors"):
            torch.compiler.precompile(
                _precompile_dynamo_dynamic,
                example_inputs=[(inference_tensor,)],
                tracer="dynamo",
                backend="eager",
            )
        with self.assertRaisesRegex(PrecompileError, "inference tensor"):
            torch.compiler.precompile(
                _precompile_dynamo_call_module,
                example_inputs=[(inference_module, torch.randn(2, 4))],
                tracer="dynamo",
                backend="eager",
            )
        with self.assertRaisesRegex(PrecompileError, "inference tensor"):
            torch.compiler.precompile(
                inference_module,
                example_inputs=[(torch.randn(2, 4),)],
                tracer="dynamo",
                backend="eager",
            )

    @parametrize("backend", ("eager", "inductor"))
    def test_tracer_dynamo_preserves_inference_mode(self, backend):
        x = torch.randn(4)
        with torch.inference_mode():
            code, cache = torch.compiler.precompile(
                _precompile_dynamo_dynamic,
                example_inputs=[(x,)],
                tracer="dynamo",
                backend=backend,
            )
            expected = _precompile_dynamo_dynamic(x)
            actual = torch.compiler.precompile.load(code, cache)(x)
        self.assertEqual(actual, expected)
        self.assertTrue(torch.is_inference(actual))

    def test_tracer_dynamo_rejects_training_inference_mode(self):
        x = torch.randn(4, requires_grad=True)
        with (
            torch.inference_mode(),
            self.assertRaisesRegex(PrecompileError, "training=True.*inference_mode"),
        ):
            torch.compiler.precompile(
                _precompile_dynamo_dynamic,
                example_inputs=[(x,)],
                tracer="dynamo",
                backend="eager",
                training=True,
            )

    def test_tracer_dynamo_recompile_limit_is_explicit(self):
        x = torch.randn(4)
        examples = [(x, f"m{i}") for i in range(3)]
        with self.assertRaisesRegex(PrecompileError, "recompile_limit=2"):
            torch.compiler.precompile(
                _precompile_dynamo_many_variants,
                example_inputs=examples,
                tracer="dynamo",
                backend="eager",
                dynamic=False,
                recompile_limit=2,
            )

        code, cache = torch.compiler.precompile(
            _precompile_dynamo_many_variants,
            example_inputs=examples,
            tracer="dynamo",
            backend="eager",
            dynamic=False,
            recompile_limit=2,
            require_complete=False,
        )
        self.assertIn("CAPTURE_COMPLETE = False", code)
        loaded = torch.compiler.precompile.load(code, cache)
        for example in examples[:2]:
            self.assertEqual(
                loaded(*example), _precompile_dynamo_many_variants(*example)
            )
        with self.assertRaisesRegex(PrecompileError, "no captured Dynamo variant"):
            loaded(*examples[2])

    def test_tracer_dynamo_recompile_limit_overrides_accumulated_limit(self):
        x = torch.randn(4)
        examples = [(x, f"m{index}") for index in range(3)]
        with torch._dynamo.config.patch(accumulated_recompile_limit=1):
            code, cache = torch.compiler.precompile(
                _precompile_dynamo_many_variants,
                example_inputs=examples,
                tracer="dynamo",
                backend="eager",
                dynamic=False,
                recompile_limit=4,
            )
        loaded = torch.compiler.precompile.load(code, cache)
        for example in examples:
            self.assertEqual(
                loaded(*example), _precompile_dynamo_many_variants(*example)
            )

    def test_tracer_dynamo_summary_and_invariants(self):
        examples = [(torch.randn(4), scale) for scale in (2, 3)]
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "invariants.txt")
            code, cache = torch.compiler.precompile(
                _precompile_dynamo_scalar,
                example_inputs=examples,
                tracer="dynamo",
                backend="eager",
                invariants=path,
            )
            with open(path, encoding="utf-8") as report:
                text = report.read()
        self.assertIn("invariant:", text)
        self.assertIn("varying:", text)
        self.assertIn("variant_examples = (0, 1)", text)
        self.assertIn("CAPTURE_COMPLETE = True", code)
        self.assertIn("POLICY_DROPPED_GUARDS", code)
        loaded = torch.compiler.precompile.load(code, cache)
        self.assertIsInstance(loaded.capture_summary, torch.compiler.PrecompileSummary)
        self.assertTrue(loaded.capture_summary.complete)
        self.assertEqual(loaded.capture_summary.variant_examples[0], (0, 1))

    def test_tracer_dynamo_custom_guard_filter_is_fail_closed(self):
        x = torch.randn(4)

        def drop_all(entries):
            return [False] * len(entries)

        with self.assertRaisesRegex(PrecompileError, "dropped guards.*dispatch"):
            torch.compiler.precompile(
                _precompile_dynamo_scalar_branch,
                example_inputs=[(x, 2)],
                tracer="dynamo",
                backend="eager",
                guard_filter_fn=drop_all,
            )

    def test_tracer_dynamo_custom_filter_cannot_restore_unserializable_guards(self):
        x = torch.randn(4)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_graph_break,
            example_inputs=[(x,)],
            tracer="dynamo",
            backend="eager",
            guard_filter_fn=lambda guards: [True] * len(guards),
        )
        self.assertEqual(
            torch.compiler.precompile.load(code, cache)(x),
            _precompile_dynamo_graph_break(x),
        )

    @parametrize("backend", ("eager", "inductor"))
    def test_tracer_dynamo_drops_unrebuildable_environment_guard(self, backend):
        module_name = "_precompile_weakref_environment"
        source = textwrap.dedent(
            """
            import weakref

            import torch

            class Value:
                pass

            value = Value()
            values = weakref.WeakValueDictionary({"value": value})

            def helper(x):
                ref = values.data["value"]
                torch._dynamo.graph_break()
                return x + (1 if ref.__callback__ is not None else 2)

            def fn(x):
                return helper(x)
            """
        )
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, f"{module_name}.py")
            with open(path, "w", encoding="utf-8") as file:
                file.write(source)
            sys.path.insert(0, directory)
            importlib.invalidate_caches()
            try:
                fixture = importlib.import_module(module_name)
                x = torch.randn(4)
                code, cache = torch.compiler.precompile(
                    fixture.fn,
                    example_inputs=[(x,)],
                    tracer="dynamo",
                    backend=backend,
                )
                loaded = torch.compiler.precompile.load(code, cache)
                self.assertEqual(loaded(x), fixture.fn(x))
            finally:
                sys.path.remove(directory)
                sys.modules.pop(module_name, None)
        callback_drops = {
            guard
            for guard in loaded.capture_summary.dropped_guards
            if guard[1].endswith(".__callback__")
        }
        weakref_drops = {
            guard
            for guard in loaded.capture_summary.dropped_guards
            if guard[0] == "TYPE_MATCH"
            and (guard[1] == "ref" or guard[1].endswith(".data['value']"))
        }
        self.assertTrue(callback_drops)
        self.assertTrue(weakref_drops)
        self.assertTrue(
            (callback_drops | weakref_drops).issubset(
                loaded.capture_summary.policy_dropped_guards
            )
        )
        self.assertTrue(
            (callback_drops | weakref_drops).isdisjoint(
                loaded.capture_summary.risky_dropped_guards
            )
        )

    def test_tracer_dynamo_does_not_assume_weakref_input_is_environment(self):
        value = _PrecompileWeakValue()
        values = weakref.WeakValueDictionary({"value": value})
        x = torch.randn(4)
        with self.assertRaisesRegex(PrecompileError, "can affect dispatch"):
            torch.compiler.precompile(
                _precompile_dynamo_weakref_input,
                example_inputs=[(x, values)],
                tracer="dynamo",
                backend="eager",
            )

    def test_tracer_dynamo_drops_generator_environment_guard(self):
        x = torch.randn(4)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_generator_environment,
            example_inputs=[(x,)],
            tracer="dynamo",
            backend="eager",
        )
        loaded = torch.compiler.precompile.load(code, cache)
        self.assertEqual(loaded(x), _precompile_dynamo_generator_environment(x))
        self.assertEqual(loaded.capture_summary.risky_dropped_guards, ())
        self.assertTrue(
            set(loaded.capture_summary.policy_dropped_guards).issubset(
                loaded.capture_summary.dropped_guards
            )
        )
        self.assertTrue(
            any(
                "_PRECOMPILE_DYNAMO_GENERATOR" in source
                for _, source in loaded.capture_summary.policy_dropped_guards
            )
        )

    def test_tracer_dynamo_does_not_drop_generator_input_guard(self):
        with self.assertRaisesRegex(PrecompileError, "can affect dispatch"):
            torch.compiler.precompile(
                _precompile_dynamo_generator_input,
                example_inputs=[(torch.randn(4), torch.Generator())],
                tracer="dynamo",
                backend="eager",
            )

    @parametrize(
        "bad",
        (
            lambda guards: [True] * (len(guards) + 1),
            lambda guards: [1] * len(guards),
        ),
    )
    def test_tracer_dynamo_validates_custom_guard_filter(self, bad):
        with self.assertRaisesRegex(PrecompileError, "guard_filter_fn"):
            torch.compiler.precompile(
                _precompile_dynamo_dynamic,
                example_inputs=[(torch.randn(4),)],
                tracer="dynamo",
                backend="eager",
                guard_filter_fn=bad,
            )

    def test_tracer_dynamo_require_no_dropped_guards(self):
        x = torch.randn(4)
        with self.assertRaisesRegex(PrecompileError, "dropped environment-contract"):
            torch.compiler.precompile(
                _precompile_dynamo_graph_break,
                example_inputs=[(x,)],
                tracer="dynamo",
                backend="eager",
                require_no_dropped_guards=True,
            )

    @parametrize("backend", ("eager", "inductor"))
    def test_tracer_dynamo_mark_unbacked_runs_across_sizes(self, backend):
        x = torch.randn(8, 4)
        mark_unbacked(x, 0)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_dynamic,
            example_inputs=[(x,)],
            tracer="dynamo",
            backend=backend,
        )
        loaded = torch.compiler.precompile.load(code, cache)
        for size in (0, 3, 16):
            runtime = torch.randn(size, 4)
            self.assertEqual(loaded(runtime), _precompile_dynamo_dynamic(runtime))

    def test_tracer_dynamo_mark_unbacked_bounds_enforced(self):
        x = torch.randn(8, 4)
        mark_unbacked(x, 0, min=4, max=16)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_dynamic,
            example_inputs=[(x,)],
            tracer="dynamo",
            backend="eager",
        )
        loaded = torch.compiler.precompile.load(code, cache)
        self.assertEqual(loaded(torch.randn(6, 4)).shape, (6, 4))
        with self.assertRaisesRegex(PrecompileError, "no captured Dynamo variant"):
            loaded(torch.randn(2, 4))

    def test_tracer_dynamo_mark_unbacked_shape_id_enforced(self):
        x = torch.randn(8, 4)
        y = torch.randn(8, 4)
        mark_unbacked(x, 0, shape_id="batch")
        mark_unbacked(y, 0, shape_id="batch")
        code, cache = torch.compiler.precompile(
            lambda a, b: a + b,
            example_inputs=[(x, y)],
            tracer="dynamo",
            backend="eager",
        )
        loaded = torch.compiler.precompile.load(code, cache)
        self.assertEqual(loaded(torch.randn(3, 4), torch.randn(3, 4)).shape, (3, 4))
        with self.assertRaises((PrecompileError, RuntimeError, AssertionError)):
            loaded(torch.randn(3, 4), torch.randn(5, 4))

    def test_tracer_dynamo_mark_unbacked_hint_override_honored(self):
        x = torch.randn(8, 4)
        mark_unbacked(x, 0, hint_override=16)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_dynamic,
            example_inputs=[(x,)],
            tracer="dynamo",
            backend="eager",
        )
        loaded = torch.compiler.precompile.load(code, cache)
        for size in (8, 32):
            runtime = torch.randn(size, 4)
            self.assertEqual(loaded(runtime), _precompile_dynamo_dynamic(runtime))

    @torch._dynamo.config.patch(
        automatic_dynamic_shapes=True, assume_static_by_default=True
    )
    def test_tracer_dynamo_training_recompiles_to_dynamic_graph(self):
        examples = [
            (torch.randn(rows, cols, requires_grad=True),)
            for rows, cols in ((2, 3), (3, 5), (5, 7))
        ]
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_dynamic,
            example_inputs=examples,
            tracer="dynamo",
            training=True,
        )

        self.assertIn("TRAINING = True", code)
        self.assertIn("DYNAMIC_GRAPH_COUNT = 1", code)
        self.assertIn("class _CompiledFunction_", code)
        self.assertIn("_inner_call_fw", code)
        self.assertIn("_inner_call_bw", code)
        for _, loaded in _default_and_inlined_loaders(code, cache, "inductor"):
            x = torch.randn(7, 9, requires_grad=True)
            ref = x.detach().clone().requires_grad_()
            expected = _precompile_dynamo_dynamic(ref)
            expected.sum().backward()
            actual = loaded(x)
            self.assertTrue(actual.requires_grad)
            actual.sum().backward()
            self.assertEqual(actual, expected)
            self.assertEqual(x.grad, ref.grad)

    def test_tracer_dynamo_training_passthrough_backward(self):
        x = torch.randn(4, requires_grad=True)
        code, cache = torch.compiler.precompile(
            _precompile_add_one,
            example_inputs=[(x,)],
            tracer="dynamo",
            training=True,
        )
        for _, loaded in _default_and_inlined_loaders(code, cache, "inductor"):
            actual_x = x.detach().clone().requires_grad_()
            expected_x = x.detach().clone().requires_grad_()
            actual = loaded(actual_x)
            expected = _precompile_add_one(expected_x)
            actual.sum().backward()
            expected.sum().backward()
            self.assertEqual(actual, expected)
            self.assertEqual(actual_x.grad, expected_x.grad)

    def test_tracer_dynamo_varargs_dispatch(self):
        x = torch.randn(4)
        y = torch.randn(4)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_varargs,
            example_inputs=[(x, y)],
            tracer="dynamo",
            backend="eager",
        )
        loaded = torch.compiler.precompile.load(code, cache)
        self.assertEqual(loaded(x, y), _precompile_dynamo_varargs(x, y))

    def test_tracer_dynamo_varkw_dispatch(self):
        x = torch.randn(4)
        kwarg_x = torch.randn(4)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_varkw,
            example_inputs=[
                torch.compiler.ExampleInput(args=(x,), kwargs={"x": kwarg_x})
            ],
            tracer="dynamo",
            backend="eager",
        )
        loaded = torch.compiler.precompile.load(code, cache)
        self.assertEqual(loaded(x, x=kwarg_x), _precompile_dynamo_varkw(x, x=kwarg_x))

    @parametrize("backend", ("eager", "inductor"))
    def test_tracer_dynamo_training_later_backward(self, backend):
        model = torch.nn.Linear(4, 3)
        x = torch.randn(5, 4)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_call_module,
            example_inputs=[(model, x)],
            tracer="dynamo",
            backend=backend,
            training=True,
        )
        for _, loaded in _default_and_inlined_loaders(code, cache, backend):
            run = copy.deepcopy(model)
            ref = copy.deepcopy(model)
            actual = loaded(run, x)
            expected = ref(x)
            actual.sum().backward()
            expected.sum().backward()
            self.assertEqual(actual, expected)
            for actual_param, expected_param in zip(run.parameters(), ref.parameters()):
                self.assertEqual(actual_param.grad, expected_param.grad)

    @parametrize("backend", ("eager", "inductor"))
    def test_tracer_dynamo_inference_stays_grad_free(self, backend):
        model = torch.nn.Linear(4, 3)
        x = torch.randn(5, 4, requires_grad=True)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_call_module,
            example_inputs=[(model, x)],
            tracer="dynamo",
            backend=backend,
        )
        for _, loaded in _default_and_inlined_loaders(code, cache, backend):
            self.assertFalse(loaded(model, x).requires_grad)

    @parametrize("backend", ("eager", "inductor"))
    def test_tracer_dynamo_autograd_grad(self, backend):
        model = torch.nn.Linear(4, 3)
        x = torch.randn(5, 4)
        target = torch.randn(5, 3)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_autograd_grad,
            example_inputs=[(model, x, target)],
            tracer="dynamo",
            backend=backend,
            training=True,
        )
        run = copy.deepcopy(model)
        ref = copy.deepcopy(model)
        actual = torch.compiler.precompile.load(code, cache)(run, x, target)
        expected = _precompile_dynamo_autograd_grad(ref, x, target)
        self.assertEqual(actual, expected)
        self.assertTrue(all(parameter.grad is None for parameter in run.parameters()))

    def test_tracer_dynamo_autograd_grad_does_not_observe_seed(self):
        def grad_step(model, x):
            saw_grad = model.weight.grad is not None
            return saw_grad, torch.autograd.grad(model(x).sum(), (model.weight,))

        model = torch.nn.Linear(4, 3)
        x = torch.randn(5, 4)
        code, cache = torch.compiler.precompile(
            grad_step,
            example_inputs=[(model, x)],
            tracer="dynamo",
            backend="eager",
            training=True,
        )
        run = copy.deepcopy(model)
        actual = torch.compiler.precompile.load(code, cache)(run, x)
        expected = grad_step(copy.deepcopy(model), x)
        self.assertEqual(actual, expected)

    @parametrize("backend", ("eager", "inductor"))
    def test_tracer_dynamo_in_function_backward(self, backend):
        model = torch.nn.Linear(4, 3)
        x = torch.randn(5, 4)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_backward,
            example_inputs=[(model, x)],
            tracer="dynamo",
            backend=backend,
            training=True,
        )
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        run = copy.deepcopy(model)
        ref = copy.deepcopy(model)
        loaded = torch.compiler.precompile.load(code, cache)
        loaded(run, x)
        loaded(run, x)
        _precompile_dynamo_backward(ref, x)
        _precompile_dynamo_backward(ref, x)
        for actual_param, expected_param in zip(run.parameters(), ref.parameters()):
            self.assertEqual(actual_param.grad, expected_param.grad)

    def test_tracer_dynamo_training_source_runs_in_fresh_process(self):
        x = torch.randn(4, requires_grad=True)
        code, _cache = torch.compiler.precompile(
            _precompile_dynamo_dynamic,
            example_inputs=[(x,)],
            tracer="dynamo",
            training=True,
        )
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as artifact:
            artifact.write(code)
            artifact_path = artifact.name
        try:
            subprocess.check_call(
                [
                    sys.executable,
                    "-c",
                    textwrap.dedent(
                        """
                        import runpy as r
                        import sys as s
                        import torch as t

                        namespace = r.run_path(s.argv[1])
                        x = t.randn(4, requires_grad=True)
                        out = namespace["forward"](x)
                        assert out.requires_grad
                        out.sum().backward()
                        t.testing.assert_close(x.grad, x.detach().cos())
                        """
                    ),
                    artifact_path,
                ]
            )
        finally:
            os.unlink(artifact_path)

    def test_training_requires_dynamo(self):
        x = torch.randn(4, requires_grad=True)
        with self.assertRaisesRegex(NotImplementedError, "tracer='dynamo'"):
            torch.compiler.precompile(
                _precompile_dynamo_dynamic,
                example_inputs=[(x,)],
                training=True,
            )

    def test_tracer_dynamo_captures_explicit_graph_break(self):
        x = torch.randn(4)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_graph_break,
            example_inputs=[(x,)],
            tracer="dynamo",
            backend="eager",
        )

        self.assertIn("FRAME_COUNT = 3", code)
        loaded = torch.compiler.precompile.load(code, cache)
        self.assertEqual(loaded(x), _precompile_dynamo_graph_break(x))
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as artifact:
            artifact.write(code)
            artifact_path = artifact.name
        try:
            subprocess.check_call(
                [
                    sys.executable,
                    "-c",
                    textwrap.dedent(
                        """
                        import runpy as r
                        import sys as s
                        import torch as t

                        namespace = r.run_path(s.argv[1])
                        x = t.randn(4)
                        expected = ((x + 1) * 2).sin()
                        t.testing.assert_close(namespace["forward"](x), expected)
                        """
                    ),
                    artifact_path,
                ]
            )
        finally:
            os.unlink(artifact_path)

    @parametrize(
        "module_type",
        (
            _PrecompileDynamoDisabledMethodModule,
            _PrecompileDynamoDataDependentModule,
            _PrecompileDynamoBreakInLoopModule,
        ),
    )
    @parametrize("backend", ("eager", "inductor"))
    def test_tracer_dynamo_graph_break_kinds(self, module_type, backend):
        model = module_type().eval()
        x = torch.randn(4, 4)
        expected = model(x)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_call_module,
            example_inputs=[(model, x)],
            tracer="dynamo",
            backend=backend,
            require_complete=False,
            require_no_risky_drops=False,
        )
        loaded = torch.compiler.precompile.load(code, cache)
        try:
            self.assertEqual(loaded(model, x), expected)
        finally:
            if hasattr(loaded, "unload"):
                loaded.unload()

    @parametrize("backend", ("eager", "inductor"))
    def test_tracer_dynamo_graph_break_recompilations(self, backend):
        calls = [(torch.randn(size), flag) for size in (3, 5) for flag in (False, True)]
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_branching_graph_break,
            example_inputs=calls,
            tracer="dynamo",
            backend=backend,
            dynamic=False,
        )
        counts = _dynamo_frame_variant_counts(code)
        self.assertTrue(any(count > 1 for _, count in counts))
        self.assertTrue(
            any(name.startswith("torch_dynamo_resume_in") for name, _ in counts)
        )
        for _, loaded in _default_and_inlined_loaders(code, cache, backend):
            for call in calls:
                self.assertEqual(
                    loaded(*call), _precompile_dynamo_branching_graph_break(*call)
                )

    @torch._dynamo.config.patch(
        automatic_dynamic_shapes=True, assume_static_by_default=True
    )
    def test_tracer_dynamo_auto_dynamic_across_graph_breaks(self):
        examples = [(torch.randn(size),) for size in (3, 5)]
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_graph_break,
            example_inputs=examples,
            tracer="dynamo",
            backend="eager",
        )
        self.assertIn("FRAME_COUNT = 3", code)
        self.assertIn("VARIANT_COUNT = 6", code)
        loaded = torch.compiler.precompile.load(code, cache)
        for size in (3, 5, 7):
            x = torch.randn(size)
            self.assertEqual(loaded(x), _precompile_dynamo_graph_break(x))

    def test_tracer_dynamo_keyword_examples(self):
        def fn(x, *, scale=1.0):
            return x * scale

        x = torch.randn(4)
        self.assertIs(
            torch.compiler.precompile.ExampleInput, torch.compiler.ExampleInput
        )
        example = torch.compiler.precompile.ExampleInput
        code, cache = torch.compiler.precompile(
            fn,
            example_inputs=[
                (x,),
                example(args=(x,), kwargs={"scale": 3.0}),
            ],
            tracer="dynamo",
            backend="eager",
        )
        loaded = torch.compiler.precompile.load(code, cache)
        self.assertEqual(loaded(x), fn(x))
        self.assertEqual(loaded(x, scale=3.0), fn(x, scale=3.0))

        for args, kwargs in (((), {}), ((x,), {"unexpected": 1.0})):
            with self.assertRaisesRegex(
                PrecompileError, "different structure|captured signature"
            ):
                loaded(*args, **kwargs)

    @parametrize("backend", ("eager", "inductor"))
    def test_tracer_dynamo_module_argument(self, backend):
        def fn(module, x):
            return module(x).relu()

        model = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        code, cache = torch.compiler.precompile(
            fn,
            example_inputs=[(model, x)],
            tracer="dynamo",
            backend=backend,
        )
        replacement = torch.nn.Linear(4, 3).eval()
        for _, loaded in _default_and_inlined_loaders(code, cache, backend):
            self.assertEqual(loaded(replacement, x), fn(replacement, x))
            with self.assertRaisesRegex(PrecompileError, "runtime module input"):
                loaded(torch.nn.Linear(4, 7).eval(), x)

    def test_tracer_dynamo_installs_unreachable_frames(self):
        from torch._dynamo.utils import counters
        from torch._precompile import _parse_artifact_metadata

        x = torch.randn(4)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_unreachable_caller,
            example_inputs=[(x,)],
            tracer="dynamo",
            backend="eager",
        )
        self.assertEqual(_parse_artifact_metadata(code)["SERVING_MODE"], "installed")

        torch._dynamo.reset()
        loaded = torch.compiler.precompile.load(
            code, cache, fn=_precompile_dynamo_unreachable_caller
        )
        counters.clear()
        with loaded:
            self.assertEqual(loaded(x), _precompile_dynamo_unreachable_caller(x))
            self.assertEqual(counters["stats"]["unique_graphs"], 0)

    def test_tracer_dynamo_installed_artifacts_are_isolated(self):
        x = torch.randn(4)
        first_code, first_cache = torch.compiler.precompile(
            _precompile_dynamo_unreachable_caller,
            example_inputs=[(x,)],
            tracer="dynamo",
            backend="eager",
        )
        first = torch.compiler.precompile.load(first_code, first_cache)
        second = torch.compiler.precompile.load(first_code, first_cache)
        self.assertEqual(first(x), _precompile_dynamo_unreachable_caller(x))
        self.assertEqual(second(x), _precompile_dynamo_unreachable_caller(x))
        first.unload()
        self.assertEqual(second(x), _precompile_dynamo_unreachable_caller(x))
        second.unload()

    def test_tracer_dynamo_installed_unload_is_terminal(self):
        x = torch.randn(4)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_unreachable_caller,
            example_inputs=[(x,)],
            tracer="dynamo",
            backend="eager",
        )
        loaded = torch.compiler.precompile.load(code, cache)
        self.assertEqual(loaded(x), _precompile_dynamo_unreachable_caller(x))
        loaded.unload()
        with self.assertRaisesRegex(RuntimeError, "has been unloaded"):
            loaded(x)

    def test_tracer_dynamo_installed_reset_requires_reload(self):
        x = torch.randn(4)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_unreachable_caller,
            example_inputs=[(x,)],
            tracer="dynamo",
            backend="eager",
        )
        loaded = torch.compiler.precompile.load(code, cache)
        self.assertEqual(loaded(x), _precompile_dynamo_unreachable_caller(x))
        torch._dynamo.reset()
        with self.assertRaisesRegex(PrecompileError, "cleared this loaded artifact"):
            loaded(x)
        loaded.unload()

    def test_tracer_dynamo_installed_unload_interrupt_is_retryable(self):
        from unittest import mock

        x = torch.randn(4)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_unreachable_caller,
            example_inputs=[(x,)],
            tracer="dynamo",
            backend="eager",
        )
        loaded = torch.compiler.precompile.load(code, cache)
        loaded.__enter__()
        artifact = loaded._loaded_forward
        with artifact.state:
            artifact.active_calls = 1
        with mock.patch.object(artifact.state, "wait", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                loaded.unload()
        self.assertTrue(artifact.loaded)
        self.assertFalse(artifact.unloading)
        with artifact.state:
            artifact.active_calls = 0
            artifact.state.notify_all()
        loaded.unload()
        self.assertFalse(artifact.loaded)

    def test_tracer_dynamo_installed_artifact_rejects_unseen_variant(self):
        x = torch.randn(4)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_unreachable_branch_caller,
            example_inputs=[(x, 0), (x, 1)],
            tracer="dynamo",
            backend="eager",
        )
        loaded = torch.compiler.precompile.load(code, cache)
        self.assertEqual(
            loaded(x, 0), _precompile_dynamo_unreachable_branch_caller(x, 0)
        )
        self.assertEqual(
            loaded(x, 1), _precompile_dynamo_unreachable_branch_caller(x, 1)
        )
        with self.assertRaisesRegex(PrecompileError, "no captured Dynamo variant"):
            loaded(x, 2)
        loaded.unload()

    def test_tracer_dynamo_installed_artifact_reuses_prepared_state(self):
        from unittest import mock

        import torch._dynamo.package as package_module

        built = []
        load_guard_manager = package_module.load_guard_manager

        def count(*args, **kwargs):
            built.append(1)
            return load_guard_manager(*args, **kwargs)

        x = torch.randn(4)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_unreachable_branch_caller,
            example_inputs=[(x, 0), (x, 1)],
            tracer="dynamo",
            backend="eager",
            dynamic=False,
        )
        with mock.patch.object(package_module, "load_guard_manager", count):
            loaded = torch.compiler.precompile.load(code, cache)
            at_load = len(built)
            try:
                loaded(x, 0)
            finally:
                loaded.unload()
        self.assertGreater(at_load, 0)
        self.assertEqual(len(built), at_load)

    def test_tracer_dynamo_installed_artifact_never_compiles_at_serve_time(self):
        x = torch.randn(4)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_unreachable_branch_caller,
            example_inputs=[(x, 0), (x, 1)],
            tracer="dynamo",
            backend="eager",
            dynamic=False,
        )
        loaded = torch.compiler.precompile.load(code, cache)
        try:
            self.assertEqual(loaded.serve_time_compiles(), 0)
            loaded(x, 0)
            self.assertEqual(loaded.serve_time_compiles(), 0)
            with torch._dynamo.config.patch(suppress_errors=True):
                with self.assertRaisesRegex(
                    PrecompileError, "no captured Dynamo variant"
                ):
                    loaded(x, 2)
            self.assertEqual(loaded.serve_time_compiles(), 0)
        finally:
            loaded.unload()

    def test_tracer_dynamo_installed_artifact_rejects_wrong_callable(self):
        model = _PrecompileDynamoBreakingModule().eval()
        x = torch.randn(5, 4)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_call_module,
            example_inputs=[(model, x)],
            tracer="dynamo",
            backend="eager",
        )
        with self.assertRaisesRegex(
            PrecompileError, "different callable|captured from"
        ):
            torch.compiler.precompile.load(
                code, cache, fn=_precompile_dynamo_wrong_call_module
            )

    @parametrize("backend", ("eager", "inductor"))
    def test_tracer_dynamo_module_graph_break_uses_installed_artifact(self, backend):
        from torch._precompile import _parse_artifact_metadata

        model = _PrecompileDynamoBreakingModule().eval()
        x = torch.randn(5, 4)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_call_module,
            example_inputs=[(model, x)],
            tracer="dynamo",
            backend=backend,
        )
        self.assertEqual(_parse_artifact_metadata(code)["SERVING_MODE"], "installed")
        replacement = _PrecompileDynamoBreakingModule().eval()
        for _, loaded in _default_and_inlined_loaders(code, cache, backend):
            with loaded:
                self.assertEqual(loaded(replacement, x), replacement(x))

    @parametrize("backend", ("eager", "inductor"))
    def test_tracer_dynamo_module_callable(self, backend):
        model = _PrecompileDynamoFoldsGlobal().eval()
        x = torch.randn(5, 4)
        code, cache = torch.compiler.precompile(
            model,
            example_inputs=[(x,)],
            tracer="dynamo",
            backend=backend,
        )
        loaded = torch.compiler.precompile.load(code, cache)
        replacement = _PrecompileDynamoFoldsGlobal().eval()
        self.assertEqual(loaded(model, x), model(x))
        self.assertEqual(loaded(replacement, x), replacement(x))

    @parametrize("backend", ("eager", "inductor"))
    @parametrize("kind", ("plain", "custom_op"))
    def test_tracer_dynamo_module_callable_with_grad_input(self, backend, kind):
        from torch.library import _scoped_library

        with _scoped_library("precompile_parity", "FRAGMENT") as lib:
            lib.define("fused_matmul(Tensor x, Tensor w) -> Tensor")
            lib.impl("fused_matmul", torch.mm, "CompositeExplicitAutograd")
            lib.impl("fused_matmul", torch.mm, "Meta")
            module_type = (
                _PrecompileDynamoPlainMatmul
                if kind == "plain"
                else _PrecompileDynamoCustomOpMatmul
            )
            model = module_type()
            x = torch.randn(8, 8, requires_grad=True)
            code, cache = torch.compiler.precompile(
                model,
                example_inputs=[(x,)],
                tracer="dynamo",
                backend=backend,
            )
            loaded = torch.compiler.precompile.load(code, cache)
            with torch.no_grad():
                self.assertEqual(loaded(model, x), model(x))

    def test_tracer_dynamo_bare_builtin_module_is_rejected(self):
        model = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        with self.assertRaisesRegex(PrecompileError, "function that calls the module"):
            torch.compiler.precompile(
                model,
                example_inputs=[(x,)],
                tracer="dynamo",
                backend="eager",
            )

    @parametrize("backend", ("eager", "inductor"))
    def test_tracer_dynamo_defaults_roundtrip(self, backend):
        def fn(model, x, scale=2.0, *, bias=1.0):
            return model(x) * scale + bias

        model = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        code, cache = torch.compiler.precompile(
            fn,
            example_inputs=[(model, x)],
            tracer="dynamo",
            backend=backend,
        )
        for _, loaded in _default_and_inlined_loaders(code, cache, backend):
            self.assertEqual(loaded(model, x), fn(model, x))

    @parametrize("backend", ("eager", "inductor"))
    def test_tracer_dynamo_multiple_module_args(self, backend):
        first = torch.nn.Linear(4, 3).eval()
        second = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)

        def fn(a, b, value):
            return a(value) + b(value)

        code, cache = torch.compiler.precompile(
            fn,
            example_inputs=[(first, second, x)],
            tracer="dynamo",
            backend=backend,
        )
        for _, loaded in _default_and_inlined_loaders(code, cache, backend):
            self.assertEqual(loaded(first, second, x), fn(first, second, x))
            replacement_a = torch.nn.Linear(4, 3).eval()
            replacement_b = torch.nn.Linear(4, 3).eval()
            self.assertEqual(
                loaded(replacement_a, replacement_b, x),
                fn(replacement_a, replacement_b, x),
            )

    @parametrize("backend", ("eager", "inductor"))
    def test_tracer_dynamo_tied_weights_roundtrip(self, backend):
        model = _PrecompileDynamoTiedWeights().eval()
        x = torch.randn(5, 4)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_call_module,
            example_inputs=[(model, x)],
            tracer="dynamo",
            backend=backend,
        )
        for _, loaded in _default_and_inlined_loaders(code, cache, backend):
            self.assertEqual(loaded(model, x), model(x))

    @parametrize("backend", ("eager", "inductor"))
    def test_tracer_dynamo_nested_and_nontensor_outputs(self, backend):
        def fn(x):
            y = x.sin()
            return y, {"twice": y * 2}, 3.14, "artifact"

        x = torch.randn(5, 4)
        code, cache = torch.compiler.precompile(
            fn,
            example_inputs=[(x,)],
            tracer="dynamo",
            backend=backend,
        )
        for _, loaded in _default_and_inlined_loaders(code, cache, backend):
            self.assertEqual(loaded(x), fn(x))

    @parametrize("backend", ("eager", "inductor"))
    def test_tracer_dynamo_input_mutation_and_output_alias(self, backend):
        def mutate(x):
            return x.add_(1)

        example = torch.zeros(4)
        code, cache = torch.compiler.precompile(
            mutate,
            example_inputs=[(example,)],
            tracer="dynamo",
            backend=backend,
        )
        for _, loaded in _default_and_inlined_loaders(code, cache, backend):
            runtime = torch.zeros(4)
            self.assertEqual(loaded(runtime), torch.ones(4))
            self.assertEqual(runtime, torch.ones(4))

        x = torch.randn(2, 3)
        code, cache = torch.compiler.precompile(
            lambda value: value.t(),
            example_inputs=[(x,)],
            tracer="dynamo",
            backend=backend,
        )
        for _, loaded in _default_and_inlined_loaders(code, cache, backend):
            runtime = x.clone()
            output = loaded(runtime)
            output.add_(1)
            self.assertEqual(runtime, x + 1)

    @parametrize("backend", ("eager", "inductor"))
    def test_tracer_dynamo_module_buffer_mutation(self, backend):
        def fresh():
            return torch.nn.BatchNorm1d(4).train()

        model = fresh()
        x = torch.randn(8, 4)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_call_module,
            example_inputs=[(model, x)],
            tracer="dynamo",
            backend=backend,
        )
        for _, loaded in _default_and_inlined_loaders(code, cache, backend):
            run = fresh()
            reference = copy.deepcopy(run)
            actual = loaded(run, x)
            expected = reference(x)
            self.assertEqual(actual, expected)
            self.assertEqual(run.running_mean, reference.running_mean)
            self.assertEqual(run.running_var, reference.running_var)

    @parametrize("backend", ("eager", "inductor"))
    def test_tracer_dynamo_returned_global_constant(self, backend):
        def fn(x):
            return x.sin(), _PRECOMPILE_DYNAMO_GLOBAL_SCALE

        x = torch.randn(5, 4)
        code, cache = torch.compiler.precompile(
            fn,
            example_inputs=[(x,)],
            tracer="dynamo",
            backend=backend,
        )
        for _, loaded in _default_and_inlined_loaders(code, cache, backend):
            self.assertEqual(loaded(x), fn(x))

    @parametrize("backend", ("eager", "inductor"))
    def test_tracer_dynamo_preserves_autocast_guard(self, backend):
        def fn(a, b):
            return a @ b

        a = torch.randn(8, 8)
        b = torch.randn(8, 8)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            expected = fn(a, b)
            code, cache = torch.compiler.precompile(
                fn,
                example_inputs=[(a, b)],
                tracer="dynamo",
                backend=backend,
            )
        for _, loaded in _default_and_inlined_loaders(code, cache, backend):
            with torch.autocast("cpu", dtype=torch.bfloat16):
                self.assertEqual(loaded(a, b), expected)
            with self.assertRaisesRegex(PrecompileError, "no captured Dynamo variant"):
                loaded(a, b)

    def test_tracer_dynamo_eager_custom_builtins(self):
        def fn(x):
            return torch.relu(x).masked_fill(x < 0, float("-inf"))

        x = torch.randn(8)
        code, cache = torch.compiler.precompile(
            fn,
            example_inputs=[(x,)],
            tracer="dynamo",
            backend="eager",
        )
        self.assertEqual(torch.compiler.precompile.load(code, cache)(x), fn(x))

    @parametrize("backend", ("eager", "inductor"))
    def test_tracer_dynamo_dtensor_subclass(self, backend):
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_gloo_available():
            self.skipTest("gloo not available")

        from torch.distributed.tensor import DeviceMesh, distribute_tensor, Replicate
        from torch.testing._internal.common_utils import find_free_port

        saved_env = {key: os.environ.get(key) for key in ("MASTER_ADDR", "MASTER_PORT")}
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        dist.init_process_group("gloo", rank=0, world_size=1)
        try:
            mesh = DeviceMesh("cpu", list(range(1)))
            model = torch.nn.Linear(4, 3).eval()
            for name, parameter in list(model.named_parameters()):
                setattr(
                    model,
                    name,
                    torch.nn.Parameter(
                        distribute_tensor(parameter.detach(), mesh, [Replicate()])
                    ),
                )
            x = distribute_tensor(torch.randn(5, 4), mesh, [Replicate()])
            expected = model(x)
            code, cache = torch.compiler.precompile(
                _precompile_dynamo_call_module,
                example_inputs=[(model, x)],
                tracer="dynamo",
                backend=backend,
            )
            for _, loaded in _default_and_inlined_loaders(code, cache, backend):
                self.assertEqual(loaded(model, x).to_local(), expected.to_local())
        finally:
            dist.destroy_process_group()
            for key, value in saved_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_tracer_dynamo_cross_tracer_cache_rejected(self):
        x = torch.randn(4)
        dynamo_code, _ = torch.compiler.precompile(
            _precompile_dynamo_dynamic,
            example_inputs=[(x,)],
            tracer="dynamo",
            backend="eager",
        )
        _, make_fx_cache = torch.compiler.precompile(
            _precompile_dynamo_dynamic,
            example_inputs=[(x,)],
            backend="eager",
        )
        with self.assertRaisesRegex(PrecompileError, "code_hash|does not match|tracer"):
            torch.compiler.precompile.load(dynamo_code, make_fx_cache)

    def test_tracer_dynamo_mismatched_code_cache_pair_rejected(self):
        code, _ = torch.compiler.precompile(
            _precompile_dynamo_dynamic,
            example_inputs=[(torch.randn(3),)],
            tracer="dynamo",
            backend="eager",
        )
        _, cache = torch.compiler.precompile(
            _precompile_dynamo_dynamic,
            example_inputs=[(torch.randn(5),)],
            tracer="dynamo",
            backend="eager",
        )
        with self.assertRaisesRegex(
            PrecompileError, "cache does not match python_code"
        ):
            torch.compiler.precompile.load(code, cache)

    def test_tracer_dynamo_corrupt_cache_bundle_degrades(self):
        x = torch.randn(4)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_dynamic,
            example_inputs=[(x,)],
            tracer="dynamo",
            backend="eager",
        )
        blob = torch.load(io.BytesIO(cache), weights_only=True)
        blob["artifact"] = [b"corrupt"]
        buffer = io.BytesIO()
        torch.save(blob, buffer)
        with self.assertLogs("torch.compiler._cache", level="WARNING") as logs:
            loaded = torch.compiler.precompile.load(code, buffer.getvalue())
        self.assertTrue(
            any("Failed to un-pickle cache artifacts" in line for line in logs.output)
        )
        self.assertEqual(loaded(x), _precompile_dynamo_dynamic(x))

    def test_tracer_dynamo_rejects_torch_version_skew(self):
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_dynamic,
            example_inputs=[(torch.randn(4),)],
            tracer="dynamo",
            backend="eager",
        )
        self.assertIn(f"_DYNAMO_TORCH_VERSION = {torch.__version__!r}", code)
        mismatched = code.replace(
            f"_DYNAMO_TORCH_VERSION = {torch.__version__!r}",
            "_DYNAMO_TORCH_VERSION = 'different-build'",
            1,
        )
        with self.assertRaisesRegex(
            PrecompileError, "produced by torch different-build"
        ):
            torch.compiler.precompile.load(mismatched, cache)

    def test_tracer_dynamo_rejects_partial_cleanly(self):
        def fn(x, scale):
            return x * scale

        with self.assertRaisesRegex((NotImplementedError, PrecompileError), "partial"):
            torch.compiler.precompile(
                functools.partial(fn, scale=2),
                example_inputs=[(torch.randn(4),)],
                tracer="dynamo",
                backend="eager",
            )

    @torch._dynamo.config.patch(
        automatic_dynamic_shapes=True, assume_static_by_default=True
    )
    @parametrize("backend", ("eager", "inductor"))
    def test_tracer_dynamo_training_module_graph_break(self, backend):
        from torch._precompile import _parse_artifact_metadata

        model = _PrecompileDynamoBreakingModule().train()
        examples = [
            (model, torch.randn(size, 4, requires_grad=True)) for size in (2, 3)
        ]
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_call_module,
            example_inputs=examples,
            tracer="dynamo",
            backend=backend,
            training=True,
        )
        self.assertEqual(_parse_artifact_metadata(code)["SERVING_MODE"], "installed")

        for _, loaded in _default_and_inlined_loaders(code, cache, backend):
            actual_model = copy.deepcopy(model)
            expected_model = copy.deepcopy(model)
            actual_input = torch.randn(7, 4, requires_grad=True)
            expected_input = actual_input.detach().clone().requires_grad_()
            expected = expected_model(expected_input)
            expected.sum().backward()
            with loaded:
                actual = loaded(actual_model, actual_input)
                actual.sum().backward()
            self.assertEqual(actual, expected)
            self.assertEqual(actual_input.grad, expected_input.grad)
            for actual_param, expected_param in zip(
                actual_model.parameters(), expected_model.parameters()
            ):
                self.assertEqual(actual_param.grad, expected_param.grad)

    @parametrize("backend", ("eager", "inductor"))
    def test_tracer_dynamo_training_across_disabled_graph_break(self, backend):
        x = torch.randn(4, requires_grad=True)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_with_disabled,
            example_inputs=[(x,)],
            tracer="dynamo",
            backend=backend,
            training=True,
        )

        if backend == "inductor":
            self.assertEqual(code.count("class _CompiledFunction_"), 2)
        for _, loaded in _default_and_inlined_loaders(code, cache, backend):
            actual_input = torch.randn(4, requires_grad=True)
            ref_input = actual_input.detach().clone().requires_grad_()
            expected = _precompile_dynamo_with_disabled(ref_input)
            expected.sum().backward()
            actual = loaded(actual_input)
            actual.sum().backward()
            self.assertEqual(actual, expected)
            self.assertEqual(actual_input.grad, ref_input.grad)

    def test_tracer_dynamo_rebinds_imported_graph_break_alias(self):
        x = torch.randn(4)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_aliased_graph_break,
            example_inputs=[(x,)],
            tracer="dynamo",
            backend="eager",
        )

        loaded = torch.compiler.precompile.load(code, cache)
        self.assertEqual(loaded(x), _precompile_dynamo_aliased_graph_break(x))
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as artifact:
            artifact.write(code)
            artifact_path = artifact.name
        try:
            subprocess.check_call(
                [
                    sys.executable,
                    "-c",
                    textwrap.dedent(
                        """
                        import runpy as r
                        import sys as s
                        import torch as t

                        namespace = r.run_path(s.argv[1])
                        x = t.randn(4)
                        t.testing.assert_close(namespace["forward"](x), (x + 1) * 2)
                        """
                    ),
                    artifact_path,
                ]
            )
        finally:
            os.unlink(artifact_path)

    def test_tracer_dynamo_rebinds_unguarded_resume_globals(self):
        x = torch.randn(4)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_empty_resume,
            example_inputs=[(x,)],
            tracer="dynamo",
            backend="eager",
        )

        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as artifact:
            artifact.write(code)
            artifact_path = artifact.name
        try:
            subprocess.check_call(
                [
                    sys.executable,
                    "-c",
                    textwrap.dedent(
                        """
                        import runpy as r
                        import sys as s
                        import torch as t

                        namespace = r.run_path(s.argv[1])
                        x = t.randn(4)
                        actual = namespace["forward"](x)
                        t.testing.assert_close(actual[0], x + 1)
                        assert actual[1] == 7
                        """
                    ),
                    artifact_path,
                ]
            )
        finally:
            os.unlink(artifact_path)

    def test_tracer_dynamo_rejects_unportable_resume_global(self):
        with self.assertRaisesRegex(PrecompileError, "transformed global"):
            torch.compiler.precompile(
                _precompile_dynamo_unportable_resume,
                example_inputs=[(torch.randn(4),)],
                tracer="dynamo",
                backend="eager",
            )

    @torch._dynamo.config.patch(
        automatic_dynamic_shapes=True, assume_static_by_default=True
    )
    def test_tracer_dynamo_captures_disabled_graph_break(self):
        examples = [(torch.randn(size, 4),) for size in (2, 3, 5)]
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_with_disabled,
            example_inputs=examples,
            tracer="dynamo",
            backend="eager",
        )

        self.assertIn("FRAME_COUNT = 2", code)
        self.assertIn("VARIANT_COUNT = 4", code)
        self.assertIn("GRAPH_COUNT = 4", code)
        self.assertIn("DYNAMIC_GRAPH_COUNT = 2", code)
        self.assertEqual(len(_dynamo_serialized_guard_summary(code)), 4)
        loaded = torch.compiler.precompile.load(code, cache)
        for size in (2, 7):
            x = torch.randn(size, 4)
            self.assertEqual(loaded(x), _precompile_dynamo_with_disabled(x))
        with self.assertRaisesRegex(PrecompileError, "no captured Dynamo variant"):
            loaded(torch.randn(1, 4))

        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as artifact:
            artifact.write(code)
            artifact_path = artifact.name
        try:
            subprocess.check_call(
                [
                    sys.executable,
                    "-c",
                    textwrap.dedent(
                        """
                        import runpy as r
                        import sys as s
                        import torch as t

                        namespace = r.run_path(s.argv[1])
                        x = t.randn(7, 4)
                        actual = namespace["forward"](x)
                        expected = (
                            t.cos(x.sin() + x.shape[0]) * 0.5 * x.shape[0]
                        )
                        t.testing.assert_close(actual, expected)
                        """
                    ),
                    artifact_path,
                ]
            )
        finally:
            os.unlink(artifact_path)

    def test_tracer_dynamo_isolates_disabled_function_globals(self):
        x = torch.randn(4)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_with_two_disabled,
            example_inputs=[(x,)],
            tracer="dynamo",
            backend="eager",
        )

        loaded = torch.compiler.precompile.load(code, cache)
        self.assertEqual(loaded(x), _precompile_dynamo_with_two_disabled(x))
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as artifact:
            artifact.write(code)
            artifact_path = artifact.name
        try:
            subprocess.check_call(
                [
                    sys.executable,
                    "-c",
                    textwrap.dedent(
                        """
                        import runpy as r
                        import sys as s
                        import torch as t

                        namespace = r.run_path(s.argv[1])
                        x = t.randn(4)
                        t.testing.assert_close(
                            namespace["forward"](x), x.sin() * 5.0
                        )
                        """
                    ),
                    artifact_path,
                ]
            )
        finally:
            os.unlink(artifact_path)

    def test_tracer_dynamo_rejects_decorated_disabled_closure(self):
        with self.assertRaisesRegex(NotImplementedError, "closure-free"):
            torch.compiler.precompile(
                _precompile_dynamo_with_decorated_disabled,
                example_inputs=[(torch.randn(4),)],
                tracer="dynamo",
                backend="eager",
            )

    def test_tracer_dynamo_disabled_global_does_not_shadow_entry(self):
        x = torch.randn(4)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_with_forward_global,
            example_inputs=[(x,)],
            tracer="dynamo",
            backend="eager",
        )
        self.assertEqual(
            torch.compiler.precompile.load(code, cache)(x),
            _precompile_dynamo_with_forward_global(x),
        )

    def test_tracer_dynamo_rejects_nested_closure(self):
        with self.assertRaisesRegex(NotImplementedError, "nested functions"):
            torch.compiler.precompile(
                _precompile_dynamo_cellvar,
                example_inputs=[(torch.randn(4), 2)],
                tracer="dynamo",
                backend="eager",
            )

    def test_tracer_dynamo_rejects_disabled_global_mutation(self):
        global _PRECOMPILE_DYNAMO_MUTATED_GLOBAL
        _PRECOMPILE_DYNAMO_MUTATED_GLOBAL = 0
        try:
            with self.assertRaisesRegex(NotImplementedError, "cannot mutate globals"):
                torch.compiler.precompile(
                    _precompile_dynamo_with_mutated_global,
                    example_inputs=[(torch.randn(4),)],
                    tracer="dynamo",
                    backend="eager",
                )
        finally:
            _PRECOMPILE_DYNAMO_MUTATED_GLOBAL = 0

    @parametrize(
        "fn",
        (
            _precompile_dynamo_stores_input_global,
            _precompile_dynamo_calls_global_mutating_helper,
        ),
    )
    def test_tracer_dynamo_rejects_input_global_mutation_before_capture(self, fn):
        global _PRECOMPILE_DYNAMO_INPUT_GLOBAL
        _PRECOMPILE_DYNAMO_INPUT_GLOBAL = None
        with self.assertRaisesRegex(NotImplementedError, "cannot mutate globals"):
            torch.compiler.precompile(
                fn,
                example_inputs=[(torch.randn(4),)],
                tracer="dynamo",
                backend="eager",
            )
        self.assertIsNone(_PRECOMPILE_DYNAMO_INPUT_GLOBAL)

    def test_tracer_dynamo_rejects_container_indirected_global_mutation(self):
        global _PRECOMPILE_DYNAMO_INPUT_GLOBAL
        _PRECOMPILE_DYNAMO_INPUT_GLOBAL = None
        with self.assertRaisesRegex(NotImplementedError, "cannot mutate globals"):
            torch.compiler.precompile(
                _precompile_dynamo_calls_container_mutating_helper,
                example_inputs=[(torch.randn(4),)],
                tracer="dynamo",
                backend="eager",
            )
        self.assertIsNone(_PRECOMPILE_DYNAMO_INPUT_GLOBAL)

    @parametrize(
        "fn",
        (
            _precompile_dynamo_mutates_global_attribute,
            _precompile_dynamo_operator_mutates_global_attribute,
            _precompile_dynamo_operator_iadd_global_attribute,
            _precompile_dynamo_setattr_global_attribute,
            _precompile_dynamo_calls_default_mutation_helper,
            _precompile_dynamo_calls_descriptor_append_default_mutation_helper,
            _precompile_dynamo_calls_descriptor_iadd_default_mutation_helper,
            _precompile_dynamo_calls_partial_descriptor_default_mutation_helper,
            _precompile_dynamo_calls_deque_appendleft_default_mutation_helper,
            _precompile_dynamo_calls_deque_rotate_default_mutation_helper,
            _precompile_dynamo_mutating_getitem,
            _precompile_dynamo_defaultdict_getitem,
            _precompile_dynamo_mutating_property,
            _precompile_dynamo_mutating_iter,
            _precompile_dynamo_mutating_bool,
            _precompile_dynamo_mutating_add,
            _precompile_dynamo_mutating_compare,
            _precompile_dynamo_mutating_unpack,
            _precompile_dynamo_mutating_format,
            _precompile_dynamo_mutating_context,
            _precompile_dynamo_mutating_len,
            _precompile_dynamo_mutating_index,
            _precompile_dynamo_mutating_contains,
            _precompile_dynamo_mutating_sum,
            _precompile_dynamo_mutating_nested_property,
            _precompile_dynamo_mutating_nested_method,
            _precompile_dynamo_mutating_iterated_method,
            _precompile_dynamo_mutating_short_circuit,
            _precompile_dynamo_mutating_star_unpack,
            _precompile_dynamo_mutating_hash,
            _precompile_dynamo_mutating_bisect,
            _precompile_dynamo_mutating_map_result,
            _precompile_dynamo_mutating_copy_result,
            _precompile_dynamo_mutating_iterator_result,
            _precompile_dynamo_mutating_descriptor_result,
            _precompile_dynamo_mutating_contextvar_result,
            _precompile_dynamo_mutating_unpack_result,
            _precompile_dynamo_mutating_yield_result,
            _precompile_dynamo_mutating_slice_result,
            _precompile_dynamo_mutating_local_function_result,
            _precompile_dynamo_heapq_mutation,
        ),
        name_fn=lambda fn: fn.__name__,
    )
    def test_tracer_dynamo_rejects_indirect_environment_mutation(self, fn):
        _PrecompileDynamoMutationHolder.state[:] = [0]
        _PrecompileDynamoMutationHolder.value = 0
        _PRECOMPILE_DYNAMO_DEFAULT_MUTATION_STATE.clear()
        _PRECOMPILE_DYNAMO_DEQUE_MUTATION_STATE.clear()
        _PRECOMPILE_DYNAMO_MUTATING_GETITEM.value = 0
        _PRECOMPILE_DYNAMO_DEFAULTDICT.clear()
        _PRECOMPILE_DYNAMO_MUTATING_PROTOCOL.value = 0
        _PRECOMPILE_DYNAMO_MUTATING_EQUAL.value = 0
        _PRECOMPILE_DYNAMO_MUTATING_RADD.value = 0
        _PrecompileDynamoMutatingClass.value = 0
        _PRECOMPILE_DYNAMO_BISECT_VALUES[:] = [1]
        _PRECOMPILE_DYNAMO_HEAP[:] = [1]
        _precompile_dynamo_stateful_helper.state.clear()
        with self.assertRaisesRegex(NotImplementedError, "cannot mutate globals"):
            torch.compiler.precompile(
                fn,
                example_inputs=[(torch.randn(4),)],
                tracer="dynamo",
                backend="eager",
            )
        self.assertEqual(_PrecompileDynamoMutationHolder.state, [0])
        self.assertEqual(_PrecompileDynamoMutationHolder.value, 0)
        self.assertEqual(_PRECOMPILE_DYNAMO_DEFAULT_MUTATION_STATE, [])
        self.assertEqual(
            _PRECOMPILE_DYNAMO_DEQUE_MUTATION_STATE,
            deque(),
        )
        self.assertEqual(_PRECOMPILE_DYNAMO_MUTATING_GETITEM.value, 0)
        self.assertEqual(_PRECOMPILE_DYNAMO_DEFAULTDICT, {})
        self.assertEqual(_PRECOMPILE_DYNAMO_MUTATING_PROTOCOL.value, 0)
        self.assertEqual(_PRECOMPILE_DYNAMO_MUTATING_EQUAL.value, 0)
        self.assertEqual(_PRECOMPILE_DYNAMO_MUTATING_RADD.value, 0)
        self.assertEqual(_PrecompileDynamoMutatingClass.value, 0)
        self.assertEqual(_PRECOMPILE_DYNAMO_BISECT_VALUES, [1])
        self.assertEqual(_PRECOMPILE_DYNAMO_HEAP, [1])
        self.assertEqual(_precompile_dynamo_stateful_helper.state, [])

    @parametrize(
        "fn",
        (_precompile_dynamo_torch_add, _precompile_dynamo_pure_add),
        name_fn=lambda fn: fn.__name__,
    )
    def test_tracer_dynamo_allows_pure_add(self, fn):
        x = torch.randn(4)
        code, cache = torch.compiler.precompile(
            fn,
            example_inputs=[(x,)],
            tracer="dynamo",
            backend="eager",
        )
        self.assertEqual(torch.compiler.precompile.load(code, cache)(x), fn(x))

    def test_tracer_dynamo_rejects_nonlocal_mutation(self):
        closure = _PRECOMPILE_DYNAMO_NONLOCAL_MUTATION.__closure__
        self.assertIsNotNone(closure)
        self.assertEqual(closure[0].cell_contents, 0)
        with self.assertRaisesRegex(NotImplementedError, "cannot mutate globals"):
            torch.compiler.precompile(
                _precompile_dynamo_calls_nonlocal_mutation,
                example_inputs=[(torch.randn(4),)],
                tracer="dynamo",
                backend="eager",
            )
        self.assertEqual(closure[0].cell_contents, 0)

    def test_tracer_dynamo_rejects_function_metadata_global_mutation(self):
        global _PRECOMPILE_DYNAMO_INPUT_GLOBAL
        _PRECOMPILE_DYNAMO_INPUT_GLOBAL = None
        with self.assertRaisesRegex(NotImplementedError, "cannot mutate globals"):
            torch.compiler.precompile(
                _precompile_dynamo_calls_function_metadata_mutating_helper,
                example_inputs=[(torch.randn(4),)],
                tracer="dynamo",
                backend="eager",
            )
        self.assertIsNone(_PRECOMPILE_DYNAMO_INPUT_GLOBAL)

    @parametrize(
        "fn",
        (
            _precompile_dynamo_calls_function_globals_mutating_helper,
            _precompile_dynamo_calls_function_closure_mutating_helper,
        ),
        name_fn=lambda fn: fn.__name__,
    )
    def test_tracer_dynamo_rejects_dynamic_function_metadata(self, fn):
        with self.assertRaisesRegex(
            PrecompileError, "input-derived.*dynamic function metadata"
        ):
            torch.compiler.precompile(
                fn,
                example_inputs=[(torch.randn(4),)],
                tracer="dynamo",
                backend="eager",
            )

    @parametrize(
        "fn",
        (
            _precompile_dynamo_calls_input_mutating_method,
            _precompile_dynamo_calls_input_mutating_getitem,
        ),
        name_fn=lambda fn: fn.__name__,
    )
    def test_tracer_dynamo_rejects_input_method_global_mutation(self, fn):
        global _PRECOMPILE_DYNAMO_INPUT_GLOBAL
        _PRECOMPILE_DYNAMO_INPUT_GLOBAL = None
        with self.assertRaisesRegex(NotImplementedError, "cannot mutate globals"):
            torch.compiler.precompile(
                fn,
                example_inputs=[
                    (_PrecompileDynamoGlobalMutatingMethod(), torch.randn(4))
                ],
                tracer="dynamo",
                backend="eager",
            )
        self.assertIsNone(_PRECOMPILE_DYNAMO_INPUT_GLOBAL)

    @parametrize(
        "fn,args",
        (
            (_precompile_dynamo_calls_mutating_factory, (torch.randn(4),)),
            (
                _precompile_dynamo_calls_conditional_mutating_factory,
                (torch.randn(4), True),
            ),
            (
                _precompile_dynamo_calls_input_factory,
                (_PrecompileDynamoGlobalMutatingFactory, torch.randn(4)),
            ),
        ),
        name_fn=lambda fn, args: fn.__name__,
    )
    def test_tracer_dynamo_rejects_constructor_global_mutation(self, fn, args):
        global _PRECOMPILE_DYNAMO_INPUT_GLOBAL
        _PRECOMPILE_DYNAMO_INPUT_GLOBAL = None
        with self.assertRaisesRegex(NotImplementedError, "cannot mutate globals"):
            torch.compiler.precompile(
                fn,
                example_inputs=[args],
                tracer="dynamo",
                backend="eager",
            )
        self.assertIsNone(_PRECOMPILE_DYNAMO_INPUT_GLOBAL)

    def test_tracer_dynamo_rejects_dynamic_disabled_globals(self):
        with self.assertRaisesRegex(
            PrecompileError, "input-derived.*dynamic global access"
        ):
            torch.compiler.precompile(
                _precompile_dynamo_with_dynamic_global,
                example_inputs=[(torch.randn(4),)],
                tracer="dynamo",
                backend="eager",
            )

    def test_tracer_dynamo_ignores_uncaptured_disabled_branch(self):
        x = torch.randn(4)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_with_dead_disabled_branch,
            example_inputs=[(x, True)],
            tracer="dynamo",
            backend="eager",
        )
        self.assertEqual(
            torch.compiler.precompile.load(code, cache)(x, True),
            _precompile_dynamo_with_dead_disabled_branch(x, True),
        )

    def test_tracer_dynamo_nested_disabled_function(self):
        x = torch.randn(4)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_with_nested_disabled,
            example_inputs=[(x,)],
            tracer="dynamo",
            backend="eager",
        )
        self.assertEqual(
            torch.compiler.precompile.load(code, cache)(x),
            _precompile_dynamo_with_nested_disabled(x),
        )

    def test_tracer_dynamo_preserves_disabled_helper_with_compiled_prefix(self):
        x = torch.randn(4)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_with_compiled_prefix_helper,
            example_inputs=[(x,)],
            tracer="dynamo",
            backend="eager",
        )
        self.assertEqual(
            torch.compiler.precompile.load(code, cache)(x),
            _precompile_dynamo_with_compiled_prefix_helper(x),
        )

    def test_tracer_dynamo_rejects_nonimportable_disabled_module(self):
        with self.assertRaisesRegex(NotImplementedError, "non-importable module"):
            torch.compiler.precompile(
                _precompile_dynamo_with_ephemeral_module,
                example_inputs=[(torch.randn(4),)],
                tracer="dynamo",
                backend="eager",
            )

    def test_tracer_dynamo_restores_existing_profiler(self):
        events = []

        def profile(_frame, event, _arg):
            events.append(event)

        previous = sys.getprofile()
        sys.setprofile(profile)
        try:
            x = torch.randn(4)
            code, cache = torch.compiler.precompile(
                _precompile_dynamo_graph_break,
                example_inputs=[(x,)],
                tracer="dynamo",
                backend="eager",
            )
            self.assertIs(sys.getprofile(), profile)
            self.assertEqual(
                torch.compiler.precompile.load(code, cache)(x),
                _precompile_dynamo_graph_break(x),
            )
        finally:
            sys.setprofile(previous)
        self.assertTrue(events)

    def test_tracer_dynamo_preserves_active_cprofile(self):
        profiler = cProfile.Profile()
        profiler.enable()
        previous = sys.getprofile()
        try:
            x = torch.randn(4)
            code, cache = torch.compiler.precompile(
                _precompile_dynamo_graph_break,
                example_inputs=[(x,)],
                tracer="dynamo",
                backend="eager",
            )
            self.assertIs(sys.getprofile(), previous)
            self.assertEqual(
                torch.compiler.precompile.load(code, cache)(x),
                _precompile_dynamo_graph_break(x),
            )
        finally:
            profiler.disable()

    def test_tracer_dynamo_rejects_unserializable_dispatch_guards(self):
        x = torch.randn(4)
        with self.assertRaisesRegex(PrecompileError, "dropped guards.*op"):
            torch.compiler.precompile(
                _precompile_dynamo_callable,
                example_inputs=[(x, torch.sin), (x, torch.cos)],
                tracer="dynamo",
                backend="eager",
            )

    def test_tracer_dynamo_rejects_single_callable_dispatch_guard(self):
        with self.assertRaisesRegex(PrecompileError, "dropped guards.*op"):
            torch.compiler.precompile(
                _precompile_dynamo_callable,
                example_inputs=[(torch.randn(4), torch.sin)],
                tracer="dynamo",
                backend="eager",
            )

    def test_tracer_dynamo_retains_scalar_dispatch_guards(self):
        examples = [(torch.randn(4), scale) for scale in (2, 3)]
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_scalar,
            example_inputs=examples,
            tracer="dynamo",
            backend="eager",
        )

        loaded = torch.compiler.precompile.load(code, cache)
        summaries = _dynamo_serialized_guard_summary(code)
        self.assertEqual(len(summaries), 2)
        self.assertEqual(
            ["CONSTANT_MATCH" in guard_types for guard_types, _, _, _ in summaries],
            [True, False],
        )
        for guard_types, _, _, _ in summaries:
            self.assertIn("TENSOR_MATCH", guard_types)
            self.assertNotIn("GLOBAL_STATE", guard_types)
        x = torch.randn(4)
        self.assertEqual(loaded(x, 2), _precompile_dynamo_scalar(x, 2))
        self.assertEqual(loaded(x, 4), _precompile_dynamo_scalar(x, 4))

    def test_tracer_dynamo_keeps_invariant_value_guard(self):
        x = torch.randn(4)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_scalar_branch,
            example_inputs=[(x, 2)],
            tracer="dynamo",
            backend="eager",
        )
        loaded = torch.compiler.precompile.load(code, cache)
        self.assertEqual(loaded(x, 2), _precompile_dynamo_scalar_branch(x, 2))
        with self.assertRaisesRegex(PrecompileError, "no captured Dynamo variant"):
            loaded(x, 3)

    def test_tracer_dynamo_preserves_relational_guards(self):
        shared = torch.ones(4)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_aliasing,
            example_inputs=[
                (torch.ones(4), torch.full((4,), 2.0)),
                (shared, shared),
            ],
            tracer="dynamo",
        )

        summaries = _dynamo_serialized_guard_summary(code)
        has_relational_inputs = any(
            types.count("TENSOR_MATCH") >= 2 for types, _, _, _ in summaries
        )
        self.assertTrue(has_relational_inputs)
        for _, loaded in _default_and_inlined_loaders(code, cache, "inductor"):
            expected_a = torch.ones(4)
            expected = _precompile_dynamo_aliasing(expected_a, torch.full((4,), 2.0))
            actual_a = torch.ones(4)
            actual = loaded(actual_a, torch.full((4,), 2.0))
            self.assertEqual(actual, expected)
            self.assertEqual(actual_a, expected_a)

            expected_shared = torch.ones(4)
            expected = _precompile_dynamo_aliasing(expected_shared, expected_shared)
            actual_shared = torch.ones(4)
            actual = loaded(actual_shared, actual_shared)
            self.assertEqual(actual, expected)
            self.assertEqual(actual_shared, expected_shared)

    def test_tracer_dynamo_rejects_storage_alias_topology_change(self):
        left_base = torch.ones(5)
        right_base = torch.ones(5)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_aliasing,
            example_inputs=[(left_base[:4], right_base[1:])],
            tracer="dynamo",
        )

        for _, loaded in _default_and_inlined_loaders(code, cache, "inductor"):
            shared = torch.ones(5)
            with self.assertRaisesRegex(PrecompileError, "storage alias"):
                loaded(shared[:4], shared[1:])

            shared_buffer = bytearray(20)
            left = torch.frombuffer(shared_buffer, dtype=torch.float32, count=4)
            right = torch.frombuffer(
                shared_buffer, dtype=torch.float32, count=4, offset=4
            )
            with self.assertRaisesRegex(PrecompileError, "storage alias"):
                loaded(left, right)

        shared = torch.ones(5)
        with self.assertRaisesRegex(PrecompileError, "storage alias"):
            torch.compiler.precompile(
                _precompile_dynamo_aliasing,
                example_inputs=[
                    (torch.ones(4), torch.ones(4)),
                    (shared[:4], shared[1:]),
                ],
                tracer="dynamo",
            )

        with self.assertRaisesRegex(PrecompileError, "storage alias"):
            torch.compiler.precompile(
                _precompile_dynamo_aliasing,
                example_inputs=[(shared[:4], shared[1:])],
                tracer="dynamo",
            )

        shared_buffer = bytearray(20)
        left = torch.frombuffer(shared_buffer, dtype=torch.float32, count=4)
        right = torch.frombuffer(shared_buffer, dtype=torch.float32, count=4, offset=4)
        with self.assertRaisesRegex(PrecompileError, "storage alias"):
            torch.compiler.precompile(
                _precompile_dynamo_aliasing,
                example_inputs=[(left, right)],
                tracer="dynamo",
            )

        with self.assertRaisesRegex(PrecompileError, "storage alias"):
            torch.compiler.precompile(
                _precompile_dynamo_box_aliasing,
                example_inputs=[(_PrecompileDynamoTensorBox(shared[:4], shared[1:]),)],
                tracer="dynamo",
            )

        code, cache = torch.compiler.precompile(
            _precompile_dynamo_box_aliasing,
            example_inputs=[
                (_PrecompileDynamoTensorBox(torch.ones(4), torch.ones(4)),)
            ],
            tracer="dynamo",
        )
        loaded = torch.compiler.precompile.load(code, cache)
        shared = torch.ones(5)
        with self.assertRaisesRegex(PrecompileError, "storage alias"):
            loaded(_PrecompileDynamoTensorBox(shared[:4], shared[1:]))

        code, cache = torch.compiler.precompile(
            _precompile_dynamo_mapping_aliasing,
            example_inputs=[
                (types.MappingProxyType({"a": torch.ones(4), "b": torch.ones(4)}),)
            ],
            tracer="dynamo",
        )
        loaded = torch.compiler.precompile.load(code, cache)
        shared = torch.ones(5)
        with self.assertRaisesRegex(PrecompileError, "storage alias"):
            loaded(types.MappingProxyType({"a": shared[:4], "b": shared[1:]}))

        code, cache = torch.compiler.precompile(
            _precompile_dynamo_list_subclass_aliasing,
            example_inputs=[
                (_PrecompileDynamoTensorList(torch.ones(4)), torch.ones(4))
            ],
            tracer="dynamo",
            backend="eager",
        )
        loaded = torch.compiler.precompile.load(code, cache)
        other = torch.ones(4)
        values = _PrecompileDynamoTensorList(torch.ones(4))
        values.hidden.set_(other)
        with self.assertRaisesRegex(PrecompileError, "storage alias"):
            loaded(values, other)

        shared = torch.ones(4)
        with self.assertRaisesRegex(PrecompileError, "storage alias"):
            torch.compiler.precompile(
                _precompile_dynamo_list_subclass_aliasing,
                example_inputs=[(_PrecompileDynamoTensorList(shared[:]), shared[:])],
                tracer="dynamo",
                backend="eager",
            )

    def test_tracer_dynamo_storage_contract_precedes_input_mutation(self):
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_rebinds_storage,
            example_inputs=[(torch.ones(4), torch.full((4,), 2.0))],
            tracer="dynamo",
        )
        for _, loaded in _default_and_inlined_loaders(code, cache, "inductor"):
            left = torch.ones(4)
            right = torch.full((4,), 2.0)
            self.assertEqual(loaded(left, right), torch.full((4,), 3.0))

    def test_tracer_dynamo_rejects_input_global_identity_guard(self):
        x = torch.randn(4)
        with self.assertRaisesRegex(PrecompileError, "input-derived"):
            torch.compiler.precompile(
                _precompile_dynamo_input_global_identity,
                example_inputs=[(x, _PRECOMPILE_DYNAMO_IDENTITY_TOKEN)],
                tracer="dynamo",
                backend="eager",
            )

    def test_tracer_dynamo_rejects_reverse_scalar_input_global_identity_guard(self):
        token = int(str(_PRECOMPILE_DYNAMO_SCALAR_IDENTITY))
        self.assertIsNot(token, _PRECOMPILE_DYNAMO_SCALAR_IDENTITY)
        with self.assertRaisesRegex(PrecompileError, "input-derived"):
            torch.compiler.precompile(
                _precompile_dynamo_scalar_identity,
                example_inputs=[(torch.randn(4), token)],
                tracer="dynamo",
                backend="eager",
            )

    def test_tracer_dynamo_rejects_reverse_code_literal_identity_guard(self):
        literal = next(
            value
            for value in _precompile_dynamo_code_literal_identity.__code__.co_consts
            if value == 1000003
        )
        token = int(str(literal))
        self.assertIsNot(token, literal)
        with self.assertRaisesRegex(PrecompileError, "input-derived"):
            torch.compiler.precompile(
                _precompile_dynamo_code_literal_identity,
                example_inputs=[(torch.randn(4), token)],
                tracer="dynamo",
                backend="eager",
            )

    def test_tracer_dynamo_rejects_python_input_alias_guard(self):
        left = int("1000005")
        right = int("1000005")
        self.assertIsNot(left, right)
        with self.assertRaisesRegex(PrecompileError, "input-derived"):
            torch.compiler.precompile(
                _precompile_dynamo_input_identity,
                example_inputs=[(torch.randn(4), left, right)],
                tracer="dynamo",
                backend="eager",
            )

    def test_tracer_dynamo_rejects_opaque_return_identity_guard(self):
        with self.assertRaisesRegex(PrecompileError, "input-derived"):
            torch.compiler.precompile(
                _precompile_dynamo_opaque_return_identity,
                example_inputs=[(torch.randn(4), object())],
                tracer="dynamo",
                backend="eager",
            )

    def test_tracer_dynamo_rejects_recursive_identity_guard(self):
        with self.assertRaisesRegex(PrecompileError, "input-derived"):
            torch.compiler.precompile(
                _precompile_dynamo_recursive_identity,
                example_inputs=[(torch.randn(4), object(), 2)],
                tracer="dynamo",
                backend="eager",
            )

    def test_tracer_dynamo_allows_pure_recursion(self):
        args = (torch.randn(4), 2)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_recursive_pure,
            example_inputs=[args],
            tracer="dynamo",
            backend="eager",
        )
        self.assertEqual(
            torch.compiler.precompile.load(code, cache)(*args),
            _precompile_dynamo_recursive_pure(*args),
        )

    def test_tracer_dynamo_allows_tensor_input_alias_guard(self):
        example = (torch.randn(4), torch.randn(2), torch.randn(2))
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_input_identity,
            example_inputs=[example],
            tracer="dynamo",
            backend="eager",
        )
        self.assertEqual(
            torch.compiler.precompile.load(code, cache)(*example),
            _precompile_dynamo_input_identity(*example),
        )

    @parametrize(
        "fn",
        (
            _precompile_dynamo_map_identity,
            _precompile_dynamo_filter_identity,
            _precompile_dynamo_reduce_identity,
        ),
        name_fn=lambda fn: fn.__name__,
    )
    def test_tracer_dynamo_rejects_native_callback_identity_guard(self, fn):
        with self.assertRaisesRegex(PrecompileError, "input-derived"):
            torch.compiler.precompile(
                fn,
                example_inputs=[(torch.randn(4), [object()])],
                tracer="dynamo",
                backend="eager",
            )

    def test_tracer_dynamo_allows_environment_only_identity(self):
        x = torch.randn(4)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_environment_only_identity,
            example_inputs=[(x,)],
            tracer="dynamo",
            backend="eager",
        )
        self.assertEqual(
            torch.compiler.precompile.load(code, cache)(x),
            _precompile_dynamo_environment_only_identity(x),
        )

    @parametrize(
        "fn,example",
        (
            (_precompile_dynamo_default_identity, (torch.randn(4),)),
            (
                _precompile_dynamo_calls_default_identity_method,
                (torch.randn(4), _PrecompileDynamoDefaultIdentityMethod()),
            ),
            (
                _precompile_dynamo_calls_environment_identity_method,
                (torch.randn(4), _PrecompileDynamoDefaultIdentityMethod()),
            ),
        ),
        name_fn=lambda fn, example: fn.__name__,
    )
    def test_tracer_dynamo_allows_environment_identity_defaults(self, fn, example):
        code, cache = torch.compiler.precompile(
            fn,
            example_inputs=[example],
            tracer="dynamo",
            backend="eager",
        )
        self.assertEqual(
            torch.compiler.precompile.load(code, cache)(*example), fn(*example)
        )

    @parametrize(
        "fn,example",
        (
            (_precompile_dynamo_tensor_equals_nan, (torch.randn(4),)),
            (_precompile_dynamo_singleton_identity, (torch.randn(4), None)),
            (_precompile_dynamo_ellipsis_default, (torch.randn(4),)),
        ),
        name_fn=lambda fn, example: fn.__name__,
    )
    def test_tracer_dynamo_allows_safe_identity_relations(self, fn, example):
        code, cache = torch.compiler.precompile(
            fn,
            example_inputs=[example],
            tracer="dynamo",
            backend="eager",
        )
        self.assertEqual(
            torch.compiler.precompile.load(code, cache)(*example), fn(*example)
        )

    @parametrize(
        "fn,example",
        (
            (_precompile_dynamo_global_index, (torch.randn(4),)),
            (
                _precompile_dynamo_global_key,
                (torch.randn(4), {_PRECOMPILE_DYNAMO_KEY: 2.0}),
            ),
        ),
        name_fn=lambda fn, example: fn.__name__,
    )
    def test_tracer_dynamo_allows_global_index_or_key(self, fn, example):
        code, cache = torch.compiler.precompile(
            fn,
            example_inputs=[example],
            tracer="dynamo",
            backend="eager",
        )
        loaded = torch.compiler.precompile.load(code, cache)
        self.assertEqual(loaded(*example), fn(*example))

    def test_tracer_dynamo_rejects_scalar_input_global_identity_guard(self):
        with self.assertRaisesRegex(PrecompileError, "input-derived"):
            torch.compiler.precompile(
                _precompile_dynamo_scalar_identity,
                example_inputs=[(torch.randn(4), _PRECOMPILE_DYNAMO_SCALAR_IDENTITY)],
                tracer="dynamo",
                backend="eager",
            )

    def test_tracer_dynamo_rejects_nested_scalar_input_global_identity_guard(self):
        with self.assertRaisesRegex(PrecompileError, "input-derived"):
            torch.compiler.precompile(
                _precompile_dynamo_nested_scalar_identity,
                example_inputs=[(torch.randn(4), [_PRECOMPILE_DYNAMO_SCALAR_IDENTITY])],
                tracer="dynamo",
                backend="eager",
            )

    def test_tracer_dynamo_rejects_helper_scalar_input_global_identity_guard(self):
        with self.assertRaisesRegex(PrecompileError, "input-derived"):
            torch.compiler.precompile(
                _precompile_dynamo_helper_scalar_identity,
                example_inputs=[(torch.randn(4), _PRECOMPILE_DYNAMO_SCALAR_IDENTITY)],
                tracer="dynamo",
                backend="eager",
            )

    @parametrize(
        "fn",
        (
            _precompile_dynamo_called_scalar_identity,
            _precompile_dynamo_closure_scalar_identity,
            _precompile_dynamo_conditional_call_arg_scalar_identity,
            _precompile_dynamo_starargs_scalar_identity,
            _precompile_dynamo_operator_scalar_identity,
            _precompile_dynamo_operator_starargs_scalar_identity,
            _precompile_dynamo_partial_scalar_identity,
            _precompile_dynamo_keyword_partial_scalar_identity,
            _precompile_dynamo_keyword_scalar_identity,
            _precompile_dynamo_container_helper_scalar_identity,
            _precompile_dynamo_list_extend_scalar_identity,
            _precompile_dynamo_dict_update_scalar_identity,
            _precompile_dynamo_walrus_scalar_identity,
            _precompile_dynamo_swap_scalar_identity,
            _precompile_dynamo_local_import_scalar_identity,
            _precompile_dynamo_method_scalar_identity,
            _precompile_dynamo_staticmethod_scalar_identity,
            _precompile_dynamo_classmethod_scalar_identity,
            _precompile_dynamo_callable_scalar_identity,
            _precompile_dynamo_init_scalar_identity,
            _precompile_dynamo_super_scalar_identity,
            _precompile_dynamo_conditional_scalar_identity,
            _precompile_dynamo_exception_scalar_identity,
        ),
        name_fn=lambda fn: fn.__name__,
    )
    def test_tracer_dynamo_rejects_called_scalar_identity_guard(self, fn):
        extra_args = (
            (True,)
            if fn
            in (
                _precompile_dynamo_conditional_scalar_identity,
                _precompile_dynamo_conditional_call_arg_scalar_identity,
            )
            else ()
        )
        with self.assertRaisesRegex(PrecompileError, "input-derived"):
            torch.compiler.precompile(
                fn,
                example_inputs=[
                    (
                        torch.randn(4),
                        _PRECOMPILE_DYNAMO_SCALAR_IDENTITY,
                        *extra_args,
                    )
                ],
                tracer="dynamo",
                backend="eager",
            )

    def test_tracer_dynamo_rejects_indirect_scalar_identity_guard(self):
        with self.assertRaisesRegex(NotImplementedError, "indirect Python call"):
            torch.compiler.precompile(
                _precompile_dynamo_metaclass_scalar_identity,
                example_inputs=[(torch.randn(4), _PRECOMPILE_DYNAMO_SCALAR_IDENTITY)],
                tracer="dynamo",
                backend="eager",
            )

    @parametrize(
        "fn",
        (
            _precompile_dynamo_calls_input_scalar_identity,
            _precompile_dynamo_calls_input_bool_identity,
        ),
        name_fn=lambda fn: fn.__name__,
    )
    def test_tracer_dynamo_rejects_input_method_scalar_identity_guard(self, fn):
        with self.assertRaisesRegex(PrecompileError, "input-derived"):
            torch.compiler.precompile(
                fn,
                example_inputs=[
                    (
                        torch.randn(4),
                        _PrecompileDynamoInputScalarIdentity(
                            _PRECOMPILE_DYNAMO_SCALAR_IDENTITY
                        ),
                    )
                ],
                tracer="dynamo",
                backend="eager",
            )

    @parametrize(
        "fn,example",
        (
            (
                _precompile_dynamo_varargs_scalar_identity,
                torch.compiler.ExampleInput(
                    (torch.randn(4), _PRECOMPILE_DYNAMO_SCALAR_IDENTITY), {}
                ),
            ),
            (
                _precompile_dynamo_kwargs_scalar_identity,
                torch.compiler.ExampleInput(
                    (torch.randn(4),),
                    {"token": _PRECOMPILE_DYNAMO_SCALAR_IDENTITY},
                ),
            ),
        ),
        name_fn=lambda fn, example: fn.__name__,
    )
    def test_tracer_dynamo_rejects_variadic_scalar_identity_guard(self, fn, example):
        with self.assertRaisesRegex(PrecompileError, "input-derived"):
            torch.compiler.precompile(
                fn,
                example_inputs=[example],
                tracer="dynamo",
                backend="eager",
            )

    @parametrize(
        "fn",
        (
            _precompile_dynamo_contains_scalar_identity,
            _precompile_dynamo_count_scalar_identity,
            _precompile_dynamo_operator_contains_scalar_identity,
            _precompile_dynamo_container_equality_scalar_identity,
        ),
        name_fn=lambda fn: fn.__name__,
    )
    def test_tracer_dynamo_rejects_membership_scalar_identity_guard(self, fn):
        with self.assertRaisesRegex(PrecompileError, "input-derived"):
            torch.compiler.precompile(
                fn,
                example_inputs=[(torch.randn(4), [_PRECOMPILE_DYNAMO_NAN_IDENTITY])],
                tracer="dynamo",
                backend="eager",
            )

    @parametrize(
        "fn",
        (
            _precompile_dynamo_contains_scalar_identity,
            _precompile_dynamo_count_scalar_identity,
            _precompile_dynamo_operator_contains_scalar_identity,
            _precompile_dynamo_container_equality_scalar_identity,
        ),
        name_fn=lambda fn: fn.__name__,
    )
    def test_tracer_dynamo_rejects_reverse_membership_scalar_identity_guard(self, fn):
        with self.assertRaisesRegex(PrecompileError, "input-derived"):
            torch.compiler.precompile(
                fn,
                example_inputs=[(torch.randn(4), [float("nan")])],
                tracer="dynamo",
                backend="eager",
            )

    @parametrize(
        "fn,container",
        (
            (_precompile_dynamo_input_container_equality, list),
            (_precompile_dynamo_input_container_membership, list),
            (_precompile_dynamo_input_container_contains, list),
            (_precompile_dynamo_input_container_count, list),
            (_precompile_dynamo_input_container_membership, deque),
        ),
        name_fn=lambda fn, container: f"{fn.__name__}_{container.__name__}",
    )
    def test_tracer_dynamo_rejects_input_container_identity_topology(
        self, fn, container
    ):
        item = float("nan")
        args = (
            (torch.randn(4), container([item]), container([item]))
            if fn is _precompile_dynamo_input_container_equality
            else (torch.randn(4), container([item]), item)
        )
        with self.assertRaisesRegex(PrecompileError, "input-derived"):
            torch.compiler.precompile(
                fn,
                example_inputs=[args],
                tracer="dynamo",
                backend="eager",
            )

    @parametrize(
        "fn,args",
        (
            (
                _precompile_dynamo_input_container_equality,
                (torch.randn(4), [1], [1]),
            ),
            (
                _precompile_dynamo_input_container_membership,
                (torch.randn(4), [1], 1),
            ),
        ),
        name_fn=lambda fn, args: fn.__name__,
    )
    def test_tracer_dynamo_allows_reflexive_input_container_ops(self, fn, args):
        code, cache = torch.compiler.precompile(
            fn,
            example_inputs=[args],
            tracer="dynamo",
            backend="eager",
        )
        self.assertEqual(torch.compiler.precompile.load(code, cache)(*args), fn(*args))

    @parametrize(
        "fn,args",
        (
            (
                _precompile_dynamo_input_sequence_index,
                (torch.randn(4), [float("nan")], 0),
            ),
            (
                _precompile_dynamo_operator_input_sequence_index,
                (torch.randn(4), [float("nan")], 0),
            ),
            (
                _precompile_dynamo_operator_environment_sequence_index,
                (torch.randn(4), 0),
            ),
        ),
        name_fn=lambda fn, args: fn.__name__,
    )
    def test_tracer_dynamo_allows_positional_sequence_index(self, fn, args):
        code, cache = torch.compiler.precompile(
            fn,
            example_inputs=[args],
            tracer="dynamo",
            backend="eager",
        )
        result = torch.compiler.precompile.load(code, cache)(*args)
        self.assertTrue(torch.isnan(result).all())

    def test_tracer_dynamo_allows_scalar_input_matching_global_value(self):
        examples = [
            (torch.randn(4), _PRECOMPILE_DYNAMO_SCALAR_VALUE),
            (torch.randn(4), 2),
        ]
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_scalar_value,
            example_inputs=examples,
            tracer="dynamo",
            backend="eager",
        )
        loaded = torch.compiler.precompile.load(code, cache)
        for x, scale in examples:
            self.assertEqual(
                loaded(x, scale), _precompile_dynamo_scalar_value(x, scale)
            )

    def test_tracer_dynamo_allows_environment_scalar_identity(self):
        examples = [(torch.randn(4), scale) for scale in (1, 2)]
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_environment_scalar_identity,
            example_inputs=examples,
            tracer="dynamo",
            backend="eager",
        )
        loaded = torch.compiler.precompile.load(code, cache)
        for x, scale in examples:
            self.assertEqual(
                loaded(x, scale),
                _precompile_dynamo_environment_scalar_identity(x, scale),
            )

    def test_tracer_dynamo_allows_pure_super_call(self):
        x = torch.randn(4)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_calls_pure_super,
            example_inputs=[(x,)],
            tracer="dynamo",
            backend="eager",
        )
        loaded = torch.compiler.precompile.load(code, cache)
        self.assertEqual(loaded(x), _precompile_dynamo_calls_pure_super(x))

    def test_tracer_dynamo_rejects_input_container_global_identity_guard(self):
        with self.assertRaisesRegex(PrecompileError, "input-derived"):
            torch.compiler.precompile(
                _precompile_dynamo_container_identity,
                example_inputs=[(_DYNAMO_CONTAINER_IDENTITY, torch.randn(3))],
                tracer="dynamo",
                backend="eager",
            )

    def test_tracer_dynamo_rejects_input_helper_attribute_identity_guard(self):
        with self.assertRaisesRegex(PrecompileError, "input-derived"):
            torch.compiler.precompile(
                _precompile_dynamo_helper_attribute_identity,
                example_inputs=[(_DYNAMO_TENSOR_DEFAULT,)],
                tracer="dynamo",
                backend="eager",
            )

    def test_tracer_dynamo_rejects_input_helper_container_attribute_identity_guard(
        self,
    ):
        with self.assertRaisesRegex(PrecompileError, "input-derived"):
            torch.compiler.precompile(
                _precompile_dynamo_helper_container_attribute_identity,
                example_inputs=[(_DYNAMO_TENSOR_DEFAULT,)],
                tracer="dynamo",
                backend="eager",
            )

    def test_tracer_dynamo_rejects_module_attribute_global_identity_guard(self):
        module = _PrecompileDynamoModuleAttributeIdentity(_DYNAMO_TENSOR_DEFAULT)
        with self.assertRaisesRegex(PrecompileError, "input-derived"):
            torch.compiler.precompile(
                _precompile_dynamo_module_attribute_identity,
                example_inputs=[(module, torch.randn(3))],
                tracer="dynamo",
                backend="eager",
            )

    def test_tracer_dynamo_rejects_module_forward_global_identity_guard(self):
        module = _PrecompileDynamoModuleForwardGlobalIdentity()
        with self.assertRaisesRegex(PrecompileError, "input callable"):
            torch.compiler.precompile(
                _precompile_dynamo_call_identity_module,
                example_inputs=[(module, _DYNAMO_TENSOR_DEFAULT)],
                tracer="dynamo",
                backend="eager",
            )

    def test_tracer_dynamo_rejects_target_module_global_identity_guard(self):
        with self.assertRaisesRegex(PrecompileError, "input callable"):
            torch.compiler.precompile(
                _PrecompileDynamoModuleForwardGlobalIdentity(),
                example_inputs=[(_DYNAMO_TENSOR_DEFAULT,)],
                tracer="dynamo",
                backend="eager",
            )

    def test_tracer_dynamo_rejects_nested_module_forward_global_identity_guard(self):
        module = _PrecompileDynamoModuleForwardGlobalIdentity()
        with self.assertRaisesRegex(PrecompileError, "input callable"):
            torch.compiler.precompile(
                _precompile_dynamo_call_nested_identity_module,
                example_inputs=[([module], _DYNAMO_TENSOR_DEFAULT)],
                tracer="dynamo",
                backend="eager",
            )

    @parametrize(
        "fn",
        (
            _precompile_dynamo_call_identity_module_helper,
            _precompile_dynamo_call_identity_module_helper_nested,
        ),
    )
    def test_tracer_dynamo_rejects_module_method_global_identity_guard(self, fn):
        module = _PrecompileDynamoModuleMethodGlobalIdentity()
        with self.assertRaisesRegex(PrecompileError, "input callable"):
            torch.compiler.precompile(
                fn,
                example_inputs=[(module, _DYNAMO_TENSOR_DEFAULT)],
                tracer="dynamo",
                backend="eager",
            )

    def test_tracer_dynamo_rejects_dynamic_module_method_dispatch(self):
        module = _PrecompileDynamoModuleMethodGlobalIdentity()
        with self.assertRaisesRegex(PrecompileError, "input-derived.*methodcaller"):
            torch.compiler.precompile(
                _precompile_dynamo_call_identity_module_methodcaller,
                example_inputs=[(module, _DYNAMO_TENSOR_DEFAULT)],
                tracer="dynamo",
                backend="eager",
            )

    def test_tracer_dynamo_rejects_module_implicit_dispatch(self):
        module = _PrecompileDynamoModuleGetitemGlobalIdentity()
        with self.assertRaisesRegex(PrecompileError, "input callable"):
            torch.compiler.precompile(
                _precompile_dynamo_getitem_identity_module,
                example_inputs=[(module, _DYNAMO_TENSOR_DEFAULT)],
                tracer="dynamo",
                backend="eager",
            )

    def test_tracer_dynamo_rejects_imported_module_method_identity_guard(self):
        module = _PrecompileDynamoModuleMethodGlobalIdentity()
        with self.assertRaisesRegex(PrecompileError, "input callable"):
            torch.compiler.precompile(
                _precompile_dynamo_call_external_module_method,
                example_inputs=[(module, _DYNAMO_TENSOR_DEFAULT)],
                tracer="dynamo",
                backend="eager",
            )

    @parametrize(
        "fn",
        (
            _precompile_dynamo_stdlib_module_dynamic_identity,
            _precompile_dynamo_stdlib_module_attrgetter_identity,
            _precompile_dynamo_stdlib_module_getattr_identity,
            _precompile_dynamo_stdlib_module_bound_getattr_identity,
        ),
    )
    def test_tracer_dynamo_rejects_dynamic_stdlib_module_alias(self, fn):
        missing = object()
        previous = getattr(math, "_precompile_token", missing)
        math._precompile_token = _DYNAMO_TENSOR_DEFAULT
        try:
            with self.assertRaisesRegex(
                (PrecompileError, NotImplementedError),
                "input-derived|dynamic global access",
            ):
                torch.compiler.precompile(
                    fn,
                    example_inputs=[(_DYNAMO_TENSOR_DEFAULT,)],
                    tracer="dynamo",
                    backend="eager",
                )
        finally:
            if previous is missing:
                del math._precompile_token
            else:
                math._precompile_token = previous

    def test_tracer_dynamo_rejects_dynamic_attribute_alias(self):
        with self.assertRaisesRegex(PrecompileError, "input-derived.*getattr"):
            torch.compiler.precompile(
                _precompile_dynamo_getattr_identity,
                example_inputs=[(_DYNAMO_TENSOR_DEFAULT,)],
                tracer="dynamo",
                backend="eager",
            )

    def test_tracer_dynamo_rejects_imported_dynamic_attribute_alias(self):
        with self.assertRaisesRegex(PrecompileError, "input-derived"):
            torch.compiler.precompile(
                _precompile_dynamo_external_getattr_identity,
                example_inputs=[(_DYNAMO_TENSOR_DEFAULT,)],
                tracer="dynamo",
                backend="eager",
            )

    @parametrize(
        "fn",
        (
            _precompile_dynamo_descriptor_identity,
            _precompile_dynamo_custom_descriptor_identity,
            _precompile_dynamo_slotted_identity,
            _precompile_dynamo_dynamic_identity,
            _precompile_dynamo_getattribute_identity,
            _precompile_dynamo_module_getattr_identity,
            _precompile_dynamo_callable_identity,
            _precompile_dynamo_deque_identity,
            _precompile_dynamo_context_identity,
            _precompile_dynamo_weak_proxy_identity,
            _precompile_dynamo_type_identity,
            _precompile_dynamo_nested_descriptor_identity,
            _precompile_dynamo_weakref_identity,
        ),
        name_fn=lambda fn: fn.__name__,
    )
    def test_tracer_dynamo_rejects_input_descriptor_identity_guard(self, fn):
        with self.assertRaisesRegex(PrecompileError, "input-derived"):
            torch.compiler.precompile(
                fn,
                example_inputs=[(_DYNAMO_TENSOR_DEFAULT,)],
                tracer="dynamo",
                backend="eager",
            )

    @parametrize(
        "fn,obj",
        (
            (_precompile_dynamo_input_getitem, _PrecompileDynamoInputGetitem()),
            (
                _precompile_dynamo_nested_input_getitem,
                _PrecompileDynamoInputObjectBox(),
            ),
            (
                _precompile_dynamo_input_getitem,
                _PrecompileDynamoInputStateGlobalAlias(_DYNAMO_TENSOR_DEFAULT),
            ),
            (_precompile_dynamo_input_getitem, _PrecompileDynamoInputList([0])),
        ),
    )
    def test_tracer_dynamo_rejects_input_object_behavior_alias(self, fn, obj):
        with self.assertRaisesRegex(PrecompileError, "input callable"):
            torch.compiler.precompile(
                fn,
                example_inputs=[(obj, _DYNAMO_TENSOR_DEFAULT)],
                tracer="dynamo",
                backend="eager",
            )

    def test_tracer_dynamo_ignores_unrelated_global_attributes(self):
        token = _DYNAMO_IDENTITY_DESCRIPTOR.unused
        x = torch.randn(3)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_unrelated_attribute,
            example_inputs=[(token, x)],
            tracer="dynamo",
            backend="eager",
        )
        loaded = torch.compiler.precompile.load(code, cache)
        other = torch.randn(3)
        self.assertEqual(
            loaded(other, x), _precompile_dynamo_unrelated_attribute(other, x)
        )

    def test_tracer_dynamo_rejects_global_tensor(self):
        with self.assertRaisesRegex(PrecompileError, "tensor-valued Python globals"):
            torch.compiler.precompile(
                _precompile_dynamo_global_tensor,
                example_inputs=[(torch.randn(3),)],
                tracer="dynamo",
                backend="eager",
            )

    @parametrize(
        "fn,args",
        (
            (_precompile_dynamo_class_tensor_state, (torch.randn(3),)),
            (
                _precompile_dynamo_input_method_tensor_default,
                (torch.randn(3), _PrecompileDynamoTensorDefaultMethod()),
            ),
        ),
        name_fn=lambda fn, args: fn.__name__,
    )
    def test_tracer_dynamo_rejects_indirect_tensor_state(self, fn, args):
        with self.assertRaisesRegex(PrecompileError, "tensor-valued"):
            torch.compiler.precompile(
                fn,
                example_inputs=[args],
                tracer="dynamo",
                backend="eager",
            )

    def test_tracer_dynamo_allows_unused_class_tensor_state(self):
        x = torch.randn(3)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_unused_class_tensor_state,
            example_inputs=[(x,)],
            tracer="dynamo",
            backend="eager",
        )
        self.assertEqual(
            torch.compiler.precompile.load(code, cache)(x),
            _precompile_dynamo_unused_class_tensor_state(x),
        )

    def test_tracer_dynamo_rejects_local_import_tensor_state(self):
        module_name = "_precompile_local_tensor_module"
        with tempfile.TemporaryDirectory() as directory:
            with open(
                os.path.join(directory, f"{module_name}.py"),
                "w",
                encoding="utf-8",
            ) as file:
                file.write(
                    textwrap.dedent(
                        """
                        import torch

                        TENSOR = torch.randn(3)

                        def helper(x, bias=TENSOR):
                            return x + bias
                        """
                    )
                )
            sys.path.insert(0, directory)
            importlib.invalidate_caches()
            try:
                for fn in (
                    _precompile_dynamo_local_import_tensor,
                    _precompile_dynamo_local_import_tensor_default,
                ):
                    with self.assertRaisesRegex(PrecompileError, "tensor-valued"):
                        torch.compiler.precompile(
                            fn,
                            example_inputs=[(torch.randn(3),)],
                            tracer="dynamo",
                            backend="eager",
                        )
            finally:
                sys.path.remove(directory)
                sys.modules.pop(module_name, None)

    def test_tracer_dynamo_rejects_indirect_global_tensor(self):
        module_name = "_precompile_indirect_global_tensor"
        source = textwrap.dedent(
            """
            import types

            import torch

            BOX = types.SimpleNamespace(value=torch.randn(3))

            def helper(x):
                return x + BOX.value

            def fn(x):
                return helper(x)
            """
        )
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, f"{module_name}.py")
            with open(path, "w", encoding="utf-8") as file:
                file.write(source)
            sys.path.insert(0, directory)
            importlib.invalidate_caches()
            try:
                fixture = importlib.import_module(module_name)
                with self.assertRaisesRegex(
                    PrecompileError, "tensor-valued Python globals"
                ):
                    torch.compiler.precompile(
                        fixture.fn,
                        example_inputs=[(torch.randn(3),)],
                        tracer="dynamo",
                        backend="eager",
                    )
            finally:
                sys.path.remove(directory)
                sys.modules.pop(module_name, None)

    def test_tracer_dynamo_rejects_imported_input_environment_identity(self):
        token = torch.randn(3)
        torch._precompile_identity_token = token
        module_name = "_precompile_identity_module"
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, f"{module_name}.py")
            with open(path, "w", encoding="utf-8") as file:
                file.write(
                    textwrap.dedent(
                        """
                        import types

                        import torch

                        TOKEN = torch._precompile_identity_token

                        class Holder:
                            def __init__(self, value):
                                self.value = value

                            def __call__(self):
                                return self.value

                        CALLABLE = Holder(TOKEN)
                        SUBMODULE = types.ModuleType("_precompile_identity_submodule")
                        SUBMODULE.CALLABLE = CALLABLE
                        USED = 2

                        def __getattr__(name):
                            if name == "DYNAMIC":
                                return TOKEN
                            raise AttributeError(name)

                        def helper(module, x):
                            return x + 1 if x is module._precompile_identity_token else x - 1

                        def global_module_fn(x):
                            return helper(torch, x)

                        MODULES = [torch]

                        def global_module_container_fn(x):
                            return helper(MODULES[0], x)

                        def class_helper(cls, x):
                            return x + 1 if x is cls._precompile_identity_token else x - 1

                        def global_class_fn(x):
                            return class_helper(torch.nn.Module, x)
                        """
                    )
                )
            package_name = "_precompile_identity_package"
            package = os.path.join(directory, package_name)
            os.makedirs(os.path.join(package, "sub"))
            for init in (package, os.path.join(package, "sub")):
                with open(os.path.join(init, "__init__.py"), "w", encoding="utf-8"):
                    pass
            with open(
                os.path.join(package, "sub", "mod.py"), "w", encoding="utf-8"
            ) as file:
                file.write("import torch\nTOKEN = torch._precompile_identity_token\n")
            sys.path.insert(0, directory)
            importlib.invalidate_caches()
            torch.nn.Module._precompile_identity_token = token
            try:
                fixture = importlib.import_module(module_name)
                for fn in (
                    _precompile_dynamo_imported_module_identity,
                    _precompile_dynamo_imported_callable_identity,
                    _precompile_dynamo_imported_nested_callable_identity,
                    _precompile_dynamo_imported_dynamic_identity,
                ):
                    with self.assertRaisesRegex(
                        PrecompileError, "input-derived identity"
                    ):
                        torch.compiler.precompile(
                            fn,
                            example_inputs=[(token,)],
                            tracer="dynamo",
                            backend="eager",
                        )

                for fn in (
                    fixture.global_module_fn,
                    fixture.global_module_container_fn,
                    fixture.global_class_fn,
                ):
                    with self.assertRaisesRegex(
                        PrecompileError, "input-derived|module object"
                    ):
                        torch.compiler.precompile(
                            fn,
                            example_inputs=[(token,)],
                            tracer="dynamo",
                            backend="eager",
                        )

                for fn in (
                    _precompile_dynamo_imported_module_alias_identity,
                    _precompile_dynamo_dotted_import_identity,
                ):
                    with self.assertRaisesRegex(
                        PrecompileError,
                        "input-derived identity|locally imported module",
                    ):
                        torch.compiler.precompile(
                            fn,
                            example_inputs=[(token,)],
                            tracer="dynamo",
                            backend="eager",
                        )

                for fn in (
                    _precompile_dynamo_imported_unrelated_attribute,
                    _precompile_dynamo_import_shadowed_in_nested_scope,
                ):
                    code, cache = torch.compiler.precompile(
                        fn,
                        example_inputs=[(token,)],
                        tracer="dynamo",
                        backend="eager",
                    )
                    loaded = torch.compiler.precompile.load(code, cache)
                    x = torch.randn(3)
                    self.assertEqual(loaded(x), fn(x))
            finally:
                sys.path.remove(directory)
                for imported in (
                    module_name,
                    package_name,
                    f"{package_name}.sub",
                    f"{package_name}.sub.mod",
                ):
                    sys.modules.pop(imported, None)
                del torch.nn.Module._precompile_identity_token
                del torch._precompile_identity_token

    def test_tracer_dynamo_ignores_unused_function_metadata_tensor(self):
        _precompile_dynamo_metadata_helper.unused = torch.randn(3)
        try:
            code, cache = torch.compiler.precompile(
                _precompile_dynamo_calls_metadata_helper,
                example_inputs=[(torch.randn(3),)],
                tracer="dynamo",
                backend="eager",
            )
            for _, loaded in _default_and_inlined_loaders(code, cache, "eager"):
                x = torch.randn(3)
                self.assertEqual(loaded(x), x + 1)
        finally:
            del _precompile_dynamo_metadata_helper.unused

    def test_tracer_dynamo_rejects_library_helper_environment_aliases(self):
        token = torch.randn(3)
        module = types.ModuleType("torch._precompile_reviewlib")
        module.token = token
        exec(
            textwrap.dedent(
                """
                import functools
                import torch

                TOKEN = token

                class Holder:
                    def reveal(self):
                        return TOKEN

                HOLDER = Holder()

                def helper(obj, x):
                    return x + 1 if x is obj.reveal() else x - 1

                def inner(x):
                    return x + 1 if x is TOKEN else x - 1

                def outer(x):
                    return inner(x)

                FUNCTIONS = [inner]

                def container_outer(x):
                    return FUNCTIONS[0](x)

                PARTIAL = functools.partial(inner)

                def partial_outer(x):
                    return PARTIAL(x)

                class Callable:
                    def __call__(self, x):
                        return inner(x)

                CALLABLE = Callable()

                def callable_outer(x):
                    return CALLABLE(x)

                class PropertyHolder:
                    @property
                    def token(self):
                        return TOKEN

                def bound_helper(obj, x):
                    return x + 1 if x is obj.token else x - 1

                BOUND_PARTIAL = functools.partial(bound_helper, PropertyHolder())

                def bound_partial_outer(x):
                    return BOUND_PARTIAL(x)

                def keyword_bound_helper(x, obj=None):
                    return x + 1 if x is obj.token else x - 1

                KEYWORD_PARTIAL = functools.partial(
                    keyword_bound_helper, obj=PropertyHolder()
                )

                def keyword_partial_outer(x):
                    return KEYWORD_PARTIAL(x)

                class PropertyCallable:
                    @property
                    def token(self):
                        return TOKEN

                    def __call__(self, x):
                        return x + 1 if x is self.token else x - 1

                PROPERTY_CALLABLE = PropertyCallable()

                def property_callable_outer(x):
                    return PROPERTY_CALLABLE(x)

                class Factory:
                    def __new__(cls, x):
                        return x + 1 if x is TOKEN_BOX[0] else x - 1

                TOKEN_BOX = [TOKEN]

                class DirectHolder:
                    def __init__(self, value):
                        self.value = value

                    def reveal(self):
                        return self.value

                    @property
                    def token(self):
                        return TOKEN_BOX[0]

                DIRECT_HOLDER = DirectHolder(TOKEN)

                def direct_method_outer(x):
                    return x + 1 if x is DIRECT_HOLDER.reveal() else x - 1

                def direct_property_outer(x):
                    return x + 1 if x is DIRECT_HOLDER.token else x - 1

                def factory_outer(x):
                    return Factory(x)

                def module_helper(mod, x):
                    return x + 1 if x is mod._precompile_review_token else x - 1

                def module_outer(x):
                    return module_helper(torch, x)

                def __getattr__(name):
                    if name == "DYNAMIC":
                        return TOKEN
                    raise AttributeError(name)

                def dynamic_module_outer(x):
                    return x + 1 if x is SELF.DYNAMIC else x - 1
                """
            ),
            module.__dict__,
        )
        module.SELF = module
        torch._precompile_reviewlib = module
        torch._precompile_review_token = token
        sys.modules[module.__name__] = module
        try:
            for fn in (
                _precompile_dynamo_library_object_identity,
                _precompile_dynamo_library_nested_helper_identity,
                _precompile_dynamo_library_container_helper_identity,
                _precompile_dynamo_library_partial_helper_identity,
                _precompile_dynamo_library_callable_helper_identity,
                _precompile_dynamo_library_bound_partial_identity,
                _precompile_dynamo_library_keyword_partial_identity,
                _precompile_dynamo_library_property_callable_identity,
                _precompile_dynamo_library_factory_identity,
                _precompile_dynamo_library_dynamic_module_identity,
                _precompile_dynamo_library_direct_method_identity,
                _precompile_dynamo_library_direct_property_identity,
            ):
                with self.subTest(fn=fn.__name__):
                    with self.assertRaisesRegex(PrecompileError, "input-derived"):
                        torch.compiler.precompile(
                            fn,
                            example_inputs=[(token,)],
                            tracer="dynamo",
                            backend="eager",
                        )
            with self.assertRaisesRegex(PrecompileError, "input-derived"):
                torch.compiler.precompile(
                    _precompile_dynamo_library_module_value_identity,
                    example_inputs=[(token,)],
                    tracer="dynamo",
                    backend="eager",
                )
        finally:
            sys.modules.pop(module.__name__, None)
            del torch._precompile_review_token
            del torch._precompile_reviewlib

    def test_tracer_dynamo_rejects_library_helper_late_import_alias(self):
        token = torch.randn(3)
        module = types.ModuleType("torch._precompile_latelib")
        inner_module = types.ModuleType("torch._precompile_lateinner")
        exec(
            textwrap.dedent(
                """
                def inner(x):
                    import _precompile_late_env

                    return (
                        x + 1
                        if x is _precompile_late_env.HOLDER.reveal()
                        else x - 1
                    )
                """
            ),
            inner_module.__dict__,
        )
        module.inner = inner_module.inner
        exec(
            textwrap.dedent(
                """
                def outer(x):
                    return inner(x)
                """
            ),
            module.__dict__,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "_precompile_late_env.py")
            with open(path, "w", encoding="utf-8") as file:
                file.write(
                    textwrap.dedent(
                        """
                        import torch

                        class Holder:
                            def __init__(self, value):
                                self.value = value

                            def reveal(self):
                                return self.value

                        HOLDER = Holder(torch._precompile_late_token)
                        """
                    )
                )
            sys.path.insert(0, directory)
            importlib.invalidate_caches()
            torch._precompile_latelib = module
            torch._precompile_late_token = token
            sys.modules[module.__name__] = module
            sys.modules[inner_module.__name__] = inner_module
            try:
                with self.assertRaisesRegex(
                    PrecompileError,
                    "input-derived identity|locally imported module",
                ):
                    torch.compiler.precompile(
                        _precompile_dynamo_library_late_import_identity,
                        example_inputs=[(token,)],
                        tracer="dynamo",
                        backend="eager",
                    )
            finally:
                sys.path.remove(directory)
                sys.modules.pop("_precompile_late_env", None)
                sys.modules.pop(module.__name__, None)
                sys.modules.pop(inner_module.__name__, None)
                del torch._precompile_late_token
                del torch._precompile_latelib

    def test_tracer_dynamo_rejects_unserializable_input_contract(self):
        inp = _UnserializableCtxInput(torch.randn(3), torch.randn(3))
        with self.assertRaisesRegex(PrecompileError, "serialize the input structure"):
            torch.compiler.precompile(
                lambda value: value.a + value.b,
                example_inputs=[(inp,)],
                tracer="dynamo",
                backend="eager",
            )

        code, cache = torch.compiler.precompile(
            lambda values: values[0] + values[1],
            example_inputs=[([torch.randn(3), torch.randn(3)],)],
            tracer="dynamo",
            backend="eager",
        )
        loaded = torch.compiler.precompile.load(code, cache)
        with self.assertRaisesRegex(
            PrecompileError, "runtime input structure cannot be serialized"
        ):
            loaded(inp)

    def test_tracer_dynamo_rejects_tensor_default(self):
        with self.assertRaisesRegex(PrecompileError, "tensor-valued function defaults"):
            torch.compiler.precompile(
                _precompile_dynamo_tensor_default,
                example_inputs=[(torch.randn(3),)],
                tracer="dynamo",
                backend="eager",
            )

    def test_tracer_dynamo_rejects_tensor_in_object_default(self):
        with self.assertRaisesRegex(PrecompileError, "tensor-valued function defaults"):
            torch.compiler.precompile(
                _precompile_dynamo_object_default,
                example_inputs=[(torch.randn(3),)],
                tracer="dynamo",
                backend="eager",
            )

    def test_tracer_dynamo_rejects_tensor_in_builtin_subclass_default(self):
        with self.assertRaisesRegex(PrecompileError, "non-literal function defaults"):
            torch.compiler.precompile(
                _precompile_dynamo_tuple_subclass_default,
                example_inputs=[(torch.randn(3),)],
                tracer="dynamo",
                backend="eager",
            )

    def test_tracer_dynamo_load_does_not_copy_unrelated_module_globals(self):
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_dynamic,
            example_inputs=[(torch.randn(4),)],
            tracer="dynamo",
            backend="eager",
        )
        loaded = torch.compiler.precompile.load(code, cache)
        self.assertNotIn("_GLOBAL_TENSOR", loaded._loaded_forward.__globals__)

    def test_tracer_dynamo_python_minor_mismatch_uses_public_error(self):
        from torch._precompile import _make_inlined_forward

        code, _ = torch.compiler.precompile(
            _precompile_dynamo_dynamic,
            example_inputs=[(torch.randn(4),)],
            tracer="dynamo",
            backend="eager",
        )
        version = tuple(sys.version_info[:2])
        incompatible = code.replace(
            f"_DYNAMO_PYTHON_VERSION = {version!r}",
            "_DYNAMO_PYTHON_VERSION = (0, 0)",
        )
        with self.assertRaisesRegex(PrecompileError, "produced on Python"):
            _make_inlined_forward(incompatible)

    def test_tracer_dynamo_preserves_key_order_guard_dependencies(self):
        forward = {"a": 2.0, "b": 3.0}
        reverse = {"b": 3.0, "a": 2.0}
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_dict_order,
            example_inputs=[(torch.ones(4), forward), (torch.ones(4), reverse)],
            tracer="dynamo",
            backend="eager",
        )

        summaries = _dynamo_serialized_guard_summary(code)
        self.assertTrue(any(key_order for _, _, key_order, _ in summaries))
        loaded = torch.compiler.precompile.load(code, cache)
        x = torch.ones(4)
        self.assertEqual(loaded(x, forward), _precompile_dynamo_dict_order(x, forward))
        self.assertEqual(loaded(x, reverse), _precompile_dynamo_dict_order(x, reverse))

    def test_tracer_dynamo_prunes_unguarded_user_object_attributes(self):
        pipeline = _PrecompileDynamoPipeline(torch.nn.Linear(8, 8))
        x = torch.randn(4, 8)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_pipeline,
            example_inputs=[(pipeline, x)],
            tracer="dynamo",
            backend="eager",
        )
        loaded = torch.compiler.precompile.load(code, cache)
        self.assertEqual(loaded(pipeline, x), _precompile_dynamo_pipeline(pipeline, x))

    def test_guard_serialization_prunes_loaded_precompile_handle(self):
        import threading

        from torch._dynamo.guards import _Missing, GuardsStatePickler

        x = torch.randn(4)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_unreachable_caller,
            example_inputs=[(x,)],
            tracer="dynamo",
            backend="eager",
        )
        loaded = torch.compiler.precompile.load(code, cache)
        inner = loaded._loaded_forward
        if inner is None:
            raise AssertionError("expected an installed artifact")
        holder = _PrecompileLockHolder()
        holder.scale = 2.0

        try:
            for handle in (loaded, inner):
                holder.installed = handle
                buf = io.BytesIO()
                GuardsStatePickler({}, {}, {}, buf).dump(holder)
                restored = pickle.loads(buf.getvalue())
                self.assertIsInstance(restored.installed, _Missing)
                self.assertEqual(restored.scale, 2.0)

                with self.assertRaisesRegex(
                    PackageError, "guard directly references a precompile handle"
                ):
                    GuardsStatePickler({id(handle): handle}, {}, {}, io.BytesIO()).dump(
                        handle
                    )

            holder.installed = threading.RLock()
            with self.assertRaisesRegex(TypeError, "cannot pickle.*RLock"):
                GuardsStatePickler({}, {}, {}, io.BytesIO()).dump(holder)
        finally:
            loaded.unload()

    @parametrize("kind", ("dtype", "int", "str", "device"))
    def test_tracer_dynamo_unguarded_interned_attribute(self, kind):
        extra = {
            "dtype": torch.float32,
            "int": 8,
            "str": "cuda",
            "device": torch.device("cpu"),
        }[kind]
        model = _PrecompileDynamoUnguardedAttribute(extra).eval()
        x = torch.randn(4, 8)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_call_module,
            example_inputs=[(model, x)],
            tracer="dynamo",
            backend="inductor",
        )
        loaded = torch.compiler.precompile.load(code, cache)
        self.assertEqual(loaded(model, x), model(x))

    def test_guard_serialization_error_names_the_value_path(self):
        from torch._dynamo.guards import _offending_value_path

        class Scope:
            pass

        holder = Scope()
        holder.deep = Scope()
        holder.deep.iterator = (index for index in range(3))
        state = Scope()
        state.output_graph = Scope()
        state.output_graph.local_scope = {"pipeline": holder}
        state.output_graph.global_scope = {}
        path = _offending_value_path(
            state, TypeError("cannot pickle 'generator' object")
        )
        self.assertIn("local_scope['pipeline'].deep.iterator", path)

    def test_tracer_invalid_raises(self):
        a, b = torch.randn(4, 4), torch.randn(4, 4)
        with self.assertRaisesRegex(ValueError, "tracer must be 'make_fx' or 'dynamo'"):
            torch.compiler.precompile(
                lambda x, y: x + y, example_inputs=[(a, b)], tracer="nope"
            )

    def test_backend_default_is_inductor(self):
        # The default lowers through Inductor: the generated code inlines the Inductor
        # output module. Use a graph_partition-agnostic marker (the ``call = runner.call``
        # form is only emitted when config.graph_partition is on, which is off in fbcode).
        m = torch.nn.Sequential(torch.nn.Linear(4, 3)).eval()
        x = torch.randn(5, 4)
        code, _ = torch.compiler.precompile(
            lambda model, x: model(x), example_inputs=[(m, x)]
        )
        self.assertIn("Inductor output code", code)

    def test_inductor_graph_partition_off(self):
        # graph_partition defaults off in fbcode; the Inductor output module then exposes
        # a top-level ``def call(args):`` instead of ``call = runner.call``. The source
        # extractor must still find it (regression: it previously matched only the
        # runner.call form, so torch.compiler.precompile crashed in fbcode).
        import torch._inductor.config as ind_config

        m = torch.nn.Sequential(torch.nn.Linear(4, 3)).eval()
        x = torch.randn(5, 4)
        with ind_config.patch(graph_partition=False):
            code, cache = torch.compiler.precompile(
                lambda model, xx: model(xx), example_inputs=[(m, x)]
            )
            self.assertNotIn("call = runner.call", code)  # non-partition form
            f_c = torch.compiler.precompile.load(code, cache)
            self.assertEqual(f_c(m, x), m(x))

    def test_inductor_caches_disabled(self):
        # Source is captured off codegen (GraphLowering.save_output_code), not the cache
        # bundle, so precompile must work even when caching is disabled -- producing a
        # runnable python_code with an empty cache, not a misleading "non-cacheable HOP"
        # error. Covers force_disable_caches and fx_graph_cache=False.
        import torch._inductor.config as ind_config

        m = torch.nn.Sequential(torch.nn.Linear(4, 3)).eval()
        x = torch.randn(5, 4)
        for patch in (
            {"force_disable_caches": True},
            {"fx_graph_cache": False},
        ):
            with ind_config.patch(**patch):
                code, cache = torch.compiler.precompile(
                    lambda model, xx: model(xx), example_inputs=[(m, x)]
                )
                # No saveable artifact when caches are off; the cache is empty.
                blob = torch.load(io.BytesIO(cache), weights_only=True)
                self.assertIsNone(blob["artifact"], patch)
                # python_code still runs standalone (JITs from inlined source).
                ns = {"__name__": "_a"}
                exec(compile(code, "<a>", "exec"), ns)
                self.assertEqual(ns["forward"](m, x), m(x), patch)
                # ...and load() falls back to the inlined path.
                self.assertEqual(
                    torch.compiler.precompile.load(code, cache)(m, x), m(x), patch
                )

    def test_inductor_cpp_wrapper_pinned_off(self):
        # cpp_wrapper would make Inductor emit a C++ ``call`` (no python module); a
        # python artifact cannot come from it, so compile_to_python pins it off. With
        # cpp_wrapper=True ambient, precompile must still produce a working python artifact.
        import torch._inductor.config as ind_config

        m = torch.nn.Sequential(torch.nn.Linear(4, 3)).eval()
        x = torch.randn(5, 4)
        with ind_config.patch(cpp_wrapper=True):
            code, cache = torch.compiler.precompile(
                lambda model, xx: model(xx), example_inputs=[(m, x)]
            )
            f_c = torch.compiler.precompile.load(code, cache)
            self.assertEqual(f_c(m, x), m(x))

    def test_example_grad_restored_when_fn_raises(self):
        # If fn runs a backward then raises during the make_fx trace, the example
        # model's .grad must be restored (the snapshot/restore is in a finally), not
        # left clobbered -- precompile does not mutate the example model's grads.
        torch.manual_seed(0)
        m = torch.nn.Linear(4, 3)
        x = torch.randn(5, 4)
        for p in m.parameters():
            self.assertIsNone(p.grad)

        def boom(model, xx):
            model(xx).sum().backward()  # populates .grad on the lifted example params
            raise ValueError("boom")

        with self.assertRaisesRegex(ValueError, "boom"):
            torch.compiler.precompile(boom, example_inputs=[(m, x)])
        for n, p in m.named_parameters():
            self.assertIsNone(p.grad, f"{n}: example .grad must be restored on failure")

    def test_unbacked_capture_with_preexisting_grad(self):
        # Regression: in the mark_unbacked path the example params are fakeified BEFORE
        # the grad clear. A model with a pre-existing .grad (the warmup-step-then-
        # precompile flow) plus a backward in fn must still capture -- the clear must
        # precede fakeify so the fakes inherit no grad -- and the real .grad is restored.
        from torch._dynamo.decorators import mark_unbacked

        torch.manual_seed(0)
        m = torch.nn.Linear(4, 3)
        x = torch.randn(8, 4)
        m(x).sum().backward()  # warmup: populate .grad before precompile
        saved = {n: p.grad.clone() for n, p in m.named_parameters()}
        mark_unbacked(x, 0)
        code, _ = torch.compiler.precompile(
            lambda mm, t: mm(t).sum().backward(), example_inputs=[(m, x)]
        )
        self.assertIn("USER_INPUT_SHAPES = [(None, 4)]", code)  # dim 0 is dynamic
        for n, p in m.named_parameters():
            self.assertEqual(p.grad, saved[n])  # warmup grad restored, not clobbered

    def test_backend_eager_no_inductor_lowering(self):
        # backend="eager" skips Inductor: the generated code has no inductor ``call``
        # entry point, and instead embeds the readable captured ATen graph and the
        # eager driver. The eager backend has no kernels to accelerate, so the cache
        # is empty -- python_code is the whole artifact.
        m = torch.nn.Sequential(torch.nn.Linear(4, 3)).eval()
        x = torch.randn(5, 4)
        code, cache = torch.compiler.precompile(
            lambda model, x: model(x), example_inputs=[(m, x)], backend="eager"
        )
        self.assertIn('backend="eager"', code)
        self.assertNotIn("call = runner.call", code)
        self.assertIn("torch.ops.aten", code)  # readable captured graph

        # The cache holds no artifact (eager caches nothing); the backend tag lives in
        # python_code (the single source of truth). The envelope still carries the
        # integrity tag, with backend='eager' to match python_code.
        self.assertIn("BACKEND = 'eager'", code)
        from torch._precompile import _CACHE_FORMAT, _CACHE_VERSION

        blob = torch.load(io.BytesIO(cache), weights_only=False)
        self.assertEqual(
            set(blob), {"artifact", "format", "version", "backend", "code_hash"}
        )
        self.assertIsNone(blob["artifact"])  # eager has no compiled blob to bundle
        self.assertEqual(blob["format"], _CACHE_FORMAT)
        self.assertEqual(blob["version"], _CACHE_VERSION)
        self.assertEqual(blob["backend"], "eager")

    def test_backend_eager_self_contained_exec(self):
        # The eager python_code execs standalone with NO cache (the captured graph
        # is inlined) and runs, matching eager.
        m = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU()).eval()
        x = torch.randn(5, 4)
        code, _cache = torch.compiler.precompile(
            lambda model, x: model(x), example_inputs=[(m, x)], backend="eager"
        )

        ns = {"__name__": "_eager"}
        exec(compile(code, "<eager>", "exec"), ns)
        self.assertEqual(ns["forward"](m, x), m(x))

    def test_preexisting_param_grad_capture_succeeds(self):
        # Precompiling a backward fn on a model whose params already carry a .grad (the
        # common warmup-step-then-precompile flow) must capture cleanly: the pre-existing
        # grad must be cleared before tracing, not baked as a constant (invariant 1).
        # Eager simply accumulates a second backward, so precompile must too.
        torch.manual_seed(0)
        m = torch.nn.Linear(4, 3)
        x = torch.randn(5, 4)
        m(x).sum().backward()  # warmup: params now carry a .grad
        self.assertIsNotNone(m.weight.grad)
        grad_before = m.weight.grad.clone()

        code, cache = torch.compiler.precompile(
            lambda model, xx: model(xx).sum().backward(), example_inputs=[(m, x)]
        )
        # Capture must not mutate the example model's pre-existing grad (restored).
        self.assertEqual(m.weight.grad, grad_before)

        run = torch.nn.Linear(4, 3)
        run.load_state_dict(m.state_dict())
        torch.compiler.precompile.load(code, cache)(run, x)  # run.grad starts None
        ref = torch.nn.Linear(4, 3)
        ref.load_state_dict(m.state_dict())
        ref(x).sum().backward()
        for (n, p), (_, rp) in zip(run.named_parameters(), ref.named_parameters()):
            self.assertEqual(p.grad, rp.grad, n)

    def test_nontensor_output_inductor_clean_error(self):
        # A non-tensor python value (float, complex, str, ...) in fn's output trips the
        # inductor backend's codegen assert; surface a clear PrecompileError (not a raw
        # InductorError) pointing to backend="eager". int / None outputs lower fine, and
        # eager handles the non-tensor value.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(2, 4)
        for bad in (3.14, 2 + 3j, "hi"):
            with self.assertRaisesRegex(PrecompileError, "non-tensor Python value"):
                torch.compiler.precompile(
                    lambda model, t, b=bad: (model(t), b), example_inputs=[(m, x)]
                )
        for extra in (7, None):
            code, cache = torch.compiler.precompile(
                lambda model, t, e=extra: (model(t), e), example_inputs=[(m, x)]
            )
            self.assertEqual(
                torch.compiler.precompile.load(code, cache)(m, x)[1], extra
            )
        ecode, ecache = torch.compiler.precompile(
            lambda model, t: (model(t), 3.14), example_inputs=[(m, x)], backend="eager"
        )
        self.assertEqual(torch.compiler.precompile.load(ecode, ecache)(m, x)[1], 3.14)

    def test_input_layout_mismatch_inductor_clean_error(self):
        # The inductor backend bakes each input's stride / memory format (invariant 6);
        # a same-shape input with a different layout must raise a clear PrecompileError
        # (not a raw assert_size_stride AssertionError) on BOTH the cached and inlined
        # paths. The eager backend is layout-flexible and accepts it.
        m = torch.nn.Linear(8, 5).eval()
        xex = torch.randn(
            8, 6
        ).t()  # example: shape (6, 8), non-contiguous stride (1, 6)
        self.assertFalse(xex.is_contiguous())
        code, cache = torch.compiler.precompile(
            lambda model, t: model(t), example_inputs=[(m, xex)]
        )
        self.assertIn("assert_size_stride", code)  # the layout guard we convert
        xrt = torch.randn(6, 8)  # same shape, contiguous -> different layout
        with self.assertRaisesRegex(PrecompileError, "memory format"):
            torch.compiler.precompile.load(code, cache)(m, xrt)  # cached path
        with self.assertRaisesRegex(PrecompileError, "memory format"):
            torch.compiler.precompile.load(code, _strip_artifact(cache))(
                m, xrt
            )  # inlined path
        # A matching (same-stride) input still works on inductor.
        xmatch = torch.randn(8, 6).t()
        self.assertEqual(
            torch.compiler.precompile.load(code, cache)(m, xmatch), m(xmatch)
        )
        # The eager backend accepts the differently-strided input.
        ecode, ecache = torch.compiler.precompile(
            lambda model, t: model(t), example_inputs=[(m, xex)], backend="eager"
        )
        self.assertEqual(torch.compiler.precompile.load(ecode, ecache)(m, xrt), m(xrt))

    def test_input_layout_mismatch_enforced_without_size_asserts(self):
        # The layout guard must be a PROACTIVE driver check, not a reliance on inductor's
        # assert_size_stride: with size_asserts=False the assert is elided, so a naive
        # try/except would silently read wrong strides. Both load paths must still raise.
        import torch._inductor.config as ind_config

        m = torch.nn.Linear(8, 5).eval()
        xex = torch.randn(8, 6).t()  # non-contiguous example, shape (6, 8)
        xrt = torch.randn(6, 8)  # same shape, contiguous -> different layout
        with ind_config.patch(size_asserts=False):
            code, cache = torch.compiler.precompile(
                lambda model, t: model(t), example_inputs=[(m, xex)]
            )
            with self.assertRaisesRegex(PrecompileError, "memory format"):
                torch.compiler.precompile.load(code, cache)(m, xrt)  # cached path
            with self.assertRaisesRegex(PrecompileError, "memory format"):
                torch.compiler.precompile.load(code, _strip_artifact(cache))(
                    m, xrt
                )  # inlined

    def test_input_shape_mismatch_clean_error(self):
        # A same-structure but wrong-SHAPE input is an invariant-3 (shape) mismatch, NOT
        # an invariant-6 layout one: the driver must say "shape" / invariant 3 and not
        # misadvise a no-op .contiguous() (both inputs here are already contiguous).
        m = torch.nn.Linear(8, 5).eval()
        xex = torch.randn(6, 8)  # contiguous example
        xrt = torch.randn(7, 8)  # contiguous, different shape (same pytree structure)
        code, cache = torch.compiler.precompile(
            lambda model, t: model(t), example_inputs=[(m, xex)]
        )
        with self.assertRaisesRegex(PrecompileError, "shape"):
            torch.compiler.precompile.load(code, cache)(m, xrt)  # cached path
        with self.assertRaisesRegex(PrecompileError, "shape"):
            torch.compiler.precompile.load(code, _strip_artifact(cache))(
                m, xrt
            )  # inlined path
        # The error must NOT mislabel a pure shape mismatch as a memory-format one.
        try:
            torch.compiler.precompile.load(code, cache)(m, xrt)
        except PrecompileError as e:
            self.assertNotIn("memory format", str(e))

    def test_size1_dim_stride_exempt_like_inductor(self):
        # A size-1 dim's stride is irrelevant (one element); inductor's assert_size_stride
        # ignores it (guards.cpp), so the proactive layout check must too -- a kept-dim
        # slice x[i:i+1] (size-1 dim with a wider stride) must RUN, not raise.
        m = torch.nn.Linear(4, 3).eval()
        xex = torch.randn(1, 4)  # contiguous, stride (4, 1)
        code, cache = torch.compiler.precompile(
            lambda model, t: model(t), example_inputs=[(m, xex)]
        )
        row = torch.randn(2, 8)[
            0:1, :4
        ]  # shape (1, 4), stride (8, 1): size-1 dim differs
        self.assertEqual(tuple(row.shape), (1, 4))
        self.assertNotEqual(row.stride(), xex.stride())
        self.assertEqual(torch.compiler.precompile.load(code, cache)(m, row), m(row))
        self.assertEqual(
            torch.compiler.precompile.load(code, _strip_artifact(cache))(m, row),
            m(row),
        )

    def test_empty_input_shape_is_still_checked(self):
        # The numel==0 exemption must relax ONLY the (meaningless) stride check, not the
        # shape check: an empty runtime input whose shape differs from the example must
        # still raise invariant 3, not silently return the traced-shape output.
        code, cache = torch.compiler.precompile(
            lambda t: t.sum(0), example_inputs=[(torch.randn(0, 4),)]
        )
        f_c = torch.compiler.precompile.load(code, cache)
        with self.assertRaisesRegex(PrecompileError, "shape"):
            f_c(torch.randn(0, 6))
        # A matching empty input runs (shape matches; stride is not checked).
        self.assertEqual(f_c(torch.randn(0, 4)), torch.randn(0, 4).sum(0))

    def test_shape_only_input_is_layout_flexible(self):
        # An input used only for its .shape (not its data) is not stride-consumed by the
        # kernel, so inductor emits no assert_size_stride for it; a transposed version
        # (same shape) must RUN, not be wrongly rejected as a memory-format mismatch.
        class M(torch.nn.Module):
            def forward(self, x, y):
                return y * x.shape[0]

        m = M().eval()
        x = torch.randn(4, 4)  # square so .t() keeps shape (4, 4)
        y = torch.randn(4, 4)
        code, cache = torch.compiler.precompile(
            lambda mm, a, b: mm(a, b), example_inputs=[(m, x, y)]
        )
        f_c = torch.compiler.precompile.load(code, cache)
        xt = x.t()  # same shape, different stride; only x.shape is consumed
        self.assertNotEqual(xt.stride(), x.stride())
        self.assertEqual(f_c(m, xt, y), m(xt, y))
        # A different x SHAPE is still rejected (x.shape[0] is baked).
        with self.assertRaisesRegex(PrecompileError, "shape"):
            f_c(m, torch.randn(5, 4), y)

    def test_dynamic_shapes_static_dim_still_checked(self):
        # The non-marked (feature) dim stays specialized: a mismatch on it is rejected,
        # while the marked (batch) dim is free.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(8, 4)
        mark_unbacked(x, 0)
        code, cache = torch.compiler.precompile(
            lambda mm, t: mm(t), example_inputs=[(m, x)]
        )
        f_c = torch.compiler.precompile.load(code, cache)
        self.assertEqual(f_c(m, torch.randn(16, 4)).shape, (16, 3))  # dynamic dim free
        with self.assertRaisesRegex(PrecompileError, "dynamic dim"):
            f_c(m, torch.randn(16, 5))  # static feature dim mismatched

    def test_dynamic_shapes_guard_required_rejected(self):
        # A graph that must guard on the dynamic dim fails LOUDLY at capture (the unbacked
        # dim cannot be guarded), as a clear PrecompileError rather than a silent artifact.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(8, 4)
        mark_unbacked(x, 0)

        def needs_guard(mm, t):
            if t.shape[0] > 4:
                return mm(t)
            return mm(t) + 1

        with self.assertRaisesRegex(PrecompileError, "guard on a dim marked with"):
            torch.compiler.precompile(needs_guard, example_inputs=[(m, x)])

    def test_dynamic_shapes_eager_rejected(self):
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(8, 4)
        mark_unbacked(x, 0)
        with self.assertRaisesRegex(
            NotImplementedError, "only supported with backend='inductor'"
        ):
            torch.compiler.precompile(
                lambda mm, t: mm(t), example_inputs=[(m, x)], backend="eager"
            )

    @parametrize("path", ("cached", "inlined"))
    def test_dtype_mismatch_rejected(self, path):
        # Each dense input's dtype is baked at capture (invariant 6); a runtime input of
        # a different dtype is rejected up front on BOTH the cached and inlined paths.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)  # float32 example
        code, cache = torch.compiler.precompile(
            lambda model, t: model(t), example_inputs=[(m, x)]
        )
        if path == "inlined":
            cache = _strip_artifact(cache)
        f_c = torch.compiler.precompile.load(code, cache)
        with self.assertRaisesRegex(PrecompileError, "dtype"):
            f_c(m, x.double())

    @unittest.skipUnless(TEST_CUDA, "needs CUDA for a cpu-vs-cuda device mismatch")
    @parametrize("path", ("cached", "inlined"))
    def test_device_mismatch_rejected(self, path):
        # Each dense input's device is baked at capture (invariant 6); a cpu-traced
        # artifact rejects a cuda input up front on BOTH load paths.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)  # cpu example
        code, cache = torch.compiler.precompile(
            lambda model, t: model(t), example_inputs=[(m, x)]
        )
        if path == "inlined":
            cache = _strip_artifact(cache)
        f_c = torch.compiler.precompile.load(code, cache)
        with self.assertRaisesRegex(PrecompileError, "device"):
            f_c(m, x.cuda())

    def test_mark_dynamic_backed_rejected(self):
        # Backed dynamic marks (mark_dynamic) have no analogue in the static/unbacked
        # capture path; precompile rejects them loudly rather than silently dropping
        # them and baking a wrong artifact (invariant 3).
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(8, 4)
        mark_dynamic(x, 0)
        with self.assertRaisesRegex(PrecompileError, "mark_dynamic"):
            torch.compiler.precompile(lambda mm, t: mm(t), example_inputs=[(m, x)])

    def test_mark_unbacked_hint_override_honored(self):
        # A mark_unbacked hint_override is a perf-only autotuning size hint (never a
        # guard), so precompile does NOT reject it; the single artifact is valid for any
        # runtime size and the hint is threaded onto the capture ShapeEnv's symbol.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(8, 4)
        mark_unbacked(x, 0, hint_override=16)
        code, cache = torch.compiler.precompile(
            lambda mm, t: mm(t), example_inputs=[(m, x)]
        )
        f_c = torch.compiler.precompile.load(code, cache)
        self.assertEqual(f_c(m, x), m(x))
        x2 = torch.randn(32, 4)
        self.assertEqual(f_c(m, x2), m(x2))

    def test_mark_unbacked_specialize_on_rejected(self):
        # A mark_unbacked specialize_on list cannot be honored (precompile produces a
        # single artifact, not per-value specializations); it is rejected at capture.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(8, 4)
        mark_unbacked(x, 0, specialize_on=[lambda t: t.shape[0] == 8])
        with self.assertRaisesRegex(PrecompileError, "specialize_on"):
            torch.compiler.precompile(lambda mm, t: mm(t), example_inputs=[(m, x)])

    def test_mark_unbacked_subclass_rejected(self):
        # A mark_unbacked dim on a tensor subclass (DTensor) cannot be honored: the
        # dynamic capture refakes a marked leaf via torch.empty, which drops the subclass
        # and would trace on a plain dense tensor. mark_unbacked stamps its marks on the
        # OUTER DTensor too (the decorator's DTensor branch falls through), so precompile
        # sees the mark and must reject it LOUDLY rather than silently tracing a
        # subclass-stripped tensor (invariant 3).
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_gloo_available():
            self.skipTest("gloo not available")

        from torch.distributed.tensor import DeviceMesh, distribute_tensor, Replicate
        from torch.testing._internal.common_utils import find_free_port

        saved_env = {k: os.environ.get(k) for k in ("MASTER_ADDR", "MASTER_PORT")}
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        dist.init_process_group("gloo", rank=0, world_size=1)
        try:
            mesh = DeviceMesh("cpu", list(range(1)))
            m = torch.nn.Linear(4, 3).eval()
            x = distribute_tensor(torch.randn(8, 4), mesh, [Replicate()])
            mark_unbacked(x, 0)
            with self.assertRaisesRegex(PrecompileError, "tensor subclass"):
                torch.compiler.precompile(lambda mm, t: mm(t), example_inputs=[(m, x)])
        finally:
            dist.destroy_process_group()
            for k, v in saved_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    @parametrize("path", ("cached", "inlined"))
    def test_shape_id_mismatched_sizes_rejected(self, path):
        # Two inputs sharing a shape_id reuse ONE unbacked symbol, so their marked dims
        # are equal by construction. A runtime call passing MISMATCHED sizes for those
        # dims violates the baked equality and is rejected with a clear PrecompileError.
        # The cached path catches it via the reconstructed artifact's assert_size_stride;
        # the inlined (artifact-stripped) path catches it via the inlined driver's own
        # assert_size_stride relabel -- exercise both so the inlined driver copy is covered.
        m = torch.nn.Linear(4, 4).eval()
        x = torch.randn(8, 4)
        y = torch.randn(8, 4)
        mark_unbacked(x, 0, shape_id="b")
        mark_unbacked(y, 0, shape_id="b")
        code, cache = torch.compiler.precompile(
            lambda mm, a, b: mm(a) + b, example_inputs=[(m, x, y)]
        )
        if path == "inlined":
            blob = torch.load(io.BytesIO(cache), weights_only=True)
            blob["artifact"] = None
            buf = io.BytesIO()
            torch.save(blob, buf)
            cache = buf.getvalue()
        f_c = torch.compiler.precompile.load(code, cache)
        with self.assertRaisesRegex(PrecompileError, "shape or memory format"):
            f_c(m, torch.randn(8, 4), torch.randn(16, 4))

    @parametrize("path", ("cached", "inlined"))
    def test_shape_id_bounds_from_both_occurrences_enforced(self, path):
        # Bounds from BOTH occurrences of a shared shape_id are applied to the single
        # shared symbol at capture: a min on one input and a max on the other are each
        # threaded onto the same unbacked symbol (see _fakeify_with_unbacked) AND baked as
        # a runtime USER_INPUT_BOUNDS guard. mark_unbacked's docstring promises a runtime
        # min/max check; this asserts it actually fires. An OUT-OF-BOUNDS size (< 2 or
        # > 64) is rejected with a PrecompileError naming the bound, while in-bounds sizes
        # (including the boundaries 2 and 64) still run and match eager. Both load paths.
        m = torch.nn.Linear(4, 4).eval()
        x = torch.randn(8, 4)
        y = torch.randn(8, 4)
        mark_unbacked(x, 0, shape_id="b", min=2)
        mark_unbacked(y, 0, shape_id="b", max=64)
        code, cache = torch.compiler.precompile(
            lambda mm, a, b: mm(a) + b, example_inputs=[(m, x, y)]
        )
        if path == "inlined":
            blob = torch.load(io.BytesIO(cache), weights_only=True)
            blob["artifact"] = None
            buf = io.BytesIO()
            torch.save(blob, buf)
            cache = buf.getvalue()
        f_c = torch.compiler.precompile.load(code, cache)
        for bs in (2, 8, 64):  # min boundary, an interior size, max boundary
            xt = torch.randn(bs, 4)
            yt = torch.randn(bs, 4)
            self.assertEqual(f_c(m, xt, yt), m(xt) + yt)
        # Below the declared min on the first occurrence's dim is rejected.
        with self.assertRaisesRegex(PrecompileError, "min=2"):
            f_c(m, torch.randn(1, 4), torch.randn(1, 4))
        # Above the declared max (from the second occurrence) is rejected.
        with self.assertRaisesRegex(PrecompileError, "max=64"):
            f_c(m, torch.randn(65, 4), torch.randn(65, 4))

    @parametrize("path", ("cached", "inlined"))
    def test_mark_unbacked_min_enforced_at_runtime(self, path):
        # mark_unbacked(x, 0, min=4) promises (in its docstring) a runtime check that the
        # dim is >= min. The capture-time torch._check on the unbacked symint never becomes
        # a runtime guard, so precompile bakes USER_INPUT_BOUNDS and the driver enforces it:
        # running the artifact at batch 2 raises a PrecompileError naming the bound on BOTH
        # load paths, while batch 8 runs and matches eager.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(8, 4)
        mark_unbacked(x, 0, min=4)
        code, cache = torch.compiler.precompile(
            lambda mm, t: mm(t), example_inputs=[(m, x)]
        )
        self.assertIn("USER_INPUT_BOUNDS = [{0: (4, None)}]", code)
        if path == "inlined":
            blob = torch.load(io.BytesIO(cache), weights_only=True)
            blob["artifact"] = None
            buf = io.BytesIO()
            torch.save(blob, buf)
            cache = buf.getvalue()
        f_c = torch.compiler.precompile.load(code, cache)
        with self.assertRaisesRegex(PrecompileError, "size 2.*min=4"):
            f_c(m, torch.randn(2, 4))
        xt = torch.randn(8, 4)
        self.assertEqual(f_c(m, xt), m(xt))

    def test_eager_backend_wrong_static_shape_rejected(self):
        # The eager driver now checks USER_INPUT_SHAPES too: a wrong static shape is
        # rejected (invariant 3).
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        code, cache = torch.compiler.precompile(
            lambda model, t: model(t), example_inputs=[(m, x)], backend="eager"
        )
        f_c = torch.compiler.precompile.load(code, cache)
        with self.assertRaisesRegex(PrecompileError, "shape"):
            f_c(m, torch.randn(7, 4))

    def test_eager_backend_dtype_mismatch_rejected(self):
        # The eager driver checks USER_INPUT_DTYPES too: a dtype mismatch is rejected
        # (invariant 6).
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        code, cache = torch.compiler.precompile(
            lambda model, t: model(t), example_inputs=[(m, x)], backend="eager"
        )
        f_c = torch.compiler.precompile.load(code, cache)
        with self.assertRaisesRegex(PrecompileError, "dtype"):
            f_c(m, x.double())

    def test_cache_integrity_tampered_backend_rejected(self):
        # The cache envelope's backend tag is an integrity check: a tampered backend
        # (here flipped to a value that does not match python_code's BACKEND) makes
        # load() raise a clear PrecompileError rather than reconstruct a foreign cache.
        m = torch.nn.Sequential(torch.nn.Linear(4, 3)).eval()
        x = torch.randn(5, 4)
        code, cache = torch.compiler.precompile(
            lambda model, t: model(t), example_inputs=[(m, x)]
        )
        blob = torch.load(io.BytesIO(cache), weights_only=True)
        blob["backend"] = "eager"  # python_code says inductor
        buf = io.BytesIO()
        torch.save(blob, buf)
        with self.assertRaisesRegex(PrecompileError, "backend"):
            torch.compiler.precompile.load(code, buf.getvalue())

    @parametrize("tag", ("format", "version"))
    def test_cache_format_version_mismatch_degrades(self, tag):
        # The cache is acceleration-only, so a FORMAT or VERSION mismatch (a foreign or
        # different-build envelope) is NOT fatal: load() DEGRADES to JIT'ing from
        # python_code rather than hard-failing. The reloaded callable must still run and
        # match eager, and load() must emit a degrade WARNING on the torch._precompile
        # logger. (A BACKEND or CODE_HASH mismatch still hard-fails -- see
        # test_cache_integrity_tampered_backend_rejected and
        # test_load_rejects_mismatched_code_cache_pair.)
        m = torch.nn.Sequential(torch.nn.Linear(4, 3)).eval()
        x = torch.randn(5, 4)
        code, cache = torch.compiler.precompile(
            lambda model, t: model(t), example_inputs=[(m, x)]
        )
        blob = torch.load(io.BytesIO(cache), weights_only=True)
        # Tamper either the format string or bump the version to a foreign value.
        blob[tag] = "not-a-precompile-cache" if tag == "format" else 999
        buf = io.BytesIO()
        torch.save(blob, buf)
        with self.assertLogs("torch._precompile", level="WARNING") as cm:
            f_c = torch.compiler.precompile.load(code, buf.getvalue())  # must not raise
        self.assertTrue(
            any("different torch build" in line for line in cm.output),
            f"expected a format/version degrade warning, got: {cm.output}",
        )
        self.assertEqual(f_c(m, x), m(x))  # JIT fallback runs and is correct

    def test_missing_calling_convention_metadata_rejected(self):
        # Syntactically valid python_code that lacks a required metadata global is not a
        # precompile artifact; load() raises a clear PrecompileError naming the gap.
        buf = io.BytesIO()
        torch.save(
            {
                "format": "torch.compiler.precompile",
                "version": 1,
                "backend": "inductor",
                "artifact": None,
            },
            buf,
        )
        with self.assertRaisesRegex(
            PrecompileError, "missing calling-convention metadata"
        ):
            torch.compiler.precompile.load("x = 1\n", buf.getvalue())

    def test_missing_dynamo_state_rejected(self):
        x = torch.randn(3)
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_dynamic,
            example_inputs=[(x,)],
            tracer="dynamo",
            backend="eager",
        )
        code = "\n".join(
            line
            for line in code.splitlines()
            if not line.startswith("_DYNAMO_STATE = ")
        )
        blob = torch.load(io.BytesIO(cache), weights_only=True)
        blob["code_hash"] = hashlib.sha256(code.encode()).hexdigest()
        buf = io.BytesIO()
        torch.save(blob, buf)
        with self.assertRaisesRegex(
            PrecompileError, "missing calling-convention metadata.*_DYNAMO_STATE"
        ):
            torch.compiler.precompile.load(code, buf.getvalue())

    @parametrize(
        "name,replacement",
        (
            ("TRAINING", "TRAINING = 'False'"),
            ("_DYNAMO_BACKEND_IDS", "_DYNAMO_BACKEND_IDS = 1"),
            ("_DYNAMO_BACKENDS", "_DYNAMO_BACKENDS = []"),
            ("_DYNAMO_PYTHON_VERSION", "_DYNAMO_PYTHON_VERSION = ()"),
            ("_DYNAMO_STATE", "_DYNAMO_STATE = 1"),
            ("_DYNAMO_TORCH_VERSION", "_DYNAMO_TORCH_VERSION = 1"),
        ),
    )
    def test_invalid_dynamo_metadata_rejected(self, name, replacement):
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_dynamic,
            example_inputs=[(torch.randn(3),)],
            tracer="dynamo",
            backend="eager",
        )
        code = "\n".join(
            replacement if line.startswith(f"{name} = ") else line
            for line in code.splitlines()
        )
        blob = torch.load(io.BytesIO(cache), weights_only=True)
        blob["code_hash"] = hashlib.sha256(code.encode()).hexdigest()
        buf = io.BytesIO()
        torch.save(blob, buf)
        with self.assertRaisesRegex(
            PrecompileError, f"invalid calling-convention metadata.*{name}"
        ):
            torch.compiler.precompile.load(code, buf.getvalue())

    def test_invalid_serialized_dynamo_state_rejected(self):
        code, cache = torch.compiler.precompile(
            _precompile_dynamo_dynamic,
            example_inputs=[(torch.randn(3),)],
            tracer="dynamo",
            backend="eager",
        )
        code = "\n".join(
            "_DYNAMO_STATE = 'not-base64'"
            if line.startswith("_DYNAMO_STATE = ")
            else line
            for line in code.splitlines()
        )
        blob = torch.load(io.BytesIO(cache), weights_only=True)
        blob["code_hash"] = hashlib.sha256(code.encode()).hexdigest()
        buf = io.BytesIO()
        torch.save(blob, buf)
        with self.assertRaisesRegex(PrecompileError, "invalid serialized Dynamo state"):
            torch.compiler.precompile.load(code, buf.getvalue())

    @parametrize(
        "corruption",
        (
            "empty_codes",
            "entry_code",
            "variant_code",
            "guards_state",
            "guards_shape",
            "disabled_freevars",
            "tensor_leaf",
            "system_info",
            "summary",
            "package",
        ),
    )
    def test_invalid_nested_dynamo_state_rejected(self, corruption):
        from torch._precompile import _parse_dynamo_state

        code, cache = torch.compiler.precompile(
            _precompile_dynamo_dynamic,
            example_inputs=[(torch.randn(3),)],
            tracer="dynamo",
            backend="eager",
        )
        state = _parse_dynamo_state(code)
        entry = state.codes[0]
        if corruption == "empty_codes":
            state = dataclasses.replace(state, codes=())
        elif corruption == "entry_code":
            state = dataclasses.replace(
                state,
                codes=(dataclasses.replace(entry, code="invalid"), *state.codes[1:]),
            )
        elif corruption == "variant_code":
            variant = dataclasses.replace(entry.variants[0], dynamo_code="invalid")
            state = dataclasses.replace(
                state,
                codes=(
                    dataclasses.replace(entry, variants=(variant, *entry.variants[1:])),
                    *state.codes[1:],
                ),
            )
        elif corruption == "guards_state":
            variant = dataclasses.replace(
                entry.variants[0], guards_state=b"not a pickle"
            )
            state = dataclasses.replace(
                state,
                codes=(
                    dataclasses.replace(entry, variants=(variant, *entry.variants[1:])),
                    *state.codes[1:],
                ),
            )
        elif corruption == "guards_shape":
            from torch._dynamo.package import load_guards_state

            guards = load_guards_state(entry.variants[0].guards_state)
            guards.output_graph.local_scope = []
            variant = dataclasses.replace(
                entry.variants[0], guards_state=pickle.dumps(guards)
            )
            state = dataclasses.replace(
                state,
                codes=(
                    dataclasses.replace(entry, variants=(variant, *entry.variants[1:])),
                    *state.codes[1:],
                ),
            )
        elif corruption == "disabled_freevars":
            from torch._dynamo.package import SerializedCode
            from torch._precompile import _DynamoDisabledFunction

            captured = 1

            def disabled(x):
                return x + captured

            state = dataclasses.replace(
                state,
                disabled_functions={
                    "disabled": _DynamoDisabledFunction(
                        code=SerializedCode.from_code_object(disabled.__code__),
                        name="disabled",
                        defaults=None,
                        kwdefaults=None,
                        module_globals={},
                        value_globals={},
                    )
                },
            )
        elif corruption == "tensor_leaf":
            if state.input_contract is None:
                raise AssertionError("expected a Dynamo input contract")
            contract_variant = state.input_contract.variants[0]
            leaf = next(i for i, value in enumerate(contract_variant.leaves) if value)
            leaves = list(contract_variant.leaves)
            leaves[leaf] = {"kind": "tensor"}
            state = dataclasses.replace(
                state,
                input_contract=dataclasses.replace(
                    state.input_contract,
                    variants=(
                        dataclasses.replace(contract_variant, leaves=tuple(leaves)),
                        *state.input_contract.variants[1:],
                    ),
                ),
            )
        elif corruption == "system_info":
            state = dataclasses.replace(state, system_info="invalid")
        elif corruption == "summary":
            if state.summary is None:
                raise AssertionError("expected a precompile summary")
            state = dataclasses.replace(
                state,
                summary=dataclasses.replace(state.summary, guarded_codes="invalid"),
            )
        else:
            state = dataclasses.replace(state, serving_mode="installed", package={})
        encoded = base64.b64encode(pickle.dumps(state)).decode()
        code = "\n".join(
            f"_DYNAMO_STATE = {encoded!r}"
            if line.startswith("_DYNAMO_STATE = ")
            else line
            for line in code.splitlines()
        )
        blob = torch.load(io.BytesIO(cache), weights_only=True)
        blob["code_hash"] = hashlib.sha256(code.encode()).hexdigest()
        buf = io.BytesIO()
        torch.save(blob, buf)
        with self.assertRaisesRegex(PrecompileError, "invalid serialized Dynamo state"):
            torch.compiler.precompile.load(code, buf.getvalue())

    def test_singleton_pickle_deepcopy_roundtrip(self):
        # torch.compiler.precompile is a process-wide singleton; pickle and deepcopy
        # must round-trip to the SAME object (it carries no per-call state), and its
        # repr is the stable public name.
        p = torch.compiler.precompile
        self.assertIs(pickle.loads(pickle.dumps(p)), p)
        self.assertIs(copy.deepcopy(p), p)
        self.assertEqual(repr(p), "torch.compiler.precompile")

    def test_standalone_runtime_artifact_execs_in_fresh_process(self):
        # A generated artifact that imports a standalone_runtime helper (here output-
        # aliasing, which emits ``from ...standalone_runtime import gen_alias_from_base``)
        # must EXEC in a FRESH process whose only prior import is ``torch`` -- a
        # regression for the runtime_wrappers <-> _dynamo circular import that a cold
        # exec used to hit. We write python_code to a temp file and exec it in a
        # subprocess that imports only torch, then runs forward().
        x = torch.randn(3, 4)
        code, _cache = torch.compiler.precompile(lambda a: a.t(), example_inputs=[(x,)])
        self.assertIn("standalone_runtime import gen_alias_from_base", code)
        with tempfile.NamedTemporaryFile(
            "w", suffix=".py", delete=False
        ) as artifact_file:
            artifact_file.write(code)
            artifact_path = artifact_file.name
        driver = textwrap.dedent(
            f"""
            import torch  # the ONLY pre-import; the artifact must self-bootstrap
            ns = {{"__name__": "_fresh_artifact"}}
            with open({artifact_path!r}) as fh:
                exec(compile(fh.read(), {artifact_path!r}, "exec"), ns)
            x = torch.randn(3, 4)
            out = ns["forward"](x)
            assert torch.equal(out, x.t()), "fresh-process artifact output mismatch"
            print("FRESH_OK")
            """
        )
        try:
            proc = subprocess.run(
                [sys.executable, "-c", driver],
                capture_output=True,
                text=True,
                timeout=300,
            )
        finally:
            if os.path.exists(artifact_path):
                os.remove(artifact_path)
        self.assertEqual(
            proc.returncode, 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
        self.assertIn("FRESH_OK", proc.stdout)

    def test_load_rejects_mismatched_code_cache_pair(self):
        # The cache envelope's code_hash (sha256 of python_code) binds a cache to the
        # EXACT python_code it accelerates. Two artifacts from the SAME backend but
        # DIFFERENT fn produce different python_code (hence different code_hash), so
        # pairing one's code with the other's cache must fail loudly rather than
        # silently run the cache's compiled graph under foreign metadata (the core
        # silent-wrong-result guard). The MATCHED pair still runs and is correct.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        codeA, cacheA = torch.compiler.precompile(
            lambda mm, t: mm(t) * 2, example_inputs=[(m, x)]
        )
        codeB, cacheB = torch.compiler.precompile(
            lambda mm, t: mm(t) + 100, example_inputs=[(m, x)]
        )
        self.assertNotEqual(codeA, codeB)
        with self.assertRaisesRegex(PrecompileError, "code_hash|does not match"):
            torch.compiler.precompile.load(codeA, cacheB)
        f_a = torch.compiler.precompile.load(codeA, cacheA)
        self.assertEqual(f_a(m, x), m(x) * 2)

    def test_non_size_stride_assertion_propagates_unchanged(self):
        # The inductor driver's forward() wraps the inlined ``call`` in a try/except
        # AssertionError that relabels ONLY inductor's own assert_size_stride failure
        # (a layout/shape mismatch) as a "shape or memory format" PrecompileError. A
        # NON-size-stride AssertionError (e.g. a user torch._assert or an internal
        # invariant) must propagate with its ORIGINAL message, not be mislabeled. A
        # call() that raises a non-layout AssertionError is hard to trigger from a real
        # compiled artifact, so doctor a real artifact's call() to raise a custom
        # assertion and re-pair its code_hash, exercising the inlined relabel guard.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        code, cache = torch.compiler.precompile(
            lambda mm, t: mm(t), example_inputs=[(m, x)]
        )
        head = code[: code.index("\ndef call(")]
        banner = code.rindex(
            "# " + "=" * 70, 0, code.index("# 2. Calling-convention metadata")
        )
        new_call = (
            '\n\ndef call(args):\n    assert False, "my custom user assertion"\n\n\n'
        )
        new_code = head + new_call + code[banner:]
        blob = torch.load(io.BytesIO(cache), weights_only=True)
        blob["artifact"] = None  # force the inlined path so the doctored call() runs
        import hashlib

        blob["code_hash"] = hashlib.sha256(new_code.encode()).hexdigest()
        buf = io.BytesIO()
        torch.save(blob, buf)
        f = torch.compiler.precompile.load(new_code, buf.getvalue())
        with self.assertRaisesRegex(AssertionError, "my custom user assertion"):
            f(m, x)
        # The original assertion must NOT be relabeled as a layout error.
        try:
            f(m, x)
        except AssertionError as e:
            self.assertNotIn("shape or memory format", str(e))

    def test_public_identity_module_and_qualname(self):
        # PrecompileError and load are public under torch.compiler.precompile, so their
        # __module__ / __qualname__ must report that public location (so Sphinx and
        # introspection anchor them under torch.compiler, not the private module).
        err = torch.compiler.precompile.PrecompileError
        self.assertEqual(err.__module__, "torch.compiler")
        self.assertEqual(err.__qualname__, "precompile.PrecompileError")
        self.assertEqual(torch.compiler.precompile.load.__module__, "torch.compiler")
        self.assertEqual(torch.compiler.precompile.load.__qualname__, "precompile.load")

    @parametrize("backend", ("inductor", "eager"))
    def test_renamed_buffer_structural_mismatch_rejected(self, backend):
        # The BUFFER_NAMES half of the structural check (invariant 2): a runtime model
        # whose PARAM names match exactly but a BUFFER is renamed (same count and shape)
        # must be rejected, since the buffer name list is part of the baked structure.
        # The cached/inlined inductor driver and the eager driver each have their own
        # _check_structure, so cover both backends.
        class WithBuf(torch.nn.Module):
            def __init__(self, bufname):
                super().__init__()
                self.lin = torch.nn.Linear(4, 3)
                self.register_buffer(bufname, torch.randn(3))
                self._bn = bufname

            def forward(self, x):
                return self.lin(x) + getattr(self, self._bn)

        m = WithBuf("buf").eval()
        x = torch.randn(5, 4)
        code, cache = torch.compiler.precompile(
            lambda mm, t: mm(t), example_inputs=[(m, x)], backend=backend
        )
        self.assertIn("BUFFER_NAMES = ['buf']", code)
        renamed = WithBuf("buf2").eval()  # same params, buffer renamed (same shape)
        f_c = torch.compiler.precompile.load(code, cache)
        with self.assertRaisesRegex(PrecompileError, "do not match the traced model"):
            f_c(renamed, x)

    def test_example_input_inplace_mutation_not_restored(self):
        # Capture EXECUTES fn once on the example inputs (invariant 3), so an in-place
        # mutation fn performs on its example user input happens at capture time and is
        # NOT restored -- only .grad is snapshotted/restored. Pin this surprising contract
        # so it stays covered: the example tensor reflects the mutation afterward.
        scratch = torch.zeros(4)
        torch.compiler.precompile(lambda a: a.add_(1.0), example_inputs=[(scratch,)])
        self.assertEqual(scratch, torch.ones(4))

    @parametrize("path", ("cached", "inlined", "eager"))
    def test_wrong_dtype_rejected_across_all_paths(self, path):
        # The same wrong-dtype input is rejected on ALL load paths -- cached (artifact),
        # inlined (artifact stripped), and eager -- each with its own driver copy of the
        # dtype check (invariant 6). Loading the SAME inductor artifact via cached and
        # inlined, plus a separate eager artifact, keeps the three drivers in agreement.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        if path == "eager":
            code, cache = torch.compiler.precompile(
                lambda mm, t: mm(t), example_inputs=[(m, x)], backend="eager"
            )
        else:
            code, cache = torch.compiler.precompile(
                lambda mm, t: mm(t), example_inputs=[(m, x)]
            )
            if path == "inlined":
                cache = _strip_artifact(cache)
        f_c = torch.compiler.precompile.load(code, cache)
        with self.assertRaisesRegex(PrecompileError, "dtype"):
            f_c(m, x.double())

    @unittest.skipUnless(TEST_CUDA, "needs CUDA for a cpu-vs-cuda device mismatch")
    def test_eager_device_mismatch_rejected(self):
        # The eager driver bakes each input's device (invariant 6): a cpu-traced eager
        # artifact rejects a cuda input up front, like the inductor backend.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)  # cpu example
        code, cache = torch.compiler.precompile(
            lambda mm, t: mm(t), example_inputs=[(m, x)], backend="eager"
        )
        f_c = torch.compiler.precompile.load(code, cache)
        with self.assertRaisesRegex(PrecompileError, "device"):
            f_c(m, x.cuda())

    def test_unserializable_in_spec_accepts_distinct_structures(self):
        # When IN_SPEC degrades to None (the input pytree spec was not serializable) the
        # structural in_spec check is SKIPPED -- a documented best-effort limit. Two
        # SAME-leaf-count, same-per-leaf-shape but STRUCTURALLY DISTINCT runtime inputs
        # are therefore both accepted without error (the only check left is leaf count /
        # per-leaf shape). Make that best-effort gap explicit.
        m = torch.nn.Linear(4, 3).eval()
        inp = _UnserializableCtxInput(torch.randn(5, 4), torch.randn(5, 4))
        code, cache = torch.compiler.precompile(
            lambda model, h: model(h.a + h.b), example_inputs=[(m, inp)]
        )
        self.assertIn("IN_SPEC = None", code)
        f_c = torch.compiler.precompile.load(code, cache)
        t = torch.randn(5, 4)
        # The traced structure (the custom node) and a plain list of the same two leaves
        # have distinct pytree structures but the same flattened leaves/shapes; both run.
        out_node = f_c(m, _UnserializableCtxInput(t, t))
        out_list = f_c(m, [t, t])
        self.assertEqual(out_node, m(t + t))
        self.assertEqual(out_list, m(t + t))

    @parametrize("path", ("cached", "inlined"))
    def test_mark_unbacked_max_enforced_at_runtime(self, path):
        # The max-only mirror of test_mark_unbacked_min_enforced_at_runtime:
        # mark_unbacked(x, 0, max=16) records USER_INPUT_BOUNDS = [{0: (None, 16)}] and
        # the driver rejects an ABOVE-max runtime size on BOTH load paths (the capture-time
        # torch._check never becomes a runtime guard on an unbacked symint), while an
        # in-bounds size runs and matches eager.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(8, 4)
        mark_unbacked(x, 0, max=16)
        code, cache = torch.compiler.precompile(
            lambda mm, t: mm(t), example_inputs=[(m, x)]
        )
        self.assertIn("USER_INPUT_BOUNDS = [{0: (None, 16)}]", code)
        if path == "inlined":
            blob = torch.load(io.BytesIO(cache), weights_only=True)
            blob["artifact"] = None
            buf = io.BytesIO()
            torch.save(blob, buf)
            cache = buf.getvalue()
        f_c = torch.compiler.precompile.load(code, cache)
        with self.assertRaisesRegex(PrecompileError, "max"):
            f_c(m, torch.randn(32, 4))
        xt = torch.randn(8, 4)
        self.assertEqual(f_c(m, xt), m(xt))

    @unittest.skipUnless(TEST_CUDA, "functionalize_rng_ops seeds via CUDA rng state")
    def test_functionalized_rng_matches_eager_cpu(self):
        # Under functionalized RNG the dropout draw is seeded from the global generator,
        # so seeding torch.manual_seed identically before the artifact run and before eager
        # makes both draw the SAME dropout mask: the artifact output is numerically EQUAL
        # to eager (a stronger check than structure-only). This runs on CPU tensors, but
        # functionalize_rng_ops still seeds via CUDARngStateHelper.get_torch_state_as_tuple,
        # which raises unless CUDA is available, so the whole test is gated on TEST_CUDA
        # (mirroring test_functionalized_rng_supported). The CUDA functionalized path uses
        # different Philox offset bookkeeping than eager, so this numeric equivalence is
        # CPU-tensor-only (see test_functionalized_rng_supported for the device-generic
        # structural check).
        import torch._functorch.config as functorch_config

        x = torch.randn(64)
        with functorch_config.patch(functionalize_rng_ops=True):
            code, cache = torch.compiler.precompile(
                lambda a: torch.nn.functional.dropout(a, 0.5, training=True),
                example_inputs=[(x,)],
            )
            f_c = torch.compiler.precompile.load(code, cache)
            torch.manual_seed(0)
            out = f_c(x)
        torch.manual_seed(0)
        ref = torch.nn.functional.dropout(x, 0.5, training=True)
        self.assertTrue((out == 0).any())  # dropout zeroed some elements
        self.assertEqual(out, ref)  # same mask under the same seed

    @parametrize("backend", ("inductor", "eager"))
    def test_param_shape_mismatch_rejected(self, backend):
        # The headline silent-wrong-result fix: the structural check (invariant 2) now
        # compares each runtime param's SHAPE against the baked example, not just its
        # name/count. A runtime model with the SAME param names but a different param
        # SHAPE (here Linear(4, K) for the traced Linear(4, M), K != M) is rejected with a
        # PrecompileError naming the offending param -- on BOTH backends, and on the
        # inductor backend's cached AND inlined load paths. Before the fix the eager
        # backend (no assert_size_stride backstop) silently returned a wrong-shaped tensor.
        m = torch.nn.Linear(4, 3).eval()  # M = 3
        x = torch.randn(5, 4)
        code, cache = torch.compiler.precompile(
            lambda model, t: model(t), example_inputs=[(m, x)], backend=backend
        )
        bad = torch.nn.Linear(4, 7).eval()  # K = 7 != 3, same param names

        for label, f_c in _default_and_inlined_loaders(code, cache, backend):
            with self.subTest(path=label):
                with self.assertRaisesRegex(PrecompileError, "weight.*shape"):
                    f_c(bad, x)

    @parametrize("backend", ("inductor", "eager"))
    def test_param_dtype_mismatch_rejected(self, backend):
        # The dtype half of the structural shape/dtype check (invariant 2): a runtime
        # model with the SAME param names and shapes but a different param DTYPE (a
        # .half() copy of the traced float32 model) is rejected with a PrecompileError
        # naming the param, on both backends, AND -- on the inductor backend -- on the
        # cached (artifact) AND inlined (artifact-stripped) load paths. The inlined
        # inductor driver has its own _check_structure dtype branch, so cover it the
        # same way test_param_shape_mismatch_rejected covers the shape branch.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        code, cache = torch.compiler.precompile(
            lambda model, t: model(t), example_inputs=[(m, x)], backend=backend
        )
        bad = torch.nn.Linear(4, 3).eval().half()  # same shape, different dtype

        for label, f_c in _default_and_inlined_loaders(code, cache, backend):
            with self.subTest(path=label):
                with self.assertRaisesRegex(PrecompileError, "weight.*dtype"):
                    f_c(bad, x)

    @parametrize("backend", ("inductor", "eager"))
    def test_buffer_shape_dtype_mismatch_rejected(self, backend):
        # The BUFFER half of the structural SHAPE/DTYPE check (invariant 2): the
        # structural loop iterates PARAM_NAMES then BUFFER_NAMES, but only the param
        # branch was exercised elsewhere. A runtime model whose PARAMS match exactly but
        # whose registered BUFFER (same name, same count) has a different SHAPE or DTYPE
        # must be rejected naming that buffer. Cover both backends, and -- on inductor --
        # the cached AND inlined driver copies (each has its own _check_structure).
        class WithBuf(torch.nn.Module):
            def __init__(self, size, dtype):
                super().__init__()
                self.lin = torch.nn.Linear(4, 3)
                # A plain buffer the graph READS, so it is lifted to a graph input and
                # survives to the structural check (a buffer never read might be elided).
                self.register_buffer("b", torch.randn(size).to(dtype))

            def forward(self, x):
                return self.lin(x) + self.b.sum()

        m = WithBuf(3, torch.float32).eval()
        x = torch.randn(5, 4)
        code, cache = torch.compiler.precompile(
            lambda model, t: model(t), example_inputs=[(m, x)], backend=backend
        )
        self.assertIn("BUFFER_NAMES = ['b']", code)
        # Same buffer name and count, but a different SHAPE / DTYPE.
        bad_shape = WithBuf(5, torch.float32).eval()
        bad_dtype = WithBuf(3, torch.float64).eval()

        for label, f_c in _default_and_inlined_loaders(code, cache, backend):
            with self.subTest(path=label):
                with self.assertRaisesRegex(PrecompileError, r"'b'.*shape"):
                    f_c(bad_shape, x)
                with self.assertRaisesRegex(PrecompileError, r"'b'.*dtype"):
                    f_c(bad_dtype, x)

    def test_param_layout_specialization_rejected_inductor(self):
        # MAJOR2 (invariant 2 inductor caveat / invariant 6): the inductor backend bakes
        # each param/buffer's LAYOUT (memory format) too, since it emits assert_size_stride
        # on every weight the graph reads. A runtime model whose weight has the SAME
        # shape+dtype but a DIFFERENT memory format (a non-contiguous view) is rejected,
        # with the broadened relabel that names a model PARAMETER/BUFFER layout. The eager
        # backend is layout-flexible and ACCEPTS the same non-contiguous weight.
        m = torch.nn.Linear(8, 5).eval()
        x = torch.randn(4, 8)
        code, cache = torch.compiler.precompile(
            lambda model, t: model(t), example_inputs=[(m, x)]
        )

        def with_noncontig_weight():
            run = torch.nn.Linear(8, 5).eval()
            run.load_state_dict(m.state_dict())
            # A non-contiguous view of the same data: same shape+dtype, different layout.
            nc = run.weight.data.t().contiguous().t()
            self.assertFalse(nc.is_contiguous())
            self.assertEqual(tuple(nc.shape), tuple(m.weight.shape))
            run.weight = torch.nn.Parameter(nc)
            return run

        def loaders():
            yield "cached", torch.compiler.precompile.load(code, cache)
            yield (
                "inlined",
                torch.compiler.precompile.load(code, _strip_artifact(cache)),
            )

        for label, f_c in loaders():
            with self.subTest(path=label):
                with self.assertRaisesRegex(
                    PrecompileError, r"memory format.*PARAMETER/BUFFER.*layout"
                ):
                    f_c(with_noncontig_weight(), x)
        # The eager backend accepts the same non-contiguous weight (layout-flexible).
        ecode, ecache = torch.compiler.precompile(
            lambda model, t: model(t), example_inputs=[(m, x)], backend="eager"
        )
        run = with_noncontig_weight()
        self.assertEqual(torch.compiler.precompile.load(ecode, ecache)(run, x), run(x))

    def test_unbacked_equality_shared_vs_independent_shape_id(self):
        # A shared shape_id binds both dimensions to one symbol, so runtime mismatches are
        # rejected. Independently marked dimensions would instead introduce a deferred
        # equality constraint that the standalone driver cannot enforce, so capture must
        # fail rather than bake the example relation.
        m = torch.nn.Linear(4, 4).eval()
        # (a) shared shape_id -> equality enforced.
        xs = torch.randn(8, 4)
        ys = torch.randn(8, 4)
        mark_unbacked(xs, 0, shape_id="b")
        mark_unbacked(ys, 0, shape_id="b")
        code_s, cache_s = torch.compiler.precompile(
            lambda mm, a, b: mm(a) + b, example_inputs=[(m, xs, ys)]
        )
        f_s = torch.compiler.precompile.load(code_s, cache_s)
        xt, yt = torch.randn(8, 4), torch.randn(8, 4)
        self.assertEqual(f_s(m, xt, yt), m(xt) + yt)  # matched sizes work
        with self.assertRaisesRegex(PrecompileError, "shape or memory format"):
            f_s(m, torch.randn(8, 4), torch.randn(16, 4))  # mismatch rejected
        # (b) independent marks -> fail closed on the unhandled equality constraint.
        xi = torch.randn(8, 4)
        yi = torch.randn(8, 4)
        mark_unbacked(xi, 0)
        mark_unbacked(yi, 0)
        with self.assertRaisesRegex(PrecompileError, "runtime shape constraints"):
            torch.compiler.precompile(
                lambda mm, a, b: mm(a) + b, example_inputs=[(m, xi, yi)]
            )

    def test_unbacked_derived_runtime_constraint_rejected(self):
        def fn(x):
            y = x.nonzero()
            torch._check(y.shape[0] > 0)
            return y

        x = torch.ones(8)
        mark_unbacked(x, 0)
        with self.assertRaisesRegex(
            PrecompileError, "deferred runtime shape constraints"
        ):
            torch.compiler.precompile(fn, example_inputs=[(x,)])

    def test_grad_identity_preserved_across_precompile(self):
        # Capture snapshots and restores the example model's .grad by the SAME object (no
        # clone), so a caller holding a prior p.grad reference -- or optimizer state keyed
        # on grad identity -- is not invalidated. Warm up a backward to populate .grad,
        # snapshot the object identity, precompile a backward step on the same model, and
        # assert p.grad is still the SAME object afterward.
        torch.manual_seed(0)
        m = torch.nn.Linear(4, 3)
        x = torch.randn(5, 4)
        m(x).sum().backward()  # warmup populates .grad
        g = m.weight.grad
        self.assertIsNotNone(g)
        torch.compiler.precompile(
            lambda mm, t: mm(t).sum().backward(), example_inputs=[(m, x)]
        )
        self.assertIs(m.weight.grad, g)  # same object, not a clone

    def test_precompile_error_public_binding(self):
        # PrecompileError is a single public type reachable two ways
        # (torch.compiler.PrecompileError and torch.compiler.precompile.PrecompileError),
        # is a real exception type, is advertised in torch.compiler.__all__, and a raised
        # instance is catchable via the public torch.compiler.PrecompileError alias.
        self.assertIs(
            torch.compiler.PrecompileError, torch.compiler.precompile.PrecompileError
        )
        self.assertIsInstance(torch.compiler.PrecompileError, type)
        self.assertIn("PrecompileError", torch.compiler.__all__)
        # A real PrecompileError (here the invariant-1 constant-tensor guard) is catchable
        # via the public torch.compiler.PrecompileError alias.
        captured = torch.randn(3)
        with self.assertRaisesRegex(torch.compiler.PrecompileError, "hard-coded"):
            torch.compiler.precompile(
                lambda x: x + captured, example_inputs=[(torch.randn(3),)]
            )

    def test_single_trust_warning_on_inlined_load(self):
        # On the inlined load path (an eager artifact has an empty cache, so there is
        # nothing to prime and load() just EXECs python_code) the untrusted-input / EXEC
        # warning must fire EXACTLY ONCE -- only _make_inlined_forward warns. Asserting
        # "exactly once" guards against the EXEC warning being duplicated on this load.
        m = torch.nn.Sequential(torch.nn.Linear(4, 3)).eval()
        x = torch.randn(5, 4)
        code, cache = torch.compiler.precompile(
            lambda model, t: model(t), example_inputs=[(m, x)], backend="eager"
        )
        with self.assertLogs("torch._precompile", level="WARNING") as cm:
            torch.compiler.precompile.load(code, cache)
        exec_warnings = [line for line in cm.output if "EXEC" in line]
        self.assertEqual(
            len(exec_warnings), 1, f"expected one EXEC warning, got: {cm.output}"
        )
        self.assertTrue(any("untrusted" in line.lower() for line in cm.output))

    def test_tied_weights_single_input_single_grad(self):
        # Invariants 1/2/5: a weight tied across two layers is interned by identity to a
        # SINGLE graph input (PARAM_NAMES lists the first name once) and accumulates ONE
        # grad -- the sum of both uses -- matching an eager backward, not one grad per name.
        class Tied(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.l1 = torch.nn.Linear(4, 4, bias=False)
                self.l2 = torch.nn.Linear(4, 4, bias=False)
                self.l2.weight = self.l1.weight  # tie: same tensor, two names

            def forward(self, x):
                return self.l2(self.l1(x))

        m = Tied()
        t = torch.randn(5, 4)
        code, cache = torch.compiler.precompile(
            lambda model, t: model(t).sum().backward(), example_inputs=[(m, t)]
        )
        self.assertIn("PARAM_NAMES = ['l1.weight']", code)  # tie collapsed to one

        ref = copy.deepcopy(m)  # deepcopy preserves the tie within the object graph
        ref(t).sum().backward()

        torch.compiler.precompile.load(code, cache)(m, t)  # one call: tied grad
        self.assertEqual(m.l1.weight.grad, ref.l1.weight.grad)
        self.assertIs(m.l1.weight, m.l2.weight)  # still one tensor at runtime

    def test_multiple_module_args_all_lifted(self):
        # The multi=True naming branch: two DIFFERENT nn.Module args are BOTH lifted, their
        # positions recorded in MODULE_POSITIONS, and their params disambiguated as m0.* /
        # m1.* (per-module prefixes). Loaded artifact matches eager m2(m1(t)).
        torch.manual_seed(0)
        m1 = torch.nn.Linear(4, 4)
        m2 = torch.nn.Linear(4, 3)
        t = torch.randn(5, 4)
        code, cache = torch.compiler.precompile(
            lambda a, b, t: b(a(t)), example_inputs=[(m1, m2, t)]
        )
        self.assertIn("MODULE_POSITIONS = [0, 1]", code)
        self.assertIn("m0.weight", code)  # first module's params prefixed m0.*
        self.assertIn("m1.weight", code)  # second module's params prefixed m1.*
        f_c = torch.compiler.precompile.load(code, cache)
        self.assertEqual(f_c(m1, m2, t), m2(m1(t)))

    def test_frozen_param_keeps_none_grad(self):
        # Invariant 5 with a mix: only params that received a gradient are harvested
        # (recorded in GRAD_PARAM_INDICES), so a frozen (requires_grad=False) param keeps
        # .grad is None while a trainable param gets a grad matching an eager backward.
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.frozen = torch.nn.Linear(4, 4)
                self.trainable = torch.nn.Linear(4, 4)
                for p in self.frozen.parameters():
                    p.requires_grad_(False)

            def forward(self, x):
                return self.trainable(self.frozen(x))

        m = M()
        t = torch.randn(5, 4)
        code, cache = torch.compiler.precompile(
            lambda model, t: model(t).sum().backward(), example_inputs=[(m, t)]
        )

        ref = copy.deepcopy(m)
        ref(t).sum().backward()

        torch.compiler.precompile.load(code, cache)(m, t)
        for p in m.frozen.parameters():
            self.assertIsNone(p.grad)  # frozen: never harvested
        for p in m.trainable.parameters():
            self.assertIsNotNone(p.grad)
        for (n, p), (_, rp) in zip(
            m.trainable.named_parameters(), ref.trainable.named_parameters()
        ):
            self.assertEqual(p.grad, rp.grad, n)

    def test_requires_grad_flip_is_noop(self):
        # Which params get a scattered grad is fixed at CAPTURE time from the example
        # model's requires_grad (invariant 5); flipping a runtime param's requires_grad
        # does NOT change what the artifact computes. Capture with params requiring grad,
        # set requires_grad=False on the runtime model, and assert the grad is STILL
        # scattered (and matches eager) -- locking the documented contract.
        torch.manual_seed(0)
        m = torch.nn.Linear(4, 3)  # params require grad at capture
        x = torch.randn(5, 4)
        code, cache = torch.compiler.precompile(
            lambda mm, t: mm(t).sum().backward(), example_inputs=[(m, x)]
        )
        run = torch.nn.Linear(4, 3)
        run.load_state_dict(m.state_dict())
        for p in run.parameters():
            p.requires_grad_(False)  # flip OFF at runtime -- must be a no-op
        torch.compiler.precompile.load(code, cache)(run, x)
        self.assertIsNotNone(run.weight.grad)  # still scattered despite the flip
        ref = torch.nn.Linear(4, 3)
        ref.load_state_dict(m.state_dict())
        ref(x).sum().backward()
        self.assertEqual(run.weight.grad, ref.weight.grad)


@skipIfTorchDynamo("precompile's make_fx capture is incompatible with dynamo wrapping")
class TestPrecompileNumerics(TestCase):
    # Numeric-correctness tests run device-generically so the same coverage
    # exercises the CUDA lowering, not just CPU.

    def test_dynamo_training_replays_mutation_onto_restricted_view(self, device):
        class Scale(torch.autograd.Function):
            @staticmethod
            def forward(ctx, tensor):
                return tensor

            @staticmethod
            def backward(ctx, grad):
                return grad * 3

        def opaque_add_(weight, value):
            weight.data.add_(value)

        with torch.library._scoped_library(
            "precompile_mutation_replay", "FRAGMENT"
        ) as lib:
            lib.define("opaque_add_(Tensor(a!) weight, Tensor value) -> ()")
            lib.impl("opaque_add_", opaque_add_, "CompositeExplicitAutograd")
            lib.impl("opaque_add_", lambda weight, value: None, "Meta")

            example_base = make_tensor(
                (4,), device=device, dtype=torch.float32, requires_grad=True
            )
            example_weight = example_base * 1.0
            example_value = make_tensor(
                (4,), device=device, dtype=torch.float32, requires_grad=True
            )
            code, cache = torch.compiler.precompile(
                _precompile_mutation_replay_step,
                example_inputs=[(example_weight, example_value)],
                tracer="dynamo",
                backend="inductor",
                training=True,
            )
            self.assertIn(
                "from torch._functorch._aot_autograd.standalone_runtime import "
                "_replay_input_mutation",
                code,
            )
            self.assertNotIn(
                "runtime_wrappers import _replay_input_mutation",
                code,
            )

            for _, loaded in _default_and_inlined_loaders(code, cache, "inductor"):
                ref_base = make_tensor(
                    (4,), device=device, dtype=torch.float32, requires_grad=True
                )
                actual_base = ref_base.detach().clone().requires_grad_()
                ref_value = make_tensor(
                    (4,), device=device, dtype=torch.float32, requires_grad=True
                )
                actual_value = ref_value.detach().clone().requires_grad_()
                ref_weight = Scale.apply(ref_base * 1.0)
                actual_weight = Scale.apply(actual_base * 1.0)
                ref_version = ref_weight._version
                actual_version = actual_weight._version

                _precompile_mutation_replay_step(ref_weight, ref_value)
                with loaded:
                    loaded(actual_weight, actual_value)
                    self.assertEqual(loaded.serve_time_compiles(), 0)

                self.assertEqual(actual_base.grad, ref_base.grad)
                self.assertEqual(actual_value.grad, ref_value.grad)
                self.assertEqual(actual_weight, ref_weight)
                self.assertEqual(
                    actual_weight._version - actual_version,
                    ref_weight._version - ref_version,
                )

    def test_torch_compile_training_preserves_undefined_tangent(self, device):
        model = _PrecompileDynamoIndependentOutputs().to(device)
        x = make_tensor((2, 4), device=device, dtype=torch.float32)
        compiled = torch.compile(model, backend="inductor", fullgraph=True)
        compiled(x)[0].sum().backward()

        self.assertIsNotNone(model.left.weight.grad)
        self.assertIsNone(model.right.weight.grad)

    @unittest.skipIf(not torch.distributed.is_available(), "requires distributed")
    def test_torch_compile_training_async_collective_undefined_tangent(self, device):
        @torch.compile(backend="inductor", fullgraph=True)
        def compiled(x, y):
            return x.sin(), AsyncCollectiveTensor(y.cos())

        x = make_tensor((4,), device=device, dtype=torch.float32, requires_grad=True)
        y = make_tensor((4,), device=device, dtype=torch.float32, requires_grad=True)
        compiled(x, y)[0].sum().backward()

        self.assertEqual(x.grad, x.cos())
        self.assertIsNone(y.grad)

    def test_dynamo_training_serializes_tangent_masks(self, device):
        from torch._dynamo.utils import counters
        from torch._inductor.utils import fresh_cache

        torch.manual_seed(0)
        model = _PrecompileDynamoIndependentOutputs().to(device)
        x = make_tensor((2, 4), device=device, dtype=torch.float32)
        backward_calls: list[torch.Tensor | None] = []
        handles = [
            layer.weight.register_hook(lambda grad: backward_calls.append(grad))
            for layer in (model.left, model.right)
        ]
        counters.clear()
        try:
            with fresh_cache():
                code, cache = torch.compiler.precompile(
                    _precompile_dynamo_undefined_tangent_step,
                    example_inputs=[(model, x, True), (model, x, False)],
                    tracer="dynamo",
                    backend="inductor",
                    training=True,
                )
        finally:
            for handle in handles:
                handle.remove()

        self.assertEqual(counters["aot_autograd"]["autograd_cache_bypass"], 0)
        self.assertEqual(len(backward_calls), 4)
        self.assertEqual(sum(grad is None for grad in backward_calls), 2)
        self.assertIn("1: (_inner_call_bw_0", code)
        self.assertIn("2: (_inner_call_bw_1", code)
        self.assertIn("_AOT_DEFAULT_BACKWARD_VARIANT_s0 = None", code)
        self.assertIn("KeptTangentInfo", code)
        self.assertEqual(code.count("Inner Inductor output code: BACKWARD variant"), 2)

        for _, loaded in _default_and_inlined_loaders(code, cache, "inductor"):
            for use_first in (True, False):
                run = copy.deepcopy(model)
                ref = copy.deepcopy(model)
                loaded(run, x, use_first)
                _precompile_dynamo_undefined_tangent_step(ref, x, use_first)
                for actual, expected in zip(run.parameters(), ref.parameters()):
                    self.assertEqual(actual.grad, expected.grad)

    @torch._dynamo.config.patch(
        automatic_dynamic_shapes=True, assume_static_by_default=True
    )
    def test_dynamo_training_custom_backward_observes_none_tangent(self, device):
        from torch._dynamo.utils import counters
        from torch._inductor.utils import fresh_cache
        from torch.library import _scoped_library

        with _scoped_library("precompile_optional_tangents", "FRAGMENT") as lib:
            lib.define("split(Tensor x, Tensor y) -> (Tensor, Tensor)")
            lib.impl(
                "split",
                lambda x, y: (x * 2, y * 3),
                "CompositeExplicitAutograd",
            )
            lib.impl(
                "split",
                lambda x, y: (torch.empty_like(x), torch.empty_like(y)),
                "Meta",
            )

            def setup_context(ctx, inputs, output):
                ctx.set_materialize_grads(False)
                ctx.save_for_backward(*inputs)

            def backward(ctx, grad_x, grad_y):
                x, y = ctx.saved_tensors
                if grad_x is None:
                    return y * 5, grad_y * y
                if grad_y is None:
                    return grad_x * x, x * 7
                return grad_x * x, grad_y * y

            torch.library.register_autograd(
                "precompile_optional_tangents::split",
                backward,
                setup_context=setup_context,
                lib=lib,
            )
            examples = []
            for size, use_first in ((2, True), (3, False), (5, True)):
                x = make_tensor(
                    (size,), device=device, dtype=torch.float32, requires_grad=True
                )
                y = make_tensor(
                    (size,), device=device, dtype=torch.float32, requires_grad=True
                )
                examples.append((x, y, use_first))
            counters.clear()
            with fresh_cache():
                code, cache = torch.compiler.precompile(
                    _precompile_dynamo_optional_tangent_step,
                    example_inputs=examples,
                    tracer="dynamo",
                    backend="inductor",
                    training=True,
                )

            self.assertEqual(counters["aot_autograd"]["autograd_cache_bypass"], 0)
            self.assertIn("DYNAMIC_GRAPH_COUNT = 1", code)
            self.assertIn("1: (_inner_call_bw_0", code)
            self.assertIn("2: (_inner_call_bw_1", code)
            caches = (cache, _strip_artifact(cache))
            for artifact_cache in caches:
                with fresh_cache():
                    loaded = torch.compiler.precompile.load(code, artifact_cache)
                    for use_first in (True, False):
                        x = make_tensor(
                            (7,),
                            device=device,
                            dtype=torch.float32,
                            requires_grad=True,
                        )
                        y = make_tensor(
                            (7,),
                            device=device,
                            dtype=torch.float32,
                            requires_grad=True,
                        )
                        actual_x = x.detach().clone().requires_grad_()
                        actual_y = y.detach().clone().requires_grad_()
                        expected_x = x.detach().clone().requires_grad_()
                        expected_y = y.detach().clone().requires_grad_()
                        loaded(actual_x, actual_y, use_first)
                        _precompile_dynamo_optional_tangent_step(
                            expected_x, expected_y, use_first
                        )
                        self.assertEqual(actual_x.grad, expected_x.grad)
                        self.assertEqual(actual_y.grad, expected_y.grad)

    @torch._dynamo.config.patch(
        automatic_dynamic_shapes=True, assume_static_by_default=True
    )
    @unittest.skipIf(not torch.distributed.is_available(), "requires distributed")
    def test_dynamo_training_async_collective_tensor_undefined_tangent(self, device):
        from torch._dynamo.utils import counters
        from torch._inductor.utils import fresh_cache

        examples = []
        for size in (2, 3, 5):
            x = make_tensor(
                (size,), device=device, dtype=torch.float32, requires_grad=True
            )
            y = make_tensor(
                (size,), device=device, dtype=torch.float32, requires_grad=True
            )
            examples.append((x, y))
        counters.clear()
        with fresh_cache():
            code, cache = torch.compiler.precompile(
                _precompile_dynamo_async_collective_tangent_step,
                example_inputs=examples,
                tracer="dynamo",
                backend="inductor",
                training=True,
            )

        self.assertEqual(counters["aot_autograd"]["autograd_cache_bypass"], 0)
        self.assertIn("DYNAMIC_GRAPH_COUNT = 1", code)
        self.assertIn("2: (_inner_call_bw_0", code)
        self.assertNotIn("tangents_2", code)
        for artifact_cache in (cache, _strip_artifact(cache)):
            with fresh_cache():
                loaded = torch.compiler.precompile.load(code, artifact_cache)
                x = make_tensor(
                    (7,), device=device, dtype=torch.float32, requires_grad=True
                )
                y = make_tensor(
                    (7,), device=device, dtype=torch.float32, requires_grad=True
                )
                loaded(x, y)
                self.assertEqual(x.grad, x.cos())
                self.assertIsNone(y.grad)

        if device == "cpu":
            with tempfile.NamedTemporaryFile(
                "w", suffix=".py", delete=False
            ) as artifact:
                artifact.write(code)
                artifact_path = artifact.name
            try:
                subprocess.check_call(
                    [
                        sys.executable,
                        "-c",
                        textwrap.dedent(
                            """
                            import runpy
                            import sys
                            import torch

                            namespace = runpy.run_path(sys.argv[1])
                            x = torch.randn(7, requires_grad=True)
                            y = torch.randn(7, requires_grad=True)
                            namespace["forward"](x, y)
                            torch.testing.assert_close(x.grad, x.cos())
                            if y.grad is not None:
                                raise AssertionError(f"expected no y.grad, got {y.grad}")
                            """
                        ),
                        artifact_path,
                    ]
                )
            finally:
                os.unlink(artifact_path)

    def test_dynamo_training_dealiases_non_differentiable_output(self, device):
        from torch._dynamo.utils import counters
        from torch._inductor.utils import fresh_cache

        example = make_tensor(
            (8,), device=device, dtype=torch.float32, requires_grad=True
        )
        counters.clear()
        with fresh_cache():
            code, cache = torch.compiler.precompile(
                _precompile_dynamo_dealias_marked_returns_step,
                example_inputs=[(example,)],
                tracer="dynamo",
                backend="inductor",
                training=True,
            )

        self.assertEqual(counters["aot_autograd"]["autograd_cache_bypass"], 0)
        self.assertIn("7: (_inner_call_bw_0", code)
        self.assertNotIn("23: (_inner_call_bw_0", code)
        self.assertIn(
            "from torch._functorch._aot_autograd.standalone_runtime import "
            "_dealias_marked_returns",
            code,
        )
        self.assertNotIn(
            "runtime_wrappers import _dealias_marked_returns",
            code,
        )
        for _, loaded in _default_and_inlined_loaders(code, cache, "inductor"):
            x = make_tensor(
                (8,), device=device, dtype=torch.float32, requires_grad=True
            )
            loaded(x)
            self.assertEqual(x.grad, 3 + x.cos())

    @onlyCUDA
    def test_make_fx_autocast_tracks_graph_devices(self, device):
        def fn(x):
            moved = x.to(device)
            return moved @ moved.t()

        x = torch.randn(4, 8)
        code, cache = torch.compiler.precompile(
            fn, example_inputs=[(x,)], backend="eager"
        )
        self.assertIn("GRAPH_DEVICES = ('cpu', 'cuda')", code)
        loaded = torch.compiler.precompile.load(code, cache)
        expected = loaded(x)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            actual = loaded(x)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertEqual(actual, expected)

    @torch._dynamo.config.patch(
        automatic_dynamic_shapes=True, assume_static_by_default=True
    )
    def test_tracer_dynamo_recompiles_to_dynamic_graph(self, device):
        from torch._inductor.utils import fresh_cache

        examples = [
            (make_tensor((size, 4), device=device, dtype=torch.float32),)
            for size in (2, 3, 5)
        ]
        with fresh_cache():
            code, cache = torch.compiler.precompile(
                _precompile_dynamo_dynamic,
                example_inputs=examples,
                tracer="dynamo",
            )

        self.assertIn("DYNAMIC_GRAPH_COUNT = 1", code)
        if device != "cpu":
            self.assertIn("@triton.jit", code)
        loaded = torch.compiler.precompile.load(code, cache)
        x = make_tensor((7, 4), device=device, dtype=torch.float32)
        self.assertEqual(loaded(x), _precompile_dynamo_dynamic(x))

    @torch._dynamo.config.patch(
        automatic_dynamic_shapes=True, assume_static_by_default=True
    )
    def test_tracer_dynamo_graph_break(self, device):
        from torch._inductor.utils import fresh_cache

        examples = [
            (make_tensor((size, 4), device=device, dtype=torch.float32),)
            for size in (2, 3, 5)
        ]
        with fresh_cache():
            code, cache = torch.compiler.precompile(
                _precompile_dynamo_with_disabled,
                example_inputs=examples,
                tracer="dynamo",
            )

        self.assertIn("FRAME_COUNT = 2", code)
        self.assertIn("GRAPH_COUNT = 4", code)
        self.assertIn("DYNAMIC_GRAPH_COUNT = 2", code)
        if device != "cpu":
            self.assertIn("@triton.jit", code)
        for _, loaded in _default_and_inlined_loaders(code, cache, "inductor"):
            for size in (2, 7):
                x = make_tensor((size, 4), device=device, dtype=torch.float32)
                self.assertEqual(loaded(x), _precompile_dynamo_with_disabled(x))
            with self.assertRaisesRegex(PrecompileError, "no captured Dynamo variant"):
                loaded(make_tensor((1, 4), device=device, dtype=torch.float32))

    def test_tracer_dynamo_installed_graph_break(self, device):
        from torch._inductor.utils import fresh_cache
        from torch._precompile import _parse_artifact_metadata

        model = _PrecompileDynamoBreakingModule().to(device).eval()
        x = make_tensor((5, 4), device=device, dtype=torch.float32)
        with fresh_cache():
            code, cache = torch.compiler.precompile(
                _precompile_dynamo_call_module,
                example_inputs=[(model, x)],
                tracer="dynamo",
            )

        self.assertEqual(_parse_artifact_metadata(code)["SERVING_MODE"], "installed")
        if device != "cpu":
            self.assertIn("@triton.jit", code)
        for _, loaded in _default_and_inlined_loaders(code, cache, "inductor"):
            self.assertEqual(loaded(model, x), model(x))
            loaded.unload()

    @torch._dynamo.config.patch(
        automatic_dynamic_shapes=True, assume_static_by_default=True
    )
    def test_tracer_dynamo_training(self, device):
        from torch._inductor.utils import fresh_cache

        examples = [
            (
                make_tensor(
                    (size, 4), device=device, dtype=torch.float32, requires_grad=True
                ),
            )
            for size in (2, 3, 5)
        ]
        with fresh_cache():
            code, cache = torch.compiler.precompile(
                _precompile_dynamo_dynamic,
                example_inputs=examples,
                tracer="dynamo",
                training=True,
            )

        self.assertIn("DYNAMIC_GRAPH_COUNT = 1", code)
        self.assertIn("_inner_call_bw", code)
        if device != "cpu":
            self.assertIn("@triton.jit", code)
        for _, loaded in _default_and_inlined_loaders(code, cache, "inductor"):
            x = make_tensor(
                (7, 4), device=device, dtype=torch.float32, requires_grad=True
            )
            ref = x.detach().clone().requires_grad_()
            expected = _precompile_dynamo_dynamic(ref)
            expected.sum().backward()
            actual = loaded(x)
            actual.sum().backward()
            self.assertEqual(actual, expected)
            self.assertEqual(x.grad, ref.grad)

    def test_plain_function(self, device):
        def f(x, y):
            return (x @ y).sin(), x + y

        a = make_tensor((4, 4), device=device, dtype=torch.float32)
        b = make_tensor((4, 4), device=device, dtype=torch.float32)
        code, cache = torch.compiler.precompile(f, example_inputs=[(a, b)])
        self.assertIsInstance(code, str)
        self.assertIsInstance(cache, bytes)

        f_c = torch.compiler.precompile.load(code, cache)
        out = f_c(a, b)
        ref = f(a, b)
        self.assertEqual(out[0], ref[0])
        self.assertEqual(out[1], ref[1])

    def test_module_params_and_buffers_are_lifted(self, device):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = torch.nn.Linear(4, 3)
                self.register_buffer("b2", torch.randn(3))

            def forward(self, x):
                return torch.relu(self.lin(x)) + self.b2

        m = M().to(device).eval()
        x = make_tensor((5, 4), device=device, dtype=torch.float32)
        code, cache = torch.compiler.precompile(
            lambda model, x: model(x), example_inputs=[(m, x)]
        )
        f_c = torch.compiler.precompile.load(code, cache)
        self.assertEqual(f_c(m, x), m(x))

    def test_multiple_module_args(self, device):
        # More than one nn.Module arg: each module's params are lifted with
        # m{i}.-prefixed names. Both modules are passed again at runtime.
        a = torch.nn.Linear(4, 4).to(device).eval()
        b = torch.nn.Linear(4, 3).to(device).eval()
        x = make_tensor((2, 4), device=device, dtype=torch.float32)
        ref = b(torch.relu(a(x)))

        code, cache = torch.compiler.precompile(
            lambda ma, mb, x: mb(torch.relu(ma(x))), example_inputs=[(a, b, x)]
        )
        self.assertIn(
            "PARAM_NAMES = ['m0.weight', 'm0.bias', 'm1.weight', 'm1.bias']", code
        )

        f_c = torch.compiler.precompile.load(code, cache)
        self.assertEqual(f_c(a, b, x), ref)

    def test_inplace_on_intermediate_is_allowed(self, device):
        # In-place ops on intermediates (e.g. nn.ReLU(inplace=True)) are fine -- they
        # do not touch any input -- and must NOT be rejected as input mutation.
        m = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU(inplace=True))
        m.to(device).eval()
        x = make_tensor((5, 4), device=device, dtype=torch.float32)
        code, cache = torch.compiler.precompile(
            lambda model, x: model(x), example_inputs=[(m, x)]
        )
        f_c = torch.compiler.precompile.load(code, cache)
        self.assertEqual(f_c(m, x), m(x))

    def test_training_backward_harvest_matches_eager(self, device):
        # A training step that calls loss.backward(): precompile scatters the
        # parameter grads onto the runtime model's .grad fields (mirroring eager
        # .backward()) and returns fn's own result (None here).
        torch.manual_seed(0)
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 3)
        ).to(device)
        loss_fn = torch.nn.MSELoss()
        # Keep magnitudes small (make_tensor defaults to a wide range) so the SGD
        # loop below converges rather than diverges.
        x = make_tensor((5, 4), device=device, dtype=torch.float32, low=-1, high=1)
        target = make_tensor((5, 3), device=device, dtype=torch.float32, low=-1, high=1)

        ref = copy.deepcopy(model)
        loss_fn(ref(x), target).backward()
        ref_grads = [p.grad.clone() for p in ref.parameters()]

        def train_step(model, x, target):
            loss_fn(model(x), target).backward()

        code, cache = torch.compiler.precompile(
            train_step, example_inputs=[(model, x, target)]
        )
        f_c = torch.compiler.precompile.load(code, cache)

        # The model is passed at runtime (no weights baked); the artifact mutates
        # model.parameters().grad in place, returning fn's result (None).
        out = f_c(model, x, target)
        self.assertIsNone(out)
        for p, rg in zip(model.parameters(), ref_grads):
            self.assertEqual(p.grad, rg)

        # Grads accumulate like eager: a second call without zeroing doubles them.
        f_c(model, x, target)
        for p, rg in zip(model.parameters(), ref_grads):
            self.assertEqual(p.grad, rg * 2)

        # A standard zero_grad / step loop reduces loss.
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        losses = []
        for _ in range(5):
            opt.zero_grad()
            f_c(model, x, target)
            losses.append(loss_fn(model(x), target).item())
            opt.step()
        self.assertLess(losses[-1], losses[0])

    def test_frozen_params_grad_matches_eager(self, device):
        # Params that do not receive a gradient -- a frozen (requires_grad=False)
        # backbone, or a param that does not contribute to the loss -- must keep
        # .grad = None after the step, exactly like eager .backward(). precompile must
        # NOT zero-fill them (regression test for the old all-params zero-fill).
        torch.manual_seed(0)
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 3)
        ).to(device)
        for p in model[0].parameters():
            p.requires_grad_(False)  # freeze the first linear
        loss_fn = torch.nn.MSELoss()
        x = make_tensor((5, 4), device=device, dtype=torch.float32, low=-1, high=1)
        target = make_tensor((5, 3), device=device, dtype=torch.float32, low=-1, high=1)

        ref = copy.deepcopy(model)
        loss_fn(ref(x), target).backward()

        def train_step(model, x, target):
            loss_fn(model(x), target).backward()

        code, cache = torch.compiler.precompile(
            train_step, example_inputs=[(model, x, target)]
        )
        f_c = torch.compiler.precompile.load(code, cache)
        f_c(model, x, target)
        for (n, p), (_, rp) in zip(model.named_parameters(), ref.named_parameters()):
            if rp.grad is None:
                self.assertIsNone(p.grad, f"{n}: expected no grad, matching eager")
            else:
                self.assertEqual(p.grad, rp.grad)

    def test_multiple_modules_backward_grad_scatter(self, device):
        # Two distinct module args + a backward: grads must scatter onto the correct
        # module's params via the cross-module GRAD_PARAM_INDICES mapping. One module
        # is partly frozen so the test also pins the index shift across modules.
        torch.manual_seed(0)
        a = torch.nn.Linear(4, 4).to(device)
        b = torch.nn.Linear(4, 3).to(device)
        a.bias.requires_grad_(False)  # a frozen param shifts later indices
        loss_fn = torch.nn.MSELoss()
        x = make_tensor((5, 4), device=device, dtype=torch.float32, low=-1, high=1)
        target = make_tensor((5, 3), device=device, dtype=torch.float32, low=-1, high=1)

        ref_a, ref_b = copy.deepcopy(a), copy.deepcopy(b)
        loss_fn(ref_b(torch.relu(ref_a(x))), target).backward()

        def train_step(ma, mb, x, target):
            loss_fn(mb(torch.relu(ma(x))), target).backward()

        code, cache = torch.compiler.precompile(
            train_step, example_inputs=[(a, b, x, target)]
        )
        f_c = torch.compiler.precompile.load(code, cache)
        f_c(a, b, x, target)
        for (n, p), (_, rp) in zip(a.named_parameters(), ref_a.named_parameters()):
            if rp.grad is None:
                self.assertIsNone(p.grad, f"a.{n}: expected no grad")
            else:
                self.assertEqual(p.grad, rp.grad, f"a.{n}")
        for (n, p), (_, rp) in zip(b.named_parameters(), ref_b.named_parameters()):
            self.assertEqual(p.grad, rp.grad, f"b.{n}")

    def test_tied_weights_lifted_once(self, device):
        # A tied weight (same tensor under multiple names) must become a single
        # lifted input: otherwise it is double-counted (double optimizer step) and
        # gradients are split rather than accumulated.
        class Tied(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.nn.Linear(4, 4, bias=False)
                self.b = torch.nn.Linear(4, 4, bias=False)
                self.b.weight = self.a.weight  # tie

            def forward(self, x):
                return self.b(torch.relu(self.a(x)))

        torch.manual_seed(0)
        m = Tied().to(device)
        x = make_tensor((3, 4), device=device, dtype=torch.float32)

        code, cache = torch.compiler.precompile(
            lambda model, x: model(x), example_inputs=[(m, x)]
        )
        f_c = torch.compiler.precompile.load(code, cache)
        self.assertEqual(f_c(m, x), m(x))
        # The tied weight is lifted once (single name), so it is one graph input.
        self.assertIn("PARAM_NAMES = ['a.weight']", code)

        # Training scatters a single grad onto the shared weight, matching eager's
        # accumulation into the tied parameter.
        ref = copy.deepcopy(m)
        ref(x).sum().backward()
        ref_grad = ref.a.weight.grad

        code, cache = torch.compiler.precompile(
            lambda model, x: model(x).sum().backward(), example_inputs=[(m, x)]
        )
        f_c = torch.compiler.precompile.load(code, cache)
        f_c(m, x)
        self.assertEqual(m.a.weight.grad, ref_grad)
        # The tie means a.weight and b.weight are the same object, so b sees it too.
        self.assertIs(m.a.weight.grad, m.b.weight.grad)

    def test_backend_eager_plain_function(self, device):
        # backend="eager" runs the captured graph as-is and matches eager.
        def f(x, y):
            return (x @ y).sin(), x + y

        a = make_tensor((4, 4), device=device, dtype=torch.float32)
        b = make_tensor((4, 4), device=device, dtype=torch.float32)
        code, cache = torch.compiler.precompile(
            f, example_inputs=[(a, b)], backend="eager"
        )
        f_c = torch.compiler.precompile.load(code, cache)
        out = f_c(a, b)
        ref = f(a, b)
        self.assertEqual(out[0], ref[0])
        self.assertEqual(out[1], ref[1])

    def test_backend_eager_module(self, device):
        m = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU())
        m.to(device).eval()
        x = make_tensor((5, 4), device=device, dtype=torch.float32)
        code, cache = torch.compiler.precompile(
            lambda model, x: model(x), example_inputs=[(m, x)], backend="eager"
        )
        f_c = torch.compiler.precompile.load(code, cache)
        self.assertEqual(f_c(m, x), m(x))

    def test_backend_eager_training_harvest(self, device):
        # The backward-harvest contract holds for the eager backend too.
        torch.manual_seed(0)
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 3)
        ).to(device)
        loss_fn = torch.nn.MSELoss()
        x = make_tensor((5, 4), device=device, dtype=torch.float32, low=-1, high=1)
        target = make_tensor((5, 3), device=device, dtype=torch.float32, low=-1, high=1)

        ref = copy.deepcopy(model)
        loss_fn(ref(x), target).backward()
        ref_grads = [p.grad.clone() for p in ref.parameters()]

        def train_step(model, x, target):
            loss_fn(model(x), target).backward()

        code, cache = torch.compiler.precompile(
            train_step, example_inputs=[(model, x, target)], backend="eager"
        )
        f_c = torch.compiler.precompile.load(code, cache)
        out = f_c(model, x, target)
        self.assertIsNone(out)
        for p, rg in zip(model.parameters(), ref_grads):
            self.assertEqual(p.grad, rg)

    def test_backend_eager_batchnorm(self, device):
        # The captured graph bakes a ``device`` constant (BatchNorm's
        # num_batches_tracked path), one of fx's custom builtins. The eager
        # standalone source must inject the full custom-builtin set, else this
        # raises NameError: name 'device' is not defined.
        def fresh():
            torch.manual_seed(0)
            m = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4))
            m.train()
            return m.to(device)

        x = make_tensor((8, 4), device=device, dtype=torch.float32)
        ref = fresh()
        ref_out = ref(x)
        ref_rm = ref[1].running_mean.clone()

        code, cache = torch.compiler.precompile(
            lambda m, xx: m(xx), example_inputs=[(fresh(), x)], backend="eager"
        )
        f_c = torch.compiler.precompile.load(code, cache)
        run = fresh()
        self.assertEqual(f_c(run, x), ref_out)
        self.assertEqual(run[1].running_mean, ref_rm)

    def test_backend_eager_inf_constant(self, device):
        # masked_fill to -inf bakes a bare ``inf`` token into gm.code (another fx
        # custom builtin); the eager standalone source must provide it.
        def f(x):
            return torch.relu(x).masked_fill(x < 0, float("-inf"))

        x = make_tensor((8,), device=device, dtype=torch.float32)
        code, cache = torch.compiler.precompile(
            f, example_inputs=[(x,)], backend="eager"
        )
        f_c = torch.compiler.precompile.load(code, cache)
        self.assertEqual(f_c(x), f(x))

    def test_batchnorm_train_with_backward(self, device):
        # Training a model containing BatchNorm exercises buffer mutation (running
        # stats) and grad harvest together; grads and running stats must match eager.
        # Inductor fuses the BN backward, so rely on assertEqual's tolerance.
        def fresh():
            torch.manual_seed(0)
            m = torch.nn.Sequential(
                torch.nn.Linear(4, 8), torch.nn.BatchNorm1d(8), torch.nn.Linear(8, 3)
            )
            m.train()
            return m.to(device)

        loss_fn = torch.nn.MSELoss()
        x = make_tensor((16, 4), device=device, dtype=torch.float32, low=-1, high=1)
        target = make_tensor(
            (16, 3), device=device, dtype=torch.float32, low=-1, high=1
        )

        ref = fresh()
        loss_fn(ref(x), target).backward()
        ref_grads = [p.grad.clone() for p in ref.parameters()]
        ref_rm = ref[1].running_mean.clone()

        def train_step(model, x, target):
            loss_fn(model(x), target).backward()

        code, cache = torch.compiler.precompile(
            train_step, example_inputs=[(fresh(), x, target)]
        )
        f_c = torch.compiler.precompile.load(code, cache)
        run = fresh()
        f_c(run, x, target)
        for p, rg in zip(run.parameters(), ref_grads):
            self.assertEqual(p.grad, rg)
        self.assertEqual(run[1].running_mean, ref_rm)

    def test_output_alias_supported(self, device):
        # An output that is a view of an input goes through AOTAutograd's output-
        # alias epilogue; precompile reproduces it.
        x = make_tensor((2, 3), device=device, dtype=torch.float32)
        code, cache = torch.compiler.precompile(lambda a: a.t(), example_inputs=[(x,)])
        f_c = torch.compiler.precompile.load(code, cache)
        self.assertEqual(f_c(x), x.t())

    def test_input_mutation_supported(self, device):
        # In-place input mutation is reflected on the passed tensor (and matches
        # eager), via AOTAutograd's mutation handling composed into the artifact.
        scratch = make_tensor((4,), device=device, dtype=torch.float32)
        code, cache = torch.compiler.precompile(
            lambda a: a.add_(1.0), example_inputs=[(scratch,)]
        )
        f_c = torch.compiler.precompile.load(code, cache)
        x = torch.zeros(4, device=device)
        out = f_c(x)
        self.assertEqual(x, torch.ones(4, device=device))
        self.assertEqual(out, torch.ones(4, device=device))

    @unittest.skipUnless(TEST_CUDA, "functionalize_rng_ops seeds via CUDA rng state")
    def test_functionalized_rng_supported(self, device):
        # Functionalized RNG (dropout) threads seed/offset; the AOT backend composes
        # the RNG wrapper in. The artifact runs and produces a valid dropout mask. Even
        # for a CPU tensor the wrapper seeds from CUDARngStateHelper.get_torch_state_as_tuple,
        # which raises unless CUDA is available, so the whole test is gated on TEST_CUDA
        # rather than on the tensor's device.
        import torch._functorch.config as functorch_config

        x = make_tensor((64,), device=device, dtype=torch.float32)
        with functorch_config.patch(functionalize_rng_ops=True):
            code, cache = torch.compiler.precompile(
                lambda a: torch.nn.functional.dropout(a, 0.5, training=True),
                example_inputs=[(x,)],
            )
            f_c = torch.compiler.precompile.load(code, cache)
            out = f_c(x)
        self.assertEqual(out.shape, x.shape)
        self.assertTrue((out == 0).any())

    def test_batchnorm_train_buffer_mutation(self, device):
        # A stateful module (BatchNorm in training mode) mutates its running stats.
        # precompile reflects that onto the runtime model's buffers and matches eager
        # -- the mutation handling comes from AOTAutograd's codegen.
        def fresh():
            torch.manual_seed(0)
            m = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4))
            m.train()
            return m.to(device)

        x = make_tensor((8, 4), device=device, dtype=torch.float32)
        code, cache = torch.compiler.precompile(
            lambda model, xx: model(xx), example_inputs=[(fresh(), x)]
        )

        ref = fresh()
        ref_out = ref(x)
        ref_rm = ref[1].running_mean.clone()
        ref_rv = ref[1].running_var.clone()
        ref_nbt = ref[1].num_batches_tracked.clone()

        f_c = torch.compiler.precompile.load(code, cache)
        run = fresh()
        out = f_c(run, x)
        self.assertEqual(out, ref_out)
        self.assertEqual(run[1].running_mean, ref_rm)
        self.assertEqual(run[1].running_var, ref_rv)
        self.assertEqual(run[1].num_batches_tracked, ref_nbt)

    def test_mutated_duplicate_input(self, device):
        # The same tensor passed twice with a mutation: make_fx resolves the aliasing
        # at trace time (the graph mutates one input and reuses the result), so the
        # artifact reproduces eager when run with the same aliasing. Storage-aliased
        # mutated inputs go through AOTAutograd's now-codegen'd synthetic-base wrapper.
        fn = lambda a, b: (a.mul_(2.0), a + b)[1]  # noqa: E731
        t = make_tensor((4,), device=device, dtype=torch.float32)
        # Clone references BEFORE precompile: capture runs fn once, mutating t.
        ref = t.clone()
        ref_out = fn(ref, ref)
        run = t.clone()

        code, cache = torch.compiler.precompile(fn, example_inputs=[(t, t)])
        f_c = torch.compiler.precompile.load(code, cache)
        out = f_c(run, run)
        self.assertEqual(out, ref_out)

    def test_dynamic_shapes_runs_across_sizes(self, device):
        # An UNBACKED-dynamic batch dim (opted in via mark_unbacked on the input): one
        # artifact runs on many runtime batch sizes (cached AND inlined paths), matching
        # eager. Device-generic so the CUDA unbacked-symint lowering is exercised.
        m = torch.nn.Sequential(
            torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 3)
        )
        m.to(device).eval()
        x = make_tensor((8, 4), device=device, dtype=torch.float32)
        mark_unbacked(x, 0)
        code, cache = torch.compiler.precompile(
            lambda mm, t: mm(t), example_inputs=[(m, x)]
        )
        self.assertIn("USER_INPUT_SHAPES = [(None, 4)]", code)  # dim 0 dynamic
        f_c = torch.compiler.precompile.load(code, cache)
        blob = torch.load(io.BytesIO(cache), weights_only=True)
        blob["artifact"] = None
        buf = io.BytesIO()
        torch.save(blob, buf)
        f_i = torch.compiler.precompile.load(code, buf.getvalue())
        for bs in (8, 16, 1):
            xt = make_tensor((bs, 4), device=device, dtype=torch.float32)
            self.assertEqual(f_c(m, xt), m(xt))  # cached path
            self.assertEqual(f_i(m, xt), m(xt))  # inlined path

    def test_dynamic_shapes_training_across_sizes(self, device):
        # Training (backward) with a dynamic batch; harvested grads match eager across
        # sizes (loss is output.sum() so no cross-input dim-equality guard is needed).
        # Device-generic so the CUDA unbacked-symint backward lowering is exercised.
        torch.manual_seed(0)
        m = torch.nn.Linear(4, 3).to(device)
        x = make_tensor((8, 4), device=device, dtype=torch.float32)
        mark_unbacked(x, 0)
        code, cache = torch.compiler.precompile(
            lambda model, t: model(t).sum().backward(), example_inputs=[(m, x)]
        )
        f_c = torch.compiler.precompile.load(code, cache)
        for bs in (8, 16, 5):
            run = torch.nn.Linear(4, 3).to(device)
            run.load_state_dict(m.state_dict())
            ref = torch.nn.Linear(4, 3).to(device)
            ref.load_state_dict(m.state_dict())
            xt = make_tensor((bs, 4), device=device, dtype=torch.float32)
            f_c(run, xt)
            ref(xt).sum().backward()
            self.assertEqual(run.weight.grad, ref.weight.grad)

    def test_dynamic_shapes_shared_shape_id(self, device):
        # Two inputs whose batch dims share a shape_id reuse ONE unbacked symbol, so a
        # cross-input matched-batch op (here an add) traces with no dim-equality guard and
        # runs across sizes. Device-generic so the CUDA lowering is exercised.
        m = torch.nn.Linear(4, 4).to(device).eval()
        x = make_tensor((8, 4), device=device, dtype=torch.float32)
        y = make_tensor((8, 4), device=device, dtype=torch.float32)
        mark_unbacked(x, 0, shape_id="b")
        mark_unbacked(y, 0, shape_id="b")
        code, cache = torch.compiler.precompile(
            lambda mm, a, b: mm(a) + b, example_inputs=[(m, x, y)]
        )
        f_c = torch.compiler.precompile.load(code, cache)
        for bs in (8, 16, 3):
            xt = make_tensor((bs, 4), device=device, dtype=torch.float32)
            yt = make_tensor((bs, 4), device=device, dtype=torch.float32)
            self.assertEqual(f_c(m, xt, yt), m(xt) + yt)

    def test_mark_unbacked_strict_honored(self, device):
        # mark_unbacked(x, 0, strict=True) is HONORED: the dim is captured as an unbacked
        # symint, so USER_INPUT_SHAPES records None for it and the single artifact runs
        # across runtime sizes, matching eager (device-generic for CUDA coverage).
        m = torch.nn.Linear(4, 3).to(device).eval()
        x = make_tensor((8, 4), device=device, dtype=torch.float32)
        mark_unbacked(x, 0, strict=True)
        code, cache = torch.compiler.precompile(
            lambda mm, t: mm(t), example_inputs=[(m, x)]
        )
        self.assertIn("USER_INPUT_SHAPES = [(None, 4)]", code)
        f_c = torch.compiler.precompile.load(code, cache)
        for bs in (8, 16, 2):
            xt = make_tensor((bs, 4), device=device, dtype=torch.float32)
            self.assertEqual(f_c(m, xt), m(xt))

    def test_unbacked_zero_batch_runs(self, device):
        # bs=0 on an unbacked dynamic dim is a valid runtime size (the symbol is >= 0);
        # the artifact runs on an empty batch and matches eager.
        m = torch.nn.Linear(4, 3).to(device).eval()
        x = make_tensor((8, 4), device=device, dtype=torch.float32)
        mark_unbacked(x, 0)
        code, cache = torch.compiler.precompile(
            lambda mm, t: mm(t), example_inputs=[(m, x)]
        )
        f_c = torch.compiler.precompile.load(code, cache)
        xt = make_tensor((0, 4), device=device, dtype=torch.float32)
        self.assertEqual(f_c(m, xt), m(xt))

    def test_channels_last_marked_input_roundtrips(self, device):
        # A channels_last-marked dynamic input round-trips at the SAME layout for a
        # LAYOUT-PRESERVING (pointwise) op: _detect_memory_format records channels_last so
        # the refaked leaf preserves it, and the artifact accepts a channels_last runtime
        # input (matching eager). (conv output has a separate inductor layout limitation,
        # so this uses a pointwise op.)
        x = make_tensor((2, 3, 4, 4), device=device, dtype=torch.float32)
        x = x.to(memory_format=torch.channels_last)
        self.assertTrue(x.is_contiguous(memory_format=torch.channels_last))
        mark_unbacked(x, 0)
        code, cache = torch.compiler.precompile(
            lambda t: torch.relu(t) * 2.0, example_inputs=[(x,)]
        )
        f_c = torch.compiler.precompile.load(code, cache)
        xt = make_tensor((5, 3, 4, 4), device=device, dtype=torch.float32)
        xt = xt.to(memory_format=torch.channels_last)
        out = f_c(xt)
        self.assertEqual(out, torch.relu(xt) * 2.0)

    def test_marked_exotic_layout_rejected(self, device):
        # _detect_memory_format cannot preserve a layout that is neither contiguous nor
        # channels_last(_3d) through the refake, so a mark_unbacked input in such a layout
        # (here a transposed, non-contiguous 2D tensor) is rejected LOUDLY at capture rather
        # than silently forced contiguous (which would bake a wrong assert_size_stride).
        # Transpose makes a non-contiguous (8, 4) tensor in neither channels_last format.
        x = make_tensor((4, 8), device=device, dtype=torch.float32).t()
        self.assertFalse(x.is_contiguous())
        mark_unbacked(x, 0)
        with self.assertRaisesRegex(PrecompileError, "memory format"):
            torch.compiler.precompile(
                lambda t: t.contiguous() * 2.0, example_inputs=[(x,)]
            )

    def test_eager_backend_input_mutation(self, device):
        # The eager backend replays the raw ATen graph, so input mutation is reflected on
        # the passed tensor and matches eager, like the inductor backend.
        scratch = make_tensor((4,), device=device, dtype=torch.float32)
        code, cache = torch.compiler.precompile(
            lambda a: a.add_(1.0), example_inputs=[(scratch,)], backend="eager"
        )
        f_c = torch.compiler.precompile.load(code, cache)
        x = torch.zeros(4, device=device)
        out = f_c(x)
        self.assertEqual(x, torch.ones(4, device=device))
        self.assertEqual(out, torch.ones(4, device=device))

    def test_eager_backend_output_alias(self, device):
        # The eager backend reproduces an output that aliases an input (a view), matching
        # eager, via the raw ATen replay.
        x = make_tensor((2, 3), device=device, dtype=torch.float32)
        code, cache = torch.compiler.precompile(
            lambda a: a.t(), example_inputs=[(x,)], backend="eager"
        )
        f_c = torch.compiler.precompile.load(code, cache)
        self.assertEqual(f_c(x), x.t())


instantiate_device_type_tests(TestPrecompileNumerics, globals())


if __name__ == "__main__":
    run_tests()
