from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any, TYPE_CHECKING

import torch
import torch.utils._pytree as pytree
from torch._C._functorch import (
    CFunctionalizeInterpreterPtr,
    CGradInterpreterPtr,
    CInterpreter,
    CJvpInterpreterPtr,
    CVmapInterpreterPtr,
    pop_dynamic_layer_stack,
    push_dynamic_layer_stack,
    RandomnessType,
    TransformType,
)
from torch.autograd.forward_ad import _set_fwd_grad_enabled


if TYPE_CHECKING:
    from collections.abc import Generator


"""
This file contains the functorch integration with PyDispatcher.

PyDispatcher does not understand functorch's DynamicLayerStack dispatching
logic because it is entirely implemented in C++ in the fallbacks for two
dispatch keys, FuncTorchDynamicLayer{Front, Back}Mode (PyDispatcher is unable
to directly reuse C++ boxed fallbacks).

Instead of trying to hammer PyDispatcher into understanding those fallbacks,
we re-implement the logic of peeking the top of the stack for an interpreter,
selecting the interpreter to dispatch on, etc, in Python. This leads to a
simpler design.

The main difference between C++ functorch and PyDispatcher's functorch logic
is that:
- C++ functorch needs to manually tweak dispatch keys to ping-pong between
  DynamicLayerFrontMode and DynamicLayerBackMode.
- PyDispatcher's functorch logic pops an Interpreter from the top of the stack
  and asks it to execute the rule associated with the Interpreter.

In C++ we do the ping-pong because e.g. vmap rules are associated with the
batched DispatchKey, but in PyDispatcher we are able to avoid this by asking
the user to register a batching rule directly to a transform that an
interpreter then invokes.
"""


# FuncTorchInterpreter is the Python version of Interpreter (recall that
# the DynamicLayerStack is a stack of interpreters).
# It is a wrapper around the actual C++ Interpreter object.
#
# Keep the methods in sync with aten/src/ATen/functorch/Interpreter.h
class FuncTorchInterpreter(ABC):
    def __init__(self, cptr: Any) -> None:
        self._cptr = cptr

    # Process an operation. eg for vmap, this is invoking a batching rule.
    # Conceptually this is analogous to Interpreter::process in C++
    @abstractmethod
    def process(self, op: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        pass

    # lower an operation from this Interpreter to the next Interpreter on the stack.
    # Concretely, this involves temporarily popping the current Interpreter.
    # Conceptually this is analogous to Interpreter::sendToNextInterpreter in C++
    def lower(self) -> contextlib.AbstractContextManager[Any]:
        return temporarily_pop_interpreter_stack()

    def level(self) -> int:
        return self._cptr.level()

    def key(self) -> TransformType:
        return self._cptr.key()

    def get_state(self) -> tuple[Any, ...]:
        raise NotImplementedError

    def check_state(self, state: tuple[Any, ...]) -> bool:
        return state == self.get_state()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("_cptr", None)
        return state


@contextlib.contextmanager
def temporarily_pop_interpreter_stack() -> Generator[None, None, None]:
    try:
        saved = pop_dynamic_layer_stack()
        yield
    finally:
        push_dynamic_layer_stack(saved)


@contextlib.contextmanager
def temporarily_clear_interpreter_stack() -> Generator[list[Any], None, None]:
    stack: list[Any] = []
    try:
        while torch._C._functorch.peek_interpreter_stack() is not None:
            stack.append(pop_dynamic_layer_stack())
        yield list(stack)
    finally:
        while stack:
            push_dynamic_layer_stack(stack.pop())


@contextlib.contextmanager
def temporarily_restore_interpreter_stack(
    stack: list[Any] | None,
) -> Generator[None, None, None]:
    pushed: list[Any] = []
    if stack is None:
        return
    try:
        for s in reversed(stack):
            push_dynamic_layer_stack(s)
            pushed.append(s)
        yield
    finally:
        for _ in reversed(pushed):
            # TODO: would be nice to assert that the layers are the same, but
            # Python object identity is not preserved
            pop_dynamic_layer_stack()


class VmapInterpreter(FuncTorchInterpreter):
    def __init__(self, cdata: CInterpreter) -> None:
        if cdata.key() != TransformType.Vmap:
            raise AssertionError(f"expected TransformType.Vmap, got {cdata.key()}")
        # NOTE: [Interpreter cdata vs cptr]
        # cdata is a generic CInterpreter. We wrap it in a CVmapInterpreterPtr
        # so that we can access methods specific to the vmap interpreter
        self._cdata = cdata

    @cached_property
    # pyrefly: ignore [bad-override]
    def _cptr(self) -> CVmapInterpreterPtr:
        return CVmapInterpreterPtr(self._cdata)

    def process(self, op: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        kernel = op.functorch_table[TransformType.Vmap]
        return kernel(self, *args, **kwargs)

    def batch_size(self) -> int:
        return self._cptr.batchSize()

    def randomness(self) -> str:
        typ = self._cptr.randomness()
        if typ == RandomnessType.Error:
            return "error"
        elif typ == RandomnessType.Same:
            return "same"
        elif typ == RandomnessType.Different:
            return "different"
        raise RuntimeError(f"Unknown RandomnessType: {typ}")

    def get_state(self) -> tuple[Any, ...]:
        return (self.key().name, self.level(), self.randomness())


@contextlib.contextmanager
def nested(
    *contexts: contextlib.AbstractContextManager[Any],
) -> Generator[tuple[contextlib.AbstractContextManager[Any], ...], None, None]:
    with contextlib.ExitStack() as stack:
        for ctx in contexts:
            stack.enter_context(ctx)
        yield contexts


class GradInterpreter(FuncTorchInterpreter):
    def __init__(self, cdata: CInterpreter) -> None:
        if cdata.key() != TransformType.Grad:
            raise AssertionError(f"expected TransformType.Grad, got {cdata.key()}")
        # See NOTE: [Interpreter cdata vs cptr]
        self._cdata = cdata

    @cached_property
    # pyrefly: ignore [bad-override]
    def _cptr(self) -> CGradInterpreterPtr:
        return CGradInterpreterPtr(self._cdata)

    def lift(
        self, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        args, kwargs = pytree.tree_map_only(
            torch.Tensor, self._cptr.lift, [args, kwargs]
        )
        return args, kwargs

    def process(self, op: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        kernel = op.functorch_table[TransformType.Grad]
        args, kwargs = self.lift(args, kwargs)
        return kernel(self, *args, **kwargs)

    # GradInterpreter has custom lower because of the no_grad interaction
    # See NOTE [grad and vjp interaction with no_grad]
    # This logic is mirrored from C++ GradInterpreterPtr::sendToNextInterpreter
    def lower(self) -> contextlib.AbstractContextManager[Any]:
        prev_grad_mode = self.prev_grad_mode()
        if not prev_grad_mode:
            return nested(torch.no_grad(), super().lower())
        return super().lower()

    def prev_grad_mode(self) -> bool:
        return self._cptr.prevGradMode()

    def get_state(self) -> tuple[Any, ...]:
        return (self.key().name, self.level(), self.prev_grad_mode())


class JvpInterpreter(FuncTorchInterpreter):
    def __init__(self, cdata: CInterpreter) -> None:
        if cdata.key() != TransformType.Jvp:
            raise AssertionError(f"expected TransformType.Jvp, got {cdata.key()}")
        # See NOTE: [Interpreter cdata vs cptr]
        self._cdata = cdata

    @cached_property
    # pyrefly: ignore [bad-override]
    def _cptr(self) -> CJvpInterpreterPtr:
        return CJvpInterpreterPtr(self._cdata)

    def lift(
        self, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        args, kwargs = pytree.tree_map_only(
            torch.Tensor, self._cptr.lift, [args, kwargs]
        )
        return args, kwargs

    def process(self, op: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        kernel = op.functorch_table[TransformType.Jvp]
        args, kwargs = self.lift(args, kwargs)
        return kernel(self, *args, **kwargs)

    # Jvp has custom lower because of the no_fwd_grad interaction
    # See NOTE [grad and vjp interaction with no_grad] for related info.
    # This logic is mirrored from C++ JvpInterpreterPtr::sendToNextInterpreter
    def lower(self) -> contextlib.AbstractContextManager[Any]:
        prev_fwd_grad_mode = self.prev_fwd_grad_mode()
        if not prev_fwd_grad_mode:
            return nested(_set_fwd_grad_enabled(False), super().lower())
        return super().lower()

    def prev_fwd_grad_mode(self) -> bool:
        return self._cptr.prevFwdGradMode()

    def get_state(self) -> tuple[Any, ...]:
        return (self.key().name, self.level(), self.prev_fwd_grad_mode())


class FunctionalizeInterpreter(FuncTorchInterpreter):
    def __init__(self, cdata: CInterpreter) -> None:
        if cdata.key() != TransformType.Functionalize:
            raise AssertionError(
                f"expected TransformType.Functionalize, got {cdata.key()}"
            )
        self._cdata = cdata

    @cached_property
    # pyrefly: ignore [bad-override]
    def _cptr(self) -> CFunctionalizeInterpreterPtr:
        return CFunctionalizeInterpreterPtr(self._cdata)

    def process(self, op: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        kernel = op.functorch_table[TransformType.Functionalize]
        return kernel(self, *args, **kwargs)

    def functionalize_add_back_views(self) -> bool:
        return self._cptr.functionalizeAddBackViews()

    def get_state(self) -> tuple[Any, ...]:
        return (self.key().name, self.level())


def coerce_cinterpreter(cinterpreter: CInterpreter) -> FuncTorchInterpreter:
    key = cinterpreter.key()
    if key == TransformType.Grad:
        return GradInterpreter(cinterpreter)
    if key == TransformType.Vmap:
        return VmapInterpreter(cinterpreter)
    if key == TransformType.Jvp:
        return JvpInterpreter(cinterpreter)
    if key == TransformType.Functionalize:
        return FunctionalizeInterpreter(cinterpreter)
    raise RuntimeError(f"NYI: PyDispatcher has not implemented support for {key}")


def retrieve_current_functorch_interpreter() -> FuncTorchInterpreter:
    interpreter = torch._C._functorch.peek_interpreter_stack()
    if interpreter is None:
        raise AssertionError("interpreter must not be None")
    return coerce_cinterpreter(interpreter)


def retrieve_all_functorch_interpreters() -> list[FuncTorchInterpreter]:
    cis = torch._C._functorch.get_interpreter_stack()
    if cis is None:
        return []
    return [coerce_cinterpreter(ci) for ci in cis]


def compare_functorch_state(states: list[tuple[Any, ...]]) -> bool:
    # There are four possible cases covered here:
    # 1. Current stack empty AND stack when generated not empty -> Invalidate
    # 2. Current stack not empty AND stack when generated empty -> Invalidate
    # 3. Current stack and generated stack empty -> Valid FX graph
    # 4. Current stack and generated stack not empty -> Valid if both states match
    peek = torch._C._functorch.peek_interpreter_stack()
    if (peek is None and len(states) != 0) or (peek is not None and len(states) == 0):
        return False

    cis = retrieve_all_functorch_interpreters()
    return len(cis) == len(states) and all(
        ci.check_state(state) for ci, state in zip(cis, states)
    )


def dispatch_functorch(op: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    interpreter = retrieve_current_functorch_interpreter()
    # In traditional PyTorch operators, DispatchKey::FuncTorchTensorWrapper's
    # unwrap_dead_tensors fallback handles unwrapping dead tensor wrappers.
    # PyDispatcher sidesteps the PyTorch dispatcher when dealing with functorch
    # transforms, so we manually unwrap the dead tensors here.
    # This logic won't need to exist when we have mode-only functorch.
    args, kwargs = pytree.tree_map_only(
        torch.Tensor, torch._C._functorch.unwrap_if_dead, (args, kwargs)
    )
    return interpreter.process(op, args, kwargs)
