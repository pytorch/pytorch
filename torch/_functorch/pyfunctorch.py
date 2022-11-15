from abc import ABC, abstractmethod
import contextlib
from typing import Any
import torch
import torch.utils._pytree as pytree
from torch._C._functorch import (
    TransformType,
    CInterpreter,
    CGradInterpreterPtr,
    CVmapInterpreterPtr,
    pop_dynamic_layer_stack,
    push_dynamic_layer_stack,
)

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
    def __init__(self, cptr: Any):
        self._cptr = cptr

    # Process an operation. eg for vmap, this is invoking a batching rule.
    # Conceptually this is analogous to Interpreter::process in C++
    @abstractmethod
    def process(self, op, args, kwargs):
        pass

    # lower an operation from this Interpreter to the next Interpreter on the stack.
    # Concretely, this involves temporarily popping the current Interpreter.
    # Conceptually this is analogous to Interpreter::sendToNextInterpreter in C++
    def lower(self):
        return temporarily_pop_interpreter_stack()

    def level(self):
        return self._cptr.level()

    def key(self):
        return self._cptr.key()


@contextlib.contextmanager
def temporarily_pop_interpreter_stack():
    try:
        saved = pop_dynamic_layer_stack()
        yield
    finally:
        push_dynamic_layer_stack(saved)


class VmapInterpreter(FuncTorchInterpreter):
    def __init__(self, cdata: CInterpreter):
        assert cdata.key() == TransformType.Vmap
        # NOTE: [Interpreter cdata vs cptr]
        # cdata is a generic CInterpreter. We wrap it in a CVmapInterpreterPtr
        # so that we can access methods specific to the vmap interpreter
        self._cdata = cdata
        self._cptr = CVmapInterpreterPtr(cdata)

    def process(self, op, args, kwargs):
        kernel = op.functorch_table[TransformType.Vmap]
        return kernel(self, *args, **kwargs)

    def batch_size(self):
        return self._cptr.batchSize()


class GradInterpreter(FuncTorchInterpreter):
    def __init__(self, cdata: CInterpreter):
        assert cdata.key() == TransformType.Grad
        # See NOTE: [Interpreter cdata vs cptr]
        self._cdata = cdata
        self._cptr = CGradInterpreterPtr(cdata)

    def lift(self, args, kwargs):
        args, kwargs = pytree.tree_map_only(torch.Tensor, self._cptr.lift, [args, kwargs])
        return args, kwargs

    def process(self, op, args, kwargs):
        kernel = op.functorch_table[TransformType.Grad]
        args, kwargs = self.lift(args, kwargs)
        return kernel(self, *args, **kwargs)

    # GradInterpreter has custom lower because of the no_grad interaction
    # See NOTE [grad and vjp interaction with no_grad]
    # This logic is mirrored from C++ GradInterpreterPtr::sendToNextInterpreter
    def lower(self):
        prev_grad_mode = self.prev_grad_mode()
        if not self.prev_grad_mode:
            return contextlib.nested(torch.no_grad(), super().lower())
        return super().lower()

    def prev_grad_mode(self):
        return self._cptr.prevGradMode()


def coerce_cinterpreter(cinterpreter: CInterpreter) -> FuncTorchInterpreter:
    key = cinterpreter.key()
    if key == TransformType.Grad:
        return GradInterpreter(cinterpreter)
    if key == TransformType.Vmap:
        return VmapInterpreter(cinterpreter)
    raise RuntimeError(f"NYI: PyDispatcher has not implemented support for {key}")


def retrieve_current_functorch_interpreter():
    interpreter = torch._C._functorch.peek_interpreter_stack()
    assert interpreter is not None
    return coerce_cinterpreter(interpreter)


def dispatch_functorch(op, args, kwargs):
    interpreter = retrieve_current_functorch_interpreter()
    return interpreter.process(op, args, kwargs)
