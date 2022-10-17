import logging
import operator
import os
import time

import sympy
from sympy import Integer

import torch
import torch.fx
from torch._decomp import get_decompositions
from torch.utils._mode_utils import no_dispatch

from . import config, ir
from .codegen.wrapper import WrapperCodeGen
from .exc import (
    LoweringException,
    MissingOperatorWithDecomp,
    MissingOperatorWithoutDecomp,
)
from .ir import Constant, FixedLayout, InputBuffer, TensorBox
from .lowering import lowerings, make_fallback, needs_realized_inputs
from .sizevars import SizeVarAllocator
from .utils import dynamo_logging, dynamo_utils
from .virtualized import V

log = logging.getLogger(__name__)


class GraphLowering(torch.fx.Interpreter):
    def symbolic_sizes_strides(self, ex: torch.Tensor):
        """
        Support dynamic shapes and dynamic strides by assigning variables
        to each dimension.  We duck-shape tensors, so if two tensors
        have the same size they get assigned the same symbolic variable.
        """
        size = [self.sizevars[i] for i in ex.size()]
        stride = [None] * len(size)
        for i, val in enumerate(ex.stride()):
            if val in (0, 1):
                stride[i] = Integer(val)
        while any(x is None for x in stride):
            candidates = {
                ex.size(i) * ex.stride()[i]: size[i] * stride[i]
                for i in range(len(size))
                if stride[i] is not None and ex.stride()[i] >= 0
            }
            # iterate over unbound strides in sorted order
            val_list = sorted(
                [(ex.stride()[i], i) for i in range(len(stride)) if stride[i] is None]
            )
            for _, i in val_list:
                if stride[i] is None and ex.stride()[i] in candidates:
                    stride[i] = candidates[ex.stride()[i]]
                    candidates[ex.size(i) * ex.stride()[i]] = size[i] * stride[i]
            if any(x is None for x in stride):
                # bind the smallest unbound stride to a new variable
                val, i = sorted(
                    [
                        (ex.stride()[i], i)
                        for i in range(len(stride))
                        if stride[i] is None
                    ]
                )[0]
                stride[i] = self.sizevars[val]
        return size, stride

    def static_sizes_strides(self, ex: torch.Tensor):
        """
        Primarily used to weights
        """
        size = [sympy.Integer(i) for i in ex.size()]
        stride = [sympy.Integer(i) for i in ex.stride()]
        return size, stride

    def __init__(self, gm: torch.fx.GraphModule, num_dynamic_inputs=None):
        super().__init__(gm)
        self.sizevars = SizeVarAllocator("s")
        self.graph_inputs = {}
        self.graph_inputs_original = {}
        self.graph_outputs = None
        self.device_types = set()
        self.buffers = []
        self.constants = {}
        self.removed_buffers = set()
        self.inplaced_to_remove = set()
        self.wrapper_code = None
        self.num_dynamic_inputs = num_dynamic_inputs
        self.num_static_inputs = None
        self.mutated_inputs = set()
        self.unaligned_buffers = set()
        self.randomness_offset = sympy.Integer(0)
        self.randomness_seeds = []
        self.name_to_buffer = {}
        self.creation_time = time.time()

    def get_dtype(self, buffer_name):
        if buffer_name in self.constants:
            return self.constants[buffer_name].dtype
        if buffer_name in self.name_to_buffer:
            return self.name_to_buffer[buffer_name].get_dtype()
        if buffer_name in self.graph_inputs:
            return self.graph_inputs[buffer_name].get_dtype()
        raise KeyError(f"could not find {buffer_name}")

    def random_seed_buffer(self, device: torch.device):
        """
        Return a device-unique 1-element tensor storing our RNG seed.
        This will get initialized at the start of each graph in
        `wrapper.py`.

        Note this is only used by cuda backends.  The CPU backend handles
        RNG seeds as a sizevar.
        """
        name = f"seed_{device.type}_{device.index}"
        if name not in self.constants:
            self.constants[name] = torch.zeros((), device=device, dtype=torch.int64)
            self.randomness_seeds.append(name)

        return ir.RandSeedBuffer(
            name=name,
            layout=ir.FixedLayout(
                device=device,
                dtype=torch.int64,
                size=[],
                stride=[],
            ),
        )

    def increment_randomness_offset(self, numel):
        """
        A global counter of how many random numbers we have handed out so far.
        """
        offset = self.randomness_offset
        self.randomness_offset = offset + numel
        return offset

    @dynamo_utils.dynamo_timed
    def run(self, *args):
        if self.num_dynamic_inputs is None:
            self.num_dynamic_inputs = len(args)
        self.num_static_inputs = len(args) - self.num_dynamic_inputs
        return super().run(*args)

    def register_buffer(self, buffer: ir.ComputedBuffer):
        name = f"buf{len(self.buffers)}"
        self.buffers.append(buffer)
        self.name_to_buffer[name] = buffer
        return name

    def realize_users_of(self, name: str):
        """
        When a buffer is mutated we need to make sure all the reads to
        the old version are realized before the mutation happens.
        """
        assert isinstance(name, str)

        def visit(value):
            if isinstance(value, (list, tuple)):
                return [visit(x) for x in value]
            if isinstance(value, ir.IRNode):
                if value.is_user_of(name):
                    value.realize()
            return value

        for key, value in self.env.items():
            try:
                visit(value)
            except Exception:
                log.warning("error in realize_users_of", exc_info=True)

    def add_tensor_constant(self, data):
        def allocate():
            for name, value in self.constants.items():
                if (
                    data.size() == value.size()
                    and data.stride() == value.stride()
                    and data.dtype == value.dtype
                    and data.device == value.device
                    and torch.eq(data, value).all()
                ):
                    return name
            name = f"constant{len(self.constants)}"
            self.constants[name] = data
            return name

        return TensorBox.create(
            ir.ConstantBuffer(
                allocate(),
                FixedLayout(data.device, data.dtype, *self.static_sizes_strides(data)),
            )
        )

    def constant_name(self, name: str, device_override: torch.device):
        """
        We AOT copy constants to the devices they are needed on.
        If device_override doesn't match the constant's device, then
        copy it and return a different name.
        """
        if self.constants[name].device == device_override or device_override is None:
            return name
        alt_name = f"{name}_{device_override.type}{device_override.index or 0}"
        if alt_name not in self.constants:
            self.constants[alt_name] = self.constants[name].to(device_override)
        return alt_name

    def placeholder(self, target, args, kwargs):
        example: torch.Tensor = super().placeholder(target, args, kwargs)
        if config.static_weight_shapes and (
            len(self.graph_inputs) < self.num_static_inputs or not config.dynamic_shapes
        ):
            # the first N inputs are weights
            sizes, strides = self.static_sizes_strides(example)
        else:
            sizes, strides = self.symbolic_sizes_strides(example)
        # TODO(jansel): handle input aliasing
        tensor = TensorBox.create(
            InputBuffer(
                target,
                FixedLayout(example.device, example.dtype, sizes, strides),
            )
        )
        self.graph_inputs[target] = tensor
        self.graph_inputs_original[target] = tensor.data.data
        if example.dim() != 0:
            self.device_types.add(example.device.type)
        return tensor

    def call_function(self, target, args, kwargs):
        if target is operator.getitem and isinstance(args[0], (list, tuple)):
            return super().call_function(target, args, kwargs)

        if target not in lowerings:
            if config.implicit_fallbacks:
                error = (
                    MissingOperatorWithDecomp
                    if get_decompositions([target])
                    else MissingOperatorWithoutDecomp
                )
                log.warning(
                    "Creating implicit fallback for:\n%s",
                    error.operator_str(target, args, kwargs),
                )
                make_fallback(target)
            elif get_decompositions([target]):
                # There isn't a good way to dynamically patch this in
                # since AOT Autograd already ran.  The error message tells
                # the user how to fix it.
                raise MissingOperatorWithDecomp(target, args, kwargs)
            else:
                raise MissingOperatorWithoutDecomp(target, args, kwargs)

        try:
            return lowerings[target](*args, **kwargs)
        except Exception as e:
            raise LoweringException(e, target, args, kwargs) from e

    def get_attr(self, target, args, kwargs):
        # this is a constant
        value = getattr(self.module, target)
        with no_dispatch():
            if value.shape == ():
                return Constant(value.item(), value.dtype, value.device)
            if len(value.shape) == 1 and value.shape[0] <= 8:
                # tensor lowering has constant inlining logic
                from .lowering import tensor

                return tensor(value.tolist(), dtype=value.dtype, device=value.device)

        return self.add_tensor_constant(value)

    def call_module(self, target, args, kwargs):
        raise AssertionError()

    def call_method(self, target, args, kwargs):
        raise AssertionError()

    def output(self, target, args, kwargs):
        result = super().output(target, args, kwargs)
        assert isinstance(result, (tuple, list)), type(result)
        assert all(
            isinstance(x, (TensorBox, ir.Constant, type(None), ir.ConstantBuffer))
            for x in result
        ), result
        self.graph_outputs = [ir.ExternKernel.realize_input(x) for x in result]
        for name, value in self.graph_inputs.items():
            value.realize()
            assert isinstance(value, TensorBox)
            value = value.data
            assert isinstance(value, ir.StorageBox)
            value_storage_box = value
            value = value.data
            if not isinstance(value, InputBuffer) or value.get_name() != name:
                # one of our inputs was mutated, need to turn that into a copy
                ir.MutationLayout.realize_into(value, self.graph_inputs_original[name])
                # replace output with mutated input
                try:
                    ind = self.graph_outputs.index(value_storage_box)
                    self.graph_outputs[ind] = self.graph_inputs_original[name]
                except ValueError:
                    pass

        self.finalize()

    def finalize(self):
        for buf in self.buffers:
            buf.decide_layout()

    def run_node(self, n: torch.fx.Node):
        with ir.IRNode.current_origins({n}):
            result = super().run_node(n)
            num_users = len(set(n.users))
            if num_users > 1 and isinstance(result, TensorBox):
                for user in n.users:
                    if user.target in needs_realized_inputs or user.op == "output":
                        result.realize_hint()

                # TODO(jansel): introduce a store vs inline choice
                result.mark_reuse(len(n.users))
        return result

    def codegen(self):
        from .scheduler import Scheduler

        self.wrapper_code = WrapperCodeGen()
        self.scheduler = Scheduler(self.buffers)
        self.scheduler.codegen()
        return self.wrapper_code.generate()

    @dynamo_utils.dynamo_timed
    def compile_to_module(self):
        from .codecache import PyCodeCache

        code = self.codegen()
        if config.debug:
            print(code)

        mod = PyCodeCache.load(code)
        for name, value in self.constants.items():
            setattr(mod, name, value)

        log.log(dynamo_logging.CODE, "Output code: %s", mod.__file__)
        V.debug.output_code(mod.__file__)
        V.debug.rename(os.path.splitext(mod.__file__)[0] + ".debug")
        return mod

    def compile_to_fn(self):
        return self.compile_to_module().call

    def get_output_names(self):
        return [
            node.get_name()
            for node in self.graph_outputs
            if not isinstance(node, ir.NoneAsConstantBuffer)
        ]
