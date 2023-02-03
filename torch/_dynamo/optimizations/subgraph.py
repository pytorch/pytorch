import functools
import importlib
import itertools
import json
import logging
import math
import operator
import os

import torch

from .. import config
from ..utils import check_is_cuda, checkpoint_params, is_jit_model, torchscript

log = logging.getLogger(__name__)


def cached(fn):
    cached_name = f"_{fn.__name__}"

    @functools.wraps(fn)
    def inner(self):
        if hasattr(self, cached_name):
            return getattr(self, cached_name)
        result = fn(self)
        setattr(self, cached_name, result)
        return result

    return inner


def load_module_fx(name):
    pymod = importlib.import_module(f"subgraphs.{name}")
    # TODO(jansel): upstream these fixes to to_folder()
    pymod.module._operator_iadd = operator.iadd
    pymod.module._operator_imul = operator.imul
    pymod.module._operator_itruediv = operator.itruediv
    pymod.module._operator_setitem = operator.setitem
    pymod.module.math_sqrt = math.sqrt
    pymod.module.device = torch.device
    pymod.module.inf = float("inf")
    return pymod.FxModule()


def load_module_jit(name):
    filename = os.path.join(config.base_dir, "subgraphs", name, "model.ts")
    if not os.path.exists(filename):
        return None
    model = torch.jit.load(filename)
    assert is_jit_model(model)
    return model


class SubGraph(object):
    @classmethod
    def load(cls, name):
        model_dir = os.path.join(config.base_dir, "subgraphs", name)
        example_inputs = torch.load(os.path.join(model_dir, "example_inputs.pt"))
        example_outputs = torch.load(os.path.join(model_dir, "example_outputs.pt"))
        metadata = json.loads(open(os.path.join(model_dir, "metadata.json")).read())
        model_fx = load_module_fx(name)
        model_jit = load_module_jit(name)
        is_cuda = metadata["is_cuda"]

        assert model_jit is not None

        torch.set_rng_state(torch.load(os.path.join(model_dir, "rng_state.pt")))
        if is_cuda:
            model_jit = model_jit.cuda()
        restore_jit = checkpoint_params(model_jit)
        if model_fx is not None:
            if is_cuda:
                model_fx = model_fx.cuda()
            restore_fx = checkpoint_params(model_fx)
        else:
            model_fx = model_jit
            restore_fx = restore_jit

        def restore():
            restore_fx()
            restore_jit()

        subgraph = cls(model_fx, example_inputs, model_dir)
        subgraph._scripted = model_jit
        subgraph._example_outputs = example_outputs
        subgraph._is_cuda = is_cuda
        subgraph.restore = restore
        return subgraph

    def __init__(self, model, example_inputs, model_dir):
        super(SubGraph, self).__init__()
        self.model = model
        self.example_inputs = example_inputs
        self.model_dir = model_dir

    def filename(self, name):
        return os.path.join(self.model_dir, name)

    @property
    @cached
    def scripted(self):
        return torchscript(self.model, self.example_inputs)

    @property
    @cached
    def example_outputs(self):
        filename = self.filename("example_outputs.pt")
        if os.path.exists(filename):
            return torch.load(filename)
        result = self.model(*self.example_inputs)
        torch.save(result, filename)
        return result

    @property
    def example_outputs_list(self):
        if self.is_tensor_output:
            return [self.example_outputs]
        return self.example_outputs

    @property
    def input_names(self):
        return [f"i{i}" for i in range(len(self.example_inputs))]

    @property
    def is_tensor_output(self):
        return not isinstance(self.example_outputs, (list, tuple))

    @property
    def output_names(self):
        return [f"o{x}" for x in range(len(self.example_outputs_list))]

    @property
    def device_index(self):
        return 0

    @property
    @cached
    def onnx_filename(self):
        filename = self.filename("onnx")
        if os.path.exists(filename):
            return filename

        try:
            torch.onnx.export(
                self.scripted,
                self.example_inputs,
                filename,
                input_names=self.input_names,
                output_names=self.output_names,
                do_constant_folding=True,
                opset_version=14,
            )
        except IndexError:
            # work around bug in constant folding pass
            torch.onnx.export(
                self.scripted,
                self.example_inputs,
                filename,
                input_names=self.input_names,
                output_names=self.output_names,
                do_constant_folding=False,
                opset_version=14,
            )
        return filename

    @property
    def is_cpu(self):
        return not self.is_cuda

    @property
    @cached
    def is_cuda(self):
        return check_is_cuda(self.model, self.example_inputs)

    @property
    def output_specs(self):
        return [
            (o.shape, o.dtype, o.layout, o.device, o.requires_grad)
            for o in self.example_outputs_list
        ]

    def empty_outputs_factory(self):
        specs = self.output_specs

        def create():
            return [
                torch.empty(
                    shape,
                    dtype=dtype,
                    layout=layout,
                    device=device,
                    requires_grad=requires_grad,
                )
                for shape, dtype, layout, device, requires_grad in specs
            ]

        return create

    def wrap_returns(self, fn):
        """Fix [Tensor()] vs Tensor() return type issues"""
        expected = self.example_outputs
        actual = fn(*self.example_inputs)
        if isinstance(expected, (list, tuple)) and not isinstance(
            actual, (list, tuple)
        ):
            assert len(expected) == 1
            if isinstance(expected, tuple):
                return lambda *args: (fn(*args),)
            else:
                return lambda *args: [fn(*args)]
        elif not isinstance(expected, (list, tuple)) and isinstance(
            actual, (list, tuple)
        ):
            assert len(actual) == 1
            return lambda *args: fn(*args)[0]
        elif isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
            assert len(actual) == len(expected)
            return fn
        else:
            return fn

    def has_dtype(self, dtype):
        for x in itertools.chain(
            self.example_inputs, self.scripted.parameters(), self.scripted.buffers()
        ):
            if x.dtype == dtype:
                return True
        return False

    def will_tensorrt_barf(self):
        return False
        # code = torch.jit.freeze(self.scripted).code
        # TODO(jansel): submit a bug report for this one, issue is in opacus_cifar10
        # if "group_norm" in code or "einsum" in code:
        #    return True
        # return self.has_dtype(torch.int64)
