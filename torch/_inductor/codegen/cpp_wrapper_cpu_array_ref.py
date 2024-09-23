# mypy: allow-untyped-defs
from itertools import count
from typing import Dict, Optional

import torch
import torch._inductor.async_compile  # noqa: F401 required to warm up AsyncCompile pools

from .. import config, ir
from .cpp_utils import cexpr
from .cpp_wrapper_cpu import CppWrapperCpu


BufferName = str


class CppWrapperCpuArrayRef(CppWrapperCpu):
    """
    Generates cpp wrapper for running on CPU and calls cpp kernels

    This class is forked from CppWrapperCpu, with a difference that tensors may be
    represented as ArrayRef, see torch/csrc/inductor/aoti_runtime/arrayref_tensor.h
    """

    def __init__(self):
        if not hasattr(self, "device"):
            self.device = "cpu"
        super().__init__()
        self.declare = "auto "
        self.declare_maybe_reference = "decltype(auto) "
        self.ending = ";"
        self.open_bracket = "{"
        self.closed_bracket = "}"
        self.comment = "//"
        self.namespace = "at::"
        self.none_str = "nullptr" if config.abi_compatible else "at::Tensor()"
        self.extern_call_ops = set()
        self.size = "sizes()"
        self.stride = "strides()"
        self.supports_intermediate_hooks = False
        self.outputs_need_copy = set()
        self.kernel_callsite_id = count()
        self.var_array_id = (
            count()
        )  # for different types of local array variable declarations
        self.declared_var_array_vars = set()
        self.int_array_id = count()  # for int array local variable declarations
        self.declared_int_array_vars = set()
        self.tmp_tensor_id = count()  # for tmp tensor local variable declarations
        self.arg_var_id = count()
        self.used_cached_devices = set()
        self.used_cached_dtypes = set()
        self.used_cached_layouts = set()
        self.cached_output_id = count()
        self.scalar_to_tensor_id = count()
        self.custom_op_wrapper_loaded = False
        self.expr_printer = cexpr
        self.allow_stack_allocation: Optional[bool] = None
        self.stack_allocated_buffers: Dict[BufferName, ir.Buffer] = {}
