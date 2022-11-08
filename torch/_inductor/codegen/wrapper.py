import collections
import dataclasses
import functools
import hashlib
from itertools import count
from typing import Any, Dict, List

from .. import codecache, config, ir
from ..codecache import cpp_compile_command, get_code_path
from ..utils import dynamo_utils, has_triton, sympy_dot, sympy_product
from ..virtualized import V
from .common import CodeGen, DeferredLine, IndentedBuffer, Kernel
from .cpp import DTYPE_TO_ATEN
from .triton import texpr

pexpr = texpr


def buffer_reuse_key(node: ir.Buffer):
    size = node.get_size()
    stride = node.get_stride()
    last_element = sympy_dot([s - 1 for s in size], stride)
    return (
        node.get_device(),
        node.get_dtype(),
        V.graph.sizevars.simplify(sympy_product(size)),
        # Detect gaps in tensor storage caused by strides
        V.graph.sizevars.size_hint(last_element),
    )


def make_buffer_reuse(old, new, del_func, declare, ending, as_strided):
    assert old.get_dtype() == new.get_dtype()
    del_line = ""
    if old.get_name() not in V.graph.get_output_names():
        del_line = del_func(old.get_name())
    if old.get_size() == new.get_size() and old.get_stride() == new.get_stride():
        return f"{declare}{new.get_name()} = {old.get_name()}{del_line}{ending}"

    return (
        f"{declare}{new.get_name()} = {as_strided}({old.get_name()}, "
        f"{V.graph.sizevars.codegen_shape_tuple(new.get_size())}, "
        f"{V.graph.sizevars.codegen_shape_tuple(new.get_stride())}){del_line}{ending}"
    )


def make_buffer_allocation(buffer):
    device = buffer.get_device()
    dtype = buffer.get_dtype()
    shape = tuple(buffer.get_size())
    stride = tuple(buffer.get_stride())
    return (
        f"{buffer.get_name()} = empty_strided("
        f"{V.graph.sizevars.codegen_shape_tuple(shape)}, "
        f"{V.graph.sizevars.codegen_shape_tuple(stride)}, "
        f"device='{device.type}', dtype={dtype})"
    )


def make_cpp_buffer_allocation(buffer):
    # TODO: map layout and device here
    dtype = buffer.get_dtype()
    shape = tuple(buffer.get_size())
    stride = tuple(buffer.get_stride())
    return (
        f"auto {buffer.get_name()} = at::empty_strided("
        f"{V.graph.sizevars.codegen_shape_tuple(shape)}, "
        f"{V.graph.sizevars.codegen_shape_tuple(stride)}, "
        f"{DTYPE_TO_ATEN[dtype]}); "
    )


class MemoryPlanningState:
    def __init__(self):
        super().__init__()
        self.reuse_pool: Dict[
            Any, List["FreeIfNotReusedLine"]
        ] = collections.defaultdict(list)

    def __contains__(self, key):
        return bool(self.reuse_pool.get(key, None))

    def pop(self, key) -> "FreeIfNotReusedLine":
        item = self.reuse_pool[key].pop()
        assert not item.is_reused
        return item

    def push(self, key, item: "FreeIfNotReusedLine"):
        assert not item.is_reused
        self.reuse_pool[key].append(item)


class MemoryPlanningLine:
    def plan(self, state: MemoryPlanningState) -> "MemoryPlanningLine":
        """First pass to find reuse"""
        return self

    def codegen(self, code: IndentedBuffer):
        """Second pass to output code"""
        pass


@dataclasses.dataclass
class AllocateLine(MemoryPlanningLine):
    node: ir.Buffer

    def plan(self, state: MemoryPlanningState):
        if self.node.get_name() in V.graph.removed_buffers:
            return NullLine()

        # try to reuse a recently freed buffer
        key = buffer_reuse_key(self.node)
        if key in state:
            free_line = state.pop(key)
            free_line.is_reused = True
            return ReuseLine(free_line.node, self.node)

        return self

    def codegen(self, code: IndentedBuffer):
        assert self.node.get_name() not in V.graph.removed_buffers
        code.writeline(make_buffer_allocation(self.node))


@dataclasses.dataclass
class CppAllocateLine(AllocateLine):
    def codegen(self, code: IndentedBuffer):
        assert self.node.get_name() not in V.graph.removed_buffers
        code.writeline(make_cpp_buffer_allocation(self.node))


@dataclasses.dataclass
class FreeIfNotReusedLine(MemoryPlanningLine):
    node: ir.Buffer
    is_reused: bool = False

    def plan(self, state: MemoryPlanningState):
        assert not self.is_reused
        if self.node.get_name() in V.graph.removed_buffers:
            return NullLine()
        state.push(buffer_reuse_key(self.node), self)
        return self

    def codegen(self, code: IndentedBuffer):
        assert self.node.get_name() not in V.graph.removed_buffers
        if not self.is_reused:
            code.writeline(f"del {self.node.get_name()}")


@dataclasses.dataclass
class ReuseLine(MemoryPlanningLine):
    node: ir.Buffer
    reused_as: ir.Buffer

    def plan(self, state: MemoryPlanningState):
        assert self.node.get_name() not in V.graph.removed_buffers
        assert self.reused_as.get_name() not in V.graph.removed_buffers
        return self

    def codegen(self, code: IndentedBuffer):
        assert self.node.get_name() not in V.graph.removed_buffers
        assert self.reused_as.get_name() not in V.graph.removed_buffers
        code.writeline(
            make_buffer_reuse(
                self.node,
                self.reused_as,
                del_func=lambda name: f"; del {name}",
                declare="",
                ending="",
                as_strided="as_strided",
            )
            + "  # reuse"
        )


@dataclasses.dataclass
class CppReuseLine(ReuseLine):
    node: ir.Buffer
    reused_as: ir.Buffer

    def codegen(self, code: IndentedBuffer):
        assert self.node.get_name() not in V.graph.removed_buffers
        assert self.reused_as.get_name() not in V.graph.removed_buffers
        code.writeline(
            make_buffer_reuse(
                self.node,
                self.reused_as,
                del_func=lambda name: f"; {name}.reset()",
                declare="auto ",
                ending=";",
                as_strided="at::as_strided",
            )
            + "  // reuse"
        )


@dataclasses.dataclass
class FreeLine(MemoryPlanningLine):
    node: ir.Buffer

    def plan(self, state: MemoryPlanningState):
        if self.node.get_name() in V.graph.removed_buffers:
            return NullLine()
        return self

    def codegen(self, code: IndentedBuffer):
        assert self.node.get_name() not in V.graph.removed_buffers
        code.writeline(f"del {self.node.get_name()}")


class NullLine(MemoryPlanningLine):
    pass


class WrapperCodeGen(CodeGen):
    """
    The outer wrapper that calls the kernels.
    """

    def __init__(self):
        super().__init__()
        self._names_iter = count()
        self.header = IndentedBuffer()
        self.prefix = IndentedBuffer()
        self.wrapper_call = IndentedBuffer()
        self.kernels = {}
        self.lines = []
        self.header.splice(
            f"""
                from ctypes import c_void_p, c_long
                import torch
                import random
                from torch import empty_strided, as_strided, device
                from {codecache.__name__} import AsyncCompile

                aten = torch.ops.aten
                assert_size_stride = torch._C._dynamo.guards.assert_size_stride
                async_compile = AsyncCompile()

            """
        )

        if has_triton():
            self.header.splice(
                f"""
                import triton
                import triton.language as tl
                from {config.inductor_import}.triton_ops.autotune import grid
                from torch._C import _cuda_getCurrentRawStream as get_cuda_stream
                """
            )

            if config.triton.convolution != "aten":
                self.header.splice(
                    f"""
                    from {config.inductor_import}.triton_ops.conv_perf_model import early_config_prune
                    from {config.inductor_import}.triton_ops.conv_perf_model import estimate_conv_time
                    from {config.inductor_import}.triton_ops.autotune import conv_heuristics
                    """
                )

            if config.triton.mm != "aten":
                self.header.splice(
                    f"""
                    from {config.inductor_import}.triton_ops.autotune import mm_heuristics
                    from {config.inductor_import}.triton_ops.autotune import mm_autotune
                    """
                )

            if config.triton.use_bmm:
                self.header.writeline(
                    f"from {config.inductor_import}.triton_ops.batched_matmul import bmm_out as triton_bmm_out"
                )

        self.set_output_refs()
        self.write_prefix()

        for name, value in V.graph.constants.items():
            # include a hash so our code cache gives different constants different files
            hashed = hashlib.sha256(repr(value).encode("utf-8")).hexdigest()
            self.header.writeline(f"{name} = None  # {hashed}")

        self.allocated = set()
        self.freed = set()
        self.write_get_cuda_stream = functools.lru_cache(None)(
            self.write_get_cuda_stream
        )

    def set_output_refs(self):
        self.output_refs = [x.codegen_reference() for x in V.graph.graph_outputs]

    def write_prefix(self):
        self.prefix.splice(
            """

            async_compile.wait(globals())
            del async_compile

            def call(args):
            """
        )
        with self.wrapper_call.indent():
            inp_len = len(V.graph.graph_inputs.keys())
            if inp_len != 0:
                lhs = f"{', '.join(V.graph.graph_inputs.keys())}{'' if inp_len != 1 else ','}"
                self.wrapper_call.writeline(f"{lhs} = args")
                self.wrapper_call.writeline("args.clear()")
            for name in V.graph.randomness_seeds:
                self.wrapper_call.writeline(
                    f"torch.randint(2**31, size=(), dtype=torch.int64, out={name})"
                )
            V.graph.sizevars.codegen(self.wrapper_call, V.graph.graph_inputs)

    def write_get_cuda_stream(self, index):
        name = f"stream{index}"
        self.writeline(f"{name} = get_cuda_stream({index})")
        return name

    def next_kernel_name(self):
        return f"kernel{next(self._names_iter)}"

    def write_allocate_line(self, buffer):
        self.writeline(AllocateLine(buffer))

    def get_deferred_line(self, name, layout):
        return DeferredLine(
            name, f"{name} = {layout.view.codegen_reference()}  # alias"
        )

    def codegen_allocation(self, buffer):
        name = buffer.get_name()
        if name in V.graph.removed_buffers or name in self.allocated:
            return
        self.allocated.add(name)

        if isinstance(
            buffer,
            (ir.ExternKernelAlloc, ir.MultiOutput),
        ):
            return

        layout = buffer.get_layout()
        if isinstance(layout, ir.MutationLayout):
            return
        if isinstance(layout, ir.AliasedLayout):
            assert isinstance(layout.view, ir.ReinterpretView)
            if not layout.maybe_guard_aligned():
                V.graph.unaligned_buffers.add(name)
            self.codegen_allocation(layout.view.data)
            allocation = self.get_deferred_line(name, layout)
            self.writeline(allocation)
            return

        self.write_allocate_line(buffer)

    def write_del_line(self, name):
        self.writeline(f"del {name}")

    def write_free_if_not_reused_line(self, buffer):
        self.writeline(FreeIfNotReusedLine(buffer))

    def codegen_free(self, buffer):
        name = buffer.get_name()

        # can be freed but not reused
        if isinstance(buffer, ir.InputBuffer):
            self.write_del_line(name)
            return

        if not self.can_reuse(buffer):
            return
        self.freed.add(name)

        layout = buffer.get_layout()
        if isinstance(layout, (ir.AliasedLayout, ir.MultiOutputLayout)):
            self.write_del_line(name)
            return

        self.write_free_if_not_reused_line(buffer)

    def can_reuse(self, buffer):
        name = buffer.get_name()
        if (
            name in V.graph.removed_buffers
            or name in V.graph.graph_inputs
            or name in V.graph.constants
            or name in self.freed
        ):
            return False
        return True

    def write_reuse_line(self, input_buffer, output_buffer):
        self.writeline(ReuseLine(input_buffer, output_buffer))

    def codegen_inplace_reuse(self, input_buffer, output_buffer):
        assert buffer_reuse_key(input_buffer) == buffer_reuse_key(output_buffer)
        self.codegen_allocation(input_buffer)
        self.freed.add(input_buffer.get_name())
        self.allocated.add(output_buffer.get_name())
        self.write_reuse_line(input_buffer, output_buffer)

    def generate_return(self):
        if self.output_refs:
            self.wrapper_call.writeline(
                "return (" + ", ".join(self.output_refs) + ", )"
            )
        else:
            self.wrapper_call.writeline("return ()")

    def generate_end(self, result):
        return

    @dynamo_utils.dynamo_timed
    def generate(self):
        result = IndentedBuffer()
        result.splice(self.header)
        result.splice(self.prefix)

        out_names = V.graph.get_output_names()
        with self.wrapper_call.indent():
            while (
                self.lines
                and isinstance(self.lines[-1], MemoryPlanningLine)
                and self.lines[-1].node.name not in out_names
            ):
                # these lines will be pointless
                self.lines.pop()

            # codegen allocations in two passes
            planning_state = MemoryPlanningState()
            for i in range(len(self.lines)):
                if isinstance(self.lines[i], MemoryPlanningLine):
                    self.lines[i] = self.lines[i].plan(planning_state)

            for line in self.lines:
                if isinstance(line, MemoryPlanningLine):
                    line.codegen(self.wrapper_call)
                else:
                    self.wrapper_call.writeline(line)

            self.generate_return()

        with result.indent():
            result.splice(self.wrapper_call)

        self.generate_end(result)

        self.add_benchmark_harness(result)

        return result.getvalue()

    def add_benchmark_harness(self, output):
        """
        Append a benchmark harness to generated code for debugging
        """
        if not config.benchmark_harness:
            return

        def add_fake_input(name, shape, stride, device, dtype):
            output.writeline(
                f"{name} = rand_strided("
                f"{V.graph.sizevars.codegen_python_shape_tuple(shape)}, "
                f"{V.graph.sizevars.codegen_python_shape_tuple(stride)}, "
                f"device='{device.type}', dtype={dtype})"
            )

        output.writelines(["", "", 'if __name__ == "__main__":'])
        with output.indent():
            output.splice(
                f"""
                from {config.dynamo_import}.testing import rand_strided
                from {config.inductor_import}.utils import print_performance
                """,
                strip=True,
            )

            for name, value in V.graph.constants.items():
                add_fake_input(
                    name, value.size(), value.stride(), value.device, value.dtype
                )

            for name, value in V.graph.graph_inputs.items():
                shape = [V.graph.sizevars.size_hint(x) for x in value.get_size()]
                stride = [V.graph.sizevars.size_hint(x) for x in value.get_stride()]
                add_fake_input(
                    name, shape, stride, value.get_device(), value.get_dtype()
                )

            output.writeline(
                f"print_performance(lambda: call([{', '.join(V.graph.graph_inputs.keys())}]))"
            )

    def define_kernel(self, name: str, kernel: str):
        self.header.splice(f"\n\n{name} = {kernel}")

    def load_kernel(self, name: str = None, kernel: str = None, arg_types: List = None):
        return

    def wrap_kernel_call(self, name, call_args):
        return "{}({})".format(name, ", ".join(call_args))

    def generate_kernel_call(self, name, call_args):
        self.writeline(
            self.wrap_kernel_call(name, call_args),
        )

    def call_kernel(self, name: str, kernel: Kernel):
        tmp = IndentedBuffer()
        kernel.call_kernel(self, tmp, name)
        for line in tmp.getvalue().split("\n"):
            line = line.strip()
            if line:
                self.writeline(line)

    def writeline(self, line):
        self.lines.append(line)


class CppWrapperCodeGen(WrapperCodeGen):
    """
    The outer wrapper that calls the kernels.
    """

    call_func_id = count()

    def __init__(self):
        self._call_func_id = next(CppWrapperCodeGen.call_func_id)
        super().__init__()

    def set_output_refs(self):
        self.output_refs = [
            x.cpp_wrapper_codegen_reference() for x in V.graph.graph_outputs
        ]

    def write_prefix(self):
        self.prefix.splice(
            """
            async_compile.wait(globals())
            del async_compile
            from torch.utils.cpp_extension import load_inline
            wrapper = (
            '''
            #include <dlfcn.h>
            #include <assert.h>
            """
        )
        with self.wrapper_call.indent():
            inputs_len = len(V.graph.graph_inputs.keys())
            if self.output_refs:
                if len(self.output_refs) == 1:
                    output_types = "at::Tensor"
                else:
                    output_return_type = "at::Tensor"
                    output_return_types = [output_return_type] * len(self.output_refs)
                    output_return_types = ", ".join(output_return_types)
                    output_types = f"std::tuple<{output_return_types}>"
            else:
                output_types = "void"

            if inputs_len != 0:
                inputs_args = ["at::Tensor"] * len(V.graph.graph_inputs.keys())
                inputs_args = ", ".join(inputs_args)
                inputs_args = f"std::tuple<{inputs_args}>"

                self.wrapper_call.writeline(
                    f"{output_types} call_{self._call_func_id}({inputs_args} args) {{"
                )
                inputs_keys_str = ", ".join(V.graph.graph_inputs.keys())
                self.wrapper_call.writeline(f"at::Tensor {inputs_keys_str};")
                self.wrapper_call.writeline(f"std::tie({inputs_keys_str}) = args;")
            else:
                self.wrapper_call.writeline(
                    f"{output_types} call_{self._call_func_id}(std::tuple<> args) {{"
                )
            for name in V.graph.randomness_seeds:
                self.wrapper_call.writeline(f"at::Tensor {name};")
                self.wrapper_call.writeline(
                    f"{name} = at::randint(std::pow(2, 31), {{}}, at::ScalarType::Long);"
                )
            V.graph.sizevars.codegen(self.wrapper_call, V.graph.graph_inputs)

    def write_allocate_line(self, buffer):
        self.writeline(CppAllocateLine(buffer))

    def write_del_line(self, name):
        self.writeline(f"{name}.reset();")
        return

    def write_free_if_not_reused_line(self, buffer):
        return

    def write_reuse_line(self, input_buffer, output_buffer):
        self.writeline(CppReuseLine(input_buffer, output_buffer))

    def get_deferred_line(self, name, layout):
        return DeferredLine(
            name, f"auto {name} = {layout.view.codegen_reference()};  // alias"
        )

    def get_kernel_path(self, code):
        ext = "so"
        extra = cpp_compile_command("i", "o")
        # \n is required to match with the CodeCache behavior
        source_code = "\n" + code.getvalue()
        _, _, kernel_path = get_code_path(source_code, ext, extra)
        return kernel_path

    def load_kernel(self, name: str = None, kernel: str = None, arg_types: List = None):
        kernel_path = self.get_kernel_path(kernel)

        self.writeline(f'auto {name}_lib = dlopen("{kernel_path}", RTLD_NOW);')
        self.writeline(f"assert({name}_lib != nullptr);")
        self.writeline(f"void (*{name})({arg_types});")
        self.writeline(f'*(void **) (&{name}) = dlsym({name}_lib, "kernel");')

    def wrap_kernel_call(self, name, call_args):
        return "{}({});".format(name, ", ".join(call_args))

    def generate_return(self):
        if self.output_refs:
            if len(self.output_refs) == 1:
                self.wrapper_call.writeline(
                    "return " + self.output_refs[0] + "; }''' )"
                )
            else:
                self.wrapper_call.writeline(
                    "return std::make_tuple("
                    + ", ".join(self.output_refs)
                    + "); }''' )"
                )
        else:
            self.wrapper_call.writeline("return; }''' )")

    def generate_end(self, result):
        shared = codecache.shared()
        cpp_flags = codecache.cpp_flags()
        optimization_flags = codecache.optimization_flags()
        ipaths, lpaths, libs = codecache.get_include_and_linking_paths()

        extra_cflags = f"{cpp_flags} {optimization_flags}"
        extra_ldflags = f"{shared} {lpaths} {libs}"
        extra_include_paths = f"{ipaths}"

        # get the hash of the wrapper code to name the extension
        wrapper_call_hash = codecache.code_hash(self.wrapper_call.getvalue())
        result.splice(
            f"""
            module = load_inline(
                name='inline_extension_{wrapper_call_hash}',
                cpp_sources=[wrapper],
                functions=['call_{self._call_func_id}'],
                extra_cflags=['{extra_cflags}'],
                extra_ldflags=['{extra_ldflags}'],
                extra_include_paths=['{extra_include_paths}'])
            """
        )
        # Wrap the func to support setting result._boxed_call = True
        result.splice(
            f"""
            def _wrap_func(f):
                def g(args):
                    return f(args)
                return g
            call = _wrap_func(module.call_{self._call_func_id})
            """
        )
