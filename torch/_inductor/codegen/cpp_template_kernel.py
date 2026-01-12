# mypy: allow-untyped-defs
import itertools
from collections.abc import Callable, Iterable
from typing import Any, Optional, Union
from unittest.mock import patch

import sympy
from sympy.parsing.sympy_parser import parse_expr

import torch
from torch._inductor.utils import do_bench_using_profiling
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.symbol import SymT

from .. import config, cpp_builder, ir, lowering as L
from ..autotune_process import CppBenchmarkRequest
from ..loop_body import LoopBody
from ..select_algorithm import PartialRender
from ..utils import sympy_index_symbol, sympy_index_symbol_with_prefix
from ..virtualized import V
from .common import REMOVED
from .cpp import CppKernel, CppKernelProxy, KernelGroup, ParallelDepth
from .cpp_utils import cexpr_index, DTYPE_TO_CPP, LocalBufferContext


def parse_expr_with_index_symbols(expr):
    if isinstance(expr, sympy.Expr):
        return expr
    elif isinstance(expr, (list, tuple)):
        return [parse_expr_with_index_symbols(e) for e in expr]
    else:
        expr = parse_expr(str(expr))
        int_symbols = {sym: sympy_index_symbol(sym.name) for sym in expr.free_symbols}
        return expr.subs(int_symbols)


def wrap_with_tensorbox(node) -> ir.TensorBox:
    return (
        ir.TensorBox.create(node) if isinstance(node, ir.Buffer) else ir.TensorBox(node)
    )


class CppTemplateKernel(CppKernel):
    def __init__(self, kernel_name, num_threads):
        super().__init__(None, num_threads)
        self.kernel_name = kernel_name
        self.render_hooks = {}
        self.local_buffers = {}

    def render(self, template, **kwargs):
        return PartialRender(
            template.render(kernel=self, **kwargs), self.render_hooks
        ).finalize_all()

    def def_kernel(
        self,
        inputs: dict[str, ir.Buffer],
        outputs: dict[str, ir.Buffer],
        aliases: Optional[dict[str, str]] = None,
        function_name: str = "",
        extra_sizevars: Optional[list[sympy.Expr]] = None,
        placeholder: str = "<DEF_KERNEL>",
    ) -> str:
        if len(function_name) == 0:
            function_name = str(self.kernel_name)
        for name, inp in inputs.items():
            if inp is not None:
                self.args.input_buffers[inp.get_name()] = name
        for name, out in outputs.items():
            self.args.output_buffers[out.get_name()] = name
        if aliases is not None:
            for alias, orig in aliases.items():
                if orig in self.args.input_buffers:
                    self.args.input_buffers[alias] = self.args.input_buffers[orig]
                if orig in self.args.output_buffers:
                    self.args.output_buffers[alias] = self.args.output_buffers[orig]

        unique_sizevars = OrderedSet(
            s
            for input in inputs.values()
            if input is not None
            for sym in itertools.chain(input.get_size(), input.get_stride())
            if isinstance(sym, sympy.Expr)
            for s in sym.free_symbols
        )
        unique_sizevars.update(
            s
            for sym in extra_sizevars or []
            if isinstance(sym, sympy.Expr)
            for s in sym.free_symbols
        )
        unique_sizevars.update(
            s
            for output in outputs.values()
            for sym in itertools.chain(output.get_size(), output.get_stride())
            if isinstance(sym, sympy.Expr)
            for s in sym.free_symbols
        )
        sizevars = sorted(unique_sizevars, key=str)
        for sizevar in sizevars:
            self.args.sizevars[sizevar] = f"k{sizevar}"

        def hook():
            # remove all aliases before generate function definition
            if aliases is not None:
                for alias in aliases:
                    if alias in self.args.input_buffers:
                        raise AssertionError(
                            f"input_buffers cannot be removed: {alias}"
                        )
                    if alias in self.args.output_buffers:
                        self.args.output_buffers[alias] = REMOVED
            cpp_argdefs, _, _ = self.args.cpp_argdefs()
            return f"void {function_name}({', '.join(cpp_argdefs)})"

        assert placeholder not in self.render_hooks
        self.render_hooks[placeholder] = hook
        return placeholder

    def call_kernel(self, name: str, node: ir.CppTemplateBuffer):
        wrapper = V.graph.wrapper_code
        _, call_args, arg_types = self.args.cpp_argdefs()
        wrapper.generate_kernel_call(name, call_args, triton=False, arg_types=arg_types)

    def dtype(self, node: ir.Buffer) -> str:
        return DTYPE_TO_CPP[node.get_dtype()]

    def acc_dtype(self, node: ir.Buffer) -> str:
        if node.get_dtype() in [torch.float32, torch.bfloat16, torch.half]:
            return "float"
        else:
            raise NotImplementedError(f"Unsupported dtype: {node.get_dtype()}")

    def size(self, node: ir.Buffer, dim: int) -> str:
        return cexpr_index(self.rename_indexing(node.get_size()[dim]))

    def stride(self, node: ir.Buffer, dim: int) -> str:
        return cexpr_index(self.rename_indexing(node.get_stride()[dim]))

    def index(self, node: ir.Buffer, indices: list[Any]) -> str:
        indexer = node.get_layout().as_fixed().make_indexer()
        index = indexer(parse_expr_with_index_symbols(indices))
        index = self.rename_indexing(index)
        outer_name = node.get_name()
        inner_name = (
            outer_name
            if outer_name in self.local_buffers
            else self.args.input(node.get_name())
        )
        return f"{inner_name}[{cexpr_index(index)}]"

    def slice_nd(self, node, ranges: list[tuple[Any, Any]]) -> ir.ReinterpretView:
        """
        Slice the given node with a list of ranges (start and end) corresponding to its dims.
        The dim is not sliced if the corresponding range is empty.
        """
        assert len(ranges) == len(node.get_size()), f"{ranges=}, {node=}"
        sliced = wrap_with_tensorbox(node)
        for dim, _range in enumerate(ranges):
            if len(_range) == 0:
                continue
            assert len(_range) == 2
            start, end = parse_expr_with_index_symbols(_range)
            sliced = L.slice_(sliced, dim, start, end, clamp=False)
        assert isinstance(sliced, ir.TensorBox)
        assert isinstance(sliced.data, ir.ReinterpretView), sliced.data
        return sliced.data

    def select(self, node, dim: int, idx: int) -> ir.ReinterpretView:
        # We avoid using L.select here because we need clamp=False so the dim after slicing
        # is 1 instead of a sympy expression of symbol - dim_size.
        node = wrap_with_tensorbox(node)
        idx = ir.View.handle_negative_index(idx, node.get_size()[dim])
        sliced = L.squeeze(L.slice_(node, dim, idx, idx + 1, clamp=False), dim)
        assert isinstance(sliced.data, ir.ReinterpretView), sliced.data
        return sliced.data

    def view(self, node, sizes: list[Any]) -> ir.IRNode:
        node = wrap_with_tensorbox(node)
        sizes = parse_expr_with_index_symbols(sizes)
        return L.view(node, sizes).data  # type: ignore[arg-type]

    def permute(self, node, dims):
        node = wrap_with_tensorbox(node)
        permuted = L.permute(node, dims).data
        assert isinstance(permuted, ir.ReinterpretView)
        return permuted

    def maybe_codegen_profile(self) -> str:
        if config.cpp.enable_kernel_profile:
            graph_id = V.graph.graph_id
            prefix = "graph_" + str(graph_id) + "_" if graph_id is not None else ""
            handle_str = (
                "torch::aot_inductor::RAIIAtenRecordFunctionHandle "
                f'record_{prefix}{self.kernel_name}_("{prefix}{self.kernel_name}", nullptr);'
            )
            return handle_str
        else:
            return ""

    def unroll_pragma(self, unroll):
        if cpp_builder.is_gcc():
            return f"#pragma GCC unroll {unroll}"
        else:
            return f"#pragma unroll {unroll}"

    def define_buffer(self, name, sizes: list[Any], dtype=torch.float) -> str:
        """Define kernel local buffer"""
        sizes = parse_expr_with_index_symbols(sizes)
        buf = ir.Buffer(
            name=name, layout=ir.FixedLayout(torch.device("cpu"), dtype, sizes)
        )
        self.local_buffers[name] = buf
        ctype = f"{DTYPE_TO_CPP[dtype]}"
        numel = f"{cexpr_index(buf.get_numel())}"
        return f"auto _{name} = std::make_unique<{ctype}[]>({numel}); auto {name} = _{name}.get();"

    def define_stack_allocated_buffer(
        self, name, sizes: list[Any], dtype=torch.float
    ) -> str:
        """Define stack-allocated buffer"""
        sizes = parse_expr_with_index_symbols(sizes)
        buf = ir.Buffer(
            name=name, layout=ir.FixedLayout(torch.device("cpu"), dtype, sizes)
        )
        self.local_buffers[name] = buf
        ctype = f"{DTYPE_TO_CPP[dtype]}"
        numel = f"{cexpr_index(buf.get_numel())}"
        return f"alignas(64) {ctype} _{name}[{numel}]; {ctype}* {name} = _{name};"

    def reinit_buffer_if_null(self, name):
        """Reinit the previously defined local buffer if it is null"""
        assert name in self.local_buffers
        buf = self.local_buffers[name]
        ctype = f"{DTYPE_TO_CPP[buf.layout.dtype]}"
        numel = f"{cexpr_index(buf.get_numel())}"
        return f"if (_{name} == nullptr) {{ _{name} = std::make_unique<{ctype}[]>({numel}); {name} = _{name}.get(); }}"

    def release_buffer(self, name):
        """Codegen the code to release the ownership of a local buffer to others"""
        assert name in self.local_buffers
        return f"_{name}.release()"

    def store_pointwise_nodes(
        self,
        dst: ir.Buffer,
        nodes: list[ir.IRNode],
        offsets: Optional[list[sympy.Expr]] = None,
        reindexers: Optional[list[Optional[Callable[[list[Any]], list[Any]]]]] = None,
    ) -> str:
        var_sizes = (tuple(dst.get_size()), ())
        var_ranges = {
            sympy_index_symbol_with_prefix(SymT.INDEX, i): sz
            for i, sz in enumerate(var_sizes[0])
        }
        if not offsets:
            offsets = [sympy.S.Zero] * len(var_sizes[0])
        if not reindexers:
            reindexers = [None] * len(nodes)
        assert len(offsets) == len(var_sizes[0])
        output_index = dst.get_layout().make_indexer()([*var_ranges.keys()])
        kernel_group = KernelGroup()
        kernel_group.args = self.args
        cpp_kernel_proxy = CppKernelProxy(kernel_group)
        bodies = []
        var_sizes_list = []
        for i, node in enumerate(nodes):
            output_name = node.get_name() if i < len(nodes) - 1 else dst.get_name()
            node = node.data if isinstance(node, ir.ComputedBuffer) else node
            assert isinstance(node, ir.Pointwise), node

            def fn(*args):
                assert len(args) == 2
                assert len(args[0]) == len(var_sizes[0])
                assert len(args[1]) == 0
                new_args = [arg + offset for arg, offset in zip(args[0], offsets)]  # type: ignore[arg-type]
                if reindexers[i] is not None:
                    new_args = reindexers[i](new_args)  # type: ignore[misc]
                V.ops.store(
                    output_name,
                    output_index,
                    node.make_loader()(new_args).value,
                )

            body = LoopBody(
                fn,
                (list(var_ranges.keys()), ()),
                var_ranges,
                list(var_ranges.keys()),
                tuple(),
            )
            bodies.append(body)
            var_sizes_list.append(var_sizes)

        cpp_kernel_proxy.codegen_loop_bodies(bodies, var_sizes_list)

        def max_parallel_depth():
            return ParallelDepth(parallel_depth=0, start_depth=0)

        # This loop is not parallelized since it is not the outermost loop.
        with patch.object(
            cpp_kernel_proxy.loop_nest, "max_parallel_depth", max_parallel_depth
        ):
            kernel_group.finalize_kernel(cpp_kernel_proxy, [])
        return kernel_group.loops_code.getvalue()

    def store_grouped_gemm_pointwise_nodes(
        self,
        dst: tuple[ir.Buffer],
        nodes: list[ir.IRNode],
        offsets: list[sympy.Expr],
        reindexers: list[Optional[Callable[[list[Any]], list[Any]]]],
        output_names: list[str],
    ) -> str:
        ref_dst = dst[0]
        var_sizes = (tuple(ref_dst.get_size()), ())
        var_ranges = {
            sympy_index_symbol_with_prefix(SymT.INDEX, i): sz
            for i, sz in enumerate(var_sizes[0])
        }
        assert offsets, "offsets should be set outside"
        assert all(len(offset) == len(var_sizes[0]) for offset in offsets)
        output_index = ref_dst.get_layout().make_indexer()([*var_ranges.keys()])
        kernel_group = KernelGroup()
        kernel_group.args = self.args
        cpp_kernel_proxy = CppKernelProxy(kernel_group)
        bodies = []
        var_sizes_list = []
        for i, node in enumerate(nodes):
            output_name = output_names[i]
            node = node.data if isinstance(node, ir.ComputedBuffer) else node
            assert isinstance(node, ir.Pointwise), node

            def fn(*args):
                assert len(args) == 2
                assert len(args[0]) == len(var_sizes[0])
                assert len(args[1]) == 0
                new_args = [arg + offset for arg, offset in zip(args[0], offsets[i])]  # type: ignore[arg-type]
                if reindexers[i] is not None:
                    new_args = reindexers[i](new_args)  # type: ignore[misc]
                V.ops.store(
                    output_name,
                    output_index,
                    node.make_loader()(new_args).value,
                )

            body = LoopBody(
                fn,
                (list(var_ranges.keys()), ()),
                var_ranges,
                list(var_ranges.keys()),
                tuple(),
            )
            bodies.append(body)
            var_sizes_list.append(var_sizes)

        cpp_kernel_proxy.codegen_loop_bodies(bodies, var_sizes_list)

        def max_parallel_depth():
            return ParallelDepth(parallel_depth=0, start_depth=0)

        # This loop is not parallelized since it is not the outermost loop.
        with patch.object(
            cpp_kernel_proxy.loop_nest, "max_parallel_depth", max_parallel_depth
        ):
            kernel_group.finalize_kernel(cpp_kernel_proxy, [])
        return kernel_group.loops_code.getvalue()

    def store_output(
        self,
        dst: ir.Buffer,
        src: ir.Buffer,
        orig_src: Optional[ir.Buffer] = None,
        epilogue_nodes: Optional[list[ir.IRNode]] = None,
        offsets: Optional[list[Any]] = None,
        reindexers: Optional[list[Optional[Callable[[list[Any]], list[Any]]]]] = None,
    ):
        """
        Store the `src` buffer to the `dst` buffer. The size of `src` and `dst` should match.
        If `epilogue_nodes` is provided, the `src` buffer is firstly computed with the epilogues
        before stored to `dst`. The `epilogues_nodes` are all pointwise.

        Notes:
        1. `src` and `dst` buffer could be the same buffer in which case we are doing in-place compute
           and stores. In case `epilogue_nodes` are not provided, we do nothing.
        2. The `epilogue_nodes`, if exist, have computations on `src` before storing to `dst` but since
           they come form the original Inductor IR, they might need to be adjusted before working with
           `src` and `dst` as outlined below:
           a) `src` or `dst` buffer could be a sub-slice of the ranges the `epilogue_nodes`work on.
              In this case, the `offsets` could be provided to adjust the indices passed to
              `epilogue_nodes` during codegen and the data ranges are also configured according to
              the sizes of `src` and `dst`.
           b) `dst` might be indexed in a different way as the `epilogue_nodes`, hence a `reindexer` is
              needed on the indices to `epilogue_nodes` to match the indexing of `dst`.
           c) If `src` is local, we need to add a local buffer for it and localize the `orig_src` buffer
              in `epilogue_nodes` with `src`.
        """
        assert isinstance(dst, (ir.Buffer, ir.ReinterpretView))
        assert dst.get_size() == src.get_size(), f"{dst=}, {src=}"
        if offsets:
            offsets = parse_expr_with_index_symbols(offsets)
        if epilogue_nodes:
            with LocalBufferContext(self.args) as scope:
                assert orig_src is not None
                if orig_src.get_name() != src.get_name():
                    scope.add_local_buffer(
                        src,
                        [
                            orig_src,
                        ],
                    )
                    epilogue_nodes = scope.localize_nodes(epilogue_nodes)
                return self.store_pointwise_nodes(
                    dst,
                    epilogue_nodes,  # type: ignore[arg-type]
                    offsets,
                    reindexers,
                )
        else:
            if dst.get_name() != src.get_name():
                # src is local
                copy = L.copy(dst, src).data.data
                with LocalBufferContext(self.args) as scope:
                    scope.add_local_buffer(src)

                    return self.store_pointwise_nodes(dst, [copy])
            else:
                assert dst.layout == src.layout, f"{dst=}, {src=}"
                return ""

    def store_outputs(
        self,
        dst: tuple[ir.Buffer],
        src: tuple[ir.IRNode],
        orig_src: Optional[tuple[ir.IRNode]] = None,
        epilogue_nodes: Optional[list[ir.IRNode]] = None,
        offsets: Optional[list[Any]] = None,
        reindexers: Optional[list[Optional[Callable[[list[Any]], list[Any]]]]] = None,
        multi_output_buffers: Optional[tuple[ir.MultiOutput, ...]] = None,
    ):
        assert isinstance(dst, Iterable)
        assert all(_dst.get_size() == _src.get_size() for _src, _dst in zip(src, dst))
        if offsets:
            offsets = parse_expr_with_index_symbols(offsets)
        gemm_num = len(src)
        final_offsets = []
        output_names = []
        if epilogue_nodes:
            if not reindexers:
                reindexers = [None] * len(epilogue_nodes)
            with LocalBufferContext(self.args) as scope:
                assert orig_src is not None
                localize_epilogue_nodes = []
                all_read_names = []
                for epilogue in epilogue_nodes:
                    all_read_names.extend(list(epilogue.get_read_names()))
                localize_epilogue_nodes.extend(scope.localize_nodes(epilogue_nodes))
                final_offsets.extend([offsets] * len(localize_epilogue_nodes))
                output_names.extend(
                    [node.get_name() for node in localize_epilogue_nodes]
                )
                for gemm_idx in range(gemm_num):
                    if orig_src[gemm_idx].get_name() != src[gemm_idx].get_name():
                        if orig_src[gemm_idx].get_name() in all_read_names or (
                            multi_output_buffers
                            and multi_output_buffers[gemm_idx].get_name()
                            in all_read_names
                        ):
                            # If any of the Epilogue nodes use this GEMM output, let's localize the GEMM output
                            global_buffers = [orig_src[gemm_idx]]
                            if (
                                multi_output_buffers
                                and multi_output_buffers[gemm_idx].get_name()
                                in all_read_names
                                and orig_src[gemm_idx].get_name() not in all_read_names
                            ):
                                # Epilogue might directly read the MultiOutput, Locallize MultiOutput to the local Buffer
                                # if this MultiOutput has not been stored by in-template epilogue
                                # otherwise, use the cse store cache if it will be stored before used
                                global_buffers.append(multi_output_buffers[gemm_idx])
                            scope.add_local_buffer(
                                src[gemm_idx],
                                global_buffers,
                            )
                        else:
                            scope.add_local_buffer(src[gemm_idx])
                            localize_epilogue_nodes.extend(
                                [L.copy(dst[gemm_idx], src[gemm_idx]).data.data]
                            )
                            reindexers.append(None)
                            output_names.append(dst[gemm_idx].get_name())
                            final_offsets.append(
                                [sympy.S.Zero] * len(dst[gemm_idx].get_size())
                            )
                res = self.store_grouped_gemm_pointwise_nodes(
                    dst,
                    localize_epilogue_nodes,
                    final_offsets,
                    reindexers,
                    output_names=output_names,
                )
                for gemm_idx in range(gemm_num):
                    if (
                        multi_output_buffers
                        and multi_output_buffers[gemm_idx].get_name() in all_read_names
                    ):
                        # If the MultiOutput is used in the Epilogue, let's remove it from args
                        multi_output_name = multi_output_buffers[gemm_idx].get_name()
                        if (
                            multi_output_name in self.args.output_buffers
                            and self.args.output_buffers[multi_output_name]
                            is not REMOVED
                        ):
                            self.remove_buffer(multi_output_name)
                return res
        else:
            if dst[0].get_name() != src[0].get_name():
                copy_list = []
                with LocalBufferContext(self.args) as scope:
                    for _src, _dst in zip(src, dst):
                        copy_list.extend([L.copy(_dst, _src).data.data])
                        scope.add_local_buffer(_src)
                        output_names.append(_dst.get_name())
                        final_offsets.append([sympy.S.Zero] * len(_dst.get_size()))
                    reindexers = [None] * len(copy_list)
                    return self.store_grouped_gemm_pointwise_nodes(
                        dst,
                        nodes=copy_list,
                        offsets=final_offsets,
                        reindexers=reindexers,
                        output_names=output_names,
                    )
            else:
                assert all(
                    _src.get_name() == _dst.get_name() for _src, _dst in zip(src, dst)
                )
                assert all(
                    _src.get_layout() == _dst.get_layout()
                    for _src, _dst in zip(src, dst)
                )
                return ""

    def check_bounds(self, expr, size, lower, upper):
        # CppTemplateKernel does not need codegen related operations
        return


class CppTemplateCaller(ir.ChoiceCaller):
    """
    CppTemplateCaller

    This class represents a caller for CPP template kernels. It is a subclass of ir.ChoiceCaller.
    Attributes:
        name (str): The name of the caller.
        category (str): The category of the caller.
        bmreq (CppBenchmarkRequest): The benchmark request for the caller.
        template_buffer (ir.CppTemplateBuffer): The template buffer for the caller.
    """

    def __init__(
        self,
        name: str,
        category: str,
        input_nodes: list[ir.Buffer],
        layout: ir.Layout,
        make_kernel_render: Callable[
            [
                ir.CppTemplateBuffer,
                bool,
                Optional[list[ir.IRNode]],
            ],
            str,
        ],
        bmreq: CppBenchmarkRequest,
        template: "CppTemplate",  # type: ignore[name-defined]  # noqa: F821
        info_kwargs: Optional[
            dict[str, Union[ir.PrimitiveInfoType, list[ir.PrimitiveInfoType]]]
        ] = None,
    ):
        super().__init__(name, input_nodes, layout, description="")
        self.category = category
        self.make_kernel_render = make_kernel_render
        self.bmreq = bmreq
        self.template = template
        self.info_kwargs = info_kwargs

    def precompile(self) -> None:
        assert self.bmreq is not None
        self.bmreq.precompile()

    def benchmark(self, *args, out) -> float:
        assert self.bmreq is not None
        if config.profile_bandwidth_with_do_bench_using_profiling:
            algo = self.bmreq.make_run_fn(*args, out=out)
            return do_bench_using_profiling(algo)
        return self.bmreq.benchmark(*args, out=out)

    def hash_key(self) -> str:
        return "-".join(
            [
                self.category,
                self.bmreq.hash_key,
            ]
        )

    def info_dict(
        self,
    ) -> dict[str, Union[ir.PrimitiveInfoType, list[ir.PrimitiveInfoType]]]:
        return {"backend": "CPP", "op_type": "unknown"}

    def output_node(self) -> ir.TensorBox:
        buffer = ir.CppTemplateBuffer(
            layout=self.layout,
            inputs=self.input_nodes,
            make_kernel_render=self.make_kernel_render,
            template=self.template,
            choice=self,
        )
        # Pass KTC annotation to the buffer for encoding
        if "ktc" in self.annotations:
            buffer.annotations["ktc"] = self.annotations["ktc"]
        return ir.TensorBox.create(buffer)
