from __future__ import annotations

import collections
import contextlib
import itertools
import re
from dataclasses import dataclass
from typing import Any, cast, TYPE_CHECKING

import sympy

import torch
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.value_ranges import bound_sympy, ValueRanges

from ... import config
from ...ir import (
    FixedLayout,
    IRNode,
    MutationLayoutSHOULDREMOVE,
    Reduction,
    ReinterpretView,
)
from ...utils import IndentedBuffer, sympy_subs, upcast_compute_type
from ...virtualized import V
from ..common import CSEProxy, CSEVariable, DeferredLine, TensorArg
from ..simd import constant_repr, SIMDKernel
from .helion_op_overrides import HelionOverrides, HelionPrinter


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence
    from typing import Self


# A tile name per dim, or 1 for a broadcast dim.
TileShape = tuple[str | int, ...]
TILE_NAMES = {"z0_": "tile_z", "y": "tile_y", "x": "tile_x", "r0_": "tile_r"}
TILE_AXES = {name: axis for axis, name in TILE_NAMES.items()}
REDUCTIONS = OrderedSet(["sum", "prod", "max", "min"])


def feeds_only_dots(node: torch.fx.Node | None) -> bool:
    """Whether the value of loop-body `node` only reaches dots, through casts."""
    return node is not None and all(
        user.target == "dot" or (user.target == "to_dtype" and feeds_only_dots(user))
        for user in node.users
    )


@dataclass(frozen=True)
class HelionTensorArg:
    # Placeholder for the parameter, named by codegen_kernel().
    ref: str
    # Buffer or ReinterpretView of a buffer.
    operand: IRNode
    output: bool


class HelionCSEProxy(CSEProxy):
    def dot(self, a: CSEVariable, b: CSEVariable) -> CSEVariable:
        return cast(HelionKernel, self.kernel).dot(a, b)

    def to_dtype(
        self,
        x: CSEVariable,
        dtype: torch.dtype,
        src_dtype: torch.dtype | None = None,
        use_compute_types: bool = True,
    ) -> CSEVariable:
        kernel = cast(HelionKernel, self.kernel)
        kwargs = {"src_dtype": src_dtype, "use_compute_types": use_compute_types}
        node = getattr(V.interpreter, "current_node", None)
        if use_compute_types and x.dtype == dtype and feeds_only_dots(node):
            # A bf16 value that only reaches dots stays bf16. Upcast, as in
            #     tmp1 = tmp0.to(torch.float32)
            #     acc = torch.addmm(acc, tmp1, tmp3)
            # it makes the dot an fp32 one (dot_precision="ieee"). The result is
            # the same: bf16 values are exact in fp32 and tf32, and the dot
            # accumulates in fp32 either way.
            return x
        if not (kernel.in_loop and x in kernel.dot_results):
            return self._default("to_dtype", (x, dtype), kwargs)
        # A bf16 matmul output saved for backward is rounded between the dot and
        # its reduction. Emitted in the K loop, the rounding reads the running sum
        # and the store after the loop cannot use it:
        #     for tile_r in hl.tile(64):
        #         acc = torch.addmm(acc, tmp1, tmp3)
        #         tmp4 = acc.to(torch.bfloat16)
        #         tmp5 = tmp4.to(torch.float32)
        #     out0[tile_y, tile_x] = tmp5.to(torch.bfloat16)
        # Helion: "Device value 'tmp5' cannot be carried between top-level loops".
        # After the loop it rounds the final sum once, as eager does. Only in the
        # K loop: an epilogue's cast, as in (a @ b).to(torch.int32) + 1, is already
        # after the loop, and in the stores it would follow the + 1 that reads it.
        with kernel.swap_buffers(kernel.stores, kernel.stores, kernel.stores):
            value = self._default("to_dtype", (x, dtype), kwargs)
        # So that tmp5 = tmp4.to(torch.float32) follows tmp4 out of the loop.
        kernel.dot_results.add(value)
        # The epilogue then reads value, not out0[tile_y, tile_x] from memory.
        kernel.outer_vars.add(value)
        return value


class HelionKernel(SIMDKernel[CSEVariable]):
    """A Helion kernel: pointwise range trees become one hl.tile loop and the
    reduction range tree an inner hl.tile loop. Loads and stores subscript
    tensors by tile, e.g. in0[tile_y, tile_r]; Helion picks block sizes and
    masks partial tiles."""

    overrides = HelionOverrides  # type: ignore[assignment]

    @staticmethod
    def supports_tiling(tiling: dict[str, sympy.Expr]) -> bool:
        return OrderedSet(tiling) <= OrderedSet(TILE_NAMES) and all(
            isinstance(numel, (int, sympy.Integer)) for numel in tiling.values()
        )

    def __init__(self, tiling: dict[str, sympy.Expr], **kwargs: Any) -> None:
        self.outer_vars: OrderedSet[CSEVariable] = OrderedSet()
        super().__init__(tiling, **kwargs)
        if not self.supports_tiling(tiling):
            raise AssertionError(f"Helion kernels do not tile {tiling}")
        nodes = tuple(self.features.scheduler_nodes())
        self.is_matmul = any(node.is_native_matmul() for node in nodes)
        # A matmul orders r0_ before x so dot operands read [y, r0_] and [r0_, x].
        order = ("z0_", "y", "r0_", "x") if self.is_matmul else ("z0_", "y", "x", "r0_")
        self.axes = tuple(axis for axis in order if axis in self.numels)
        self.outer_axes = tuple(axis for axis in self.axes if axis != "r0_")
        roots = {tree.prefix: tree for tree in self.range_trees}
        self.symbols = {axis: roots[axis].index_sym() for axis in self.axes}
        self.tile_ranges = {self.symbols[axis]: self.numels[axis] for axis in self.axes}
        self.arguments: dict[tuple[Any, ...], HelionTensorArg] = {}
        # One buffer, so that a masked load follows the computation of its mask.
        self.loads = self.compute
        self.load_masks: list[CSEVariable] = []
        self.accumulator_count = 0
        self.has_fp32_dot = False
        # Dot accumulators and the casts of them that run after the K loop.
        self.dot_results: OrderedSet[CSEVariable] = OrderedSet()
        # Values stored outside the current reduction loop; see disable_reduction().
        self.outer_stores: dict[str, CSEVariable] = {}

    def __enter__(self) -> Self:
        super().__enter__()
        self.exit_stack.enter_context(
            V.set_ops_handler(HelionCSEProxy(self, HelionOverrides()))
        )
        return self

    @property
    def in_loop(self) -> bool:
        return self.inside_reduction and self.features.is_reduction()

    @property
    def scope_axes(self) -> tuple[str, ...]:
        # A matmul's K loop only holds dot operands, which do not broadcast.
        return self.axes if self.in_loop and not self.is_matmul else self.outer_axes

    def create_cse_var(self, *args: Any, **kwargs: Any) -> CSEVariable:
        var = super().create_cse_var(*args, **kwargs)
        if not (hasattr(self, "features") and self.in_loop):
            self.outer_vars.add(var)
        return var

    def dtype_to_str(self, dtype: torch.dtype) -> str:
        return str(dtype)

    def index_axes(self, index: sympy.Expr) -> list[str]:
        return [axis for axis in self.axes if self.symbols[axis] in index.free_symbols]

    def value_shape(self, axes: Sequence[str] | None = None) -> TileShape:
        """Shape of a value over `axes` that broadcasts against the tiles in scope,
        e.g. (tile_y, 1) for {y} in a loop over (y, x)."""
        scope = self.scope_axes
        axes = scope if axes is None else axes
        present = [axis for axis in scope if axis in axes]
        if not present:
            return ()
        start = scope.index(present[0])
        return tuple(
            TILE_NAMES[axis] if axis in present else 1 for axis in scope[start:]
        )

    def index_tensor(self, axis: str) -> str:
        name = f"{TILE_NAMES[axis]}.index"
        scope = self.scope_axes
        axes = scope[scope.index(axis) :]
        if len(axes) == 1:
            return name
        # reshape() rather than slicing: Helion's tcgen05 store path proves the
        # layout of a store subscript only through reshape, +, *, // and %.
        shape = ", ".join(TILE_NAMES[a] if a == axis else "1" for a in axes)
        return f"{name}.reshape([{shape}])"

    def align(self, value: CSEVariable, target: TileShape) -> str:
        shape = tuple(value.shape or ())
        present = [TILE_AXES[str(dim)] for dim in shape if dim != 1]
        target_axes = [TILE_AXES[str(dim)] for dim in target if dim != 1]
        if any(axis not in target_axes for axis in present) or present != sorted(
            present, key=target_axes.index
        ):
            raise NotImplementedError(f"Cannot broadcast Helion {shape} to {target}")
        if not present:
            return str(value)
        names = [TILE_NAMES[axis] for axis in present]
        dims = target[target.index(names[0]) :]
        aligned = tuple(dim if dim in names else 1 for dim in dims)
        if aligned == shape:
            return str(value)
        return f"{value}.reshape([{', '.join(map(str, aligned))}])"

    def in_scope(self, value: CSEVariable) -> CSEVariable:
        axes = [TILE_AXES[str(dim)] for dim in value.shape or () if dim != 1]
        target = self.value_shape(axes)
        if target == tuple(value.shape or ()):
            return value
        line = self.align(value, self.value_shape())
        return self.cse.generate(self.compute, line, dtype=value.dtype, shape=target)

    def tile_index(self, index: sympy.Expr) -> sympy.Expr:
        """`index` over the range tree roots, i.e. the tile indices."""
        index = sympy.sympify(index)
        entries = {
            symbol: self.range_tree_nodes[symbol].expr
            for symbol in index.free_symbols
            if symbol in self.range_tree_nodes
        }
        index = sympy.expand(sympy_subs(index, entries))
        return V.graph.sizevars.simplify_with_ranges(index, self.tile_ranges)

    def tensor_arg(self, operand: IRNode, *, output: bool = False) -> HelionTensorArg:
        layout = operand.get_layout()
        key = (
            operand.get_name(),
            tuple(layout.size),
            tuple(layout.stride),
            layout.offset,
        )
        if key not in self.arguments:
            ref = f"__arg{len(self.arguments)}__"
            self.arguments[key] = HelionTensorArg(ref, operand, output)
            if output:
                self.args.output(operand.get_name())
            else:
                self.args.input(operand.get_name())
        return self.arguments[key]

    def tiled_view(
        self, buffer: IRNode, index: sympy.Expr, axes: Sequence[str]
    ) -> IRNode | None:
        """A view of `buffer` with one dim per axis in `axes`, if `index` is affine
        in those tiles and the view stays inside the storage."""
        layout = buffer.get_layout()
        expanded = sympy.expand(index)
        sizes = [self.numels[axis] for axis in axes]
        strides = [expanded.coeff(self.symbols[axis]) for axis in axes]
        offset = expanded.xreplace({self.symbols[axis]: 0 for axis in self.axes})
        if not all(
            isinstance(value, (int, sympy.Integer))
            for value in (*sizes, *strides, offset)
        ):
            return None
        expected = offset + sum(self.symbols[a] * s for a, s in zip(axes, strides))
        end = offset + sum((size - 1) * stride for size, stride in zip(sizes, strides))
        # The storage size can be symbolic, e.g. x[0] @ w with x [s, 40, 72] and a
        # dynamic s: end = 2879 < 2880 * s holds since s >= 2.
        if (
            offset < 0
            or any(stride < 0 for stride in strides)
            or not V.graph.sizevars.statically_known_lt(end, layout.storage_size())
            or not V.graph.sizevars.statically_known_equals(index, expected)
        ):
            return None
        if (sizes, strides, offset) == (
            list(layout.size),
            list(layout.stride),
            layout.offset,
        ):
            return buffer
        layout = FixedLayout(
            layout.device, layout.dtype, sizes, strides, offset, layout.is_pinned
        )
        return ReinterpretView(data=buffer, layout=layout)

    def subscript(
        self,
        buffer: IRNode,
        index: sympy.Expr,
        axes: Sequence[str],
        *,
        output: bool = False,
        flat: bool = True,
    ) -> tuple[HelionTensorArg, list[str], TileShape] | None:
        """The parameter and subscript that read `buffer` at `index`, and its shape.

        Tries the buffer's own dims (dropping size-1 dims the index skips), then
        a tiled_view(), then if `flat` a flat_subscript().
        """
        layout = buffer.get_layout()
        candidates = [
            [
                axis
                for axis in axes
                if V.graph.sizevars.statically_known_equals(size, self.numels[axis])
            ]
            + ([None] if size == 1 else [])
            for size in layout.size
        ]
        for dims in itertools.product(*candidates):
            if [dim for dim in dims if dim is not None] != list(axes):
                continue
            expected = layout.offset + sum(
                self.symbols[dim] * stride
                for dim, stride in zip(dims, layout.stride)
                if dim is not None
            )
            if V.graph.sizevars.statically_known_equals(index, expected):
                kept = [i for i, dim in enumerate(dims) if dim is not None]
                operand = buffer
                if len(kept) < len(dims):
                    squeezed = FixedLayout(
                        layout.device,
                        layout.dtype,
                        [layout.size[i] for i in kept],
                        [layout.stride[i] for i in kept],
                        layout.offset,
                        layout.is_pinned,
                    )
                    operand = ReinterpretView(data=buffer, layout=squeezed)
                tiles = tuple(TILE_NAMES[dim] for dim in dims if dim is not None)
                arg = self.tensor_arg(operand, output=output)
                return arg, list(tiles) or ["()"], tiles
        if (view := self.tiled_view(buffer, index, axes)) is not None:
            tiles = tuple(TILE_NAMES[axis] for axis in axes)
            return self.tensor_arg(view, output=output), list(tiles) or ["()"], tiles
        if not flat:
            return None
        return self.flat_subscript(buffer, index, axes, output=output)

    def flat_subscript(
        self,
        buffer: IRNode,
        index: sympy.Expr,
        axes: Sequence[str],
        *,
        output: bool = False,
    ) -> tuple[HelionTensorArg, list[str], TileShape]:
        """A flat view of the storage read at `index`, like a Triton pointer offset."""
        layout = buffer.get_layout()
        size = layout.storage_size()
        operand = buffer
        if (list(layout.size), list(layout.stride), layout.offset) != ([size], [1], 0):
            flat = FixedLayout(layout.device, layout.dtype, [size], [1], 0)
            operand = ReinterpretView(data=buffer, layout=flat)
        arg = self.tensor_arg(operand, output=output)
        return arg, [self.index_tensor_expr(index, axes)], self.value_shape(axes)

    def disable_reduction(self) -> contextlib.AbstractContextManager[None]:
        outside = super().disable_reduction()

        @contextlib.contextmanager
        def ctx() -> Iterator[None]:
            with outside:
                yield
            # Back in a reduction loop: values stored outside it are read
            # through in_scope() instead of the CSE store cache.
            for name, value in [*self.cse.store_cache.items()]:
                if value in self.outer_vars:
                    self.outer_stores[name] = value
                    del self.cse.store_cache[name]

        return ctx()

    def codegen_body(self) -> None:
        if not (self.compute or self.stores):
            return
        if self.in_loop:
            self.body.writeline(f"for tile_r in hl.tile({self.numels['r0_']}):")
            with self.body.indent():
                self.body.splice(self.compute)
            self.body.splice(self.stores)
            # Values from the loop cannot be carried to the next top-level loop.
            self.cse.invalidate(self.outer_vars)
        else:
            self.body.splice(self.compute)
            self.body.splice(self.stores)
        self.compute.clear()
        self.stores.clear()

    def codegen_kernel(self) -> str:
        # No buffer or kernel names, so identical schedules share one module.
        self.codegen_body()
        code = IndentedBuffer()
        code.writelines(
            [
                "import torch",
                "import helion",
                "import helion.language as hl",
                "",
            ]
        )
        settings = (
            f'backend="{config.helion.backend}", static_shapes=True, '
            f'autotune_effort="{config.helion.autotune_effort}"'
        )
        if self.has_fp32_dot:
            tf32 = torch.backends.cuda.matmul.fp32_precision == "tf32"
            settings += f', dot_precision="{"tf32" if tf32 else "ieee"}"'
        code.writeline(f"@helion.kernel({settings})")
        names = self.kernel_arg_names()
        signature = ", ".join(f"{name}: torch.Tensor" for name in names.values())
        code.writeline(f"def kernel({signature}) -> None:")
        with code.indent():
            tiles = ", ".join(TILE_NAMES[axis] for axis in self.outer_axes)
            sizes = ", ".join(str(self.numels[axis]) for axis in self.outer_axes)
            sizes = sizes if len(self.outer_axes) == 1 else f"[{sizes}]"
            code.writeline(f"for {tiles} in hl.tile({sizes}):")
            with code.indent():
                code.splice(self.body)
        return re.sub(r"__arg\d+__", lambda ref: names[ref[0]], code.getvalue())

    def kernel_args(self) -> list[HelionTensorArg]:
        """Inputs, then outputs whose stores were not removed."""
        removed = (
            self.removed_buffers
            | self.inplaced_to_remove
            | V.graph.removed_buffers
            | V.graph.inplaced_to_remove
        )
        args = [
            arg
            for arg in self.arguments.values()
            if not (arg.output and arg.operand.get_name() in removed)
        ]
        return sorted(args, key=lambda arg: arg.output)

    def kernel_arg_names(self) -> dict[str, str]:
        counts: collections.Counter[str] = collections.Counter()
        names = {}
        for arg in self.kernel_args():
            prefix = "out" if arg.output else "in"
            names[arg.ref] = f"{prefix}{counts[prefix]}"
            counts[prefix] += 1
        return names

    def call_args(self) -> list[str]:
        # An input whose storage an output took over is passed as that output.
        # Reuse can chain inside one kernel: with buf1 reusing buf0 and buf2
        # reusing buf1, the wrapper runs `buf1 = buf0; del buf0` and
        # `buf2 = buf1; del buf1`, so buf0 is passed as buf2.
        reused = {inp: out for out, inp in self.inplace_update_buffers.items()}
        args = []
        for arg in self.kernel_args():
            operand = arg.operand
            name = operand.get_name()
            while name in reused:
                name = reused[name]
            if name != operand.get_name():
                target = V.graph.get_buffer(name)
                operand = ReinterpretView(data=target, layout=operand.get_layout())
            args.append(
                operand.codegen_reference()
                if isinstance(operand, ReinterpretView)
                else operand.get_name()
            )
        return args

    def codegen_nan_check(self) -> None:
        # Check each buffer, as Triton does, not call_args(): a flat operand is
        # passed as a view of its whole storage span, which can include memory
        # outside the tensor.
        wrapper = V.graph.wrapper_code
        _, call_args, signatures, _ = self.args.python_argdefs()
        for arg, signature in zip(call_args, signatures):
            if isinstance(signature, TensorArg):
                wrapper.writeline(f"assert not {arg}.isnan().any().item()")
                wrapper.writeline(f"assert not {arg}.isinf().any().item()")

    def call_kernel(
        self, name: str, node: IRNode | None = None, deallocate_ws: bool = True
    ) -> None:
        wrapper = V.graph.wrapper_code
        wrapper.writeline(wrapper.wrap_kernel_call(name, self.call_args()))

    def load(self, name: str, index: sympy.Expr) -> CSEVariable:
        if name in self.outer_stores:
            return self.in_scope(self.outer_stores[name])
        buffer = V.graph.get_buffer(name)
        index = self.tile_index(index)
        axes = self.index_axes(index)
        node = self.current_node
        if self.in_loop and node is not None and node.is_native_matmul():
            return self.load_dot_operand(buffer, index, axes)
        mask, mask_axes = self.load_mask(axes)
        if not mask_axes <= OrderedSet(axes):
            axes = [axis for axis in self.axes if axis in axes or axis in mask_axes]
            arg, coordinates, shape = self.flat_subscript(buffer, index, axes)
        elif (found := self.subscript(buffer, index, axes)) is not None:
            arg, coordinates, shape = found
        else:
            raise AssertionError("expected a flat subscript")
        dtype = upcast_compute_type(buffer.get_dtype())
        cast_suffix = "" if dtype == buffer.get_dtype() else f".to({dtype})"
        subscript = ", ".join(coordinates)
        if mask is None:
            line = f"{arg.ref}[{subscript}]{cast_suffix}"
        else:
            # Helion masks only lanes past the tile range; the body of
            # ops.masked() must not read where its mask is false.
            mask = self.align(mask, shape)
            index = "" if coordinates == ["()"] else subscript
            line = f"hl.load({arg.ref}, [{index}], extra_mask={mask}){cast_suffix}"
        value = self.cse.generate(self.loads, line, dtype=dtype, shape=shape)
        aligned = self.value_shape(axes)
        if aligned == shape:
            return value
        return self.cse.generate(
            self.loads,
            self.align(value, self.value_shape()),
            dtype=dtype,
            shape=aligned,
        )

    def load_dot_operand(
        self, buffer: IRNode, index: sympy.Expr, axes: Sequence[str]
    ) -> CSEVariable:
        """A whole matrix tile in its storage dtype: [z, y, r0_] or [z, r0_, x]."""
        if ("x" in axes) == ("y" in axes):
            raise NotImplementedError(f"Cannot read a Helion dot operand at {index}")
        batch = ["z0_"] if "z0_" in self.numels else []
        operand = [*batch, "r0_", "x"] if "x" in axes else [*batch, "y", "r0_"]
        found = OrderedSet(axes) <= OrderedSet(operand) and self.subscript(
            buffer, index, operand, flat=False
        )
        if not found:
            raise NotImplementedError(f"Cannot read a Helion dot operand at {index}")
        arg, coordinates, shape = found
        line = f"{arg.ref}[{', '.join(coordinates)}]"
        return self.cse.generate(
            self.loads, line, dtype=buffer.get_dtype(), shape=shape
        )

    def store(
        self,
        name: str,
        index: sympy.Expr,
        value: CSEVariable,
        mode: str | None = None,
    ) -> None:
        if mode is not None:
            raise NotImplementedError(f"Helion kernels do not support {mode} stores")
        buffer = V.graph.get_buffer(name)
        layout = buffer.get_layout()
        if isinstance(layout, MutationLayoutSHOULDREMOVE):
            buffer = layout.get_buffer()
        index = self.tile_index(index)
        # A size-1 tile the index skips stays in the subscript.
        value_axes = [TILE_AXES[str(dim)] for dim in value.shape or () if dim != 1]
        axes = [
            axis
            for axis in self.axes
            if self.symbols[axis] in index.free_symbols
            or (axis in value_axes and self.numels[axis] == 1)
        ]
        found = self.subscript(buffer, index, axes, output=True)
        if found is None:
            raise AssertionError("expected a flat subscript")
        arg, coordinates, shape = found
        dtype = buffer.get_dtype()
        cast_suffix = "" if value.dtype == dtype else f".to({dtype})"
        buffer_code = self.compute if self.in_loop else self.stores
        line = f"{arg.ref}[{', '.join(coordinates)}] = {self.align(value, shape)}"
        # DeferredLine, so that remove_kernel_local_buffers() can drop it.
        buffer_code.writeline(DeferredLine(name, line + cast_suffix))

    def index_expr(self, expr: sympy.Expr, dtype: torch.dtype) -> CSEVariable:
        index = self.tile_index(expr)
        dtype = upcast_compute_type(dtype)
        if not index.free_symbols:
            value = int(index) if index.is_Integer else float(index)
            line = HelionOverrides.constant(value, dtype)
            return self.cse.generate(self.compute, line, dtype=dtype, shape=())
        axes = self.index_axes(index)
        return self.cse.generate(
            self.compute,
            f"({self.index_tensor_expr(index, axes)}).to({dtype})",
            dtype=dtype,
            shape=self.value_shape(axes),
        )

    def index_tensor_expr(self, index: sympy.Expr, axes: Sequence[str]) -> str:
        if not OrderedSet(axes) <= OrderedSet(self.scope_axes) or (
            index.free_symbols - OrderedSet(self.symbols[axis] for axis in axes)
        ):
            raise NotImplementedError(f"Unsupported Helion index: {index}")
        # tile.index is int32; widen it if any subexpression could overflow.
        ranges = {
            self.symbols[axis]: ValueRanges(0, self.numels[axis] - 1) for axis in axes
        }
        bounds = [bound_sympy(term, ranges) for term in sympy.preorder_traversal(index)]
        widen = any(not (-(2**31) <= b.lower and b.upper < 2**31) for b in bounds)
        suffix = ".to(torch.int64)" if widen else ""
        replacements = {
            self.symbols[axis]: sympy.Symbol(self.index_tensor(axis) + suffix)
            for axis in axes
        }
        return HelionPrinter().doprint(index.xreplace(replacements), simplify=False)

    def load_mask(
        self, axes: Sequence[str]
    ) -> tuple[CSEVariable | None, OrderedSet[str]]:
        """The extra_mask of a load over `axes`, and its axes. A mask over none of
        the load's axes cannot move its address and is left to masked()."""
        load_axes = OrderedSet(axes)
        guards = []
        guard_axes: OrderedSet[str] = OrderedSet()
        for mask in self.load_masks:
            mask_axes = OrderedSet(
                TILE_AXES[str(dim)] for dim in mask.shape or () if dim != 1
            )
            if mask_axes <= load_axes or mask_axes & load_axes:
                guards.append(mask)
                guard_axes |= mask_axes
        if not guards:
            return None, guard_axes
        result = guards[0]
        for guard in guards[1:]:
            result = V.ops.logical_and(result, guard)
        return result, guard_axes

    def masked(
        self,
        mask: CSEVariable,
        body: Callable[[], CSEVariable],
        other: bool | float | int,
    ) -> CSEVariable:
        self.load_masks.append(mask)
        try:
            result = body()
        finally:
            self.load_masks.pop()
        if result.dtype is None:
            raise AssertionError(f"expected a dtype for masked value {result}")
        return V.ops.where(mask, result, V.ops.constant(other, result.dtype))

    def accumulator(self, init: bool | float | int, dtype: torch.dtype) -> CSEVariable:
        tiles = tuple(TILE_NAMES[axis] for axis in self.outer_axes)
        name = f"acc{self.accumulator_count or ''}"
        self.accumulator_count += 1
        acc = CSEVariable(name, ValueRanges.unknown(), dtype, tiles)
        self.outer_vars.add(acc)
        shape = ", ".join(tiles)
        if init == 0 and dtype != torch.bool:
            self.body.writeline(f"{acc} = hl.zeros([{shape}], dtype={dtype})")
        else:
            value = constant_repr(init)
            self.body.writeline(f"{acc} = hl.full([{shape}], {value}, dtype={dtype})")
        return acc

    def dot(self, lhs: CSEVariable, rhs: CSEVariable) -> CSEVariable:
        # Each dot has its own accumulator, so horizontally fused dots stay apart.
        self.has_fp32_dot |= torch.float32 in (lhs.dtype, rhs.dtype)
        acc = self.accumulator(0, torch.float32)
        op = "baddbmm" if "z0_" in self.numels else "addmm"
        self.compute.writeline(f"{acc} = torch.{op}({acc}, {lhs}, {rhs})")
        self.dot_results.add(acc)
        return acc

    def reduction(
        self,
        dtype: torch.dtype,
        src_dtype: torch.dtype,
        reduction_type: str,
        value: CSEVariable | tuple[CSEVariable, ...],
    ) -> CSEVariable | tuple[CSEVariable, ...]:
        if reduction_type == "dot":
            return value
        if self.is_matmul or reduction_type not in REDUCTIONS:
            raise AssertionError(f"Helion kernels do not support {reduction_type}")
        if not isinstance(value, CSEVariable):
            raise AssertionError(f"expected one value to reduce, got {value}")
        if "tile_r" not in map(str, value.shape or ()):
            # Broadcast over tile_r, so that each block reduces one copy per lane.
            axes = [TILE_AXES[str(dim)] for dim in value.shape or () if dim != 1]
            shape = self.value_shape([*axes, "r0_"])
            zeros = f"hl.zeros([{', '.join(map(str, shape))}], dtype={value.dtype})"
            line = f"{self.align(value, self.value_shape())} + {zeros}"
            value = self.cse.generate(
                self.compute, line, dtype=value.dtype, shape=shape
            )
        init = Reduction.default_accumulator(
            reduction_type, upcast_compute_type(src_dtype)
        )
        if not isinstance(init, (bool, float, int)):
            raise AssertionError(f"expected one initial value, got {init}")
        acc = self.accumulator(init, upcast_compute_type(dtype))
        block = {
            "sum": f"{acc} + torch.sum({value}, -1)",
            "prod": f"{acc} * torch.prod({value}, -1)",
            "max": f"torch.maximum({acc}, torch.amax({value}, -1))",
            "min": f"torch.minimum({acc}, torch.amin({value}, -1))",
        }[reduction_type]
        self.compute.writeline(f"{acc} = {block}")
        return acc
