from __future__ import annotations

from .ir_base import (
    Callable,
    ComputedBuffer,
    ConstantBuffer,
    Dep,
    Expr,
    FlexibleLayout,
    IRNode,
    IndentedBuffer,
    InputBuffer,
    Layout,
    Operation,
    OpsValue,
    OrderedSet,
    OutputSpec,
    Sequence,
    ShapeAsConstantBuffer,
    V,
    _IntLike,
    cache_on_self_and_args,
    config,
    dataclasses,
    dependencies,
    indent,
    is_cpu,
    overload,
    sympy,
    torch,
)
from .ir_compute import Pointwise, Reduction, Scan, Sort
from .ir_views import ReinterpretView


@dataclasses.dataclass
class MutableBox(IRNode):
    """
    TensorBox / StorageBox allow in-place mutation of Tensors
    """

    data: IRNode

    def has_exceeded_max_reads(self) -> bool:
        return self.data.has_exceeded_max_reads()

    def get_device(self) -> torch.device | None:
        return self.data.get_device()

    def make_loader(self) -> Callable[[Sequence[Expr]], OpsValue]:
        return self.data.make_loader()

    def make_indexer(self) -> Callable[[Sequence[Expr]], Expr]:
        return self.data.make_indexer()

    def get_stride(self) -> Sequence[_IntLike]:
        return self.data.get_stride()

    def get_name(self) -> str:
        return self.data.get_name()

    def has_large_inner_fn(self, threshold: int | None = None) -> bool:
        return self.data.has_large_inner_fn(threshold)

    def mark_reuse(self, users: int) -> None:
        return self.data.mark_reuse(users)

    def realize_hint(self) -> None:
        return self.data.realize_hint()

    def unwrap_view(self) -> IRNode:
        return self.data.unwrap_view()

    def is_input_buffer(self) -> bool:
        return self.data.is_input_buffer()

    def freeze_layout(self) -> None:
        return self.data.freeze_layout()

    def freeze_layout_with_stride_order(
        self, order: Sequence[int], allow_padding: bool = False
    ) -> None:
        return self.data.freeze_layout_with_stride_order(order, allow_padding)

    def freeze_layout_with_fill_order(self, order: Sequence[int]) -> None:
        return self.data.freeze_layout_with_fill_order(order)

    def freeze_layout_with_same_order(self, stride: Sequence[_IntLike]) -> None:
        return self.data.freeze_layout_with_same_order(stride)

    def freeze_layout_with_exact_strides(
        self, exact_strides: Sequence[_IntLike], allow_padding: bool = False
    ) -> None:
        return self.data.freeze_layout_with_exact_strides(exact_strides, allow_padding)

    def get_read_writes(self) -> dependencies.ReadWrites:
        return self.data.get_read_writes()

    def get_reads(self) -> OrderedSet[Dep]:
        return self.data.get_reads()

    def num_reads(self) -> int:
        return self.data.num_reads()

    def get_storage_numel(self) -> _IntLike:
        return self.data.get_storage_numel()

    def get_reduction_type(self) -> str | None:
        return self.data.get_reduction_type()

    def get_reduction_size(self) -> Sequence[Expr]:
        return self.data.get_reduction_size()

    def is_extern(self) -> bool:
        return self.data.is_extern()

    def is_no_op(self) -> bool:
        return self.data.is_no_op()

    def constant_to_device(self, device: torch.device) -> IRNode:
        return self.data.constant_to_device(device)

    def get_mutation_names(self) -> Sequence[str]:
        return self.data.get_mutation_names()

    def get_operation_name(self) -> str:
        return self.data.get_operation_name()

    def get_inputs_that_alias_output(self) -> Sequence[str]:
        return self.data.get_inputs_that_alias_output()

    def realize(self) -> str | None:
        return self.data.realize()

    @cache_on_self_and_args("MutableBox")
    def get_free_symbol_uses(
        self, unbacked_only: bool = False
    ) -> OrderedSet[sympy.Symbol]:
        return self.data.get_free_symbol_uses(unbacked_only)

    def get_read_names(self) -> OrderedSet[str]:
        return self.data.get_read_names()

    def get_defining_op(self) -> Operation | None:
        return self.data.get_defining_op()

    def codegen_reference(self, writer: IndentedBuffer | None = None) -> str:
        return self.data.codegen_reference(writer)

    @property
    def layout(self) -> OutputSpec:
        # we intentionally call get_output_spec (rather than get_layout) since Buffer.layout is an OutputSpec
        return self.data.get_output_spec()

    def get_layout(self) -> Layout:
        return self.data.get_layout()

    def get_output_spec(self) -> OutputSpec:
        return self.data.get_output_spec()

    def get_size(self) -> Sequence[Expr]:
        return self.data.get_size()

    @property
    def dtype(self) -> torch.dtype:
        return self.data.dtype

    def __str__(self) -> str:
        if isinstance(self.data, MutableBox):
            line0 = f"{type(self).__name__}({type(self.data).__name__}("
            endl = "))"
            inner = self.data.data
        else:
            line0 = f"{type(self).__name__}("
            inner = self.data
            endl = ")"

        lines = [
            line0,
            indent(str(inner)),
            endl,
        ]
        return "\n".join(lines)

    __repr__ = __str__


class TensorBox(MutableBox):
    @overload
    @staticmethod
    def create(data: ShapeAsConstantBuffer) -> ShapeAsConstantBuffer: ...
    @overload
    @staticmethod
    def create(data: IRNode) -> TensorBox: ...

    @staticmethod
    def create(data: IRNode):
        if isinstance(data, ShapeAsConstantBuffer):
            return data
        return TensorBox(StorageBox(data))


class StorageBox(MutableBox):
    """
    StorageBox allow in-place mutation of Tensors
    """

    def is_input_buffer(self) -> bool:
        if isinstance(self.data, (InputBuffer, ReinterpretView)):
            return self.data.get_name() in V.graph.graph_inputs
        return False

    def is_module_buffer(self) -> bool:
        return (
            isinstance(self.data, (ConstantBuffer))
            and self.data.get_name() in V.graph.constants
        )

    def realize(self) -> str | None:
        if IRNode.is_realized_node(self.data):
            return self.data.get_name()

        assert isinstance(self.data, (Pointwise, Reduction, Scan, Sort)), type(
            self.data
        )
        origin_node = self.data.get_origin_node()
        traceback = self.data.get_traceback()
        device = self.data.get_device()
        assert device is not None

        self.data = ComputedBuffer(
            name=None,
            layout=FlexibleLayout(
                device=device,
                dtype=self.data.get_dtype(),
                size=self.data.get_size(),
                is_pinned=False,
            ),
            data=self.data,
        )
        self.data.name = V.graph.register_buffer(self.data)
        V.graph.register_operation(self.data)
        self.data.origins = self.origins
        self.data.origin_node = origin_node
        self.data.traceback = traceback
        return self.data.name

    def realize_hint(self) -> None:
        """
        Called on buffers we expect to be forced to realize later.
        """
        if (
            isinstance(self.data, (Pointwise, Reduction))
            and self.data.inner_fn_opcount().nontrivial_read_count > 1
        ):
            self.realize()

    def has_accumulated_enough_reads_by_size(self, threshold: int) -> bool:
        from torch._inductor.utils import is_nonfreeable_buffers

        size_of_reads = [
            V.graph.get_dep_size_hint(dep)
            for dep in self.get_reads()
            if not is_nonfreeable_buffers(dep)
        ]
        if not size_of_reads:
            return False
        total_size = sum(size_of_reads)
        max_size = max(size_of_reads)
        min_size = min(size_of_reads)
        return (
            total_size >= threshold
            and total_size / max_size >= 2
            and max_size == min_size
        )

    def has_exceeded_max_reads(self) -> bool:
        return isinstance(self.data, Pointwise) and (
            self.num_reads() > config.realize_acc_reads_threshold
            or self.has_large_inner_fn()
            or (
                config.realize_acc_reads_size_threshold is not None
                and self.has_accumulated_enough_reads_by_size(
                    config.realize_acc_reads_size_threshold
                )
            )
        )

    def should_realize_on_reuse(self, users: int) -> bool:
        """
        A heuristic to decide if we should realize a tensor
        that is used multiple times.
        """
        if users > 1 and isinstance(self.data, (Pointwise, Reduction)):
            if is_cpu(self.data):
                # Heuristic for realizing reused result of heavy ops on cpu
                opcount = self.data.inner_fn_opcount()
                heavy_ops = ["exp", "sigmoid"]  # a list of heavy ops
                if any(x in opcount.used_ops for x in heavy_ops):
                    return True
            return (
                self.num_reads() > config.realize_reads_threshold
                or self.has_large_inner_fn()
            )
        return False

    def mark_reuse(self, users: int) -> None:
        if self.should_realize_on_reuse(users):
            self.realize()

    def num_reads(self) -> int:
        return self.data.num_reads()
