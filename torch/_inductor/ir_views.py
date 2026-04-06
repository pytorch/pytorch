from __future__ import annotations

from .ir_base import (
    Any,
    Buffer,
    Callable,
    CleanDiv,
    ConstantBuffer,
    Dep,
    Expr,
    FixedLayout,
    FlexibleLayout,
    FloorDiv,
    GPU_ALIGN_BYTES,
    GuardOnDataDependentSymNode,
    IRNode,
    IndentedBuffer,
    Integer,
    Layout,
    ModularIndexing,
    OpsValue,
    OrderedSet,
    Sequence,
    SymT,
    Symbol,
    V,
    _IntLike,
    _T,
    _V,
    cache_on_self_and_args,
    extract_read_writes,
    free_unbacked_symbols,
    fuse_reindexing,
    get_dtype_size,
    get_free_symbols,
    has_free_unbacked_symbols,
    ir_dataclass,
    ops,
    override,
    patch,
    sympy,
    sympy_index_symbol_with_prefix,
    sympy_product,
    sympy_subs,
    torch,
)
from .ir_compute import Pointwise


def is_storage_and_layout(x: IRNode) -> bool:
    try:
        as_storage_and_layout(x, freeze=False)
        return True
    except NotImplementedError:
        return False


def is_contiguous_storage_and_layout(x: IRNode) -> bool:
    try:
        _buffer, layout = as_storage_and_layout(x, freeze=False)
        # pad the stride here so we will NOT claim an tensor as contiguous
        # if a padding is gonna happen.
        if layout.should_pad_strides():
            layout.pad_strides()
        return layout.is_contiguous()
    except NotImplementedError:
        return False


def as_storage_and_layout(
    x: IRNode,
    freeze: bool = True,
    want_contiguous: bool = False,
    stride_order: Sequence[int | Integer] | None = None,
    allow_padding: bool = False,
    exact_strides: Sequence[int | Integer] | None = None,
) -> tuple[StorageBox, Layout]:
    """
    Try to simplify x into a StorageBox and a Layout.

    allow_padding only affect how we apply stride_order. When allow_padding
    is True, we have the freedom to add padding when applying the stride_order.
    """
    if isinstance(x, TensorBox):
        return as_storage_and_layout(
            x.data,
            freeze=freeze,
            want_contiguous=want_contiguous,
            stride_order=stride_order,
            allow_padding=allow_padding,
            exact_strides=exact_strides,
        )
    if isinstance(x, StorageBox):
        _, layout = as_storage_and_layout(
            x.data,
            freeze=freeze,
            want_contiguous=want_contiguous,
            stride_order=stride_order,
            allow_padding=allow_padding,
            exact_strides=exact_strides,
        )
        return x, x.data.get_layout()
    if isinstance(x, Buffer):
        if freeze:
            if want_contiguous:
                x.freeze_layout()
                assert x.get_layout().is_contiguous()
            elif stride_order is not None:
                x.freeze_layout_with_stride_order(
                    stride_order, allow_padding=allow_padding
                )
            elif exact_strides is not None:
                x.freeze_layout_with_exact_strides(
                    exact_strides, allow_padding=allow_padding
                )
            else:
                x.decide_layout()
        return StorageBox(x), x.get_layout()
    if isinstance(x, ReinterpretView):
        # making the base of x contiguous or stride_ordered will not necessarily make
        # the ReinterpretView either, so don't pass along those arguments
        buffer, _ = as_storage_and_layout(
            x.data,
            freeze=freeze,
        )
        return buffer, x.layout
    raise NotImplementedError


def is_stride_order_storage_and_layout(
    x: IRNode, stride_order: Sequence[int | Integer]
) -> bool:
    try:
        _buffer, layout = as_storage_and_layout(x, freeze=False)
        return layout.is_stride_ordered(stride_order)
    except NotImplementedError:
        return False


def is_unaligned(node: IRNode) -> bool:
    if isinstance(node, (TensorBox, StorageBox)):
        return is_unaligned(node.data)

    if isinstance(node, ReinterpretView):
        layout = node.layout
        has_unaligned_layout = not V.graph.sizevars.statically_known_multiple_of(
            layout.offset * get_dtype_size(layout.dtype), GPU_ALIGN_BYTES
        )
        return is_unaligned(node.data) or has_unaligned_layout

    if isinstance(node, Buffer):
        return node.get_name() in V.graph.unaligned_buffers

    # assume to be aligned otherwise
    return False


@ir_dataclass
class BaseView(IRNode):
    data: IRNode

    @cache_on_self_and_args("BaseView")
    def get_free_symbol_uses(self, unbacked_only: bool = False) -> OrderedSet[Symbol]:
        return self.data.get_free_symbol_uses(unbacked_only)

    def make_reindexer(self) -> Callable[[Sequence[Expr]], Sequence[Expr]]:
        raise NotImplementedError(f"make_reindexer NYI on {self}")

    def make_indexer(self) -> Callable[[Sequence[Expr]], Expr]:
        inner = self.data.make_indexer()
        reindex = self.make_reindexer()

        def indexer(idx: Sequence[Expr]) -> Expr:
            return inner(reindex(idx))

        return indexer

    def make_loader(self) -> Callable[[Sequence[Expr]], OpsValue]:
        inner = self.data.make_loader()
        reindex = self.make_reindexer()

        def loader(idx: Sequence[Expr]) -> OpsValue:
            return inner(reindex(idx))

        return loader

    @property
    def dtype(self) -> torch.dtype:
        return self.data.get_dtype()

    def get_layout(self) -> Layout:
        return self.data.get_layout()

    def get_device(self) -> torch.device | None:
        return self.data.get_device()

    def get_origin_node(self) -> torch.fx.Node | None:
        return None

    def get_name(self) -> str:
        return self.data.get_name()

    def get_pointwise_size(self) -> Sequence[Expr]:
        return self.get_size()

    def mark_reuse(self, users: int) -> None:
        return self.data.mark_reuse(users)

    def has_exceeded_max_reads(self) -> bool:
        return self.data.has_exceeded_max_reads()

    def realize(self) -> str | None:
        return self.data.realize()

    def realize_hint(self) -> None:
        self.data.realize_hint()

    def get_storage_numel(self) -> _IntLike:
        return self.data.get_storage_numel()

    def is_extern(self) -> bool:
        return self.data.is_extern()

    def is_module_buffer(self) -> bool:
        assert isinstance(self.data, BaseView), type(self.data)
        return self.data.is_module_buffer()

    def get_read_names(self) -> OrderedSet[str]:
        return self.data.get_read_names()

    def get_reads(self) -> OrderedSet[Dep]:
        with patch.object(FlexibleLayout, "allow_indexing", True):
            return extract_read_writes(
                self.make_loader(),
                self.get_size(),
            ).reads

    def unwrap_view(self) -> IRNode:
        x: IRNode = self
        while isinstance(x, BaseView):
            x = x.data
        return x

    def constant_to_device(self, device: torch.device) -> IRNode:
        """Move this to a given device. Requires that all reads are to constants."""
        loader = self.make_loader()
        loader = patch.object(ConstantBuffer, "override_device", device)(loader)
        return Pointwise(
            device=device,
            dtype=self.get_dtype(),
            inner_fn=loader,
            ranges=self.get_size(),
        )


@ir_dataclass
class ExpandView(BaseView):
    size: Sequence[Expr]

    @staticmethod
    def _normalize_size(x: IRNode, new_size: Sequence[_IntLike]) -> Sequence[_IntLike]:
        """Replace `-1` with correct sizes"""
        sizevars = V.graph.sizevars
        new_size = [sympy.expand(s) for s in new_size]
        old_size = x.get_size()
        old_size = [None] * (len(new_size) - len(old_size)) + list(old_size)
        assert len(new_size) == len(old_size)
        for i in range(len(new_size)):
            if new_size[i] == -1:
                assert old_size[i] is not None
                new_size[i] = old_size[i]
            elif old_size[i] is None or V.graph.sizevars.is_size_one_or_false(
                old_size[i]
            ):
                pass
            elif not has_free_unbacked_symbols(
                old_size
            ) and not has_free_unbacked_symbols(new_size):
                # Sanity check: Expect broadcast compatibility
                #
                # NB: new_size[i] == old_size[i] is expected to already be
                # guarded because the meta formula was expected to have taught
                # us this equality.
                v1 = new_size[i]
                v2 = old_size[i]
                assert v1 is not None
                assert v2 is not None
                diff = v1 - v2
                assert (
                    sizevars.optimization_hint(
                        diff,
                        fallback=0,
                    )
                    == 0
                ), (
                    f"Broadcast failed in ExpandView({x.get_size()}, {new_size}) on dimension {i}"
                )
        return new_size

    @classmethod
    def create(cls, x: IRNode, new_size: Sequence[_IntLike]) -> BaseView:
        new_size = cls._normalize_size(x, new_size)

        if is_storage_and_layout(x):
            storage, old_layout = as_storage_and_layout(x)
            skip = len(new_size) - len(old_layout.size)
            assert skip >= 0
            new_stride = [sympy.S.Zero] * skip
            for stride, size in zip(old_layout.stride, old_layout.size):
                new_stride.append(
                    stride
                    if not V.graph.sizevars.is_size_one_or_false(size)
                    else sympy.S.Zero
                )
            new_layout = FixedLayout(
                old_layout.device,
                old_layout.dtype,
                list(new_size),
                new_stride,
                old_layout.offset,
                old_layout.is_pinned,
            )
            return ReinterpretView(data=storage, layout=new_layout)

        return ExpandView(data=x, size=new_size)

    def get_size(self) -> Sequence[Expr]:
        return self.size

    def make_reindexer(
        self,
    ) -> Callable[[Sequence[Expr]], Sequence[Expr]]:
        target = self.get_size()
        actual = self.data.get_size()
        skip = len(target) - len(actual)

        def reindex(
            index: Sequence[Expr],
        ) -> Sequence[Expr]:
            index = list(index[skip:])
            assert len(index) == len(actual)
            for i in range(len(actual)):
                if actual[i] == 1:
                    # zero out broadcast dimension
                    index[i] = sympy.S.Zero
            return index

        return reindex


@ir_dataclass
class PermuteView(BaseView):
    dims: list[Expr]

    @classmethod
    def create(cls, x: IRNode, dims: Sequence[int]) -> BaseView:
        dims = cls._map_neg_dims(dims)
        assert OrderedSet(dims) == OrderedSet(range(len(dims)))

        if is_storage_and_layout(x):
            storage, old_layout = as_storage_and_layout(x)
            new_layout = FixedLayout(
                old_layout.device,
                old_layout.dtype,
                [old_layout.size[i] for i in dims],
                [old_layout.stride[i] for i in dims],
                old_layout.offset,
                old_layout.is_pinned,
            )
            return ReinterpretView(data=storage, layout=new_layout)

        return PermuteView(data=x, dims=dims)

    @classmethod
    def _map_neg_dims(cls, dims: Sequence[int]) -> list[int]:
        return [dim if dim >= 0 else len(dims) + dim for dim in dims]

    def get_size(self) -> Sequence[Expr]:
        assert OrderedSet(self._map_neg_dims(self.dims)) == OrderedSet(
            range(len(self.dims))
        )
        size = self.data.get_size()
        return [size[i] for i in self.dims]

    def make_reindexer(
        self,
    ) -> Callable[[Sequence[Expr]], Sequence[Expr]]:
        inv = {j: i for i, j in enumerate(self.dims)}
        inv = [inv[i] for i in range(len(self.dims))]
        assert OrderedSet(inv) == OrderedSet(range(len(self.dims)))

        def reindex(
            index: Sequence[Expr],
        ) -> Sequence[Expr]:
            return [index[i] for i in inv]

        return reindex


@ir_dataclass
class SqueezeView(BaseView):
    @classmethod
    def create(cls, x: IRNode, *, dim: int | None = None) -> IRNode:
        if is_storage_and_layout(x):
            storage, old_layout = as_storage_and_layout(x)
            new_size = []
            new_stride = []
            if dim is not None:
                assert isinstance(dim, int), type(dim)
                assert 0 <= dim and dim < len(old_layout.size)

            for i, (size, stride) in enumerate(zip(old_layout.size, old_layout.stride)):
                if dim is None:
                    # Only append if dim is not squeezed out
                    if not V.graph.sizevars.is_size_one_or_false(size):
                        new_size.append(size)
                        new_stride.append(stride)
                else:
                    if i != dim:
                        new_size.append(size)
                        new_stride.append(stride)
                    else:
                        assert size == 1, "expected squeezed size to be 1"

            new_layout = FixedLayout(
                old_layout.device,
                old_layout.dtype,
                new_size,
                new_stride,
                old_layout.offset,
                old_layout.is_pinned,
            )
            return ReinterpretView(data=storage, layout=new_layout)

        if dim is None:
            return View.create(
                x,
                [
                    s
                    for s in x.get_size()
                    if not V.graph.sizevars.is_size_one_or_false(s)
                ],
            )
        else:
            assert x.get_size()[dim] == 1
            return View.create(x, [s for i, s in enumerate(x.get_size()) if i != dim])

    @staticmethod
    def squeezer(
        size: Sequence[Expr],
    ) -> tuple[list[int], Callable[[Sequence[Expr]], tuple[Expr, ...]]]:
        new_size = [s for s in size if s != 1]
        not_one = [i for i, s in enumerate(size) if s != 1]
        length = len(size)

        def reindex(index: Sequence[Expr]) -> tuple[Expr, ...]:
            assert len(index) == len(not_one), f"{index} {not_one}"
            new_index: list[Expr] = [sympy.S.Zero] * length
            for idx, s in zip(not_one, index):
                new_index[idx] = s
            return tuple(new_index)

        return new_size, reindex

    def __init__(self, data: Any) -> None:
        raise AssertionError("use SqueezeView.create()")


@ir_dataclass
class GenericView(BaseView):
    size: Sequence[Expr]
    reindex: Callable[[Sequence[Expr]], Sequence[Expr]]

    def make_reindexer(
        self,
    ) -> Callable[[Sequence[Expr]], Sequence[Expr]]:
        return self.reindex

    def reindex_str(self) -> str:
        index_old = [
            sympy_index_symbol_with_prefix(SymT.INDEX, n) for n in range(len(self.size))
        ]
        index_new = list(self.reindex(index_old))
        return f"lambda {', '.join(map(str, index_old))}: {index_new}"

    def __str__(self) -> str:
        return self.str_helper(
            [self.data, f"size={self.size}", f"reindex={self.reindex_str()}"]
        )

    __repr__ = __str__

    @classmethod
    def create(
        cls,
        x: IRNode,
        new_size: Sequence[Expr],
        reindex: Callable[[Sequence[Expr]], Sequence[Expr]],
    ) -> BaseView:
        return cls(data=x, size=list(new_size), reindex=reindex)

    def get_size(self) -> Sequence[Expr]:
        return self.size


@ir_dataclass
class View(GenericView):
    """
    This class handles tensor reshaping by computing appropriate index transformations
    to map the new shape back to the original storage layout.
    """

    @staticmethod
    def handle_negative_index(idx: Expr, size: Expr) -> Expr:
        idx = sympy.expand(idx)
        size = sympy.expand(size)
        evaluate_expr = V.graph.sizevars.shape_env.evaluate_expr
        if evaluate_expr(sympy.Lt(idx, 0)):
            idx = idx + size
        return idx

    @classmethod
    @override
    def create(cls, x: IRNode, new_size: Sequence[Expr]) -> IRNode:  # type: ignore[override]
        assert isinstance(new_size, Sequence), type(new_size)
        old_size, new_size = cls.resolve_negative_size(x.get_size(), new_size)

        # Skip pointless views
        if V.graph.sizevars.statically_known_list_equals(old_size, new_size):
            return x

        unbacked_symbols_in_sizes = (
            len(free_unbacked_symbols(old_size)) > 0
            or len(free_unbacked_symbols(new_size)) > 0
        )
        is_contiguous = is_contiguous_storage_and_layout(x)

        def create_reinterpret_view(
            inp: IRNode, new_size: Sequence[Expr], new_stride: Sequence[Expr]
        ) -> ReinterpretView:
            storage, old_layout = as_storage_and_layout(inp, want_contiguous=True)
            new_layout = FixedLayout(
                old_layout.device,
                old_layout.dtype,
                new_size,
                new_stride,
                old_layout.offset,
                old_layout.is_pinned,
            )
            return ReinterpretView(data=storage, layout=new_layout)

        def handle_unbacked_or_dynamic_reshape(
            x: IRNode,
        ) -> IRNode:
            """
            Handle the case where view is not possible with current strides.
            Try dynamic_reshape_indexer first; if it fails with unbacked
            symbols (guard_or_false can't resolve comparisons), fall back
            to making the tensor contiguous.
            """
            nonlocal old_size, new_size
            try:
                reindex = cls.dynamic_reshape_indexer(old_size, new_size)
                return cls(data=x, size=list(new_size), reindex=reindex)
            except GuardOnDataDependentSymNode:
                # dynamic_reshape_indexer cannot handle unbacked SymInts
                # because guard_or_false can't resolve size comparisons.
                # https://github.com/pytorch/pytorch/issues/145561
                x = ExternKernel.require_contiguous(x)
                return create_reinterpret_view(
                    x, new_size, FlexibleLayout.contiguous_strides(new_size)
                )

        if 0 in new_size:

            def fake_reindex(index: Any) -> tuple[int, ...]:
                return tuple([0] * len(old_size))

            return cls(data=x, size=list(new_size), reindex=fake_reindex)

        # TODO: a new class for FixedTransferLayout that output layout is constrained by input layout
        elif is_contiguous:
            # Input is contiguous, output can use contiguous strides
            return create_reinterpret_view(
                x, new_size, FlexibleLayout.contiguous_strides(new_size)
            )

        # Input is non-contiguous. Check if we can get storage/layout.
        if not is_storage_and_layout(x):
            # Can't get storage/layout (e.g., for Pointwise nodes)
            return handle_unbacked_or_dynamic_reshape(x)

        # Try to compute valid output strides.
        storage, old_layout = as_storage_and_layout(x, freeze=False)

        old_stride = old_layout.stride

        # Convert sympy exprs to SymInt for _compute_stride, then convert back
        old_size_symint = V.graph.sizevars.to_symints_or_ints(old_size)
        old_stride_symint = V.graph.sizevars.to_symints_or_ints(old_stride)
        new_size_symint = V.graph.sizevars.to_symints_or_ints(new_size)

        from torch._subclasses.fake_impls import _compute_stride

        # Use size_oblivious=True for unbacked symbols to avoid DDE errors
        new_stride_symint = _compute_stride(
            old_size_symint,
            old_stride_symint,
            new_size_symint,
            size_oblivious=unbacked_symbols_in_sizes,
        )

        if new_stride_symint is not None:
            # Convert SymInt back to sympy expressions
            new_stride = [
                s.node.expr if hasattr(s, "node") else sympy.Integer(s)
                for s in new_stride_symint
            ]
            # View is possible with computed strides
            new_layout = FixedLayout(
                old_layout.device,
                old_layout.dtype,
                new_size,
                new_stride,
                old_layout.offset,
                old_layout.is_pinned,
            )
            return ReinterpretView(data=storage, layout=new_layout)

        # View not possible with current strides
        return handle_unbacked_or_dynamic_reshape(x)

    @staticmethod
    def resolve_negative_size(
        old_size: Sequence[Expr], new_size: Sequence[Expr]
    ) -> tuple[list[Expr], list[Expr]]:
        new_size = [V.graph.sizevars.simplify(x) for x in new_size]
        old_size = [V.graph.sizevars.simplify(x) for x in old_size]

        new_size = list(new_size)
        for i in range(len(new_size)):
            if new_size[i] == -1:
                new_size[i] = sympy.S.One
                new_size[i] = CleanDiv(sympy_product(old_size), sympy_product(new_size))
                break

        V.graph.sizevars.check_equals(sympy_product(old_size), sympy_product(new_size))
        return old_size, new_size

    @classmethod
    def dynamic_reshape_indexer(
        cls,
        old_size: Sequence[_IntLike],
        new_size: Sequence[_IntLike],
        dense_dim: int | None = None,
    ) -> Callable[[Sequence[_T]], Sequence[_V]]:
        try:
            reindex = cls._dynamic_reshape_indexer(old_size, new_size, dense_dim)
        except (AssertionError, GuardOnDataDependentSymNode, IndexError):
            # optimistic algorithm failed, lets do a fallback
            flat = [sympy_product(old_size)]
            reindex1 = cls._dynamic_reshape_indexer(old_size, flat)
            reindex2 = cls._dynamic_reshape_indexer(flat, new_size)
            reindex = fuse_reindexing(reindex1, reindex2)
        return reindex

    @staticmethod
    def _dynamic_reshape_indexer(
        old_size: Sequence[Expr],
        new_size: Sequence[Expr],
        dense_dim: int | None = None,
    ) -> Callable[[Sequence[Expr]], Sequence[Expr]]:
        """
        Perform a reshape entirely by modifying indexing math
        """
        guard_or_false = V.graph.sizevars.guard_or_false

        def compare_sizes(a: Expr, b: Expr) -> int:
            """
            Compare two symbolic sizes, returning -1 if a < b, 0 if a == b, 1 if a > b.

            For unbacked symbols, guard_or_false returns False, so we fall back
            to divisibility checks.
            """
            if guard_or_false(sympy.Eq(a, b)):
                return 0
            if guard_or_false(sympy.Lt(a, b)):
                return -1
            if guard_or_false(sympy.Gt(a, b)):
                return 1

            # Divisibility fallback for unbacked symbols:
            # e.g. comparing u0 vs u0*u1, statically_known_multiple_of(u0*u1, u0)
            # returns True so we take the merge path (return -1).
            # The merge reindex for old=[u0, u1] -> new=[u0*u1] is:
            #   for k in range(u0 * u1):
            #     i = k // u1
            #     j = k % u1
            #     z[k] = x[i, j] + 1
            # Two cases where this could seem wrong but is still safe:
            #   u1=1: the merge reindex is correct since the extra dim is
            #         size 1 (a no-op). e.g. k // 1 = k, k % 1 = 0.
            #   u0=0: loop is range(0), no kernel runs, result doesn't matter.

            if V.graph.sizevars.statically_known_multiple_of(b, a):
                return -1
            if V.graph.sizevars.statically_known_multiple_of(a, b):
                return 1

            raise GuardOnDataDependentSymNode(sympy.Eq(a, b))

        # TODO: These symbols may not escape, if they don't assert so and
        # treat them as temporary
        vars = [
            sympy_index_symbol_with_prefix(SymT.VIEW, i) for i in range(len(new_size))
        ]

        stack_new = list(zip(vars, new_size))
        stack_old = list(old_size)

        # process the dense dim first
        reordering_dense_dim = (
            dense_dim is not None
            and dense_dim != len(stack_old) - 1
            and len(new_size) == 1
        )
        if reordering_dense_dim:
            assert dense_dim is not None  # mypy
            old_dim = stack_old.pop(dense_dim)
            stack_old.append(old_dim)

        view_expr = []
        while stack_new and stack_old:
            size_old = stack_old.pop()
            var, size_new = stack_new.pop()
            if size_old == 1:
                view_expr.append(sympy.S.Zero)
                stack_new.append((var, size_new))  # re-add
            elif size_new == 1:
                stack_old.append(size_old)  # re-add
            elif compare_sizes(size_new, size_old) == 0:
                view_expr.append(var)
            elif compare_sizes(size_new, size_old) < 0:
                while compare_sizes(size_new, size_old) < 0:
                    var2, size_new2 = stack_new.pop()
                    var = var2 * size_new + var
                    size_new = size_new * size_new2
                view_expr.append(var)
                V.graph.sizevars.check_equals(size_new, size_old)
            elif compare_sizes(size_new, size_old) > 0:
                divisor = sympy.S.One
                modulus = size_old
                view_expr.append(ModularIndexing(var, divisor, modulus))
                divisor = divisor * modulus
                while compare_sizes(size_new, size_old) > 0:
                    modulus = stack_old.pop()
                    view_expr.append(ModularIndexing(var, divisor, modulus))
                    divisor = divisor * modulus
                    size_old = size_old * modulus
                V.graph.sizevars.check_equals(size_new, size_old)
            else:
                raise AssertionError

        while stack_old:
            size_old = stack_old.pop()
            V.graph.sizevars.check_equals(size_old, 1)
            view_expr.append(sympy.S.Zero)

        while stack_new:
            var, size_new = stack_new.pop()
            V.graph.sizevars.check_equals(size_new, 1)

        if dense_dim is not None and len(new_size) == 1:
            view_expr.reverse()
            # Move the last expression (dense dim) to its original position
            dense_expr = view_expr.pop()
            view_expr.insert(dense_dim, dense_expr)
        else:
            view_expr.reverse()

        assert len(view_expr) == len(old_size)

        def reindex(
            index: Sequence[Expr],
        ) -> Sequence[Expr]:
            assert len(index) == len(vars), (len(index), len(vars))
            replacements = dict(zip(vars, index))
            return tuple(sympy_subs(x, replacements) for x in view_expr)

        return reindex


@ir_dataclass
class ReinterpretView(BaseView):
    """Pretend our storage has a different layout"""

    layout: Layout

    def __post_init__(self) -> None:
        super().__post_init__()
        if isinstance(self.data, BaseView):
            object.__setattr__(self, "data", self.data.unwrap_view())

    def __str__(self) -> str:
        return self.str_helper(
            [
                self.data,
                self.layout,
            ]
        )

    __repr__ = __str__

    def get_name(self) -> str:
        return self.data.get_name()

    def get_device(self) -> torch.device | None:
        return self.layout.device

    def get_origin_node(self) -> torch.fx.Node | None:
        return None

    @property
    def dtype(self) -> torch.dtype:
        return self.layout.dtype

    def get_size(self) -> Sequence[Expr]:
        return list(self.layout.size)

    def get_stride(self) -> Sequence[Expr]:
        return list(self.layout.stride)

    def make_loader(self) -> Callable[[Sequence[Expr]], OpsValue]:
        def loader(index: Sequence[Expr]) -> OpsValue:
            indexer = self.layout.make_indexer()
            tmp_loader = ops.load(self.get_name(), indexer(index))
            if self.layout.dtype != self.data.dtype:
                return ops.to_dtype_bitcast(tmp_loader, self.dtype, self.data.dtype)
            else:
                return tmp_loader

        return loader

    def make_indexer(self) -> Callable[[Sequence[Expr]], Expr]:
        return self.layout.make_indexer()

    def get_layout(self) -> Layout:
        return self.layout

    def freeze_layout(self) -> None:
        pass

    @cache_on_self_and_args("ReinterpretView")
    def get_free_symbol_uses(
        self, unbacked_only: bool = False
    ) -> OrderedSet[sympy.Symbol]:
        return (
            get_free_symbols(self.layout.size, unbacked_only)
            | get_free_symbols(self.layout.stride, unbacked_only)
            | get_free_symbols(self.layout.offset, unbacked_only)
        )

    def codegen_reference(self, writer: IndentedBuffer | None = None) -> str:
        # reinterpret_tensor is similar to as_strided except:
        # - offset is added to the existing offset (rather than replacing it)
        # - view tracking is disabled similar to unsafe_view
        return V.graph.wrapper_code.codegen_reinterpret_view(
            self.data,
            self.layout.size,
            self.layout.stride,
            self.layout.offset,
            writer.writeline if writer is not None else V.graph.wrapper_code.writeline,
            dtype=self.layout.dtype,
        )

    def num_reads(self) -> int:
        return 1


@ir_dataclass
class DtypeView(BaseView):
    """Pretend our storage has a different type"""

    target_dtype: torch.dtype

    @classmethod
    def create(cls, x: IRNode, new_dtype: torch.dtype) -> BaseView:
        if is_storage_and_layout(x):
            storage, old_layout = as_storage_and_layout(x)
            new_layout = FixedLayout(
                old_layout.device,
                new_dtype,
                old_layout.size,
                old_layout.stride,
                old_layout.offset,
                old_layout.is_pinned,
            )
            return ReinterpretView(data=storage, layout=new_layout)
        return DtypeView(data=x, target_dtype=new_dtype)

    def __str__(self) -> str:
        return self.str_helper([self.data, self.target_dtype])

    __repr__ = __str__

    @property
    def dtype(self) -> torch.dtype:
        return self.target_dtype

    def get_size(self) -> Sequence[Expr]:
        return self.data.get_size()

    def make_loader(self) -> Callable[[Sequence[Expr]], OpsValue]:
        inner = self.data.make_loader()

        def loader(idx: Sequence[Expr]) -> OpsValue:
            return ops.to_dtype_bitcast(inner(idx), self.target_dtype, self.data.dtype)

        return loader


class SliceView(View):
    @classmethod
    def normalize_start_end(
        cls, x: IRNode, dim: int, start: int, end: int
    ) -> tuple[int, int]:
        """
        Normalize start and end such that both are in the range
        [0, x.get_size()[dim]] and start <= end.
        """
        sizevars = V.graph.sizevars
        dim_size = x.get_size()[dim]

        if any(free_unbacked_symbols(x) for x in (start, end, dim_size)):
            min_func = sympy.Min
            max_func = sympy.Max
        else:
            min_func = sizevars.evaluate_min
            max_func = sizevars.evaluate_max

        def clamp(x: Expr, lower: int, upper: int) -> Expr:
            clamped_lower = (
                x if sizevars.statically_known_geq(x, lower) else max_func(x, lower)
            )
            clamped_full = (
                clamped_lower
                if sizevars.statically_known_leq(clamped_lower, upper)
                else min_func(clamped_lower, upper)
            )
            return clamped_full

        def clamp_wrap(
            val: int | None, lower: int, upper: int, default: Expr | int
        ) -> Expr | int:
            if val is None:
                # TODO(rec): can this really happen?
                return default
            val = cls.handle_negative_index(val, dim_size)
            return clamp(val, lower, upper)

        start = clamp_wrap(start, 0, dim_size, 0)
        end = clamp_wrap(end, start, dim_size, dim_size)
        return start, end

    @classmethod
    def create(  # type: ignore[override]
        cls,
        x: IRNode,
        dim: int,
        start: int,
        end: int,
        step: int = 1,
        clamp: bool = True,
    ) -> IRNode:
        step = sympy.expand(step)
        assert isinstance(step, Expr) or step > 0, step
        try:
            if start == 0 and end >= 2**63 - 1 and step == 1:
                return x
        except TypeError:
            pass

        new_size = list(x.get_size())

        # NB: Ordinarily we default to clamping.
        # We only don't clamp for split_with_sizes. For split_with_sizes, sizes should be already valid
        # failing in this situation is ok, since invalid sizes could trigger silent errors.
        if clamp:
            start, end = cls.normalize_start_end(x, dim, start, end)

        new_size[dim] = FloorDiv(end - start + (step - 1), step)

        if is_storage_and_layout(x):
            # Fast path
            storage, old_layout = as_storage_and_layout(x)
            new_stride = list(old_layout.stride)
            new_stride[dim] = new_stride[dim] * step
            new_layout = FixedLayout(
                old_layout.device,
                old_layout.dtype,
                new_size,
                new_stride,
                old_layout.offset + old_layout.stride[dim] * start,
                old_layout.is_pinned,
            )
            return ReinterpretView(data=storage, layout=new_layout)

        def reindex(
            index: Sequence[Expr],
        ) -> Sequence[Expr]:
            assert len(index) == len(new_size), f"wrong ndim {index} {new_size}"
            index = list(index)
            index[dim] = index[dim] * step + start
            return index

        # redirect to a generic view
        return SliceView(data=x, size=new_size, reindex=reindex)
