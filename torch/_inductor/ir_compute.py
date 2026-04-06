from __future__ import annotations

from .ir_base import (
    Any,
    BackendFeature,
    Callable,
    ComputedBuffer,
    ConstantBuffer,
    Dep,
    DeviceProperties,
    Expr,
    FlexibleLayout,
    FloorDiv,
    IRNode,
    Integer,
    Literal,
    Mod,
    Never,
    OpCountResult,
    OpCounterCSE,
    OpsValue,
    OrderedSet,
    ReductionHint,
    ReductionType,
    Sequence,
    StoreMode,
    SupportsFloat,
    SupportsInt,
    SymT,
    Symbol,
    SymbolUsageCollectorOpsHandler,
    TypeAlias,
    V,
    _IntLike,
    _NumLike,
    _P,
    _is_static,
    cache_on_self,
    cache_on_self_and_args,
    config,
    dependencies,
    dtype_from_size,
    extract_free_symbols,
    extract_input_node_reduction_ranges,
    extract_read_writes,
    functools,
    get_free_symbols,
    has_triton,
    ir_dataclass,
    is_boolean_dtype,
    is_float_dtype,
    is_gpu,
    itertools,
    log,
    ops,
    ops_wrapper,
    partial,
    patch,
    sympy,
    sympy_index_symbol_with_prefix,
    sympy_product,
    torch,
    triton_version,
)


@ir_dataclass
class Loops(IRNode):
    device: torch.device
    dtype: torch.dtype
    inner_fn: Callable[..., Any]
    ranges: Sequence[_IntLike]

    @cache_on_self_and_args("Loops")
    def get_free_symbol_uses(
        self, unbacked_only: bool = False
    ) -> OrderedSet[sympy.Symbol]:
        return OrderedSet().union(
            *(get_free_symbols(e, unbacked_only) for e in self.ranges),
            self.inner_fn_free_symbols(unbacked_only),
        )

    def _to_str(self, names: Sequence[str]) -> str:
        return self.str_helper(
            [
                f"'{self.device.type}'",
                str(self.dtype),
                self.inner_fn_str(),
            ]
            + [f"{name}={getattr(self, name)}" for name in names]
            + [f"origin_node={self.origin_node!r}"]
        )

    def __str__(self) -> str:
        return self._to_str(("ranges",))

    __repr__ = __str__

    def get_device(self) -> torch.device | None:
        return self.device

    def get_origin_node(self) -> torch.fx.Node | None:
        return self.origin_node

    def get_size(self) -> Sequence[Expr]:
        return self.ranges

    def get_pointwise_size(self) -> Sequence[Expr]:
        return self.ranges

    @classmethod
    def create(cls, *args: Any, **kwargs: Any) -> TensorBox:
        origin_node = kwargs.pop("origin_node", None)
        tb = kwargs.pop("traceback", None)
        r = cls(*args, **kwargs)
        # Need to explicitly set origin_node here to propagate it down.
        # todo(chilli): I think it would be better for IRNode to directly set
        # origin_node
        r._post_init_setattr("origin_node", origin_node)
        r._post_init_setattr("traceback", tb or r.traceback)
        return TensorBox.create(r)

    @staticmethod
    def _index(ranges: Sequence[_IntLike], prefix: SymT = SymT.INDEX) -> Sequence[Expr]:
        return [
            sympy.S.Zero if s == 1 else sympy_index_symbol_with_prefix(prefix, n)
            for n, s in enumerate(ranges)
        ]

    @cache_on_self
    def inner_fn_opcount(self) -> OpCountResult:
        opcounter = OpCounterCSE(V.MockHandler())
        with (
            V.set_ops_handler(opcounter),
            patch.object(FlexibleLayout, "allow_indexing", True),
        ):
            self.inner_fn(*self.inner_fn_args())
            return opcounter.getvalue()

    def inner_fn_args(self) -> Sequence[Sequence[_IntLike]]:
        return (self._index(self.ranges),)

    @cache_on_self
    def inner_fn_str(self) -> str:
        return V.KernelFormatterHandler.ir_to_string(
            self.inner_fn, *self.inner_fn_args()
        )

    def has_large_inner_fn(self, threshold: int | None = None) -> bool:
        if threshold is None:
            threshold = 0
        threshold = max(threshold, config.realize_opcount_threshold)
        return self.inner_fn_opcount().num_ops > threshold

    def inner_fn_free_symbols(self, unbacked_only: bool = False) -> OrderedSet[Symbol]:
        index = self._index(self.ranges)
        return extract_free_symbols(self.inner_fn, index, unbacked_only=unbacked_only)

    # returns a list of op names which references the symbol
    def collect_inner_fn_symbol_usage(self, symbol: Symbol) -> OrderedSet[str]:
        index = self._index(self.ranges)
        handler = SymbolUsageCollectorOpsHandler(symbol)
        with (
            V.set_ops_handler(handler),
        ):
            self.inner_fn(index)
        return handler.usages

    def get_reads(self) -> OrderedSet[Dep]:
        with patch.object(FlexibleLayout, "allow_indexing", True):
            if self.get_reduction_type():
                return extract_read_writes(
                    self.make_loader(),
                    self.get_size(),
                    self.get_reduction_size(),
                ).reads
            else:
                return extract_read_writes(
                    self.make_loader(),
                    self.get_size(),
                ).reads

    def get_read_names(self) -> OrderedSet[str]:
        return OrderedSet(self.inner_fn_opcount().read_buffers)

    def num_reads(self) -> int:
        return len(self.inner_fn_opcount().read_buffers)

    def get_reduction_size(self) -> Sequence[Expr]:
        raise NotImplementedError(
            f"get_reduction_size() is not implemented by {type(self)}!"
        )

    def get_reduction_type(self) -> str | None:
        raise NotImplementedError(
            f"get_reduction_type() is not implemented by {type(self)}!"
        )

    def constant_to_device(self, device: torch.device) -> IRNode:
        raise NotImplementedError(
            f"constant_to_device() is not implemented by {type(self)}!"
        )


def nop_loader_fn(idx: Expr | Sequence[Expr], *, dtype: torch.dtype) -> OpsValue:
    if dtype.is_floating_point:
        return ops.constant(float("nan"), dtype)
    else:
        return ops.constant(0, dtype)


@ir_dataclass
class Pointwise(Loops):
    def make_loader(self) -> Callable[[Sequence[Expr]], OpsValue]:
        # Make zero-element loops into a no-op
        if self.is_zero_elements():
            return partial(nop_loader_fn, dtype=self.dtype)

        return self.inner_fn

    def __str__(self) -> str:
        return self._to_str(("ranges",))

    __repr__ = __str__

    def get_reduction_size(self) -> Sequence[sympy.Expr]:
        return []

    def get_reduction_type(self) -> str | None:
        return None

    def store_output(
        self,
        output_name: str | None,
        indexer: Callable[[Sequence[Expr]], Never],
        vars: Sequence[Expr],
    ) -> None:
        loader = self.make_loader()
        return ops.store(output_name or "unnamed", indexer(vars), loader(vars))

    def constant_to_device(self, device: torch.device) -> IRNode:
        """Move this to a given device. Requires that all reads are to constants."""
        loader = self.make_loader()
        loader = patch.object(ConstantBuffer, "override_device", device)(loader)
        return Pointwise(
            device=device,
            dtype=self.dtype,
            inner_fn=loader,
            ranges=self.ranges,
        )


@ir_dataclass
class Scatter(Pointwise):
    output_indexer: Callable[[Sequence[Expr]], Expr]
    scatter_mode: StoreMode = None

    def constant_to_device(self, device: torch.device) -> IRNode:
        """Move this to a given device. Requires that all reads are to constants."""
        loader = self.make_loader()
        loader = patch.object(ConstantBuffer, "override_device", device)(loader)
        return Scatter(
            device=device,
            dtype=self.dtype,
            inner_fn=loader,
            ranges=self.ranges,
            output_indexer=self.output_indexer,
            scatter_mode=self.scatter_mode,
        )

    def store_output(
        self,
        output_name: str | None,
        indexer: Callable[[Sequence[Expr]], Never],
        vars: Sequence[Expr],
    ) -> Any:
        loader = self.make_loader()
        if output_name is None:
            output_name = "unnamed"
        return ops.store(
            output_name,
            indexer(self.output_indexer(vars)),
            loader(vars),
            mode=self.scatter_mode,
        )


REDUCTION_COMBINE_FN: dict[str, Callable[..., OpsValue]] = {
    "any": ops_wrapper("logical_or"),
    "max": ops_wrapper("maximum"),
    "min": ops_wrapper("minimum"),
    "prod": ops_wrapper("mul"),
    "sum": ops_wrapper("add"),
    "dot": ops_wrapper("add"),
    "xor_sum": ops_wrapper("bitwise_xor"),
}


def get_reduction_combine_fn(
    reduction_type: str, dtype: torch.dtype, arg_break_ties_left: bool = True
) -> Callable[..., object]:
    if reduction_type in REDUCTION_COMBINE_FN:
        return REDUCTION_COMBINE_FN[reduction_type]

    elif reduction_type in ("argmax", "argmin"):

        def argmax_combine_fn(
            a: tuple[object, object], b: tuple[object, object]
        ) -> tuple[OpsValue, OpsValue]:
            a_value, a_index = a
            b_value, b_index = b

            if reduction_type == "argmin":
                mask = ops.lt(a_value, b_value)
            else:
                mask = ops.gt(a_value, b_value)

            equal = ops.eq(a_value, b_value)
            if is_float_dtype(dtype):
                a_isnan = ops.ne(a_value, a_value)
                b_isnan = ops.ne(b_value, b_value)
                mask = ops.logical_or(mask, ops.gt(a_isnan, b_isnan))
                equal = ops.logical_or(equal, ops.logical_and(a_isnan, b_isnan))

            tie = (
                ops.lt(a_index, b_index)
                if arg_break_ties_left
                else ops.gt(a_index, b_index)
            )
            mask = ops.logical_or(mask, ops.logical_and(equal, tie))
            return (
                ops.where(mask, a_value, b_value),
                ops.where(mask, a_index, b_index),
            )

        return argmax_combine_fn

    elif reduction_type == "welford_combine":

        def welford_combine_fn(
            a: tuple[OpsValue, OpsValue, OpsValue],
            b: tuple[OpsValue, OpsValue, OpsValue],
        ) -> tuple[OpsValue, OpsValue, OpsValue]:
            a_mean, a_m2, a_weight = a
            b_mean, b_m2, b_weight = b

            delta = b_mean - a_mean
            new_weight = a_weight + b_weight
            w2_over_w = b_weight / new_weight
            return (
                a_mean + delta * w2_over_w,
                a_m2 + b_m2 + delta * delta * a_weight * w2_over_w,
                new_weight,
            )

        return welford_combine_fn

    else:
        raise NotImplementedError(f"unknown reduction_type={reduction_type}")


@ir_dataclass
class Reduction(Loops):
    reduction_ranges: Sequence[_IntLike]
    reduction_type: ReductionType
    # self.dtype represents the dst dtype
    src_dtype: torch.dtype
    reduction_hint: ReductionHint

    def __str__(self) -> str:
        return self._to_str(("ranges", "reduction_ranges", "reduction_type"))

    __repr__ = __str__

    @cache_on_self_and_args("Reduction")
    def get_free_symbol_uses(self, unbacked_only: bool = False) -> OrderedSet[Symbol]:
        return super().get_free_symbol_uses(unbacked_only) | OrderedSet().union(
            *(get_free_symbols(e, unbacked_only) for e in self.reduction_ranges)
        )

    def get_reduction_size(self) -> Sequence[Expr]:
        return self.reduction_ranges

    def get_reduction_type(self) -> str | None:
        return self.reduction_type

    def store_reduction(
        self,
        output_name: str | None,
        indexer: Callable[[Sequence[Expr]], Never],
        vars: Sequence[Expr],
        reduction_vars: Sequence[Symbol],
    ) -> None:
        value = ops.reduction(
            self.dtype,
            self.src_dtype,
            self.reduction_type,
            self.inner_fn(vars, reduction_vars),
        )
        ops.store_reduction(output_name or "unnamed", indexer(vars), value)

    def index_length(self) -> int:
        return len(self.ranges) + len(self.reduction_ranges)

    def inner_fn_args(self) -> Sequence[Sequence[Expr]]:
        index = self._index(self.ranges)
        rindex = self._index(self.reduction_ranges, SymT.R0_INDEX)
        return (index, rindex)

    def inner_fn_free_symbols(self, unbacked_only: bool = False) -> OrderedSet[Symbol]:
        index = self._index(self.ranges)
        rindex = self._index(self.reduction_ranges, SymT.R0_INDEX)
        return extract_free_symbols(
            self.inner_fn, index, rindex, unbacked_only=unbacked_only
        )

    def constant_to_device(self, device: torch.device) -> IRNode:
        """Move this to a given device. Requires that all reads are to constants."""
        loader = self.make_loader()
        loader = patch.object(ConstantBuffer, "override_device", device)(loader)
        return Reduction(
            device=device,
            dtype=self.dtype,
            inner_fn=loader,
            ranges=self.ranges,
            reduction_ranges=self.reduction_ranges,
            reduction_type=self.reduction_type,
            src_dtype=self.src_dtype,
            reduction_hint=ReductionHint.DEFAULT,
        )

    @staticmethod
    def num_splits(
        device: torch.device,
        dst_dtype: torch.dtype,
        src_dtype: torch.dtype,
        inner_fn: Callable[_P, OpsValue],
        ranges: Sequence[_IntLike],
        reduction_ranges: Sequence[_IntLike],
        reduction_type: ReductionType | Literal["scan"],
        reduction_numel: Expr,
        input_node: IRNode | None = None,
    ) -> tuple[ReductionHint, _IntLike]:
        # Use optimization_hint when all unbacked symbols have explicit hints,
        # otherwise fall back conservatively.
        exprs = [reduction_numel, sympy_product(ranges)]
        if not V.graph.sizevars.all_unbacked_explicitly_hinted(exprs):
            return ReductionHint.DEFAULT, 1
        reduction_numel_hint = V.graph.sizevars.optimization_hint(reduction_numel)
        numel_hint = V.graph.sizevars.optimization_hint(sympy_product(ranges))

        should_split = reduction_type == "scan" or (
            not V.graph.has_feature(device, BackendFeature.REDUCE_TO_SINGLE_ELEMENT)
            and reduction_type
            not in (
                "argmax",
                "argmin",
            )
            and config.split_reductions
        )

        if reduction_type == "dot":
            # Don't split when doing native matmul
            return ReductionHint.DEFAULT, 1

        props = DeviceProperties.create(device)
        num_sm = props.multi_processor_count
        min_elements_per_thread = 32
        if should_split:
            inner_reduction_splits: Callable[[int, int], int] = functools.partial(
                V.choices.reduction_split_factor, device, inner_reduction=True
            )
            outer_reduction_splits: Callable[[int, int], int] = functools.partial(
                V.choices.reduction_split_factor, device, inner_reduction=False
            )
        else:

            def inner_reduction_splits(
                reduction_numel_hint: int,
                numel_hint: int,
            ) -> int:
                return 1

            outer_reduction_splits = inner_reduction_splits

        # easy cases
        if numel_hint == 1:
            split = inner_reduction_splits(reduction_numel_hint, numel_hint)
            if split == 1:
                # No need to split.
                return ReductionHint.INNER, split
            if input_node is not None and isinstance(input_node, TensorBox):
                with patch.object(FlexibleLayout, "allow_indexing", True):
                    (
                        new_ranges,
                        new_reduction_ranges,
                    ) = extract_input_node_reduction_ranges(input_node)
                if new_ranges is not None and new_reduction_ranges is not None:
                    extracted_numel_hint = (
                        V.graph.sizevars.replace_backed_symbols_with_hints(
                            sympy_product(new_ranges + new_reduction_ranges)
                        )
                    )
                    if reduction_numel_hint == extracted_numel_hint:
                        log.debug(
                            "Use previous IRNode's range and reduction_ranges instead of split. "
                            "current ranges: %s, current reduction ranges: %s, current split: %d, "
                            "new ranges: %s, new reduction ranges: %s",
                            ranges,
                            reduction_ranges,
                            split,
                            new_ranges,
                            new_reduction_ranges,
                        )
                        # If the input_node or its dependent nodes are also Reduction nodes,
                        # use reduction_sizes of this node or its dependent nodes directly.
                        return ReductionHint.INNER, -1
            return ReductionHint.INNER, split
        if (
            reduction_numel_hint <= min_elements_per_thread
            or numel_hint >= num_sm * 2 * 32
        ):
            return ReductionHint.DEFAULT, 1

        r = Reduction(
            device=device,
            dtype=dst_dtype,
            inner_fn=inner_fn,
            ranges=ranges,
            reduction_ranges=reduction_ranges,
            reduction_type=reduction_type if reduction_type != "scan" else "sum",
            src_dtype=src_dtype,
            reduction_hint=ReductionHint.DEFAULT,
        )

        def get_read_indices(r: Reduction) -> tuple[Sequence[Expr], bool]:
            device = r.get_device()
            assert device is not None
            cb = ComputedBuffer(
                name=None,
                layout=FlexibleLayout(
                    device=device,
                    dtype=r.get_dtype(),
                    size=r.get_size(),
                ),
                data=r,
            )
            read_writes = cb.get_read_writes()
            # try finding the full size producer
            # TODO this will fail for something like ((1, N) * (N, 1)).sum()
            # this would also possibly be wrong for producers with the different contiguity but we hope those cases are rare
            assert read_writes.range_vars is not None
            range_vars = [
                r
                for r in read_writes.range_vars
                if isinstance(r, Expr) and not isinstance(r, sympy.Number)
            ]
            indices = []
            changed = False
            for md in sorted(read_writes.reads, key=lambda x: x.name):
                if all(r in md.index.free_symbols for r in range_vars):
                    indices.append(md.index)
                    if md.name in V.graph.name_to_buffer:
                        buf = V.graph.name_to_buffer[md.name]
                        original_stride = getattr(buf.layout, "stride", None)
                        buf.decide_layout()
                        if getattr(buf.layout, "stride", None) != original_stride:
                            changed = True
            return indices, changed

        indices, changed = get_read_indices(r)
        if changed:
            indices, _ = get_read_indices(r)

        if len(indices) == 0:
            # TODO determine splits when all inputs are broadcast
            return ReductionHint.DEFAULT, 1

        (_, reduction_vars), ranges1 = dependencies.index_vars_squeeze(
            r.get_size(), r.get_reduction_size()
        )
        num_outer = 0
        num_inner = 0
        for i in indices:
            j = V.graph.sizevars.simplify_with_ranges(i, ranges1)
            strides = V.graph.sizevars.stride_hints(
                j, reduction_vars, list(ranges1.keys())
            )
            # A 0 stride does not make a reduction contiguous.
            # This can happen when the reduction ranges contains a 1.
            outer = all(s == 0 or s > 1 for s in strides)
            if outer:
                num_outer += 1
            else:
                num_inner += 1
        if num_inner > num_outer:
            return ReductionHint.INNER, inner_reduction_splits(
                reduction_numel_hint, numel_hint
            )
        else:
            return ReductionHint.OUTER, outer_reduction_splits(
                reduction_numel_hint, numel_hint
            )

    @staticmethod
    def _unroll_reduction_fn(
        inner_fn: Callable[[Sequence[_IntLike], Sequence[_IntLike]], OpsValue],
        reduction_ranges: Sequence[_IntLike],
        reduction_type: str,
        src_dtype: torch.dtype,
    ) -> Callable[[Sequence[_IntLike]], OpsValue]:
        """Convert inner_fn from a reduction to an pointwise"""
        reduction_ranges = V.graph.sizevars.guard_int_seq(reduction_ranges)

        combine_fn = get_reduction_combine_fn(reduction_type, src_dtype)

        def fn(index: Sequence[_IntLike]) -> Any:
            return functools.reduce(
                combine_fn,
                (
                    value_fn(index, rindex)
                    for rindex in itertools.product(
                        *[range(x) for x in reduction_ranges]
                    )
                ),
            )

        value_fn: Callable[[Sequence[_IntLike], Sequence[_IntLike]], Any]
        if reduction_type in ("argmin", "argmax"):
            flatten_index = _fixed_indexer(
                reduction_ranges,
                FlexibleLayout.contiguous_strides(reduction_ranges),
            )

            def value_fn(
                index: Sequence[_IntLike], rindex: Sequence[_IntLike]
            ) -> tuple[OpsValue, OpsValue]:
                rindex = [sympy.expand(i) for i in rindex]
                return (
                    inner_fn(index, rindex),
                    ops.index_expr(flatten_index(rindex), torch.int64),
                )

            return lambda index: fn(index)[1]
        else:
            value_fn = inner_fn
            return fn

    @classmethod
    # pyrefly: ignore [bad-override]
    def create(
        cls,
        device: torch.device,
        dst_dtype: torch.dtype,
        src_dtype: torch.dtype,
        inner_fn: Callable[..., Any],
        ranges: Sequence[Expr],
        reduction_ranges: Sequence[Expr],
        reduction_type: ReductionType,
        reduction_hint: ReductionHint = ReductionHint.DEFAULT,
        input_node: IRNode | None = None,
    ) -> TensorBox:
        """
        Create a reduction node. May split the reduction to multiple layers to expose
        more parallelism.
        """
        reduction_numel = V.graph.sizevars.simplify(sympy_product(reduction_ranges))

        if reduction_numel == 0:
            # N.B. This is a hack to generate the literal of the given type
            # Ideally, we should be fixing `def constant` in triton.py
            # but it breaks due to hardcoded dtypes in other places
            def py_cnst(val: object) -> bool | float | int:
                if dst_dtype == torch.bool:
                    return bool(val)
                elif dst_dtype.is_floating_point:
                    assert isinstance(val, SupportsFloat), type(val)
                    return float(val)
                else:
                    assert isinstance(val, SupportsInt), type(val)
                    return int(val)

            rtypes_to_inits = {
                "sum": py_cnst(0),
                "xor_sum": py_cnst(0),
                "prod": py_cnst(1),
                "any": py_cnst(0),
                # "all" is desugared to `!any(!val)`
            }

            assert reduction_type in rtypes_to_inits, (
                f"{reduction_type} not supported for zero-dimension tensors!"
            )

            def const_fn(index: int) -> OpsValue:
                return ops.constant(rtypes_to_inits[reduction_type], dst_dtype)

            return Pointwise.create(
                device=device,
                dtype=src_dtype,
                inner_fn=const_fn,
                ranges=list(ranges),
            )

        if reduction_numel == 1:
            # this reduction is actually a pointwise op
            if reduction_type in ("argmin", "argmax"):

                def fn(index: int) -> OpsValue:
                    return ops.constant(0, dst_dtype)

            else:

                def fn(index: int) -> OpsValue:
                    reduction_index = [sympy.S.Zero for _ in reduction_ranges]
                    return inner_fn(index, reduction_index)

            return Pointwise.create(
                device=device, dtype=dst_dtype, inner_fn=fn, ranges=ranges
            )

        if (
            isinstance(reduction_numel, Integer)
            and int(reduction_numel) < config.unroll_reductions_threshold
            and (sympy_product(ranges) != 1 or is_gpu(device.type))
            and reduction_type != "dot"
        ):
            # When native matmul, don't unroll the dot reduction.

            # NB: This works around https://github.com/pytorch/pytorch/issues/140457
            # since turning reductions into pointwise ops can exacerbate this problem
            return Pointwise.create(
                device=device,
                dtype=dst_dtype,
                inner_fn=cls._unroll_reduction_fn(
                    inner_fn, reduction_ranges, reduction_type, src_dtype
                ),
                ranges=ranges,
            )

        # triton doesn't support reduce to single element well, so break it up
        hint, split = cls.num_splits(
            device,
            dst_dtype,
            src_dtype,
            inner_fn,
            ranges,
            reduction_ranges,
            reduction_type,
            reduction_numel,
            input_node,
        )

        def _maybe_increase_split(split: int) -> int:
            # don't apply min_num_split constraint for static shape case.
            if _is_static(reduction_numel):
                return split
            if split > 1:
                return max(split, config.min_num_split)
            else:
                return split

        split = _maybe_increase_split(split)

        # intermediate reduction in split can contain complex indexing,
        # and num_splits will fail to correctly set the hint
        # reuse the passed hint if available
        if reduction_hint == ReductionHint.DEFAULT:
            reduction_hint = hint
        if split == -1:
            assert input_node is not None
            with patch.object(FlexibleLayout, "allow_indexing", True):
                new_ranges, new_reduction_ranges = extract_input_node_reduction_ranges(
                    input_node
                )
            assert new_ranges is not None
            assert new_reduction_ranges is not None
            return cls.create_multilayer_existing_ranges(
                device,
                dst_dtype,
                src_dtype,
                inner_fn,
                ranges,
                reduction_ranges,
                new_ranges,
                new_reduction_ranges,
                reduction_type,
                reduction_hint,
            )
        elif split > 1:
            # triton doesn't support reduce to single element well, so break it up
            out = cls.create_multilayer(
                device,
                dst_dtype,
                src_dtype,
                inner_fn,
                ranges,
                reduction_ranges,
                reduction_type,
                split,
                reduction_hint,
                input_node,
            )

            # Find the reduction that get split
            split_reduction = None
            if config.triton.mix_order_reduction and isinstance(out, TensorBox):

                def _find_split_reduction(
                    cur_node: TensorBox,
                ) -> ComputedBuffer | None:
                    read_names = cur_node.get_read_names()
                    if len(read_names) != 1:
                        return None

                    bufname = next(iter(read_names))
                    if bufname not in V.graph.name_to_buffer:
                        return None
                    buf = V.graph.name_to_buffer[bufname]
                    if not isinstance(buf, ComputedBuffer):
                        return None

                    assert buf.data.get_reduction_type() is not None

                    return buf

                split_reduction = _find_split_reduction(out)

            if split_reduction:
                # If a reduction is split to more than 2 layers,
                # say there are 3 layers,
                # we always have the correct setting for layer1 (top layer).
                # The setting on layer2 may be incorrect but it's fine
                # since they are never get used.
                # TODO: should we skip setting these fields for layer2
                assert isinstance(split_reduction.data, Reduction), (
                    f"{type(split_reduction.data)}"
                )
                split_reduction._split_size = split_reduction.data.reduction_ranges[0]
                split_reduction._original_inner_fn = inner_fn
                split_reduction._original_ranges = ranges
                split_reduction._original_reduction_ranges = reduction_ranges
            return out

        out = TensorBox.create(
            Reduction(
                device=device,
                dtype=dst_dtype,
                inner_fn=inner_fn,
                ranges=ranges,
                reduction_ranges=reduction_ranges,
                reduction_type=reduction_type,
                src_dtype=src_dtype,
                reduction_hint=reduction_hint,
            )
        )
        return out

    @staticmethod
    def default_accumulator(
        reduction_type: str, dtype: torch.dtype
    ) -> _NumLike | Sequence[_NumLike]:
        if reduction_type in ("max", "argmax"):
            if is_float_dtype(dtype):
                return float("-inf")
            elif is_boolean_dtype(dtype):
                return False
            else:
                return torch.iinfo(dtype).min
        if reduction_type in ("min", "argmin"):
            if is_float_dtype(dtype):
                return float("inf")
            elif is_boolean_dtype(dtype):
                return True
            else:
                return torch.iinfo(dtype).max

        zero = False if is_boolean_dtype(dtype) else 0
        one = True if is_boolean_dtype(dtype) else 1
        return {
            "sum": zero,
            "prod": one,
            "dot": zero,
            "xor_sum": zero,
            "any": zero,
            "welford_reduce": (zero, zero, zero),
            "welford_combine": (zero, zero, zero),
            "online_softmax_reduce": (float("-inf"), zero),
        }[reduction_type]

    @staticmethod
    def default_value(
        reduction_type: str, dtype: torch.dtype
    ) -> _NumLike | Sequence[_NumLike]:
        if reduction_type == "welford_reduce":
            return 0
        return Reduction.default_accumulator(reduction_type, dtype)

    @staticmethod
    def _multilayer_second_step_hint(
        split: _IntLike, numel_hint: int, reduction_hint: ReductionHint
    ) -> ReductionHint:
        if split == -1:
            return reduction_hint
        if split <= 512 and numel_hint <= 512 and reduction_hint == ReductionHint.OUTER:
            return ReductionHint.OUTER_TINY
        if (
            split <= 1024
            and numel_hint <= 256
            and reduction_hint == ReductionHint.OUTER
        ):
            return ReductionHint.OUTER_TINY

        return reduction_hint

    @classmethod
    def check_for_split_dense_dim_reindexing(
        cls, reduction_numel: _IntLike, input_node: IRNode | None
    ) -> int | None:
        """
        If we are reducing over the full tensor, and it is non-dense in the last dimension,
        reindex so we reduce over the dense dimension. initially just handle complete
        reduction case
        """
        if input_node is None:
            return None

        if not V.graph.sizevars.statically_known_equals(
            input_node.get_numel(), reduction_numel
        ):
            return None

        input_node.realize()
        try:
            # finalize layout
            as_storage_and_layout(input_node)
        except NotImplementedError:
            return None

        strides = input_node.get_stride()

        for i, s in enumerate(strides[:-1]):
            if V.graph.sizevars.statically_known_equals(s, 1):
                return i

        return None

    @classmethod
    def _multilayer_wrap_loader(
        cls,
        loader: Callable[..., OpsValue],
        reduction_ranges: Sequence[_IntLike],
        reduction_numel: _IntLike,
        split: _IntLike,
        block_size: _IntLike,
        default: _NumLike | Sequence[_NumLike],
        input_node: IRNode | None = None,
    ) -> Callable[..., object]:
        dense_index = cls.check_for_split_dense_dim_reindexing(
            reduction_numel, input_node
        )
        reindex = View.dynamic_reshape_indexer(
            reduction_ranges, [reduction_numel], dense_index
        )
        need_mask = not V.graph.sizevars.statically_known_true(
            sympy.Eq(Mod(reduction_numel, split), 0)
        )

        def wrapper_fn(
            index: Sequence[Symbol], reduction_index: Sequence[Symbol]
        ) -> OpsValue:
            (reduction_index,) = reduction_index
            *new_index, reduction_block = index
            indices = block_size * reduction_block + reduction_index

            def body() -> OpsValue:
                return loader(new_index, reindex([indices]))

            if need_mask:
                index_dtype = dtype_from_size(reduction_numel)
                mask = ops.lt(
                    ops.index_expr(indices, index_dtype),
                    ops.index_expr(reduction_numel, index_dtype),
                )
                return ops.masked(mask, body, default)
            else:
                return body()

        return wrapper_fn

    @classmethod
    def _multilayer_wrap_loader_existing_ranges(
        cls,
        loader: Callable[[Sequence[Expr], Sequence[Expr]], OpsValue],
        original_ranges: Sequence[Expr],
        original_reduction_ranges: Sequence[Expr],
        new_ranges: Sequence[Integer],
        new_reduction_ranges: Sequence[Integer],
    ) -> Callable[[Sequence[sympy.Expr], Sequence[sympy.Expr]], OpsValue]:
        assert all(r == 1 for r in original_ranges), (
            f"Only enabled for numel_hint == 1, found {original_ranges=}"
        )
        reindex = View.dynamic_reshape_indexer(
            original_reduction_ranges, tuple(new_ranges) + tuple(new_reduction_ranges)
        )

        def wrapper_fn(
            merged_index: Sequence[Expr],
            new_reduction_index: Sequence[Expr],
        ) -> OpsValue:
            original_idx = merged_index[: len(original_ranges)]
            new_index = merged_index[len(original_ranges) :]
            return loader(
                original_idx,
                reindex(tuple(new_index) + tuple(new_reduction_index)),
            )

        return wrapper_fn

    @classmethod
    def create_multilayer_helper(
        cls,
        device: torch.device,
        dst_dtype: torch.dtype,
        src_dtype: torch.dtype,
        wrapper_fn: Callable[..., Any],
        original_ranges: Sequence[Expr],
        original_reduction_ranges: Sequence[Expr],
        new_ranges: list[Expr],
        new_reduction_ranges: list[Integer],
        reduction_type: ReductionType,
        split: _IntLike,
        reduction_hint: ReductionHint,
    ) -> TensorBox:
        """
        Break a large reduction up into multiple smaller reductions
        recursively
        """
        # triton will automatically compute reductions in fp32 if reducing over fp16/bf16
        # within the kernel. keep the intermediate in fp32 so as to keep the whole reduction
        # in fp32 and not reduce precision by breaking up the kernel into multiple layers
        intermediate_dtype = (
            dst_dtype
            if dst_dtype not in (torch.float16, torch.bfloat16)
            else torch.float
        )
        intermediate = Reduction.create(
            device,
            intermediate_dtype,
            src_dtype,
            wrapper_fn,
            new_ranges,
            new_reduction_ranges,
            reduction_type,
            reduction_hint,
        )
        intermediate.realize()
        intermediate_loader = intermediate.make_loader()

        def intermediate_fn(
            index: Sequence[_IntLike], reduction_index: Sequence[_IntLike]
        ) -> OpsValue:
            return intermediate_loader([*index, *reduction_index])

        numel_hint = V.graph.sizevars.optimization_hint(sympy_product(original_ranges))
        reduction_hint = cls._multilayer_second_step_hint(
            split, numel_hint, reduction_hint
        )

        assert original_ranges == new_ranges[: len(original_ranges)]
        return TensorBox.create(
            Reduction(
                device=device,
                dtype=dst_dtype,
                inner_fn=intermediate_fn,
                ranges=original_ranges,
                reduction_ranges=new_ranges[len(original_ranges) :],
                reduction_type=reduction_type,
                src_dtype=src_dtype,
                reduction_hint=reduction_hint,
            )
        )

    @classmethod
    def create_multilayer(
        cls,
        device: torch.device,
        dst_dtype: torch.dtype,
        src_dtype: torch.dtype,
        inner_fn: Callable[..., Any],
        ranges: Sequence[Expr],
        reduction_ranges: Sequence[Expr],
        reduction_type: ReductionType,
        split: _IntLike,
        reduction_hint: ReductionHint,
        input_node: IRNode | None = None,
    ) -> TensorBox:
        """
        Break a large reduction up into multiple smaller reductions
        recursively
        """
        # TODO(jansel): realize the reduction so we can do dynamic indexing
        reduction_numel = sympy_product(reduction_ranges)
        block_size = FloorDiv(reduction_numel + (split - 1), split)
        default = cls.default_value(reduction_type, dst_dtype)
        wrapper_fn = cls._multilayer_wrap_loader(
            inner_fn,
            reduction_ranges,
            reduction_numel,
            split,
            block_size,
            default,
            input_node,
        )

        return cls.create_multilayer_helper(
            device,
            dst_dtype,
            src_dtype,
            wrapper_fn,
            ranges,
            reduction_ranges,
            [*ranges, split],
            [block_size],
            reduction_type,
            split,
            reduction_hint,
        )

    @classmethod
    def create_multilayer_existing_ranges(
        cls,
        device: torch.device,
        dst_dtype: torch.dtype,
        src_dtype: torch.dtype,
        inner_fn: Callable[..., Any],
        original_ranges: Sequence[Expr],
        original_reduction_ranges: Sequence[Expr],
        new_ranges: list[Integer],
        new_reduction_ranges: list[Integer],
        reduction_type: ReductionType,
        reduction_hint: ReductionHint,
    ) -> TensorBox:
        """
        Break a large reduction up into multiple smaller reductions
        recursively
        """
        wrapper_fn = cls._multilayer_wrap_loader_existing_ranges(
            inner_fn,
            original_ranges,
            original_reduction_ranges,
            new_ranges,
            new_reduction_ranges,
        )
        return cls.create_multilayer_helper(
            device,
            dst_dtype,
            src_dtype,
            wrapper_fn,
            original_ranges,
            original_reduction_ranges,
            [*original_ranges, *new_ranges],
            new_reduction_ranges,
            reduction_type,
            -1,
            reduction_hint,
        )


def _fixed_indexer(
    size: Sequence[int],
    stride: Sequence[int] | None = None,
    offset: Expr = Integer(0),
) -> Callable[[Sequence[Expr]], Expr]:
    """A closure containing math to read a given element"""

    def indexer(index: Sequence[int]) -> int:
        assert stride is not None and len(index) == len(stride)
        assert len(index) == len(size)
        result = offset
        for idx, st, sz in zip(index, stride, size):
            if sz != 1:
                result = result + idx * st
        return result

    return indexer


INNER_FN_TY: TypeAlias = Callable[[Sequence[Expr], Sequence[Expr]], OpsValue]


class MultiOutputReduction(Reduction):
    output_index: int

    def __init__(
        self,
        device: torch.device,
        dst_dtype: torch.dtype,
        inner_fns: INNER_FN_TY | Sequence[INNER_FN_TY],
        ranges: Sequence[Integer],
        reduction_ranges: Sequence[Integer],
        reduction_type: ReductionType,
        src_dtype: torch.dtype,
        reduction_hint: ReductionHint,
        output_index: int,
    ):
        if callable(inner_fns):
            inner_fns = (inner_fns,)

        loader: Callable[[Sequence[Expr], Sequence[Expr]], Any]
        if len(inner_fns) == 1:
            loader = inner_fns[0]
        else:

            def loader(
                idx: Sequence[Expr], reduction_idx: Sequence[Expr]
            ) -> tuple[OpsValue, ...]:
                return tuple(fn(idx, reduction_idx) for fn in inner_fns)

        super().__init__(
            device=device,
            dtype=dst_dtype,
            inner_fn=loader,
            ranges=ranges,
            reduction_ranges=reduction_ranges,
            reduction_type=reduction_type,
            src_dtype=src_dtype,
            reduction_hint=reduction_hint,
        )
        self.output_index = output_index

    def store_reduction(
        self,
        output_name: str | None,
        indexer: Callable[[Sequence[Expr]], Never],
        vars: Sequence[Expr],
        reduction_vars: Sequence[Symbol],
    ) -> Any:
        values = ops.reduction(
            self.dtype,
            self.src_dtype,
            self.reduction_type,
            self.inner_fn(vars, reduction_vars),
        )
        assert isinstance(values, (tuple, list)), type(values)
        value = values[self.output_index]
        return ops.store_reduction(output_name or "unnamed", indexer(vars), value)


class OnlineSoftmaxReduction(MultiOutputReduction):
    @classmethod
    def create(  # type: ignore[override]
        cls,
        device: torch.device,
        dst_dtype: torch.dtype,
        src_dtype: torch.dtype,
        inner_fn: Callable[..., Any],
        ranges: Sequence[Expr],
        reduction_ranges: Sequence[Expr],
        num_output: int,
        reduction_hint: ReductionHint = ReductionHint.DEFAULT,
        input_node: IRNode | None = None,
    ) -> Sequence[TensorBox]:
        """
        Create the reduction disregarding splitting.
        """
        results = tuple(
            TensorBox.create(
                MultiOutputReduction(
                    device,
                    dst_dtype,
                    inner_fn,
                    ranges,
                    reduction_ranges,
                    "online_softmax_reduce",
                    src_dtype,
                    reduction_hint,
                    output_idx,
                )
            )
            for output_idx in range(num_output)
        )
        for t in results:
            t.realize()
        return results


class WelfordReduction(MultiOutputReduction):
    @classmethod
    def create(  # type: ignore[override]
        cls,
        device: torch.device,
        dtype: torch.dtype,
        inner_fns: Sequence[Callable[..., Any]],
        ranges: list[Integer],
        reduction_ranges: list[Integer],
        reduction_type: ReductionType,
        reduction_hint: ReductionHint = ReductionHint.DEFAULT,
    ) -> Sequence[TensorBox]:
        assert reduction_type in ("welford_reduce", "welford_combine")
        assert not config.mtia.disable_welford_reduction, (
            "welford reduction usage is explicitly disabled, please check you config"
        )

        reduction_numel = V.graph.sizevars.simplify(sympy_product(reduction_ranges))

        def const(val: int) -> TensorBox:
            def inner_fn(idx: Sequence[Expr]) -> OpsValue:
                return ops.constant(
                    val,
                    dtype,
                )

            return Pointwise.create(
                device=device,
                dtype=dtype,
                inner_fn=inner_fn,
                ranges=list(ranges),
            )

        if reduction_numel == 0:
            mean = const(0)
            m2 = const(0)
            weight = const(0)
            return mean, m2, weight

        if reduction_numel == 1:

            def copy(
                loader: Callable[[Sequence[Expr], Sequence[Expr]], OpsValue],
            ) -> TensorBox:
                def inner_fn(idx: Sequence[Expr]) -> OpsValue:
                    reduction_index = [sympy.S.Zero for _ in reduction_ranges]
                    return loader(idx, reduction_index)

                return Pointwise.create(
                    device=device,
                    dtype=dtype,
                    inner_fn=inner_fn,
                    ranges=list(ranges),
                )

            if reduction_type == "welford_reduce":
                return copy(inner_fns[0]), const(0), const(1)
            else:
                return tuple(copy(fn) for fn in inner_fns)

        # TODO: Unrolled reduction
        # if (
        #     isinstance(reduction_numel, Integer)
        #     and int(reduction_numel)
        #     < config.unroll_reductions_threshold
        #     and sympy_product(ranges) != 1
        # ):
        #     return Pointwise.create(
        #         device,
        #         dst_dtype,
        #         cls._unroll_reduction_fn(
        #             inner_fn, reduction_ranges, reduction_type, src_dtype,
        #         ),
        #         ranges,
        #     )

        # triton doesn't support reduce to single element well, so break it up
        hint, split = Reduction.num_splits(
            device,
            dtype,
            dtype,
            inner_fns[0],
            ranges,
            reduction_ranges,
            reduction_type=reduction_type,
            reduction_numel=reduction_numel,
        )
        # intermediate reduction in split can contain complex indexing,
        # and num_splits will fail to correctly set the hint
        # reuse the passed hint if available
        if reduction_hint == ReductionHint.DEFAULT:
            reduction_hint = hint
        if split > 1:
            # triton doesn't support reduce to single element well, so break it up
            return cls.create_multilayer(
                device,
                dtype,
                inner_fns,
                ranges,
                reduction_ranges,
                reduction_type,
                split,
                reduction_hint,
            )

        results = [
            TensorBox.create(
                WelfordReduction(
                    device,
                    dtype,
                    inner_fns,
                    ranges,
                    reduction_ranges,
                    reduction_type,
                    dtype,
                    reduction_hint,
                    output_idx,
                )
            )
            for output_idx in range(3)
        ]
        for t in results:
            t.realize()
        return results

    @staticmethod
    def default_value(
        reduction_type: str, dtype: torch.dtype
    ) -> _NumLike | Sequence[_NumLike]:
        return (0, 0, 0)

    @classmethod
    def create_multilayer(  # type: ignore[override]
        cls,
        device: torch.device,
        dtype: torch.dtype,
        inner_fns: Sequence[Callable[..., Any]],
        ranges: list[Integer],
        reduction_ranges: list[Integer],
        reduction_type: ReductionType,
        split: _IntLike,
        reduction_hint: ReductionHint,
    ) -> Sequence[TensorBox]:
        """
        Break a large reduction up into multiple smaller reductions
        recursively
        """
        reduction_numel = sympy_product(reduction_ranges)
        need_mask = not V.graph.sizevars.statically_known_true(
            sympy.Eq(Mod(reduction_numel, split), 0)
        )

        if need_mask and reduction_type != "welford_combine":
            # If we need mask, then "welford_reduce" doesn't work because
            # masked inputs shouldn't count towards the welford weight

            def constant(
                idx: Sequence[Expr], reduction_idx: Sequence[Expr], value: int
            ) -> OpsValue:
                return ops.constant(value, dtype)

            return cls.create_multilayer(
                device=device,
                dtype=dtype,
                inner_fns=(
                    inner_fns[0],
                    partial(constant, value=0),
                    partial(constant, value=1),
                ),
                ranges=ranges,
                reduction_ranges=reduction_ranges,
                reduction_type="welford_combine",
                split=split,
                reduction_hint=reduction_hint,
            )

        block_size = FloorDiv(reduction_numel + (split - 1), split)
        intermediates = WelfordReduction.create(
            device,
            dtype,
            tuple(
                cls._multilayer_wrap_loader(
                    loader,
                    reduction_ranges,
                    reduction_numel,
                    split,
                    block_size,
                    default=0,
                )
                for loader in inner_fns
            ),
            [*ranges, split],
            [block_size],
            reduction_type,
            reduction_hint,
        )
        for i in intermediates:
            i.realize()

        def intermediate_loader_fn(
            index: Sequence[Expr],
            reduction_index: Sequence[Expr],
            loader: Callable[[Sequence[Expr]], OpsValue],
        ) -> OpsValue:
            return loader([*index, *reduction_index])

        # numel_hint is only used to choose between ReductionHint.OUTER vs
        # OUTER_TINY, which is a performance heuristic, not a correctness decision.
        numel_hint = V.graph.sizevars.optimization_hint(sympy_product(ranges))
        reduction_hint = cls._multilayer_second_step_hint(
            split, numel_hint, reduction_hint
        )
        return WelfordReduction.create(
            device,
            dtype,
            tuple(
                partial(intermediate_loader_fn, loader=i.make_loader())
                for i in intermediates
            ),
            ranges,
            [split],
            # welford_reduce turns one input into three outputs, which are combined with welford_combine
            "welford_combine",
            reduction_hint,
        )


@ir_dataclass
class Scan(Loops):
    scan_ranges: list[Integer]
    size: list[Integer]
    combine_fn: Callable[[tuple[Any, ...], tuple[Any, ...]], tuple[Any, ...]]
    reindex: Callable[[Sequence[_IntLike], Sequence[_IntLike]], Sequence[_IntLike]]
    reduction_hint: ReductionHint
    output_index: int
    # output_index indexes the following tuples
    dtypes: tuple[torch.dtype, ...]
    inner_fns: tuple[Callable[..., Any], ...]

    # HACK we mimic reduction

    @cache_on_self_and_args("Scan")
    def get_free_symbol_uses(self, unbacked_only: bool = False) -> OrderedSet[Symbol]:
        # TODO: Can combine_fn/reindex close over unbacked symbols? If so, we
        # need to explicitly represent the closure so we can pull out unbacked
        # symbols here
        return (
            super().get_free_symbol_uses(unbacked_only)
            | OrderedSet().union(
                *(get_free_symbols(e, unbacked_only) for e in self.scan_ranges)
            )
            | OrderedSet().union(
                *(get_free_symbols(e, unbacked_only) for e in self.size)
            )
        )

    def __post_init__(self) -> None:
        assert len(self.ranges) + len(self.scan_ranges) == len(self.size)
        super().__post_init__()

    def store_reduction(
        self,
        output_name: str | None,
        indexer: Callable[[Sequence[_IntLike]], Never],
        vars: Sequence[Expr],
        scan_vars: Sequence[Symbol],
    ) -> Any:
        idx = self.reindex(vars, scan_vars)
        values = tuple(inner_fn(idx) for inner_fn in self.inner_fns)
        result = ops.scan(self.dtypes, self.combine_fn, values)
        return ops.store(
            output_name or "unnamed", indexer(idx), result[self.output_index]
        )

    def get_reduction_type(self) -> str | None:
        # return self.scan_op
        return "custom"

    def get_reduction_size(self) -> Sequence[Expr]:
        return self.scan_ranges

    def get_size(self) -> Sequence[Expr]:
        return self.size

    def get_pointwise_size(self) -> Sequence[Expr]:
        return self.ranges

    def index_length(self) -> int:
        return len(self.ranges) + len(self.scan_ranges)

    def inner_fn_args(self) -> Sequence[Sequence[_IntLike]]:
        index = self._index(self.ranges)
        rindex = self._index(self.scan_ranges, SymT.R0_INDEX)
        idx = self.reindex(index, rindex)
        return (idx,)

    def inner_fn_free_symbols(self, unbacked_only: bool = False) -> OrderedSet[Symbol]:
        index = self._index(self.ranges)
        rindex = self._index(self.scan_ranges, SymT.R0_INDEX)
        idx = self.reindex(index, rindex)
        return extract_free_symbols(self.inner_fn, idx, unbacked_only=unbacked_only)

    @classmethod
    def create(  # type: ignore[override]
        cls,
        device: torch.device,
        dtypes: tuple[torch.dtype, ...],
        inner_fns: tuple[Callable[[Sequence[Expr]], Any], ...],
        size: list[Integer],
        axis: int,
        combine_fn: Callable[[tuple[Any, ...], tuple[Any, ...]], tuple[Any, ...]],
        reduction_hint: ReductionHint = ReductionHint.DEFAULT,
        *,
        # Whether we have the option to fallback to aten
        can_fallback_to_aten: bool = True,
        **kwargs: Any,
    ) -> Sequence[TensorBox | None]:
        pointwise_ranges = [*size[:axis], *size[axis + 1 :]]
        scan_ranges = [size[axis]]

        if not V.graph.has_feature(device, BackendFeature.SCAN):
            return [None] * len(dtypes)

        if len(dtypes) > 1 and not V.graph.has_feature(
            device, BackendFeature.TUPLE_REDUCTION
        ):
            return [None] * len(dtypes)

        sizevars = V.graph.sizevars
        scan_numel = sizevars.simplify(sympy_product(scan_ranges))

        assert len(dtypes) == len(inner_fns)

        # Scan with a single element is just a copy
        if sizevars.statically_known_true(sympy.Le(scan_numel, 1)):
            return [
                Pointwise.create(
                    device=device,
                    dtype=dtypes[output_index],
                    inner_fn=inner_fns[output_index],
                    ranges=size,
                )
                for output_index in range(len(dtypes))
            ]

        reduction_hint, num_splits = cls.num_splits(
            device=device,
            dtype=dtypes[0],
            inner_fn=inner_fns[0],
            axis=axis,
            pointwise_ranges=pointwise_ranges,
            scan_ranges=scan_ranges,
            combine_fn=combine_fn,
            scan_numel=scan_numel,
        )
        scan_type = Scan
        if num_splits > 1:
            supports_split = (
                # pyrefly: ignore [unsupported-operation]
                torch.version.hip is None or (has_triton and triton_version >= "3.3.0")
            ) and (len(dtypes) == 1)
            if not supports_split:
                if can_fallback_to_aten:
                    # Fallback to ATen
                    return [None] * len(dtypes)
                else:
                    num_splits = 1
            else:
                scan_type = SplitScan

        def reindex(index: Sequence[Expr], scan_index: Sequence[Expr]) -> list[Expr]:
            assert len(scan_index) == len(scan_ranges)
            assert len(index) == len(pointwise_ranges)
            return [*index[:axis], *scan_index, *index[axis:]]

        results = [
            TensorBox.create(
                scan_type(
                    device=device,
                    dtype=dtypes[output_index],
                    dtypes=dtypes,
                    inner_fn=inner_fns[output_index],
                    inner_fns=inner_fns,
                    size=size,
                    ranges=pointwise_ranges,
                    scan_ranges=scan_ranges,
                    combine_fn=combine_fn,
                    reindex=reindex,
                    reduction_hint=reduction_hint,
                    output_index=output_index,
                    **kwargs,
                )
            )
            for output_index in range(len(dtypes))
        ]

        for result in results:
            result.realize()

        return results

    @classmethod
    def num_splits(
        cls,
        device: torch.device,
        dtype: torch.dtype,
        inner_fn: Callable[[Sequence[Expr]], OpsValue],
        axis: int,
        pointwise_ranges: list[Integer],
        scan_ranges: list[Integer],
        combine_fn: Callable[[tuple[Any, ...], tuple[Any, ...]], tuple[Any, ...]],
        scan_numel: Expr,
    ) -> tuple[ReductionHint, _IntLike]:
        # TODO: custom splitting heuristic for scan
        def wrapper_fn(idx: Sequence[Expr], reduction_idx: Sequence[Expr]) -> OpsValue:
            return inner_fn([*idx[:axis], *reduction_idx, *idx[axis:]])

        return Reduction.num_splits(
            device=device,
            dst_dtype=dtype,
            src_dtype=dtype,
            inner_fn=wrapper_fn,
            ranges=pointwise_ranges,
            reduction_ranges=scan_ranges,
            reduction_type="scan",
            reduction_numel=scan_numel,
        )


# This signifies a scan op that should go through TritonSplitScanKernel codegen on CUDA.
@ir_dataclass
class SplitScan(Scan):
    pass


@ir_dataclass
class Sort(Loops):
    # Sorts a tuple of key, value pairs
    sort_ranges: list[Integer]
    size: list[Integer]
    reindex: Callable[[Sequence[Expr], Sequence[Expr]], Sequence[Expr]]
    reduction_hint: ReductionHint
    output_index: int
    # output_index indexes the following tuples
    dtypes: tuple[torch.dtype, ...]
    inner_fns: tuple[Callable[..., Any], ...]

    stable: bool
    descending: bool

    # HACK we mimic reduction

    @cache_on_self_and_args("Sort")
    def get_free_symbol_uses(self, unbacked_only: bool = False) -> OrderedSet[Symbol]:
        return (
            super().get_free_symbol_uses(unbacked_only)
            | OrderedSet().union(
                *(get_free_symbols(e, unbacked_only) for e in self.sort_ranges)
            )
            | OrderedSet().union(
                *(get_free_symbols(e, unbacked_only) for e in self.size)
            )
        )

    def __post_init__(self) -> None:
        assert len(self.ranges) + len(self.sort_ranges) == len(self.size)
        super().__post_init__()

    def store_reduction(
        self,
        output_name: str | None,
        indexer: Callable[[Sequence[Expr]], Expr],
        vars: Sequence[Expr],
        reduction_vars: Sequence[Expr],
    ) -> Any:
        idx = self.reindex(vars, reduction_vars)
        values = tuple(inner_fn(idx) for inner_fn in self.inner_fns)
        result = ops.sort(self.dtypes, values, self.stable, self.descending)
        return ops.store(
            output_name or "unnamed", indexer(idx), result[self.output_index]
        )

    def get_reduction_type(self) -> str | None:
        return "sort"

    def get_reduction_size(self) -> Sequence[Expr]:
        return self.sort_ranges

    def get_size(self) -> Sequence[Expr]:
        return self.size

    def get_pointwise_size(self) -> Sequence[Expr]:
        return self.ranges

    def index_length(self) -> int:
        return len(self.ranges) + len(self.sort_ranges)

    def inner_fn_args(self) -> Sequence[Sequence[Expr]]:
        index = self._index(self.ranges)
        rindex = self._index(self.sort_ranges, SymT.R0_INDEX)
        idx = self.reindex(index, rindex)
        return (idx,)

    def inner_fn_free_symbols(self, unbacked_only: bool = False) -> OrderedSet[Symbol]:
        index = self._index(self.ranges)
        rindex = self._index(self.sort_ranges, SymT.R0_INDEX)
        idx = self.reindex(index, rindex)
        return extract_free_symbols(self.inner_fn, idx, unbacked_only=unbacked_only)

    @classmethod
    def create(  # type: ignore[override]
        cls,
        device: torch.device,
        dtypes: tuple[torch.dtype, ...],
        inner_fns: tuple[Callable[[list[Expr]], Any], ...],
        size: list[Integer],
        axis: int,
        stable: bool,
        descending: bool,
        reduction_hint: ReductionHint = ReductionHint.DEFAULT,
        **kwargs: Any,
    ) -> Sequence[TensorBox | None]:
        pointwise_ranges = [*size[:axis], *size[axis + 1 :]]
        sort_ranges = [size[axis]]

        if not V.graph.has_feature(device, BackendFeature.SORT):
            return [None] * len(dtypes)

        sizevars = V.graph.sizevars
        sort_numel = sizevars.simplify(sympy_product(sort_ranges))

        # Heuristic, smallest rblock where triton usually outperforms aten.sort
        # It also isn't bandwidth bound so fusion is unlikely to help.
        max_rblock = 512
        is_persistent_kernel = (
            config.triton.persistent_reductions
            and sizevars.statically_known_true(sympy.Le(sort_numel, max_rblock))
        )
        if not is_persistent_kernel:
            # We only support persistent triton kernels
            return [None] * len(dtypes)

        assert len(dtypes) == len(inner_fns)

        # Sort with a single element is just a copy
        if sizevars.statically_known_true(sympy.Le(sort_numel, 1)):
            return [
                Pointwise.create(
                    device=device,
                    dtype=dtypes[output_index],
                    inner_fn=inner_fns[output_index],
                    ranges=size,
                )
                for output_index in range(len(dtypes))
            ]

        def reindex(index: Sequence[Expr], sort_index: Sequence[Expr]) -> list[Expr]:
            assert len(sort_index) == len(sort_ranges)
            assert len(index) == len(pointwise_ranges)
            return [*index[:axis], *sort_index, *index[axis:]]

        results = [
            TensorBox.create(
                Sort(
                    device=device,
                    dtype=dtypes[output_index],
                    dtypes=dtypes,
                    inner_fn=inner_fns[output_index],
                    inner_fns=inner_fns,
                    size=size,
                    ranges=pointwise_ranges,
                    sort_ranges=sort_ranges,
                    reindex=reindex,
                    reduction_hint=reduction_hint,
                    output_index=output_index,
                    stable=stable,
                    descending=descending,
                    **kwargs,
                )
            )
            for output_index in range(len(dtypes))
        ]

        for result in results:
            result.realize()

        return results
