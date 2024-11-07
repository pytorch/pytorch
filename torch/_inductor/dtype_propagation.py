# mypy: allow-untyped-defs
import functools
from typing import Optional, Protocol, Sequence, Tuple, TypeVar, Union

import torch
from torch._inductor.virtualized import V
from torch._prims_common import ELEMENTWISE_TYPE_PROMOTION_KIND


T = TypeVar("T")


class DTypeArg(Protocol):
    @property
    def dtype(self) -> torch.dtype: ...


# Inputs need to be cacheable (e.g., not a CSEVar) in order for the cache to be effective
# So first decompose CSEVars -> tuple before calling this


@functools.lru_cache(None)
def get_promoted_dtype(
    *args: Sequence[Tuple[torch.dtype, bool]],
    type_promotion_kind: ELEMENTWISE_TYPE_PROMOTION_KIND,
):
    def construct_input(inp):
        if inp[1]:
            return torch.empty(1, dtype=inp[0])
        else:
            return torch.empty([1], dtype=inp[0])

    inps = [construct_input(arg) for arg in args]
    _, dtype = torch._prims_common.elementwise_dtypes(
        *inps, type_promotion_kind=type_promotion_kind
    )
    return dtype


def promote_types(
    args: Sequence[Union[DTypeArg, torch.types.Number, str]],
    type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND,
):
    dtype_prop_candidates = []

    for arg in args:
        if isinstance(arg, str):
            # comes from templates.. TODO
            continue

        if isinstance(arg, torch._prims_common.Number):
            dtype_prop_candidates.append((torch.tensor(arg).dtype, True))
            continue

        dtype_prop_candidates.append((arg.dtype, False))

    dtype = get_promoted_dtype(
        *dtype_prop_candidates,
        type_promotion_kind=type_promotion_kind,
    )

    return dtype


class DtypePropagationOpsHandler:
    """
    Propagate dtype from args to output
    """

    def __init__(self) -> None:
        for op, rule in torch._inductor.utils.op_dtype_propagation_rules.items():
            fn = (
                functools.partial(self.return_dtype, dtype=rule.override_return_dtype)
                if rule.override_return_dtype
                else functools.partial(
                    self.op_dtype_rule, type_promotion_kind=rule.type_promotion_kind
                )
            )
            setattr(self, op, fn)

        # Set pointwise operation rules
        for op in torch._inductor.codegen.common.pointwise_overrides_data.values():
            if not hasattr(self, op.name):
                setattr(
                    self,
                    op.name,
                    functools.partial(
                        self.op_dtype_rule, type_promotion_kind=op.type_promotion_kind
                    ),
                )

        # Set boolean operation rules
        for op in torch._inductor.utils.boolean_ops():
            if not hasattr(self, op):
                setattr(
                    self, op, functools.partial(self.return_dtype, dtype=torch.bool)
                )

        from torch._inductor.ops_handler import OpsHandler

        ops_set = {s for s in dir(OpsHandler) if s[0] != "_"}
        unimplemented_ops = ops_set - set(dir(self))
        torch._check(
            len(unimplemented_ops) == 0,
            lambda: f"Unimplemented dtype rule for ops: {unimplemented_ops}",
        )

    # metaprogrammed in __init__

    @staticmethod
    def op_dtype_rule(*args, type_promotion_kind) -> torch.dtype:
        return promote_types(args, type_promotion_kind=type_promotion_kind)

    @staticmethod
    def return_dtype(*args, dtype) -> torch.dtype:
        return dtype

    # op rules

    @staticmethod
    def constant(value, dtype) -> torch.dtype:
        return dtype

    @staticmethod
    def load_seed(name, offset) -> torch.dtype:
        return torch.float32

    @staticmethod
    def randint64(seed, offset, low, high) -> torch.dtype:
        return torch.int64

    @staticmethod
    def masked(mask, body, other) -> torch.dtype:
        # TODO: inspect body to propagate dtype
        return torch.float32

    @staticmethod
    def where(a, b, c) -> torch.dtype:
        return promote_types([b, c])

    @staticmethod
    def index_expr(expr, dtype) -> torch.dtype:
        return dtype

    @staticmethod
    def to_dtype(
        x, dtype: torch.dtype, src_dtype: Optional[torch.dtype] = None
    ) -> torch.dtype:
        return dtype

    @staticmethod
    def to_dtype_bitcast(x, dtype: torch.dtype, src_dtype: torch.dtype) -> torch.dtype:
        return dtype

    @staticmethod
    def gelu(x) -> torch.dtype:
        return x.dtype

    @staticmethod
    def mul(a, b) -> torch.dtype:
        return promote_types([a, b])

    @staticmethod
    def div(a, b) -> torch.dtype:
        return promote_types([a, b])

    @staticmethod
    def truediv(a, b) -> torch.dtype:
        return promote_types(
            [a, b],
        )

    @staticmethod
    def pow(a, b) -> torch.dtype:
        return promote_types([a, b])

    @staticmethod
    def mod(a, b) -> torch.dtype:
        return promote_types([a, b])

    @staticmethod
    def indirect_indexing(x, size, check=True, wrap_neg=True) -> torch.dtype:
        return torch.int64

    @staticmethod
    def randn(seed, offset) -> torch.dtype:
        return torch.float

    @staticmethod
    def rand(seed, offset) -> torch.dtype:
        return torch.float

    @staticmethod
    def store_reduction(name, index, value) -> torch.dtype:
        return V.graph.get_dtype(name)

    @staticmethod
    def reduction(dtype, src_dtype, reduction_type, value) -> torch.dtype:
        return dtype

    @staticmethod
    def store(name, index, value, mode=None) -> torch.dtype:
        return V.graph.get_dtype(name)

    @staticmethod
    def load(name, index) -> torch.dtype:
        return V.graph.get_dtype(name)

    @staticmethod
    def or_(x0: T, x1: T) -> T:
        raise RuntimeError("TODO: Implement or_")

    @staticmethod
    def floor(x0: T) -> T:
        raise RuntimeError("TODO: Implement floor")

    @staticmethod
    def ceil_to_int(x: T, dtype: torch.dtype) -> T:
        raise RuntimeError("TODO: Implement ceil_to_int")

    @staticmethod
    def int_truediv(x0: T, x1: T) -> T:
        raise RuntimeError("TODO: Implement int_truediv")

    @staticmethod
    def scan(
        dtypes: Tuple[torch.dtype, ...],
        combine_fn: Callable[[Tuple[T, ...], Tuple[T, ...]], Tuple[T, ...]],
        values: Tuple[T, ...],
    ) -> Tuple[T, ...]:
        raise RuntimeError("TODO: Implement scan")

    @staticmethod
    def invert(x0: T) -> T:
        raise RuntimeError("TODO: Implement invert")

    @staticmethod
    def matmul(x0: T, x1: T) -> T:
        raise RuntimeError("TODO: Implement matmul")

    @staticmethod
    def fmod(x0: T, x1: T) -> T:
        raise RuntimeError("TODO: Implement fmod")

    @staticmethod
    def round_to_int(x: T, dtype: torch.dtype) -> T:
        raise RuntimeError("TODO: Implement round_to_int")

    @staticmethod
    def xor(x0: T, x1: T) -> T:
        raise RuntimeError("TODO: Implement xor")

    @staticmethod
    def identity(x: T) -> T:
        raise RuntimeError("TODO: Implement identity")

    @staticmethod
    def frexp(x: T):
        raise RuntimeError("TODO: Implement frexp")

    @staticmethod
    def sort(
        dtypes: Tuple[torch.dtype, ...],
        values: Tuple[T, ...],
        stable: bool,
        descending: bool,
    ) -> Tuple[T, ...]:
        raise RuntimeError("TODO: Implement sort")

    @staticmethod
    def trunc(x0: T) -> T:
        raise RuntimeError("TODO: Implement trunc")

    @staticmethod
    def bucketize(
        values: T,
        boundaries: Tuple[str, sympy.Expr, sympy.Expr, sympy.Expr],
        boundary_indices: T,
        indexing_dtype: torch.dtype,
        right: bool,
    ) -> T:
        raise RuntimeError("TODO: Implement bucketize")

    @staticmethod
    def rshift(x0: T, x1: T) -> T:
        raise RuntimeError("TODO: Implement rshift")

    @staticmethod
    def round(x0: T) -> T:
        raise RuntimeError("TODO: Implement round")

    @staticmethod
    def getitem(x0: T, x1: T) -> T:
        raise RuntimeError("TODO: Implement getitem")

    @staticmethod
    def trunc_to_int(x: T, dtype: torch.dtype) -> T:
        raise RuntimeError("TODO: Implement trunc_to_int")

    @staticmethod
    def floor_to_int(x: T, dtype: torch.dtype) -> T:
        raise RuntimeError("TODO: Implement floor_to_int")

    @staticmethod
    def truncdiv(x0: T, x1: T) -> T:
        raise RuntimeError("TODO: Implement truncdiv")

    @staticmethod
    def floordiv(x0: T, x1: T) -> T:
        raise RuntimeError("TODO: Implement floordiv")

    @staticmethod
    def round_decimal(x0: T, x1: T) -> T:
        raise RuntimeError("TODO: Implement round_decimal")

    @staticmethod
    def lshift(x0: T, x1: T) -> T:
        raise RuntimeError("TODO: Implement lshift")

    @staticmethod
    def libdevice_abs(x0: T) -> T:
        raise RuntimeError("TODO: Implement libdevice_abs")
