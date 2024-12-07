# mypy: allow-untyped-defs
import functools
from typing import (
    Callable,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TYPE_CHECKING,
    TypeVar,
    Union,
)

import sympy


if TYPE_CHECKING:
    from torch._inductor.loop_body import LoopBodyBlock

import torch
from torch._inductor.virtualized import V
from torch._prims_common import ELEMENTWISE_TYPE_PROMOTION_KIND

from .utils import upcast_compute_type
from .virtualized import OpsValue


T = TypeVar("T")


class DTypeVar(Protocol):
    @property
    def dtype(self) -> torch.dtype:
        ...


DTypeArg = Union[DTypeVar, torch.types.Number, str, OpsValue]


# Inputs need to be cacheable (e.g., not a CSEVar) in order for the cache to be effective
# So first decompose CSEVars -> tuple before calling this


@functools.lru_cache(None)
def get_promoted_dtype(
    *args: Sequence[Tuple[torch.dtype, bool]],
    type_promotion_kind: Optional[ELEMENTWISE_TYPE_PROMOTION_KIND] = None,
):
    def construct_input(inp):
        if inp[1]:
            return torch.empty(1, dtype=inp[0])
        else:
            return torch.empty([1], dtype=inp[0])

    inps = [construct_input(arg) for arg in args]
    _, dtype = torch._prims_common.elementwise_dtypes(
        *inps,
        type_promotion_kind=(
            type_promotion_kind
            if type_promotion_kind
            else ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
        ),
    )
    return dtype


def promote_types(
    args: Sequence[DTypeArg],
    type_promotion_kind: Optional[ELEMENTWISE_TYPE_PROMOTION_KIND] = None,
):
    dtype_prop_candidates = []

    for arg in args:
        if isinstance(arg, str):
            # comes from templates.. TODO
            continue

        if isinstance(arg, OpsValue):
            arg = arg.value
            assert isinstance(arg, torch._prims_common.Number) or hasattr(arg, "dtype")

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

    # Singleton DtypePropagationOpsHandler, because we meta program over a number of op rules.
    # Those are only defined after other inductor state has run.

    _instance: Optional["DtypePropagationOpsHandler"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

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
    def op_dtype_rule(
        *args: DTypeArg, type_promotion_kind: ELEMENTWISE_TYPE_PROMOTION_KIND
    ) -> torch.dtype:
        return promote_types(args, type_promotion_kind=type_promotion_kind)

    @staticmethod
    def return_dtype(*args: DTypeArg, dtype: torch.dtype) -> torch.dtype:
        return dtype

    # op rules

    @staticmethod
    def constant(value: torch.types.Number, dtype: torch.dtype) -> torch.dtype:
        return upcast_compute_type(dtype)

    @staticmethod
    def load_seed(name: str, offset: int) -> torch.dtype:
        return upcast_compute_type(V.graph.get_dtype(name))

    @staticmethod
    def randint64(seed: int, offset: int, low: int, high: int) -> torch.dtype:
        return torch.int64

    @staticmethod
    def masked(mask: DTypeArg, body: "LoopBodyBlock", other: DTypeArg) -> torch.dtype:
        # TODO - we avoid calling this in codegen, needs work for non codegen use cases
        loads = body.graph.find_nodes(op="call_method", target="load")
        if len(loads) <= 1:
            return promote_types([other])

        return upcast_compute_type(V.graph.get_dtype(loads[-1].args[1]))

    @staticmethod
    def where(a: DTypeArg, b: DTypeArg, c: DTypeArg) -> torch.dtype:
        return promote_types([b, c])

    @staticmethod
    def index_expr(expr: sympy.Expr, dtype: torch.dtype) -> torch.dtype:
        # TODO - TODO - rationalize index_expr. The dtype is not always used and we are inconsistent about int32 or int64
        # in lowerings. cpp just uses the dtype
        if dtype not in (torch.int32, torch.int64) or not hasattr(
            V.kernel, "index_dtype"
        ):
            return upcast_compute_type(dtype)

        return torch.int32 if V.kernel.index_dtype == "tl.int32" else torch.int64

    @staticmethod
    def to_dtype(
        x: DTypeArg,
        dtype: torch.dtype,
        src_dtype: Optional[torch.dtype] = None,
        use_compute_types=True,
    ) -> torch.dtype:
        return upcast_compute_type(dtype) if use_compute_types else dtype

    @staticmethod
    def to_dtype_bitcast(
        x: DTypeArg, dtype: torch.dtype, src_dtype: torch.dtype
    ) -> torch.dtype:
        return upcast_compute_type(dtype)

    @staticmethod
    def gelu(x: DTypeArg) -> torch.dtype:
        return promote_types([x])

    @staticmethod
    def mul(a: DTypeArg, b: DTypeArg) -> torch.dtype:
        return promote_types([a, b])

    @staticmethod
    def div(a: DTypeArg, b: DTypeArg) -> torch.dtype:
        return promote_types([a, b])

    @staticmethod
    def truediv(a: DTypeArg, b: DTypeArg) -> torch.dtype:
        return promote_types([a, b])

    @staticmethod
    def pow(a: DTypeArg, b: DTypeArg) -> torch.dtype:
        return promote_types([a, b])

    @staticmethod
    def mod(a: DTypeArg, b: DTypeArg) -> torch.dtype:
        return promote_types([a, b])

    @staticmethod
    def indirect_indexing(
        x: DTypeArg, size: int, check: bool = True, wrap_neg: bool = True
    ) -> torch.dtype:
        return torch.int64

    @staticmethod
    def randn(seed: int, offset: int) -> torch.dtype:
        return torch.float

    @staticmethod
    def rand(seed: int, offset: int) -> torch.dtype:
        return torch.float

    @staticmethod
    def store_reduction(name: str, index, value: DTypeArg) -> None:
        return None

    @staticmethod
    def reduction(
        dtype: torch.dtype, src_dtype: torch.dtype, reduction_type: str, value: DTypeArg
    ) -> torch.dtype:
        return dtype

    @staticmethod
    def store(name: str, index, value: DTypeArg, mode: Optional[str] = None) -> None:
        return None

    @staticmethod
    def load(name: str, index) -> torch.dtype:
        return upcast_compute_type(V.graph.get_dtype(name))

    @staticmethod
    def floor(x: DTypeArg) -> torch.dtype:
        return promote_types(
            [x], type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
        )

    @staticmethod
    def ceil_to_int(x: DTypeArg, dtype: torch.dtype) -> torch.dtype:
        return dtype

    @staticmethod
    def int_truediv(x: DTypeArg, y: DTypeArg) -> torch.dtype:
        return promote_types(
            [x, y], type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
        )

    @staticmethod
    def scan(
        dtypes: Tuple[torch.dtype, ...],
        combine_fn: Callable[[Tuple[T, ...], Tuple[T, ...]], Tuple[T, ...]],
        values: Tuple[T, ...],
    ) -> Tuple[torch.dtype, ...]:
        return dtypes

    @staticmethod
    def fmod(x: DTypeArg, y: DTypeArg) -> torch.dtype:
        return promote_types([x, y])

    @staticmethod
    def round_to_int(x: DTypeArg, dtype: torch.dtype) -> torch.dtype:
        return dtype

    @staticmethod
    def identity(x: DTypeArg) -> torch.dtype:
        return promote_types([x])

    @staticmethod
    def frexp(x: DTypeArg) -> Tuple[torch.dtype, torch.dtype]:
        # TODO - need to handle multiple outputs
        return (promote_types([x]), torch.int32)

    @staticmethod
    def sort(
        dtypes: Tuple[torch.dtype, ...],
        values: Tuple[T, ...],
        stable: bool,
        descending: bool,
    ) -> Tuple[torch.dtype, ...]:
        return dtypes

    @staticmethod
    def trunc(x: DTypeArg) -> torch.dtype:
        return promote_types([x])

    @staticmethod
    def bucketize(
        values: DTypeArg,
        boundaries: Tuple[str, sympy.Expr, sympy.Expr, sympy.Expr],
        boundary_indices: DTypeArg,
        indexing_dtype: torch.dtype,
        right: bool,
    ) -> torch.dtype:
        return indexing_dtype

    @staticmethod
    def rshift(x: DTypeArg, y: DTypeArg) -> torch.dtype:
        return promote_types([x])

    @staticmethod
    def round(x: DTypeArg) -> torch.dtype:
        return promote_types(
            [x], type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
        )

    @staticmethod
    def getitem(x: DTypeArg, y: DTypeArg) -> torch.dtype:
        raise RuntimeError("Unexpected op: getitem")

    @staticmethod
    def trunc_to_int(x: DTypeArg, dtype: torch.dtype) -> torch.dtype:
        return dtype

    @staticmethod
    def floor_to_int(x: DTypeArg, dtype: torch.dtype) -> torch.dtype:
        return dtype

    @staticmethod
    def truncdiv(x: DTypeArg, y: DTypeArg) -> torch.dtype:
        return promote_types([x, y])

    @staticmethod
    def floordiv(x: DTypeArg, y: DTypeArg) -> torch.dtype:
        return promote_types([x, y])

    @staticmethod
    def round_decimal(x: DTypeArg, y: DTypeArg) -> torch.dtype:
        # TODO - dont see it anywhere..
        return promote_types([x])

    @staticmethod
    def halide_clamp(value, size, check):
        # TODO - way of registering dtype for op in backend
        return torch.int32

    @staticmethod
    def inline_asm_elementwise(
        *inputs, asm, constraints=None, dtype=torch.float32, is_pure=True, pack=1
    ):
        return dtype

    @staticmethod
    def lshift(x: DTypeArg, y: DTypeArg) -> torch.dtype:
        return promote_types([x])

    @staticmethod
    def libdevice_abs(x: DTypeArg) -> torch.dtype:
        return promote_types([x])

    @staticmethod
    def invert(x: DTypeArg) -> torch.dtype:
        raise RuntimeError("Unexpected op: invert")

    @staticmethod
    def matmul(x: DTypeArg, y: DTypeArg) -> torch.dtype:
        raise RuntimeError("Unexpected op: matmul")
