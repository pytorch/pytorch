# mypy: allow-untyped-defs
import functools
from typing import Optional, Protocol, Sequence, Tuple, TypeVar, Union

import torch
from torch._prims_common import ELEMENTWISE_TYPE_PROMOTION_KIND


T = TypeVar("T")


class DTypeArg(Protocol):
    @property
    def dtype(self) -> torch.dtype:
        ...


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

    # metaprogrammed in __init__

    @staticmethod
    def op_dtype_rule(*args, type_promotion_kind) -> torch.dtype:
        return promote_types(args, type_promotion_kind=type_promotion_kind)

    @staticmethod
    def return_dtype(*args, dtype) -> torch.dtype:
        return dtype

    # op rules

    @staticmethod
    def default_handler(*args):
        # Fallback to FP32 dtype
        return torch.float32

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
