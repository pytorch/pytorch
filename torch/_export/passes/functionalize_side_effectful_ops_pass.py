import copy
from typing import Optional

import torch
from torch._export.pass_base import (
    _ExportPassBaseDeprecatedDoNotUse,
    Argument,
    PassResult,
)
from torch._export.pass_infra.node_metadata import NodeMetadata
from torch._export.pass_infra.proxy_value import ProxyValue
from torch._ops import OpOverload


aten = torch.ops.aten

_NON_FUNCTIONAL_TO_FUNCTIONAL_SIDE_EFFECTFUL_FUNCS: dict[OpOverload, OpOverload] = {
    aten.sym_constrain_range.default: aten._functional_sym_constrain_range.default,
    aten._assert_async.msg: aten._functional_assert_async.msg,
}


class _FunctionalizeSideEffectfulOpsPass(_ExportPassBaseDeprecatedDoNotUse):
    """
    Functionalize ops with side effect in graph module by replacing the op with
    functional version of it. A new dependency token (`dep_token`) will be
    created and propagated through functional ops to output.
    For example:
    ```
    def f(x):
        sym_constrain_range(x.shape[0], min=1, max=3)
        return x.add(3)
    ```
    Will be transformed to:
    ```
    def f(x):
        dep_token0 = _make_dep_token()
        dep_token1 = _functional_sym_constrain_range(
            x.shape[0], min=1, max=3, dep_token=dep_token0
        )

        return x.add(3), dep_token1
    ```
    """

    def __init__(self) -> None:
        super().__init__()
        self._dep_token: Optional[ProxyValue] = None
        self._next_dep_token_index: Optional[int] = None

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        # Early return if no non-functional assertions.
        if not any(
            n.target in _NON_FUNCTIONAL_TO_FUNCTIONAL_SIDE_EFFECTFUL_FUNCS
            for n in graph_module.graph.nodes
        ):
            return PassResult(graph_module=graph_module, modified=False)

        gm = copy.deepcopy(graph_module)
        self._dep_token = None
        self._next_dep_token_index = None
        return super().call(gm)

    def call_operator(
        self,
        op: OpOverload,
        args: tuple[Argument, ...],
        kwargs: dict[str, Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        if op not in _NON_FUNCTIONAL_TO_FUNCTIONAL_SIDE_EFFECTFUL_FUNCS:
            return super().call_operator(op, args, kwargs, meta)

        if self._dep_token is None:
            self._dep_token = super().call_operator(
                aten._make_dep_token,
                args=(),
                kwargs={},
                meta=self._create_dummy_node_metadata(),
            )
            self._dep_token.node.name = "dep_token0"
            self._next_dep_token_index = 1

        self._dep_token = super().call_operator(
            _NON_FUNCTIONAL_TO_FUNCTIONAL_SIDE_EFFECTFUL_FUNCS[op],
            args=args,
            kwargs={**kwargs, "dep_token": self._dep_token},
            meta=meta,
        )
        assert self._next_dep_token_index is not None
        self._dep_token.node.name = f"dep_token{self._next_dep_token_index}"
        self._next_dep_token_index += 1

        return self._dep_token

    def output(self, results: list[Argument], meta: NodeMetadata) -> ProxyValue:
        assert self._dep_token is not None

        return super().output(results=(*results, self._dep_token), meta=meta)  # type: ignore[arg-type]
