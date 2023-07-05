from collections import defaultdict
from typing import Callable, Dict, Optional

import torch
import torch._decomp as decomp
from torch._ops import OpOverload
import torch._functorch.config as config

aten = torch.ops.aten
_decompositions: Dict[str, Dict[OpOverload, Callable]] = defaultdict(dict)


class DepTokenStateTracker:
    """
    A singleton / global context manager which is used to:
    - Record the global flag that assertion ops functionalization is enabled
      during compilation if the context is entered.
    - Record most recent token which is propagated through functional assertion
      ops.
    """

    _dep_token = None
    _entered: bool = False

    def __enter__(self) -> "DepTokenStateTracker":
        assert (
            not DepTokenStateTracker._entered
        ), "Cannot reenter singleton context `DepTokenStateTracker`."
        self.reset()
        DepTokenStateTracker._entered = True
        return self

    def __exit__(self, exc_type, exc_cal, exc_tb) -> None:
        assert DepTokenStateTracker._entered, (
            "Cannot exit singleton context `DepTokenStateTracker` which is not "
            "entered previously."
        )
        self.reset()
        DepTokenStateTracker._entered = False
        return None

    @classmethod
    def reset(cls) -> None:
        cls._dep_token = None

    @classmethod
    def set_dep_token(cls, dep_token) -> None:
        assert cls._entered
        cls._dep_token = dep_token

    @classmethod
    def get_dep_token(cls):
        return cls._dep_token


class FunctionalAssertionsHelper:
    @staticmethod
    def can_functionalize_asserts(needs_autograd: bool) -> bool:
        if config.functionalize_assertion_ops:
            # The following assertions is aim to limit assertions functionalization
            # to simple case for now.
            # Only handle forward graph.
            assert (
                not needs_autograd
            ), "Cannot functionalize assertion ops when grad is enabled"
            # Avoid rng ops functionalization which also add extra outputs.
            assert (
                not config.functionalize_rng_ops
            ), "Cannot functionalize assertion ops when RNG functionalization is enabled"

        return config.functionalize_assertion_ops

    @staticmethod
    def functionalization_enabled() -> bool:
        return DepTokenStateTracker._entered

    @staticmethod
    def get_decompositions() -> Dict[str, Dict[OpOverload, Callable]]:
        return _decompositions

    @staticmethod
    def get_state_tracker() -> DepTokenStateTracker:
        return DepTokenStateTracker()

    @staticmethod
    def create_compile_tracing_wrapper(func) -> Callable:
        def _traced_forward(*args):
            outs = func(*args)
            dep_token = DepTokenStateTracker.get_dep_token()
            return outs if dep_token is None else outs + (dep_token,)

        return _traced_forward

    @staticmethod
    def check_trace_joint(trace_joint: bool) -> None:
        # Don't functionalize asserts when `trace_joint` is enabled for now.
        if config.functionalize_assertion_ops:
            assert (
                not trace_joint
            ), "Cannot functionalize assertion ops when trace joint is enabled"

    @staticmethod
    def create_asserts_dep_token_output(
        gm: torch.fx.GraphModule, num_outputs_dep_token: int
    ) -> Optional[str]:
        if num_outputs_dep_token == 0:
            return None
        assert num_outputs_dep_token == 1

        output_args = next(
            n for n in reversed(gm.graph.nodes) if n.op == "output"
        ).args[0]
        dep_token_arg = output_args[-1]

        assert dep_token_arg.target in (
            aten._make_dep_token.default,
            aten._functional_assert_async.msg,
        )

        return dep_token_arg.name

    @classmethod
    def get_num_outputs_dep_token(cls) -> int:
        return (
            1
            if cls.functionalization_enabled()
            and DepTokenStateTracker.get_dep_token() is not None
            else 0
        )


def _register_decomposition(aten_op):
    return decomp.register_decomposition(aten_op, _decompositions)


@_register_decomposition(aten._assert_async.msg)
def _assert_async_msg(val, assert_msg) -> None:
    dep_token = DepTokenStateTracker.get_dep_token()
    if dep_token is None:
        dep_token = aten._make_dep_token()

    DepTokenStateTracker.set_dep_token(
        aten._functional_assert_async.msg(val, assert_msg, dep_token)
    )
