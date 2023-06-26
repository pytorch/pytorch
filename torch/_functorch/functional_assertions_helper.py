from collections import defaultdict
from typing import Callable, Dict

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
        if cls._dep_token is None:
            cls._dep_token = aten._make_dep_token()

        return cls._dep_token


class FunctionalAssertionsHelper:
    @staticmethod
    def can_functionalize(needs_autograd: bool) -> bool:
        if config.functionalize_assertion_ops:
            # The following assertions is aim to limit assertions functionalization
            # to simple case for now.
            # Only handle forward graph.
            assert not needs_autograd
            # Avoid rng ops functionalization which also add extra outputs.
            assert not config.functionalize_rng_ops

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
            return outs + (DepTokenStateTracker.get_dep_token(),)

        return _traced_forward


def _register_decomposition(aten_op):
    return decomp.register_decomposition(aten_op, _decompositions)


@_register_decomposition(aten._assert_async.msg)
def _assert_async_msg(val, assert_msg) -> None:
    DepTokenStateTracker.set_dep_token(
        aten._functional_assert_async.msg(
            val, assert_msg, DepTokenStateTracker.get_dep_token()
        )
    )
