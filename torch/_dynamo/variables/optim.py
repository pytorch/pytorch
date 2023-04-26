from typing import Dict, List

from torch._guards import TracingContext
from ..source import AttrSource
from .base import VariableTracker


class OptimizerVariable(VariableTracker):
    def __init__(self, optimizer_key: int, **kwargs):
        super().__init__(**kwargs)
        self.optimizer_key = optimizer_key
        assert self.source

    def var_getattr(self, tx, name):
        options = VariableTracker.propagate(self)
        guards = options.get("guards", set())
        if name in ("step", "zero_grad"):
            return OptimizerMethodVariable(
                self, name, source=AttrSource(self.source, name)
            )


class OptimizerMethodVariable(VariableTracker):
    def __init__(self, optimizer: VariableTracker, method_name: str, **kwargs):
        super().__init__(**kwargs)
        self.optimizer = optimizer
        self.method_name = method_name
        assert method_name in (
            "step",
            "zero_grad",
        ), f"Optimizer.{method_name} not supported in train_step compile"
        assert self.source

    def call_function(
        self,
        tx,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        from .builder import wrap_fx_proxy

        options = VariableTracker.propagate(self, args, kwargs.values())

        optimizer_proxy = tx.output.create_proxy(
            "get_attr",
            self.optimizer.optimizer_key,
            tuple(),
            {},
        )
        optimizer_proxy.node.meta["example_value"] = None
        tc = TracingContext.train_step_context(assert_if_missing=True)

        if self.method_name == "step":
            assert len(args) == 0, f"no args supported for optimizer.step(), {args}"
            assert (
                len(kwargs) == 0
            ), f"no kwargs supported for optimizer.step(), {kwargs}"
            # Keep track of which optimizers we step, so we can ensure they also get zero-grad'd
            tc.optimizers_stepped.add(id(self.optimizer))
        elif self.method_name == "zero_grad":
            assert (
                len(args) == 0
            ), f"no args supported for optimizer.zero_grad(), {args}"
            if len(kwargs):
                assert len(kwargs) == 1
                assert (
                    "set_to_none" in kwargs and kwargs["set_to_none"] is True
                ), "must set_to_none to keep parity between compile/eager"
            tc.optimizers_zeroed_grad.add(id(self.optimizer))

        return wrap_fx_proxy(
            tx,
            tx.output.create_proxy(
                "call_method",
                self.method_name,
                args=(optimizer_proxy,),
                kwargs=kwargs,
            ),
            **options,
        )
