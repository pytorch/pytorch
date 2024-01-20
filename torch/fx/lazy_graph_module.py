from torch.fx import GraphModule
from contextlib import contextmanager
import os

_use_lazy_graph_module = False

@contextmanager
def use_lazy_graph_module(should_use: bool):
    try:
        global _use_lazy_graph_module
        prior = _use_lazy_graph_module
        _use_lazy_graph_module = should_use
        yield
    finally:
        _use_lazy_graph_module = prior

def get_graph_module_cls():
    return LazyGraphModule if _use_lazy_graph_module else GraphModule

class LazyGraphModule(GraphModule):
    @classmethod
    def from_graphmodule(cls, gm: GraphModule):
        if isinstance(gm, LazyGraphModule):
            return gm
        else:
            return LazyGraphModule(gm, gm.graph)

    def real_recompile(self):
        if self._needs_recompile():
             self._real_recompile()

    @classmethod
    def _needs_recompile(cls):
        return cls.forward is cls._lazy_forward

    def _lazy_forward(self, *args, **kwargs):
        self._real_recompile()
        assert not self._needs_recompile()

        # call `__call__` rather than 'forward' since recompilation may
        # install a wrapper for `__call__` to provide a customized error
        # message.
        return self(*args, **kwargs)

    forward = _lazy_forward

    def _real_recompile(self):
        return super().recompile()

    @classmethod
    def recompile(cls):
        cls.forward = cls._lazy_forward 

    @property
    def code(self) -> str:
        self.real_recompile()
        return super().code()
