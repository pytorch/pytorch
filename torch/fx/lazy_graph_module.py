from contextlib import contextmanager

from torch.fx import GraphModule
from ._compatibility import compatibility

_use_lazy_graph_module = False
_force_skip_lazy_graph_module = False  # used in unit test to skip lazy graph module


@compatibility(is_backward_compatible=False)
@contextmanager
def use_lazy_graph_module(should_use: bool):
    try:
        global _use_lazy_graph_module
        prior = _use_lazy_graph_module
        _use_lazy_graph_module = should_use and not _force_skip_lazy_graph_module
        yield
    finally:
        _use_lazy_graph_module = prior


@compatibility(is_backward_compatible=False)
def get_graph_module_cls():
    return LazyGraphModule if _use_lazy_graph_module else GraphModule


@compatibility(is_backward_compatible=False)
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
        # Call self.real_recompile() rather than self._real_recompile() here.
        # The _lazy_forward method may be saved and call repeatedly.
        # Calling self.real_recompile can make sure we skip recompilation if
        # we have already done so.
        self.real_recompile()
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
        return super().code

    def __str__(self) -> str:
        """
        str(GraphModule) will access the _code attribute. Make sure recompile
        happens so _code attribute is available.
        """
        self.real_recompile()
        return super().__str__()
