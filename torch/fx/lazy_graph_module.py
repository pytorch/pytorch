from contextlib import contextmanager

from torch.fx import GraphModule
from torch.fx.graph_module import _format_import_block, reduce_graph_module
from torch.package import sys_importer
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

    def __reduce__(self):
        """
        Follow GraphModule.__reduce__ but call 'self._real_recompile' rather
        than 'self.recompile' since for a LazyGraphModule, self.recompile just
        mark the need of recompilation and does not return the PythonCode object.
        """
        python_code = self._real_recompile()
        dict_without_graph = self.__dict__.copy()
        import_block = _format_import_block(python_code.globals, sys_importer)
        del dict_without_graph["_graph"]
        return (reduce_graph_module, (dict_without_graph, import_block))

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
