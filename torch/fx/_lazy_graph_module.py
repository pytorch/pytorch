from contextlib import contextmanager

from torch.fx import GraphModule
from torch.fx.graph_module import _format_import_block, reduce_graph_module
from torch.package import sys_importer
from ._compatibility import compatibility

_use_lazy_graph_module_flag = False
_force_skip_lazy_graph_module_flag = False


@compatibility(is_backward_compatible=False)
@contextmanager
def _force_skip_lazy_graph_module():
    """
    Skip using lazy graph module disregarding the setting of _use_lazy_graph_module.
    Use to skip _LazyGraphModule when testing inductor torchscript related backend.

    torch.jit.script a _LazyGraphModule results in following error:
        https://gist.github.com/shunting314/5143654c8084aed84ecd19b818258a69
    """
    try:
        global _force_skip_lazy_graph_module_flag
        prior = _force_skip_lazy_graph_module_flag
        _force_skip_lazy_graph_module_flag = True
        yield
    finally:
        _force_skip_lazy_graph_module_flag = prior


@compatibility(is_backward_compatible=False)
@contextmanager
def _use_lazy_graph_module(should_use: bool):
    try:
        global _use_lazy_graph_module_flag
        prior = _use_lazy_graph_module_flag
        _use_lazy_graph_module_flag = (
            should_use and not _force_skip_lazy_graph_module_flag
        )
        yield
    finally:
        _use_lazy_graph_module_flag = prior


@compatibility(is_backward_compatible=False)
def get_graph_module_cls():
    return _LazyGraphModule if _use_lazy_graph_module_flag else GraphModule


@compatibility(is_backward_compatible=False)
class _LazyGraphModule(GraphModule):
    @classmethod
    def from_graphmodule(cls, gm: GraphModule):
        if isinstance(gm, _LazyGraphModule):
            return gm
        else:
            return _LazyGraphModule(gm, gm.graph)

    @staticmethod
    def force_recompile(gm):
        """
        Sometimes we need force a recompile as a workaround
        - we want to do the real recompilation before symbolic_trace to avoid error:
            https://gist.github.com/shunting314/75549c2e82ae07ac1139c94a3583d259
        """
        if isinstance(gm, _LazyGraphModule):
            gm.real_recompile()

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
        than 'self.recompile' since for a _LazyGraphModule, self.recompile just
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
