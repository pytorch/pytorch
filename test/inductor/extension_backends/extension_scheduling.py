from torch._inductor.codegen import cpp, wrapper
from torch._inductor.codegen.scheduling import Scheduling
from torch._inductor.virtualized import V


class ExtensionWrapperCodegen(wrapper.WrapperCodeGen):
    def __init__(self):
        super().__init__()


class ExtensionScheduling(Scheduling):
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self._scheduling = cpp.CppScheduling(scheduler)

    def can_fuse_vertical(self, node1, node2):
        return True

    def can_fuse_horizontal(self, node1, node2):
        return True

    def group_fn(self, *args, **kwargs):
        sizes = args[0]
        return tuple(tuple(map(V.graph.sizevars.simplify, s)) for s in sizes)

    def codegen_nodes(self, nodes):
        self._scheduling.codegen_nodes(nodes)

    def codegen_sync(self, *args, **kwargs):
        pass

    def flush(self, *args, **kwargs):
        self._scheduling.flush()
