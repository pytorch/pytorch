from torch._inductor.codegen import cpp, cpp_wrapper_cpu, wrapper
from torch._inductor.scheduler import BaseScheduling
from torch._inductor.virtualized import V


class ExtensionWrapperCodegen(wrapper.PythonWrapperCodegen):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ExtensionCppWrapperCodegen(cpp_wrapper_cpu.CppWrapperCpu):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ExtensionScheduling(BaseScheduling):
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self._scheduling = cpp.CppScheduling(scheduler)

    def can_fuse_vertical(self, node1, node2):
        return True

    def can_fuse_horizontal(self, node1, node2):
        return True

    def group_fn(self, sizes):
        return tuple(tuple(map(V.graph.sizevars.simplify, s)) for s in sizes)

    def codegen_template(self, template_node, epilogue_nodes):
        pass

    def codegen_node(self, node):
        self._scheduling.codegen_node(node)

    def codegen_sync(self):
        pass

    def flush(self):
        self._scheduling.flush()
