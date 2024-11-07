# Owner(s): ["oncall: package/deploy"]

from torch.fx import Tracer


class TestAllLeafModulesTracer(Tracer):
    def is_leaf_module(self, m, qualname):
        return True
