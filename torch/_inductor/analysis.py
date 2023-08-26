from torch._inductor import scheduler

from . import ir

def get_runtime_snode(snode: "BaseSchedulerNode"):
    """
    Gets the runtime of Scheduler node. Currently somewhat of a placeholder, todo to be replaced by more sophisticated approaches.

    TODO(yf225): potentially replace with a more sophisticated cost model (e.g. analytical)
    """
    if isinstance(snode.node, ir.AllReduce):
        return 5
    if isinstance(snode.node, ir.CollectiveKernel):
        return 15
    if isinstance(snode.node, ir.MultiOutput):
        return 0
    if isinstance(snode.node, ir.Wait):
        return 0
    if isinstance(snode.node, ir.ExternKernel):
        return 10
    return 1
