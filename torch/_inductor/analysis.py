from torch._inductor import scheduler

from . import ir

def get_snode_runtime(snode: "BaseSchedulerNode"):
    """
    Gets the runtime of Scheduler node. Currently somewhat of a placeholder.

    TODO: replace with a more sophisticated cost model (e.g. analytical)
    """
    # Collective kernels
    if isinstance(snode.node, ir.CollectiveKernel):
        if isinstance(snode.node, ir.AllReduce):
            return 5
        else:
            return 15
    elif isinstance(snode.node, ir.Wait):
        return 0
    # Compute kernels
    elif isinstance(snode.node, ir.ExternKernel):
        return 10
    # All other kernels
    return 1
