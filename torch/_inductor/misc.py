"""
TODO: I'll cleanup this
"""

from torch._inductor.codegen.triton import EnableReduction, DisableReduction
from torch._inductor.scheduler import SchedulerNode

def dump_node_schedule(node_schedule):
    print(f"Node schedule with {len(node_schedule)} nodes")
    for idx, node in enumerate(node_schedule):
        print(f" {idx:3}:")
        if node is EnableReduction:
            print("enable reduction")
        elif node is DisableReduction:
            print("disable reduction")
        elif isinstance(node, SchedulerNode):
            is_red = node.is_reduction()
            print(f"schedule node {'red' if is_red else 'pw'}")
            if is_red:
                print(f"orig red hint {node.node.data.reduction_hint}")
            print("ReadDep:")
            for dep in node.read_writes.reads:
                print(dep)
            print("WriteDep:")
            for dep in node.read_writes.writes:
                print(dep)
        else:
            raise RuntimeError(f"Unrecognized node type: {type(node)}")
