from collections import defaultdict
import torch
from torch._subclasses.fake_tensor import fake_id_to_stack

def assert_no_aliased_graph_inputs(graph: torch.fx.Graph) -> None:
    """
    Assert that there is no aliased graph inputs that share the same storage.
    TODO(yf225): we should only run this in traceable FSDP unit tests.
    """
    storage_id_to_graph_inputs = defaultdict(list)
    for node in graph.nodes:
        if node.op == "placeholder" and isinstance(node.meta.get('val', None), torch.Tensor):
            storage_id_to_graph_inputs[id(node.meta['val'].untyped_storage())].append(node)
    for storage_id, aliased_graph_inputs in storage_id_to_graph_inputs.items():
        if len(aliased_graph_inputs) > 1:
            print(f"""
Found aliased graph inputs: {aliased_graph_inputs},
type(val): {[type(node.meta['val']) for node in aliased_graph_inputs]},
val.shape: {[node.meta['val'].shape for node in aliased_graph_inputs]},
id(val): {[id(node.meta['val']) for node in aliased_graph_inputs]},
sid: {storage_id}
""")
            # for node in aliased_graph_inputs:
            #     if node.meta['val'].shape == torch.Size([32000, 4096]):
            #         print(f"""stacks: {"\n\n".join([fake_id_to_stack[id(node.meta['val'])] for node in aliased_graph_inputs])}""")
