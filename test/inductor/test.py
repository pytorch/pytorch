import torch.distributed._symmetric_memory as symm_mem

import torch

device = torch.device("cuda:0")
allocator = symm_mem.get_mempool_allocator(device)
mempool = torch.cuda.MemPool(allocator)
symm_mem.empty(shape)


class Forward(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Tensors created within this context are symmetric
        # with torch.cuda.use_mem_pool(mempool):  <-- This is a must, not optim
        #     y = foo(x)
        #  y = cust_op(x)
        #     ...

        # This op requires y to be a symmetric tensor
        torch.ops.symm_mem.one_shot_all_reduce(x, "sum", "0")
        return x


model = Forward()
x = torch.rand([4, 4], device=device)

exported_program = torch.export.export(model, (x,))

# Export the FX graph
print("FX Graph:")
print(exported_program.graph_module.graph)

# Optionally save the graph to a file
with open("fx_graph.txt", "w") as f:
    f.write(str(exported_program.graph_module.graph))

 @register_lowering(
            torch.ops.symm_mem.one_shot_all_reduce, lowering_dict=custom_lowering_dict
        )
        def foo_lowering(x):
            with torch.cuda.MemPool(mempool):
                y = x.copy()
            return torch.ops.symm_mem.one_shot_all_reduce(y, "sum", "0")
