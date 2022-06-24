import torch
# from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
# from torch.distributed.fsdp.wrap import ParamExecOrderWrapPolicy, always_wrap_policy

class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(6, 6))
        self.layer0 = torch.nn.Linear(6, 6)
        self.layer1 = torch.nn.Linear(6, 6, bias=False)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Linear(6, 3, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(3, 6, bias=False),
        )
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # `layer0` -> `layer2` -> `layer1`
        # the forward execution order is NOT consistent with the model definition order.
        z = self.relu(self.layer0(x))
        z = self.relu(self.layer2(z))
        z = z @ self.weight
        z = self.relu(self.layer1(z))
        return z

# module = Model()
# # fsdp_model = FSDP(module, auto_wrap_policy=always_wrap_policy)
# trace_graph: torch.fx.Graph = torch.fx.symbolic_trace(module).graph
# print(trace_graph)
# params = []
# for node in list(trace_graph.nodes):
#     print(node.format_node())
#     # print(node, node.op, node.target, node.args, )
#     # if node.op == "call_module":
#     #     targets = node.target.split(".")
#     #     out = module
#     #     for t in targets:
#     #         out = getattr(out, t)
#     #     print(out)
#     #     for p in out.parameters():
#     #         params.append(p)
#     # elif node.op == "call_function":
#     #     for inputnode in node.args:
#     #         if


class A():
    def __init__(self):
        self.a = 1
        self.b = 2
    def print(self):
        print(f"{self.a}, {self.b}")

def patch_a(aclass):
    printf = aclass.print
    def patched_print():
        print(f"{aclass.a}, {aclass.b}, patch!!!")
    aclass.print = patched_print
    return aclass

a = A()
a.print()
b = patch_a(a)
b.print()
b.a = 1111
b.print()
a.print()