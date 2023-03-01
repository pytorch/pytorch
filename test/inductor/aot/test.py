import torch
import torch._dynamo
import torch._inductor
import torch._inductor.config

torch._inductor.config.aot_codegen_output_prefix = "aot_inductor_output"


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.ones(32, 64)

    def forward(self, x):
        x = torch.relu(x + self.weight)
        return x


inp = torch.randn((32, 64), device="cpu")
module, _ = torch._dynamo.export(Net(), inp)
so_path = torch._inductor.aot_compile(module, [inp])
print(so_path)
