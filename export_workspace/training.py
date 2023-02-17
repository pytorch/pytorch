import torch
import torch._dynamo

device = "cpu"
dtype = torch.float

class TestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)

    def forward(self, x):
        out =  self.linear(x)
        loss = out.sum()
        return loss

x = torch.rand(8, 8, device=device, dtype=dtype, requires_grad=False)
model = TestModule()

gm, guard = torch._dynamo.export(model, x, aten_graph=True, tracing_mode="real", training_mode=True)

gm.print_readable()
