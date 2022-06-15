import torch


class Mod(torch.nn.Module):
    def __init__(self):
        super(Mod, self).__init__()
        self.a = torch.nested_tensor([torch.rand([2]), torch.rand([4])])

    def forward(self):
        return self.a + 1

torch.jit.script(Mod())
    