import torch

class M(torch.nn.Module):
    def __init__(self):
        super(M, self).__init__()

    def forward(self, a, b):
        c = torch.div(a, b)
        return c

s = torch.jit.script(M())
print(s.graph)
torch.jit.save(s, "file.pt")
print("FINISHED SAVING")
scripted = torch.jit.load("file.pt")
