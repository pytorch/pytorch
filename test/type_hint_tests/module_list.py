import torch

# ModuleList with elements of type Module
class BarModule(torch.nn.Module):
    pass

ml: torch.nn.ModuleList = torch.nn.ModuleList([FooModule(), BarModule()])
ml[0].children() == []
