import torch


# ModuleList with elements of type Module
class FooModule(torch.nn.Module):
    pass


class BarModule(torch.nn.Module):
    pass


ml: torch.nn.ModuleList = torch.nn.ModuleList([FooModule(), BarModule()])
ml[0].children() == []  # noqa: B015
reveal_type(ml)  # E: {ModuleList}
