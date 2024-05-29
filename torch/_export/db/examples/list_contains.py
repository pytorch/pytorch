import torch

from torch._export.db.case import export_case


@export_case(
    example_inputs=(torch.randn(3, 2),),
    tags={"torch.dynamic-shape", "python.data-structure", "python.assert"},
)
class ListContains(torch.nn.Module):
    """
    List containment relation can be checked on a dynamic shape or constants.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        assert x.size(-1) in [6, 2]
        assert x.size(0) not in [4, 5, 6]
        assert "monkey" not in ["cow", "pig"]
        return x + x
