import torch
from torch.nn import functional as F
from torch.nn.modules.module import _IncompatibleKeys

class Linear(torch.nn.Linear):
    mask: torch.Tensor
    inplace_masking: bool

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mask = torch.ones(self.weight.shape)
        self.register_buffer('mask', mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight_pre_hook(), self.bias)

    def weight_pre_hook(self):
        return self.weight * self.mask

    def _get_name(self):
        return 'SparseLinear'

    def load_state_dict(self, state_dict, strict=True):
        r"""We allow loading from both sparse and dense layers."""
        if 'mask' not in state_dict:
            state_dict['mask'] = torch.ones(state_dict['weight'].shape)
        return super().load_state_dict(state_dict, strict=strict)

    @classmethod
    def from_dense(cls, dense):
        sparse = cls(dense.in_features, dense.out_features, (dense.bias is not None))
        sparse.load_state_dict(dense.state_dict())
        return sparse
