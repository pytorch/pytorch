
import torch
from torch.nn import *
class FxModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.load_state_dict(torch.load(r'foo/state_dict.pt'))

    
    
    def forward(self, primals_1, tangents_1):
        cos = torch.ops.aten.cos(primals_1)
        sin = torch.ops.aten.sin(cos);  cos = None
        neg = torch.ops.aten.neg(sin);  sin = None
        mul = torch.ops.aten.mul(tangents_1, neg);  tangents_1 = neg = None
        sin_1 = torch.ops.aten.sin(primals_1);  primals_1 = None
        neg_1 = torch.ops.aten.neg(sin_1);  sin_1 = None
        mul_1 = torch.ops.aten.mul(mul, neg_1);  mul = neg_1 = None
        return [mul_1]
        
