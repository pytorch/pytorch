import torch
from typing import List

from torch.testing import \
    (floating_types, floating_types_and, 
     floating_and_complex_types, floating_and_complex_types_and,
     all_types_and_complex_and, all_types_and_complex)
from torch.testing._internal.common_utils import make_tensor
from .core import OpInfo

def sample_inputs_foreach(self, device, dtype, N, *, noncontiguous=False):
    tensors = [make_tensor((N - i, N - i), device, dtype, noncontiguous=noncontiguous) for i in range(N)]
    return tensors

def get_foreach_method_names(name):
    # get torch inplace reference function
    op_name = "_foreach_" + name
    inplace_op_name = "_foreach_" + name + "_"

    op = getattr(torch, op_name, None)
    inplace_op = getattr(torch, inplace_op_name, None)

    ref = getattr(torch, name, None)
    ref_inplace = getattr(torch.Tensor, name + "_", None)
    return op, inplace_op, ref, ref_inplace


class ForeachFuncInfo(OpInfo):
    """Early version of a specialized OpInfo for foreach functions"""
    def __init__(self,
                 name,
                 dtypes=floating_and_complex_types(),
                 dtypesIfCPU=all_types_and_complex(),
                 dtypesIfCUDA=floating_and_complex_types_and(torch.half),
                 dtypesIfROCM=None,
                 safe_casts_outputs=True,
                 sample_inputs_func=sample_inputs_foreach,
                 **kwargs):
        super().__init__(
            "_foreach_" + name,
            dtypes=dtypes,
            dtypesIfCPU=dtypesIfCPU,
            dtypesIfCUDA=dtypesIfCUDA,
            dtypesIfROCM=dtypesIfROCM,
            safe_casts_outputs=safe_casts_outputs,
            sample_inputs_func=sample_inputs_func,
            **kwargs
        )

        foreach_method, foreach_method_inplace, torch_ref_method, torch_ref_inplace = get_foreach_method_names(name)
        self.method_variant = foreach_method
        self.inplace_variant = foreach_method_inplace
        self.ref = torch_ref_method
        self.ref_inplace = torch_ref_inplace


foreach_unary_op_db: List[OpInfo] = [
    ForeachFuncInfo('exp'),
    ForeachFuncInfo('acos'),
    ForeachFuncInfo('asin'),
    ForeachFuncInfo('atan'),
    ForeachFuncInfo('cos'),
    ForeachFuncInfo('cosh'),
    ForeachFuncInfo('log'),
    ForeachFuncInfo('log10'),
    ForeachFuncInfo('log2'),
    ForeachFuncInfo('tan'),
    ForeachFuncInfo('tanh'),
    ForeachFuncInfo('sin'),
    ForeachFuncInfo('sinh'),

    ForeachFuncInfo(
        'neg',
        dtypes=all_types_and_complex(),
        dtypesIfCPU=all_types_and_complex(),
        dtypesIfCUDA=all_types_and_complex(),
        sample_inputs_func=sample_inputs_foreach,
        safe_casts_outputs=False,
    ),

    ForeachFuncInfo(
        'sqrt',
        dtypes=floating_types(),
        dtypesIfCPU=floating_and_complex_types_and(torch.bfloat16),
        dtypesIfCUDA=floating_and_complex_types_and(torch.half),
    ),

    ForeachFuncInfo(
        'ceil',
        dtypes=floating_types(),
        dtypesIfCPU=floating_types_and(torch.bfloat16),
        dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
    ),

    ForeachFuncInfo(
        'erf',
        dtypes=floating_types(),
        dtypesIfCPU=floating_types_and(torch.bfloat16),
        dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
    ),

    ForeachFuncInfo(
        'erfc',
        dtypes=floating_types(),
        dtypesIfCPU=floating_types_and(torch.bfloat16),
        dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
    ),

    ForeachFuncInfo(
        'expm1',
        dtypes=floating_types(),
        dtypesIfCPU=floating_types_and(torch.bfloat16),
        dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
    ),

    ForeachFuncInfo(
        'floor',
        dtypes=floating_types(),
        dtypesIfCPU=floating_types_and(torch.bfloat16),
        dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
    ),

    ForeachFuncInfo(
        'log1p',
        dtypes=floating_types(),
        dtypesIfCPU=floating_types_and(torch.bfloat16),
        dtypesIfCUDA=floating_types_and(torch.half),
    ),

    ForeachFuncInfo(
        'round',
        dtypes=floating_types(),
        dtypesIfCPU=floating_types_and(torch.bfloat16),
        dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
    ),

    ForeachFuncInfo(
        'frac',
        dtypes=floating_types(),
        dtypesIfCPU=floating_types_and(torch.bfloat16),
        dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
    ),

    ForeachFuncInfo(
        'reciprocal',
        dtypes=floating_types(),
        dtypesIfCPU=floating_types_and(torch.bfloat16),
        dtypesIfCUDA=floating_types_and(torch.half),
    ),

    ForeachFuncInfo(
        'sigmoid',
        dtypes=floating_types(),
        dtypesIfCPU=floating_types_and(torch.bfloat16),
        dtypesIfCUDA=floating_types_and(torch.half),
    ),

    ForeachFuncInfo(
        'trunc',
        dtypes=floating_types(),
        dtypesIfCPU=floating_types_and(torch.bfloat16),
        dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
    ),

    ForeachFuncInfo(
        'abs',
        dtypes=all_types_and_complex_and(torch.bfloat16, torch.half, torch.bool),
        dtypesIfCPU=all_types_and_complex_and(torch.bfloat16, torch.half),
        dtypesIfCUDA=all_types_and_complex_and(torch.bfloat16, torch.half, torch.bool),
        safe_casts_outputs=False,
        supports_forward_ad=True,
    ),
]
