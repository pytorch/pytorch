"""
This is the centralized file for all PyTorch operator upgraders.
Each function definition here needs to following requirements:
1. The body of the function must be Torchscriptable
2. The naming convention of the upgraders should be:
    <op_name>_<op_overload>_upgrader_<old_version>_<new_version>
3. The name of the upgrader must be present in operator_versions.yaml
"""
import torch
import yaml
from typing import Union, List, Optional

@torch.jit.script
def div_Tensor_0_3(self: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    if (self.is_floating_point() or other.is_floating_point()):
        return self.true_divide(other)
    return self.divide(other, rounding_mode='trunc')

@torch.jit.script
def div_Scalar_0_3(self: torch.Tensor, other: Union[int, float, complex]) -> torch.Tensor:
    if (self.is_floating_point() or isinstance(other, float)):
        return self.true_divide(other)
    return self.divide(other, rounding_mode='trunc')

# # TODO: not present in the schema
# def div_0_3(self: number, other: number) -> number:
#     return self / other

@torch.jit.script
def div_out_0_3(self: torch.Tensor, other: torch.Tensor, *, out: torch.Tensor) -> torch.Tensor:
    if (self.is_floating_point() or other.is_floating_point() or out.is_floating_point()):
        return self.true_divide(other, out=out)
    return self.divide(other, rounding_mode='trunc', out=out)

@torch.jit.script
def div__Tensor_0_3(self: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    if (self.is_floating_point() or other.is_floating_point()):
        return self.true_divide_(other)
    return self.divide_(other, rounding_mode='trunc')

@torch.jit.script
def div__Scalar_0_3(self: torch.Tensor, other: Union[int, float, complex]) -> torch.Tensor:
    if (self.is_floating_point() or isinstance(other, float)):
        return self.true_divide_(other)
    return self.divide_(other, rounding_mode='trunc')

# TODO: some issues with kwarg
# @torch.jit.script
# def full_names_0_4(size: List[int], fill_value: Union[int, float, complex], *,
#              dtype: Optional[int], layout: Optional[int], device: Optional[torch.device],
#              pin_memory: Optional[bool]) -> torch.Tensor:
#     if dtype is None:
#         fill_value = float(fill_value)
#     return torch.full(size, fill_value, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory)

# @torch.jit.script
# def full_out_0_4(size: List[int], fill_value: Union[int, float, complex], *, out: torch.Tensor) -> torch.Tensor:
#     return torch.full(size, fill_value, out=out)

def format_bytecode(code_table):
    def listify(t):
        return list(map(listify, t)) if isinstance(t, (list, tuple)) else t
    code_table_dict = {}
    for code in code_table:
        code_list = list(code)
        code_table_dict[code_list[0]] = listify(code_list[1:])
    return code_table_dict

yaml_content = [
    {"div_Tensor_0_3": format_bytecode(torch._C.MobileCode(div_Tensor_0_3.graph, "div_Tensor_0_3").bytecode_table())},
    {"div_Scalar_0_3": format_bytecode(torch._C.MobileCode(div_Scalar_0_3.graph, "div_Scalar_0_3").bytecode_table())},
    {"div_out_0_3": format_bytecode(torch._C.MobileCode(div_out_0_3.graph, "div_out_0_3").bytecode_table())},
    {"div__Tensor_0_3": format_bytecode(torch._C.MobileCode(div__Tensor_0_3.graph, "div__Tensor_0_3").bytecode_table())},
    {"div__Scalar_0_3": format_bytecode(torch._C.MobileCode(div__Scalar_0_3.graph, "div__Scalar_0_3").bytecode_table())},
]

stream = open('upgraders.yaml', 'w')
yaml.dump(yaml_content, stream)
