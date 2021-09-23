"""
This is the centralized file for all PyTorch operator upgraders.
Each function definition here needs to following requirements:

1. The body of the function must be Torchscriptable
2. The naming convention of the upgraders should be:
    <op_name>_upgrader_<old_version>_<new_version>
3. The name of the upgrader must be present in operator_versions.yaml

"""

def div_0_3(self: Tensor, other: Tensor) -> Tensor:
    if (self.is_floating_point() or other.is_floating_point()):
        return self.true_divide(other)
    return self.divide(other, rounding_mode='trunc')

def div_0_3(self: Tensor, other: number) -> Tensor:
    if (self.is_floating_point() or isinstance(other, float)):
        return self.true_divide(other)
    return self.divide(other, rounding_mode='trunc')

def div_0_3(self: number, other: number) -> number:
    return self / other

def div_0_3(self: Tensor, other: Tensor, *, out: Tensor) -> Tensor:
    if (self.is_floating_point() or other.is_floating_point() or out.is_floating_point()):
        return self.true_divide(other, out=out)
    return self.divide(other, rounding_mode='trunc', out=out)

def div__0_3(self: Tensor, other: Tensor) -> Tensor:
    if (self.is_floating_point() or other.is_floating_point()):
        return self.true_divide_(other)
    return self.divide_(other, rounding_mode='trunc')

def div__0_3(self: Tensor, other: number) -> Tensor:
    if (self.is_floating_point() or isinstance(other, float)):
        return self.true_divide_(other)
    return self.divide_(other, rounding_mode='trunc')

def full_0_4(size: List[int], fill_value: number, *, dtype: Optional[int]=None,
             layout: Optional[int]=None, device: Optional[Device]=None,
             pin_memory: Optional[bool]=None) -> Tensor:
    if dtype is None:
        fill_value = float(fill_value)
    return torch.full(size, fill_value, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory)

def full_0_4(size: List[int], fill_value: number, *, out: Tensor) -> Tensor:
    return torch.full(size, fill_value, out=out)
