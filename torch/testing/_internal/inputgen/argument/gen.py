import math
from typing import Optional, Tuple

import torch
from torch.testing._internal.common_dtype import floating_types, integral_types
from torch.testing._internal.inputgen.argument.engine import MetaArg
from torch.testing._internal.inputgen.variable.gen import VariableGenerator
from torch.testing._internal.inputgen.variable.space import VariableSpace


class TensorGenerator:
    def __init__(
        self, dtype: Optional[torch.dtype], structure: Tuple, space: VariableSpace
    ):
        self.dtype = dtype
        self.structure = structure
        self.space = space

    def gen(self):
        if self.dtype is None:
            return None
        vg = VariableGenerator(self.space)
        min_val = vg.gen_min()
        max_val = vg.gen_max()
        if min_val == float("-inf"):
            min_val = None
        if max_val == float("inf"):
            max_val = None
        # TODO(mcandales): Implement a generator that actually supports any given space
        return self.get_random_tensor(
            size=self.structure, dtype=self.dtype, high=max_val, low=min_val
        )

    def get_random_tensor(self, size, dtype, high=None, low=None):
        high = 100 if high is None else high
        low = -high if low is None else low
        size = tuple(size)
        if dtype == torch.bool:
            if not self.space.contains(0):
                return torch.full(size, True, dtype=dtype)
            else:
                return torch.randint(low=0, high=2, size=size, dtype=dtype)

        if dtype in integral_types():
            low = math.ceil(low)
            high = math.floor(high) + 1
        elif dtype in floating_types():
            low = math.ceil(8 * low)
            high = math.floor(8 * high) + 1
        else:
            raise ValueError(f"Unsupported Dtype: {dtype}")

        if dtype == torch.uint8:
            if not self.space.contains(0):
                return torch.randint(low=max(1, low), high=high, size=size, dtype=dtype)
            else:
                return torch.randint(low=max(0, low), high=high, size=size, dtype=dtype)

        t = torch.randint(low=low, high=high, size=size, dtype=dtype)
        if not self.space.contains(0):
            pos = torch.randint(low=max(1, low), high=high, size=size, dtype=dtype)
            t = torch.where(t == 0, pos, t)

        if dtype in integral_types():
            return t
        if dtype in floating_types():
            return t / 8


class ArgumentGenerator:
    def __init__(self, meta: MetaArg):
        self.meta = meta

    def gen(self):
        if self.meta.optional:
            return None
        elif self.meta.argtype.is_tensor():
            return TensorGenerator(
                dtype=self.meta.dtype,
                structure=self.meta.structure,
                space=self.meta.value,
            ).gen()
        elif self.meta.argtype.is_tensor_list():
            return [
                TensorGenerator(
                    dtype=self.meta.dtype[i],
                    structure=self.meta.structure[i],
                    space=self.meta.value,
                ).gen()
                for i in range(len(self.meta.dtype))
            ]
        else:
            return self.meta.value
