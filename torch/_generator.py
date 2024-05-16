from __future__ import annotations

from typing import Callable, Optional, Tuple

from typing_extensions import TypeVarTuple, Unpack

import torch

ConstructorArgTs = TypeVarTuple("ConstructorArgTs")


def _rebuild(
    func: Callable[[Unpack[ConstructorArgTs]], Generator],
    args: Tuple[Unpack[ConstructorArgTs]],
    state: Tuple[int, Optional[int], torch.Tensor],
) -> Generator:
    generator = func(*args)
    generator.__setstate__(state)
    return generator


# NB: If you add a new method to Generator, you must update
# torch/_C/__init__.pyi.in to add a type annotation for your method;
# otherwise, it will not show up in autocomplete.
class Generator(torch._C.Generator):
    def __reduce_ex__(
        self,
        proto: int,
    ) -> Tuple[Callable[[Unpack[ConstructorArgTs]], Generator], Tuple[Unpack[ConstructorArgTs]]]:
        func = type(self)
        args = (self.device,)
        state = (
            self.initial_seed(),
            self.get_offset() if self.device.type == "cpu" else None,
            self.get_state(),
        )
        return (_rebuild, (func, args, state))

    def __setstate__(self, state: Tuple[int, Optional[int], torch.Tensor]) -> None:
        seed, offset, state_tensor = state
        if offset is not None:
            self.set_offset(offset)
        self.manual_seed(seed).set_state(state_tensor)
