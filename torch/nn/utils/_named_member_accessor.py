# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Iterable, List, Tuple

import torch


class NamedMemberAccessor:
    """
    A class that provides a way to access the submodules and parameters/buffers
    of a module. It provides caching mechanism to speed up submodule lookups.
    This is useful for functional programming to manipulate the module state.
    """

    def __init__(self, module: "torch.nn.Module") -> None:
        self.module = module
        self.memo: Dict[str, torch.nn.Module] = {}

    # Nested attribute access

    def get_submodule(self, name: str) -> "torch.nn.Module":
        """
        Return the submodule specified by the given path.
        For example, to get the submodule mod.layer1.conv1,
        use accessor.get_submodule("layer1.conv1")

        Compare to mod.get_submodule("layer1.conv1"), this method will cache the
        intermediate submodule access to speed up future lookups.
        """
        if not name:
            return self.module

        try:
            return self.memo[name]
        except KeyError:
            prefix, dot, attr = name.rpartition(".")
            if dot:
                module = self.get_submodule(prefix)
            else:
                module = self.module
            try:
                submodule = getattr(module, attr)
            except AttributeError as ex:
                raise AttributeError(
                    f"{module._get_name()} has no attribute `{attr}`"
                ) from ex
            if not isinstance(submodule, torch.nn.Module):
                raise TypeError(f"{submodule} is not an instance of torch.nn.Module")
            self.memo[name] = submodule
            return submodule

    def get_tensor(self, name: str) -> torch.Tensor:
        """
        Get the tensor specified by the given path to value.
        For example, to get the attribute mod.layer1.conv1.weight,
        use accessor.get_tensor('layer1.conv1.weight')

        Compare to mod.get_parameter("layer1.conv1.weight"), this method will
        cache the intermediate submodule access to speed up future lookups.
        """
        prefix, _, attr = name.rpartition(".")
        submodule = self.get_submodule(prefix)
        try:
            tensor = getattr(submodule, attr)
        except AttributeError as ex:
            raise AttributeError(
                f"{submodule._get_name()} has no attribute `{name}`"
            ) from ex
        if not isinstance(tensor, torch.Tensor) and tensor is not None:
            raise TypeError(f"{tensor} is not an instance of torch.Tensor")
        return tensor  # type: ignore[return-value]

    def set_tensor(self, name: str, value: torch.Tensor) -> None:
        """
        Set the attribute specified by the given path to value.
        For example, to set the attribute mod.layer1.conv1.weight,
        use accessor.set_tensor("layer1.conv1.weight", value)
        """
        prefix, _, attr = name.rpartition(".")
        setattr(self.get_submodule(prefix), attr, value)

    def del_tensor(self, name: str) -> None:
        """
        Delete the attribute specified by the given path.
        For example, to delete the attribute mod.layer1.conv1.weight,
        use accessor.del_tensor("layer1.conv1.weight")
        """
        prefix, _, attr = name.rpartition(".")
        submodule = self.get_submodule(prefix)
        try:
            delattr(submodule, attr)
        except AttributeError as ex:
            raise AttributeError(
                f"{submodule._get_name()} has no attribute `{name}`"
            ) from ex

    # Batched operations

    def get_tensors(self, names: Iterable[str]) -> List[torch.Tensor]:
        """
        Get the tensors specified by the given paths.
        For example, to get the attributes mod.layer1.conv1.weight and
        mod.layer1.conv1.bias, use accessor.get_tensors(["layer1.conv1.weight",
        "layer1.conv1.bias"])
        """
        return [self.get_tensor(name) for name in names]

    def set_tensors(self, names: Iterable[str], values: Iterable[torch.Tensor]) -> None:
        """
        Set the attributes specified by the given paths to values.
        For example, to set the attributes mod.layer1.conv1.weight and
        mod.layer1.conv1.bias, use accessor.set_tensors(["layer1.conv1.weight",
        "layer1.conv1.bias"], [weight, bias])
        """
        if not isinstance(names, (list, tuple)):
            names = list(names)
        if not isinstance(values, (list, tuple)):
            values = list(values)
        assert len(names) == len(values), "names and values must have the same length"

        for name, value in zip(names, values):
            self.set_tensor(name, value)

    def del_tensors(self, names: Iterable[str]) -> None:
        """
        Delete the attributes specified by the given paths.
        For example, to delete the attributes mod.layer1.conv1.weight and
        mod.layer1.conv1.bias, use accessor.del_tensors(["layer1.conv1.weight",
        "layer1.conv1.bias"])
        """
        for name in names:
            self.del_tensor(name)

    # Shortcut methods

    def named_parameters(
        self,
        remove_duplicate: bool = True,
    ) -> Iterable[Tuple[str, torch.Tensor]]:
        """
        Iterate over all the parameters in the module.
        """
        yield from self.module.named_parameters(remove_duplicate=remove_duplicate)

    def named_buffers(
        self,
        remove_duplicate: bool = True,
    ) -> Iterable[Tuple[str, torch.Tensor]]:
        """
        Iterate over all the buffers in the module.
        """
        yield from self.module.named_buffers(remove_duplicate=remove_duplicate)

    def named_tensors(
        self,
        remove_duplicate: bool = True,
    ) -> Iterable[Tuple[str, torch.Tensor]]:
        """
        Iterate over all the tensors in the module.
        """
        yield from self.module.named_parameters(remove_duplicate=remove_duplicate)
        yield from self.module.named_buffers(remove_duplicate=remove_duplicate)

    def named_modules(
        self,
        remove_duplicate: bool = True,
    ) -> Iterable[Tuple[str, "torch.nn.Module"]]:
        """
        Iterate over all the modules in the module.
        """
        yield from self.module.named_modules(remove_duplicate=remove_duplicate)
