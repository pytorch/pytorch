# represents all kernels used by an Executorch model.
# It maintains a Dict[OperatorName, Dict[ETKernelKey, BackendMetadata]] structure.
from collections import defaultdict, namedtuple
from dataclasses import dataclass
from typing import Dict, Union

from torchgen.model import (
    BackendIndex,
    BackendMetadata,
    DispatchKey,
    NativeFunction,
    NativeFunctionsGroup,
    OperatorName,
)
from torchgen.utils import assert_never

ETParsedYaml = namedtuple("ETParsedYaml", ["native_functions", "kernel_index"])


@dataclass(frozen=True)
class ETKernelKey:
    default: bool = False
    # TODO: jackkhuu to add more fields


@dataclass(frozen=True)
class ETKernelIndex:
    index: Dict[OperatorName, Dict[ETKernelKey, BackendMetadata]]

    def has_kernels(self, g: Union[NativeFunction, NativeFunctionsGroup]) -> bool:
        m = self.get_kernels(g)
        return m is not None

    def get_kernels(
        self, g: Union[NativeFunction, NativeFunctionsGroup]
    ) -> Dict[ETKernelKey, BackendMetadata]:
        if isinstance(g, NativeFunction):
            f = g
        elif isinstance(g, NativeFunctionsGroup):
            f = g.functional
        else:
            assert_never(g)
        if f.func.name not in self.index:
            return {}
        return self.index[f.func.name]

    @staticmethod
    def grow_from_backend_indices(
        kernel_index: Dict[OperatorName, Dict[ETKernelKey, BackendMetadata]],
        backend_indices: Dict[DispatchKey, Dict[OperatorName, BackendMetadata]],
    ) -> None:
        for dk in backend_indices:
            index = backend_indices[dk]
            for op, backend_metadata in index.items():
                if op in kernel_index:
                    kernel_index[op][ETKernelKey(default=True)] = backend_metadata
                else:
                    kernel_index[op] = {ETKernelKey(default=True): backend_metadata}

    @staticmethod
    def from_backend_indices(
        backend_indices: Dict[DispatchKey, Dict[OperatorName, BackendMetadata]]
    ) -> "ETKernelIndex":
        kernel_index: Dict[
            OperatorName, Dict[ETKernelKey, BackendMetadata]
        ] = defaultdict(dict)
        ETKernelIndex.grow_from_backend_indices(kernel_index, backend_indices)
        return ETKernelIndex(kernel_index)

    def grow(
        self, backend_indices: Dict[DispatchKey, Dict[OperatorName, BackendMetadata]]
    ) -> "ETKernelIndex":
        ETKernelIndex.grow_from_backend_indices(self.index, backend_indices)
        return self

    def _to_backend_index(self) -> BackendIndex:
        """
        WARNING: this will be deprecated once all the codegen places know how to handle ETKernelIndex.
        """
        index: Dict[OperatorName, BackendMetadata] = {}
        for op in self.index:
            kernel_dict = self.index[op]
            assert (
                len(kernel_dict.values()) == 1
            ), f"Can't convert ETKernelIndex to BackendIndex because {op} has more than one kernels. Got {kernel_dict}"
            index[op] = kernel_dict[ETKernelKey(default=True)]
        return BackendIndex(
            dispatch_key=DispatchKey.CPU,
            use_out_as_primary=False,
            device_guard=False,
            external=False,
            index=index,
        )
