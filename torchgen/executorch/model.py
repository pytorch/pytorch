# Represents all kernels used by an Executorch model.
# It maintains a Dict[OperatorName, Dict[ETKernelKey, BackendMetadata]] structure.

from __future__ import annotations

import itertools
from collections import defaultdict, namedtuple
from dataclasses import dataclass
from enum import IntEnum

from torchgen.model import (
    BackendIndex,
    BackendMetadata,
    DispatchKey,
    NativeFunction,
    NativeFunctionsGroup,
    OperatorName,
)
from torchgen.utils import assert_never


KERNEL_KEY_VERSION = 1


# TODO: Duplicated Subset from codegen.tool.gen_oplist, remove declaration in codegen
class ScalarType(IntEnum):
    Byte = 0
    Char = 1
    Short = 2
    Int = 3
    Long = 4
    Float = 6
    Double = 7
    Bool = 11


ETParsedYaml = namedtuple("ETParsedYaml", ["native_functions", "kernel_index"])


@dataclass(frozen=True)
class ETKernelKeyOpArgMeta:
    arg_name: str
    dtype: str
    # The order of the dimensions if entry is a Tensor
    dim_order: tuple[int, ...]

    def to_native_string(self) -> str:
        dtype_str = ScalarType[self.dtype].value
        dim_str = str(self.dim_order)[1:-1].replace(" ", "")
        return f"{dtype_str};{dim_str}"


@dataclass(frozen=True)
class ETKernelKey:
    # Field undefined is default = True
    arg_meta: tuple[ETKernelKeyOpArgMeta, ...] = ()

    # Indicator for this kernel being used as a catch all
    default: bool = False

    version: int = KERNEL_KEY_VERSION

    @staticmethod
    def gen_from_yaml(
        args: dict[str, tuple[str, str]],
        type_alias_map: dict[str, list[str]],  # TODO: Support unwrapped str val
        dim_order_alias_map: dict[str, list[int]],
    ) -> list[ETKernelKey]:
        """Generate ETKernelKeys from arg kernel specs
        Multiple ETKernelKeys are returned due to dtype permutations from utilizing
        type_alias_map (actualizing each potential type permutation as a KernelKey)

        Args:
            args: Mapping from argument name to kernel specs
                Kernel specs are a tuple of (dtype, dim_order).
                Currently tuple entries must be aliased via the alias map arguments
            type_alias_map: Mapping from type alias to potential type enums
                i.e { T0 : [Double, Int] } means T0 can be either Double or Int
                Used for lookup by args
            dim_order_alias_map: Mapping from alias to a list of dimension orders
                Used for lookup by args
        """
        # Cast to dim order to int
        dim_order_alias_map = {
            k: [int(alias) for alias in v] for k, v in dim_order_alias_map.items()
        }
        kernel_keys = []

        # Get all used Dtype Alias
        dtype_alias_used = set()
        for type_alias, dim_order in args.values():
            # Enforce usage of alias initially
            # TODO: Support inlined arguments
            assert type_alias in type_alias_map, "Undefined type alias: " + str(
                type_alias
            )
            assert (
                dim_order in dim_order_alias_map
            ), f"Undefined dim_order alias: {dim_order}"
            dtype_alias_used.add(type_alias)

        # Generate all permutations of dtype alias values
        alias_dtypes = [
            [(alias, dtype) for dtype in type_alias_map[alias]]
            for alias in dtype_alias_used
        ]
        alias_permutations = [
            dict(permutation) for permutation in list(itertools.product(*alias_dtypes))
        ]

        # Using each alias value permutation, generate kernel keys
        op_arg_cache = {}
        for permutation in alias_permutations:
            arg_list = []
            for arg_name, arg_spec in args.items():
                dtype = permutation[arg_spec[0]]
                dim_order = dim_order_alias_map[arg_spec[1]]  # type: ignore[assignment]
                if (
                    cache_key := (arg_name, dtype, tuple(dim_order))
                ) not in op_arg_cache:
                    op_arg_cache[cache_key] = ETKernelKeyOpArgMeta(*cache_key)  # type: ignore[arg-type]

                arg_list.append(op_arg_cache[cache_key])
            kernel_keys.append(ETKernelKey(tuple(arg_list)))

        return kernel_keys

    def to_native_string(self) -> str:
        if self.default:
            return "default"
        return (
            "v"
            + str(KERNEL_KEY_VERSION)
            + "/"
            + "|".join([arg.to_native_string() for arg in self.arg_meta])
        )


@dataclass(frozen=True)
class ETKernelIndex:
    index: dict[OperatorName, dict[ETKernelKey, BackendMetadata]]

    def has_kernels(self, g: NativeFunction | NativeFunctionsGroup) -> bool:
        m = self.get_kernels(g)
        return m is not None

    def get_kernels(
        self, g: NativeFunction | NativeFunctionsGroup
    ) -> dict[ETKernelKey, BackendMetadata]:
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
        kernel_index: dict[OperatorName, dict[ETKernelKey, BackendMetadata]],
        backend_indices: dict[DispatchKey, dict[OperatorName, BackendMetadata]],
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
        backend_indices: dict[DispatchKey, dict[OperatorName, BackendMetadata]],
    ) -> ETKernelIndex:
        kernel_index: dict[OperatorName, dict[ETKernelKey, BackendMetadata]] = (
            defaultdict(dict)
        )
        ETKernelIndex.grow_from_backend_indices(kernel_index, backend_indices)
        return ETKernelIndex(kernel_index)

    def grow(
        self, backend_indices: dict[DispatchKey, dict[OperatorName, BackendMetadata]]
    ) -> ETKernelIndex:
        ETKernelIndex.grow_from_backend_indices(self.index, backend_indices)
        return self

    def _to_backend_index(self) -> BackendIndex:
        """
        WARNING: this will be deprecated once all the codegen places know how to handle ETKernelIndex.
        """
        index: dict[OperatorName, BackendMetadata] = {}
        for op in self.index:
            kernel_dict = self.index[op]
            assert (
                len(kernel_dict.values()) == 1
            ), f"Can't convert ETKernelIndex to BackendIndex because {op} has more than one kernels. Got {kernel_dict}"
            index[op] = kernel_dict.get(
                ETKernelKey(default=True),
                BackendMetadata(kernel="", structured=False, cpp_namespace=""),
            )
        return BackendIndex(
            dispatch_key=DispatchKey.CPU,
            use_out_as_primary=False,
            device_guard=False,
            external=False,
            index=index,
        )

    # Note duplicate ETKernelKey from index_b will clobber the metadata from index_a
    @staticmethod
    def merge_indices(index_a: ETKernelIndex, index_b: ETKernelIndex) -> ETKernelIndex:
        combined = defaultdict(dict, index_a.index.copy())

        for op, entry in index_b.index.items():
            for key, metadata in entry.items():
                combined[op][key] = metadata

        return ETKernelIndex(combined)
