from __future__ import annotations

from collections import defaultdict, namedtuple
from typing import Any

import yaml

from torchgen.executorch.model import ETKernelIndex, ETKernelKey
from torchgen.gen import LineLoader, parse_native_yaml
from torchgen.model import (
    BackendMetadata,
    DispatchKey,
    FunctionSchema,
    NativeFunction,
    OperatorName,
)
from torchgen.utils import NamespaceHelper


# Parse native_functions.yaml into a sequence of NativeFunctions and ET Backend Indices.
ETParsedYaml = namedtuple("ETParsedYaml", ["native_functions", "et_kernel_indices"])

# Fields in native_functions.yaml used to determine which kernels should be used
ET_FIELDS = ["kernels", "type_alias", "dim_order_alias"]


def parse_from_yaml(ei: dict[str, object]) -> dict[ETKernelKey, BackendMetadata]:
    """Given a loaded yaml representing kernel assignment information, extract the
    mapping from `kernel keys` to `BackendMetadata` (the latter representing the kernel instance)

    Args:
        ei: Dict keys {kernels, type_alias, dim_order_alias}
            See ETKernelKey for description of arguments
    """
    e = ei.copy()
    if (kernels := e.pop("kernels", None)) is None:
        return {}

    type_alias: dict[str, list[str]] = e.pop("type_alias", {})  # type: ignore[assignment]
    dim_order_alias: dict[str, list[str]] = e.pop("dim_order_alias", {})  # type: ignore[assignment]
    dim_order_alias.pop("__line__", None)

    kernel_mapping: dict[ETKernelKey, BackendMetadata] = {}

    for entry in kernels:  # type: ignore[attr-defined]
        arg_meta = entry.get("arg_meta")
        if arg_meta is not None:
            arg_meta.pop("__line__")

        kernel_name = entry.get("kernel_name")
        namespace_helper = NamespaceHelper.from_namespaced_entity(
            kernel_name, max_level=3
        )
        kernel_namespace = namespace_helper.get_cpp_namespace(default="at")
        backend_metadata = BackendMetadata(
            kernel=namespace_helper.entity_name,
            structured=False,
            cpp_namespace=(kernel_namespace + "::native"),
        )

        kernel_keys = (
            [ETKernelKey((), default=True)]
            if arg_meta is None
            else ETKernelKey.gen_from_yaml(arg_meta, type_alias, dim_order_alias)  # type: ignore[arg-type]
        )

        for kernel_key in kernel_keys:
            assert kernel_key not in kernel_mapping, (
                "Duplicate kernel key: " + str(kernel_key) + " " + str(e)
            )
            kernel_mapping[kernel_key] = backend_metadata

    return kernel_mapping


def parse_et_yaml_struct(es: object) -> ETKernelIndex:
    """Given a loaded yaml representing a list of operators, for each op extract the mapping
    of `kernel keys` to `BackendMetadata` (the latter representing the kernel instance
    that should be used by the kernel key).
    """
    indices: dict[OperatorName, dict[ETKernelKey, BackendMetadata]] = {}
    for ei in es:  # type: ignore[attr-defined]
        e = ei.copy()

        funcs = e.pop("func")
        assert isinstance(funcs, str), f"not a str: {funcs}"
        namespace_helper = NamespaceHelper.from_namespaced_entity(
            namespaced_entity=funcs, max_level=1
        )
        opname = FunctionSchema.parse(namespace_helper.entity_name).name

        assert opname not in indices, f"Duplicate func found in yaml: {opname} already"

        if len(index := parse_from_yaml(e)) != 0:
            indices[opname] = index

    return ETKernelIndex(indices)


def extract_kernel_fields(es: object) -> dict[OperatorName, dict[str, Any]]:
    """Given a loaded yaml representing a list of operators, extract the
    kernel key related fields indexed by the operator name.
    """
    fields: dict[OperatorName, dict[str, Any]] = defaultdict(dict)
    for ei in es:  # type: ignore[attr-defined]
        funcs = ei.get("func")
        assert isinstance(funcs, str), f"not a str: {funcs}"
        namespace_helper = NamespaceHelper.from_namespaced_entity(
            namespaced_entity=funcs, max_level=1
        )
        opname = FunctionSchema.parse(namespace_helper.entity_name).name

        for field in ET_FIELDS:
            if (value := ei.get(field)) is not None:
                fields[opname][field] = value

    return fields


def parse_et_yaml(
    path: str,
    tags_yaml_path: str,
    ignore_keys: set[DispatchKey] | None = None,
    skip_native_fns_gen: bool = False,
) -> tuple[list[NativeFunction], dict[OperatorName, dict[str, Any]]]:
    """Parse native_functions.yaml into NativeFunctions and an Operator Indexed Dict
    of fields to persist from native_functions.yaml to functions.yaml
    """
    with open(path) as f:
        es = yaml.load(f, Loader=LineLoader)

    et_kernel = extract_kernel_fields(es)

    # Remove ET specific fields from entries for BC compatibility
    strip_et_fields(es)

    native_yaml = parse_native_yaml(
        path,
        tags_yaml_path,
        ignore_keys,
        skip_native_fns_gen=skip_native_fns_gen,
        loaded_yaml=es,
    )
    return native_yaml.native_functions, et_kernel


def strip_et_fields(es: object) -> None:
    """Given a loaded yaml representing a list of operators,
    remove ET specific fields from every entries for BC compatibility
    """
    for entry in es:  # type: ignore[attr-defined]
        for field in ET_FIELDS:
            entry.pop(field, None)
