from collections import defaultdict, namedtuple
from dataclasses import dataclass
from typing import (
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)

import yaml

from torchgen.executorch.model import ETKernelIndex, ETKernelKey

from torchgen.gen import parse_native_yaml, LineLoader
from torchgen.model import BackendMetadata, DispatchKey, FunctionSchema
from torchgen.utils import (
    NamespaceHelper,
)

# Parse native_functions.yaml into a sequence of NativeFunctions and ET Backend Indices.
ETParsedYaml = namedtuple("ETParsedYaml", ["native_functions", "et_kernel_indices"])

def parse_from_yaml(ei: Dict[str, object]) -> Dict[ETKernelKey, BackendMetadata]:
    """ Given a loaded yaml representing kernel assignment information, extract the
    mapping from `kernel keys` to `BackendMetadata` (the latter representing the kernel instance)

    Args:
        ei: Dict keys {kernels, type_alias, dim_order_alias}
            See ETKernelKey for description of arguments
    """
    e = ei.copy()
    if (kernels := e.pop("kernels", None)) is None:
        return {}

    type_alias = e.pop("type_alias", None)
    dim_order_alias = e.pop("dim_order_alias", None)
    assert type_alias is not None and dim_order_alias is not None, "type_alias and dim_order_alias cannot be None: " + str(ei)

    kernel_mapping: Dict[ETKernelKey, BackendMetadata] = {}

    for entry in kernels:
        arg_meta = entry.get("arg_meta")
        if arg_meta is not None:
            arg_meta.pop('__line__')

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

        kernel_keys = [ETKernelKey((), default=True)] if arg_meta is None else ETKernelKey.gen_from_yaml(arg_meta, type_alias, dim_order_alias)

        for kernel_key in kernel_keys:
            assert kernel_key not in kernel_mapping, "Duplicate kernel key: " + str(kernel_key) + " " + str(e)
            kernel_mapping[kernel_key] = backend_metadata

    return kernel_mapping

def parse_et_yaml_struct(es: object) -> ETKernelIndex:
    """ Given a loaded yaml representing a list of operators, for each op extract the mapping
    of `kernel keys` to `BackendMetadata` (the latter representing the kernel instance
    that should be used by the kernel key).
    """
    indices: Dict[OperatorName, Dict[ETKernelKey, BackendMetadata]] = {}
    for ei in es:
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

def parse_et_yaml(
    path: str,
    tags_yaml_path: str,
    ignore_keys: Optional[Set[DispatchKey]] = None,
    skip_native_fns_gen: bool = False,
) -> ETParsedYaml:
    native_yaml = parse_native_yaml(path, tags_yaml_path, ignore_keys, skip_native_fns_gen=skip_native_fns_gen)

    with open(path, "r") as f:
        es = yaml.load(f, Loader=LineLoader)
    return ETParsedYaml(native_yaml.native_functions, parse_et_yaml_struct(es))
