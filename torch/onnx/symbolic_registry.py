import warnings
import importlib
from inspect import getmembers, isfunction
from typing import Dict, Tuple, Any, Union

# The symbolic registry "_registry" is a dictionary that maps operators
# (for a specific domain and opset version) to their symbolic functions.
# An operator is defined by its domain, opset version, and opname.
# The keys are tuples (domain, version), (where domain is a string, and version is an int),
# and the operator's name (string).
# The map's entries are as follows : _registry[(domain, version)][op_name] = op_symbolic
_registry: Dict[Tuple[str, int], Dict] = {}

_symbolic_versions: Dict[Union[int, str], Any] = {}
from torch.onnx.symbolic_helper import _onnx_stable_opsets, _onnx_main_opset
for opset_version in _onnx_stable_opsets + [_onnx_main_opset]:
    module = importlib.import_module("torch.onnx.symbolic_opset{}".format(opset_version))
    _symbolic_versions[opset_version] = module


def register_version(domain, version):
    if not is_registered_version(domain, version):
        global _registry
        _registry[(domain, version)] = {}
    register_ops_in_version(domain, version)


def register_ops_helper(domain, version, iter_version):
    for domain, op_name, op_func in get_ops_in_version(iter_version):
        if not is_registered_op(op_name, domain, version):
            register_op(op_name, op_func, domain, version)


def register_ops_in_version(domain, version):
    # iterates through the symbolic functions of
    # the specified opset version, and the previous
    # opset versions for operators supported in
    # previous versions.

    # Opset 9 is the base version. It is selected as the base version because
    #   1. It is the first opset version supported by PyTorch export.
    #   2. opset 9 is more robust than previous opset versions. Opset versions like 7/8 have limitations
    #      that certain basic operators cannot be expressed in ONNX. Instead of basing on these limitations,
    #      we chose to handle them as special cases separately.
    # Backward support for opset versions beyond opset 7 is not in our roadmap.

    # For opset versions other than 9, by default they will inherit the symbolic functions defined in
    # symbolic_opset9.py.
    # To extend support for updated operators in different opset versions on top of opset 9,
    # simply add the updated symbolic functions in the respective symbolic_opset{version}.py file.
    # Checkout topk in symbolic_opset10.py, and upsample_nearest2d in symbolic_opset8.py for example.
    iter_version = version
    while iter_version != 9:
        register_ops_helper(domain, version, iter_version)
        if iter_version > 9:
            iter_version = iter_version - 1
        else:
            iter_version = iter_version + 1

    register_ops_helper(domain, version, 9)


def get_ops_in_version(version):
    members = getmembers(_symbolic_versions[version])
    domain_opname_ops = []
    for obj in members:
        if isinstance(obj[1], type) and hasattr(obj[1], "domain"):
            ops = getmembers(obj[1], predicate=isfunction)
            for op in ops:
                domain_opname_ops.append((obj[1].domain, op[0], op[1]))  # type: ignore[attr-defined]

        elif isfunction(obj[1]):
            if obj[0] == "_len":
                obj = ("len", obj[1])
            if obj[0] == "_list":
                obj = ("list", obj[1])
            if obj[0] == "_any":
                obj = ("any", obj[1])
            if obj[0] == "_all":
                obj = ("all", obj[1])
            domain_opname_ops.append(("", obj[0], obj[1]))
    return domain_opname_ops

def is_registered_version(domain, version):
    global _registry
    return (domain, version) in _registry


def register_op(opname, op, domain, version):
    if domain is None or version is None:
        warnings.warn("ONNX export failed. The ONNX domain and/or version to register are None.")
    global _registry
    if not is_registered_version(domain, version):
        _registry[(domain, version)] = {}
    _registry[(domain, version)][opname] = op


def is_registered_op(opname, domain, version):
    if domain is None or version is None:
        warnings.warn("ONNX export failed. The ONNX domain and/or version are None.")
    global _registry
    return (domain, version) in _registry and opname in _registry[(domain, version)]

def unregister_op(opname, domain, version):
    global _registry
    if is_registered_op(opname, domain, version):
        del _registry[(domain, version)][opname]
        if not _registry[(domain, version)]:
            del _registry[(domain, version)]
    else:
        warnings.warn("The opname " + opname + " is not registered.")

def get_op_supported_version(opname, domain, version):
    iter_version = version
    while iter_version <= _onnx_main_opset:
        ops = [(op[0], op[1]) for op in get_ops_in_version(iter_version)]
        if (domain, opname) in ops:
            return iter_version
        iter_version += 1
    return None

def get_registered_op(opname, domain, version):
    if domain is None or version is None:
        warnings.warn("ONNX export failed. The ONNX domain and/or version are None.")
    global _registry
    if not is_registered_op(opname, domain, version):
        raise UnsupportedOperatorError(domain, opname, version)
    return _registry[(domain, version)][opname]

class UnsupportedOperatorError(RuntimeError):
    def __init__(self, domain, opname, version):
        supported_version = get_op_supported_version(opname, domain, version)
        if domain in ["", "aten", "prim", "quantized"]:
            msg = f"Exporting the operator {domain}::{opname} to ONNX opset version {version} is not supported. "
            if supported_version is not None:
                msg += (f"Support for this operator was added in version {supported_version}, "
                        "try exporting with this version.")
            else:
                msg += "Please feel free to request support or submit a pull request on PyTorch GitHub."
        else:
            msg = (f"ONNX export failed on an operator with unrecognized namespace {domain}::{opname}. "
                   "If you are trying to export a custom operator, make sure you registered "
                   "it with the right domain and version.")
        super().__init__(msg)
