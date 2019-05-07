import warnings
import importlib

_registry = {}

_symbolic_versions = {}
from torch.onnx.symbolic_helper import _onnx_stable_opsets
for opset_version in _onnx_stable_opsets:
    module = importlib.import_module('torch.onnx.symbolic_opset{}'.format(opset_version))
    _symbolic_versions[opset_version] = module

def register_version(domain, version):
    if not is_registered_version(domain, version):
        global _registry
        _registry[(domain, version)] = {}
    register_ops_in_version(domain, version)


def register_ops_in_version(domain, version):
    # iterates through the symbolic functions of
    # the specified opset version, and the previous
    # opset versions for operators supported in
    # previous versions
    iter_version = version
    while iter_version >= 9:
        version_ops = get_ops_in_version(iter_version)
        for opname in version_ops:
            if not is_registered_op(opname, domain, version):
                register_op(opname, version_ops[opname], domain, version)
        iter_version = iter_version - 1


def get_ops_in_version(version):
    return _symbolic_versions[version].__dict__


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


def get_registered_op(opname, domain, version):
    if domain is None or version is None:
        warnings.warn("ONNX export failed. The ONNX domain and/or version are None.")
    global _registry
    return _registry[(domain, version)][opname]

def is_registered_custom_version(domain, version):
    while version > 0:
        if is_registered_version(domain, version):
            return True
        version = version - 1
    return False

def is_registered_custom_op(opname, domain, version):
    if domain is None or version is None:
        warnings.warn("ONNX export failed. The ONNX domain and/or version are None.")
    while version > 0:
        if is_registered_op(opname, domain, version):
            return True
        version = version - 1
    return False

def get_registered_custom_op(opname, domain, version):
    if domain is None or version is None:
        warnings.warn("ONNX export failed. The ONNX domain and/or version are None.")
    while version > 0:
        if is_registered_custom_op(opname, domain, version):
            get_registered_op(opname, domain, version)
        version = version - 1
    warnings.warn("ONNX export failed. "
                  "The registered op {} does not exist for domain {} and version {}."
                  .format(opname, domain, version))
