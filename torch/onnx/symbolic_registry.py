import torch
import torch.onnx

import torch.onnx.symbolic_opset9
import torch.onnx.symbolic_opset10

_registry = {}

_symbolic_versions = {
    9 : torch.onnx.symbolic_opset9,
    10 : torch.onnx.symbolic_opset10
}


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
        raise RuntimeError("ONNX export failed. The ONNX domain and/or version to register are None.")
    global _registry
    if not is_registered_version(domain, version):
        _registry[(domain, version)] = {}
    _registry[(domain, version)][opname] = op


def is_registered_op(opname, domain, version):
    if domain is None or version is None:
        raise RuntimeError("ONNX export failed. The ONNX domain and/or version are None.")
    global _registry
    return (domain, version) in _registry and opname in _registry[(domain, version)]


def get_registered_op(opname, domain, version):
    if domain is None or version is None:
        raise RuntimeError("ONNX export failed. The ONNX domain and/or version are None.")
    global _registry
    return _registry[(domain, version)][opname]
