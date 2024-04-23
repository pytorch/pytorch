import torch._C._lazy


def render_ir_graph(tensors):
    """Return a text dump of the LTC IR graph in dot format for the tensors.
    The text can be processed by tools like dot to be rendered in pdf,png etc."""
    return torch._C._lazy._get_tensors_dot(tensors)


def dump_ir(tensors, ir_format):
    """Return a dump of the tensors in the specified format.
    Valid format are
    - text: for LTC IR
    - backend: for the activate backend IR
    """
    if ir_format == "text":
        return torch._C._lazy._get_tensors_text(tensors)
    elif ir_format == "backend":
        return torch._C._lazy._get_tensors_backend(tensors)
    else:
        raise RuntimeError(f"Unrecognized IR format: {ir_format}")
