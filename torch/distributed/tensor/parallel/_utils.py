import warnings
from typing import Tuple, Union

from torch.distributed._tensor import DeviceMesh
from torch.distributed._tensor.placement_types import Placement
from torch.distributed.device_mesh import _mesh_resources
from torch.export.unflatten import InterpreterModule
try:
    from torch._dynamo.external_utils import is_compiling as is_torchdynamo_compiling
except Exception:
    def is_torchdynamo_compiling():  # type: ignore[misc]
        return False

LayoutsType = Union[Placement, Tuple[Placement, ...]]


def _deprecate_warnings(func_name: str, extra_msg: str) -> None:
    """
    Inject common validation logics for `_prepare_input` funcs via this decorator.

    Include verifying that input needs to be either a :class:`Tensor` or :class:`DTensor`
    and only 1D :class:`DeviceMesh` is passed in.
    """
    # TODO: Will follow up with dynamo POC to make warnings.warn working with dynamo.
    if not is_torchdynamo_compiling():
        warnings.warn(f"{func_name} is deprecated and will be removed soon. {extra_msg}")


def _validate_tp_mesh_dim(
    device_mesh: DeviceMesh,
) -> None:
    """
    Check whether TP mesh dimension is valid or not.

    Args:
        device_mesh (:class:`DeviceMesh`):
            The `device_mesh` where we perform
            Tensor Parallelism on.

    Return:
        `True` if the mesh dimension
        is valid, `False` otherwise.
    """
    if device_mesh.ndim > 1:
        raise ValueError(f"Tensor Parallel only accepts a 1D DeviceMesh, but found {device_mesh.ndim}D!"
                         "If you have a 2-D or N-D device_mesh, consider passing in device_mesh[\"tp\"]")

    parent_mesh = _mesh_resources.get_parent_mesh(device_mesh)
    if parent_mesh:
        # if parent_mesh.ndim != 2:
        #     raise RuntimeError(
        #         f"Found TP device_mesh has a parent mesh with dims {parent_mesh.ndim}",
        #         "Currently we only support 2D TP composition with DP.",
        #     )

        tp_mesh_dim = _mesh_resources.get_parent_mesh_dim(device_mesh)
        if tp_mesh_dim != 1:
            raise RuntimeError(
                f"Found TP device_mesh on the {tp_mesh_dim} dimension of its parent mesh.",
                "Currently we only support intranode TP and TP needs to be the innermost dimension on its parent mesh.",
            )

def _check_tp_module_type(module, allowed_type):
    """
    Perform a check on `module` to see if it is of the allowed type.

    Logically similar to running `isinstance(module, allowed_type)` but makes an additional check for module
    of type InterpreterModule (from Pipeline Parallel tracing frontend) to find metadata on the node indicating it
    was of the allowed type before tracing.
    """

    # foo = list(module.graph.nodes)[2]
    # foo.meta
    # [rank0]:(Pdb) [rank0]:(Pdb) [rank0]:{'stack_trace': '  File "/data/users/whc/torchtrain/torchtitan/models/llama/model.py", line 429, in forward\n    h = self.tok_embeddings(tokens)\n', 'nn_module_stack': {'L__self__': ('', 'torchtitan.models.llama.model.Transformer'), 'L__self___tok_embeddings': ('tok_embeddings', 'torch.nn.modules.sparse.Embedding')}, 'source_fn_stack': [('l__self___tok_embeddings', <class 'torch.nn.modules.sparse.Embedding'>)], 'original_aten': <OpOverload(op='aten.embedding', overload='default')>, 'from_node': [('h', 'L__self___tok_embeddings')], 'seq_nr': 437, 'torch_fn': ('embedding_1', 'function.embedding'), 'val': FakeTensor(..., device='meta', size=(4, 2048, 256)), 'tensor_meta': TensorMetadata(shape=torch.Size([4, 2048, 256]), dtype=torch.float32, requires_grad=False, stride=(524288, 256, 1), memory_format=torch.contiguous_format, is_quantized=False, qparams={})}
    # foo.meta['nn_module_stack']
    # [rank0]:(Pdb) [rank0]:{'L__self__': ('', 'torchtitan.models.llama.model.Transformer'), 'L__self___tok_embeddings': ('tok_embeddings', 'torch.nn.modules.sparse.Embedding')}
    # foo.meta.keys()
    # [rank0]:(Pdb) [rank0]:dict_keys(['stack_trace', 'nn_module_stack', 'source_fn_stack', 'original_aten', 'from_node', 'seq_nr', 'torch_fn', 'val', 'tensor_meta'])

    def has_allowed_metadata(module):
        if isinstance(module, InterpreterModule):
            for node in module.graph.nodes:
                if hasattr(node, "meta"):
                    meta = node.meta
                    if 'nn_module_stack' not in meta:
                        continue
                    submod_name, submod_type = list(meta['nn_module_stack'].values())[-1]
                    module, classname = submod_type.rsplit('.', 1)
                    import importlib
                    submod_class = getattr(importlib.import_module(module), classname)
                    ret = issubclass(submod_class, allowed_type)
                    print(f"moduleof type {submod_type} is instance of {allowed_type}: {ret}")
                    # import torch
                    # torch.distributed.breakpoint()
                    # print(1)
                    return ret
    return isinstance(module, allowed_type) or has_allowed_metadata(module)
