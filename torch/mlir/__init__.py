import torch

import torch_mlir
from torch_mlir.passmanager import PassManager
from torch_mlir.dialects.torch.importer.jit_ir import ClassAnnotator, ModuleBuilder

def export(model, args):
    r"""
    Exports a model into MLIR format. ``model`` should be a 
    :class:`torch.jit.ScriptModule`.

    Args:
        model (torch.jit.ScriptModule): the model to be exported.
        args (tuple of arguments): args to be passed to the model, as if via
            ``model(*args)``.

    Returns:
      A ``torch_mlir.ir.Module`` with the MLIR representation of the model.
    """

    # TODO: Implement more sugar, analogous to what torch.onnx.export does.
    # - Accepting the various formats for `model`:
    #   - torch.nn.Module, torch.jit.ScriptModule or torch.jit.ScriptFunction
    # - Accepting the various formats for `args`:
    #   - just one tensor, a tuple, or also with kwargs.
    # - dynamic_axes
    # - ... any others that are critical.

    # Run the model once to check that the args are valid.
    model(*args)

    annotator = ClassAnnotator()
    annotator.exportNone(model._c._type())
    annotator.exportPath(model._c._type(), ["forward"])
    arg_annotations = [None] # Implicit `self`.
    for arg in args:
        # For now, we assume all argument tensors have value semantics.
        if not isinstance(arg, torch.Tensor):
            arg_annotations.append(None)
            continue
        arg_annotations.append((list(arg.shape), arg.dtype, True))
    annotator.annotateArgs(model._c._type(), ["forward"], arg_annotations)

    mb = ModuleBuilder()
    mb.import_module(model._c, annotator)
    with mb.module.context:
        pm = PassManager.parse("torchscript-module-to-torch-backend-pipeline")
        pm.run(mb.module)
    return mb.module

    
