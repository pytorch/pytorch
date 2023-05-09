from .decomp import Decompose
from .functionalization import Functionalize, RemoveInputMutation
from .fx_to_onnxscript import export_fx_to_onnxscript
from .modularization import Modularize
from .shape_inference import ShapeInferenceWithFakeTensor
from .virtualization import MovePlaceholderToFront, ReplaceGetAttrWithPlaceholder

__all__ = [
    "export_fx_to_onnxscript",
    "Decompose",
    "Functionalize",
    "Modularize",
    "MovePlaceholderToFront",
    "RemoveInputMutation",
    "ReplaceGetAttrWithPlaceholder",
    "ShapeInferenceWithFakeTensor",
]
