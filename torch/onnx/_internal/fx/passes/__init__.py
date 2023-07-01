from .decomp import Decompose
from .functionalization import Functionalize, RemoveInputMutation
from .modularization import Modularize
from .readability import RestoreParameterAndBufferNames
from .shape_inference import ShapeInferenceWithFakeTensor
from .virtualization import MovePlaceholderToFront, ReplaceGetAttrWithPlaceholder

__all__ = [
    "Decompose",
    "Functionalize",
    "Modularize",
    "MovePlaceholderToFront",
    "RemoveInputMutation",
    "RestoreParameterAndBufferNames",
    "ReplaceGetAttrWithPlaceholder",
    "ShapeInferenceWithFakeTensor",
]
