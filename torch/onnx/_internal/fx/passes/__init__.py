from .decomp import Decompose
from .functionalization import Functionalize, RemoveInputMutation
from .modularization import Modularize
from .shape_inference import ShapeInferenceWithFakeTensor
from .virtualization import MovePlaceholderToFront, ReplaceGetAttrWithPlaceholder

__all__ = [
    "Decompose",
    "Functionalize",
    "Modularize",
    "MovePlaceholderToFront",
    "RemoveInputMutation",
    "ReplaceGetAttrWithPlaceholder",
    "ShapeInferenceWithFakeTensor",
]
