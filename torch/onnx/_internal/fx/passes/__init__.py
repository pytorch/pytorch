from .decomp import Decompose
from .functionalization import Functionalize, RemoveInputMutation
from .shape_inference import ShapeInferenceWithFakeTensor
from .virtualization import MovePlaceholderToFront, ReplaceGetAttrWithPlaceholder

__all__ = [
    "Decompose",
    "Functionalize",
    "MovePlaceholderToFront",
    "RemoveInputMutation",
    "ReplaceGetAttrWithPlaceholder",
    "ShapeInferenceWithFakeTensor",
]
