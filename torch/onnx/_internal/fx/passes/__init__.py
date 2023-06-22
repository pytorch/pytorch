from .decomp import Decompose
from .functionalization import Functionalize, RemoveInputMutation
from .shape_inference import ShapeInferenceWithFakeTensor
from .type_promotion import ExplicitTypePromotionPass
from .virtualization import MovePlaceholderToFront, ReplaceGetAttrWithPlaceholder

__all__ = [
    "Decompose",
    "ExplicitTypePromotionPass",
    "Functionalize",
    "MovePlaceholderToFront",
    "RemoveInputMutation",
    "ReplaceGetAttrWithPlaceholder",
    "ShapeInferenceWithFakeTensor",
]
