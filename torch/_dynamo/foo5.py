from .exc import (unimplemented_v2)
from . import graph_break_hints

def testing(self):
    unimplemented_v2 (
        gb_type="new graph break",
        context="testing",
        explanation="testing",
        hints=[*graph_break_hints.USER_ERROR],
    )
