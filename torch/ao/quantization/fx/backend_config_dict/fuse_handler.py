from ..fusion_patterns import ModuleReLUFusion
from typing import Callable

# TODO: move ModuleReLUFusion here
def get_fuse_handler_cls():
    return ModuleReLUFusion
