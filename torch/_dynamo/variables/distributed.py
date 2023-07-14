import collections
import contextlib
import functools
import importlib
import inspect
import itertools
import random
import threading
import types
from typing import Dict, List

import torch.nn

from .. import variables
from ..allowed_functions import is_allowed
from ..exc import unimplemented
from ..guards import GuardBuilder
from ..source import AttrSource, ODictGetItemSource, RandomValueSource
from ..utils import (
    all_hook_names,
    build_checkpoint_variable,
    check_constant_args,
    get_custom_getattr,
    is_namedtuple_cls,
    is_utils_checkpoint,
    istype,
    namedtuple_fields,
    object_has_getattribute,
)
from .base import MutableLocal, VariableTracker
from .ctx_manager import GenericContextWrappingVariable, NullContextVariable
from .dicts import ConstDictVariable


class PlacementVariable(VariableTracker):
    def __init__(self, placement: str):
        super().__init__()
        self.placement = placement
