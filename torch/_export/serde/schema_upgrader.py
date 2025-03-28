import copy
import torch.utils._pytree as pytree
from torch._export.serde.union import _Union
from dataclasses import is_dataclass, fields
from typing import Any

UPGRADE_FN_MAP = {}

def upgraded_from(from_cls: Any):
    '''
    Decorator to register an upgraded cls for a target_cls.
    '''
    def wrapper(upgraded_cls: Any):
        global UPGRADE_MAP
        assert hasattr(upgraded_cls, "upgrade")
        assert from_cls not in UPGRADE_FN_MAP, f"upgrade function for class has been registered: {from_cls}. Registered upgrade fn {UPGRADE_FN_MAP[from_cls]}."
        UPGRADE_FN_MAP[from_cls] = upgraded_cls.upgrade
        print("registering", from_cls)
        return upgraded_cls
    return wrapper

# NOTE [Traversing schema upgrader]
# We choose a top-down upgrade apporach, the implication is that the upgrade function
# 1. take an old schema as input
# 2. return a partially upgraded new schema object whose recursive field hasn't
#    been udpated after upgrade function returns.
def upgrade_to_latest_recursive(root: Any):
    root = copy.deepcopy(root)

    def _upgrade_to_latest(obj: Any):
        latest_obj = obj
        while type(latest_obj) in UPGRADE_FN_MAP:
            latest_obj = UPGRADE_FN_MAP[type(latest_obj)](latest_obj)
        return latest_obj

    def _need_recurse(obj: Any):
        return is_dataclass(obj)

    def _maybe_filter_fields(obj: Any):
        if isinstance(obj, _Union):
            return tuple(field for field in fields(obj) if field.name == obj._type)
        else:
            return fields(obj)

    root = _upgrade_to_latest(root)

    if _need_recurse(root):
        for field in _maybe_filter_fields(root):
            field_name = field.name
            val = getattr(root, field_name)
            new_val = pytree.tree_map(upgrade_to_latest_recursive, val)
            setattr(root, field_name, new_val)

    return root
