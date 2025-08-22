import gc
import types
import typing
import weakref

import torch


# Things we never want to flag as leaks
_SKIP_TYPES = (
    types.FrameType,
    types.GeneratorType,
    types.FunctionType,
    types.MethodType,
    types.CodeType,
    types.ModuleType,
)


def _is_tracked_fake(obj: typing.Any) -> bool:
    return isinstance(obj, torch.fx.experimental.symbolic_shapes.TrackedFake)


def _is_gm_meta_like_dict(d: dict) -> bool:
    # Hope gm.meta was a custom dict we can assert on
    return "val" in d


def _dict_is_attr_of_tracked_fake(d: dict) -> bool:
    """
    Python 3.10 quirk: sometimes the referrer is obj.__dict__ instead of obj.
    Check if this dict is exactly the __dict__ of a TrackedFake.
    """
    for parent in gc.get_referrers(d):
        if (
            hasattr(parent, "__dict__")
            and parent.__dict__ is d
            and _is_tracked_fake(parent)
        ):
            return True
    return False


def find_legit_leaks_from_referrers(active_fakes: weakref.WeakSet) -> weakref.WeakSet:
    legit_leak: weakref.WeakSet = weakref.WeakSet()

    for act in active_fakes:
        # Track by id to avoid processing duplicate referrers
        seen = set()
        # Assume it's a leak unless we find only ignorable referrers
        flagged = False

        for r in gc.get_referrers(act):
            rid = id(r)
            if rid in seen:
                continue
            seen.add(rid)

            # Fast-path: skip obvious non-owners
            if r is globals() or r is locals():
                continue
            if isinstance(r, _SKIP_TYPES):
                continue
            if _is_tracked_fake(r):
                # TrackedFake should be ignored
                continue

            # Handle dicts carefully (Python 3.10 sometimes shows __dict__)
            if isinstance(r, dict):
                if _is_gm_meta_like_dict(r):
                    continue
                if _dict_is_attr_of_tracked_fake(r):
                    continue
                flagged = True
                break

            # Any other referrer we don't explicitly whitelist counts as a leak
            flagged = True
            break

        if flagged:
            legit_leak.add(act)

    return legit_leak
