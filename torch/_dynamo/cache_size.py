import logging
import types
import weakref
from dataclasses import dataclass

from . import config

log = logging.getLogger(__name__)


@dataclass
class CacheSizeRelevantForFrame:
    """
    We track the number of cache entries that have same id_match objects as the
    given frame.

    TODO(janimesh) - Consider adding a map from tuple_of_match_ids to count -
    https://github.com/pytorch/pytorch/pull/107496#discussion_r1304564682 - this
    could be useful for debugging as well.
    """

    # Total number of CacheEntry objects in the Dynamo linked list
    num_cache_entries: int = 0

    # Number of CacheEntry objects having same ID_MATCH'd objects as given frame.
    num_cache_entries_in_bucket: int = 0

    def will_compilation_exceed(self, limit: int) -> bool:
        return self.will_compilation_exceed_bucket(limit)

    def will_compilation_exceed_bucket(self, limit: int) -> bool:
        return self.num_cache_entries_in_bucket >= limit


def _get_weakref_from_f_locals(frame: types.FrameType, local_name: str):
    obj = frame.f_locals.get(local_name, None)
    weak_id = None
    try:
        weak_id = weakref.ref(obj)
    except TypeError:
        pass  # cannot weakref bool object
    return weak_id


def _is_same_cache_bucket(frame: types.FrameType, cache_entry) -> bool:
    """
    Checks if the ID_MATCH'd objects saved on cache_entry are same as the ones
    in frame.f_locals, and if the config hash used to compile the cache entry's
    optimized code is the same as the frame's.
    """
    from .eval_frame import get_saved_else_current_config_hash

    if not cache_entry:
        return False

    if cache_entry.check_fn.config_hash != get_saved_else_current_config_hash():
        return False

    for (
        local_name,
        weakref_from_cache_entry,
    ) in cache_entry.check_fn.id_matched_objs.items():
        if weakref_from_cache_entry() is not None:
            weakref_from_frame = _get_weakref_from_f_locals(frame, local_name)
            if weakref_from_frame != weakref_from_cache_entry:
                return False

    # Also covers the case where no ID_MATCH objects are saved in frame.f_locals
    return True


def compute_cache_size(
    frame: types.FrameType, cache_entry
) -> CacheSizeRelevantForFrame:
    # Walk the linked list to calculate the cache size
    num_cache_entries = 0
    num_cache_entries_in_bucket = 0

    while cache_entry:
        num_cache_entries += 1
        # Track the number of cache entries in the same bucket:
        # 1. having same ID_MATCH'd objects as that of frame.f_locals.
        # 2. having the same config hash as the frame's
        # This will be used later to compare against the
        # cache_size_limit.
        if _is_same_cache_bucket(frame, cache_entry):
            num_cache_entries_in_bucket += 1
        cache_entry = cache_entry.next

    return CacheSizeRelevantForFrame(num_cache_entries, num_cache_entries_in_bucket)


def is_recompilation(cache_size: CacheSizeRelevantForFrame) -> bool:
    """
    If the frame (earlier parsed by compute_cache_size) has more than 1 cache
    entry with same ID_MATCH'd objects, then its a recompilation.
    """
    # Note that you can have multiple entries in the cache but still not a
    # recompile, e.g., you can have 64 nn module instances, each one having an
    # ID_MATCH guard, and each one having just 1 cache entry in the cache.  In
    # this case, we can have 64 entries in the cache, but no recompilation
    # because there is only one entry for each id_matched_obj.
    return cache_size.will_compilation_exceed(1)


def exceeds_cache_size_limit(cache_size: CacheSizeRelevantForFrame) -> bool:
    """
    Checks if we are exceeding the cache size limit.
    """
    return cache_size.will_compilation_exceed(config.cache_size_limit)
