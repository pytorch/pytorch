import logging
import weakref
from dataclasses import dataclass
from typing import Any, Optional

from torch._guards import CompileId
from . import config
from .types import DynamoFrameType


log: logging.Logger = logging.getLogger(__name__)
"""
[Note on cache size limit]

Background - TorchDynamo cache is a linked list. Each cache entry is a
(guard_manager, out_code, next pointer). These are stored on the f_code's co_extra
scratch space. When a frame is invoked, we walk this linked list and run
guard_manager in each cache_entry to decide if the frame needs recompilation. If none
of the guard_manager's returns True, we recompile and add a new entry. To ensure we
don't end up recompiling infinitely, we put limits on the cache size.

There are two limits
1) recompile_limit
2) accumulated_recompile_limit


Earlier we used to have only limit - maximum number of entries in 1 cache line
(which is now represented by (2) above). So, why do we need two limits? Lets try
to understand that.

In general, we want our cache limit value to be a small number (e.g. 8 or even
lower). This ensures that for frames that cause too many recompilation fall to
eager quickly. However, there is another problem that prevents us from lowering
the value of recompile_limit. This is due to ID_MATCH'd guards. Today, we put
ID_MATCH guards on nn module if there is a graph break. This means we will have
many recompilations for the same code object because the ID_MATCH guard fails
for different instances of the nn module. This is a common pattern in how models
are authored. Therefore, this requires us to keep the recompile_limit high.

We resolve this by introducing these two limits. The first limit (1) limits the
number of cache entries that have an ID_MATCH'd guard for an nn module instance.
And, (2)nd limit becomes a safeguard mechanism to have a maximum compilations
for a code object. One important question is - what is the limit for the code
object that does not have any ID_MATCH guard? For such code objects, we choose
(1) as the cache size limit.

Lets take an example to understand how these limits help. Suppose, we have 16
instances of a nn module and we ID_MATCH on the self object. Further, suppose
the inputs to these functions have varying batch size, leading to one
recompilation. In total, there will be 32 recompilations, and therefore 32 cache
entries on the forward code object. In the older case when we had only 1 limit,
our cache size limit must be >= 32 to capture all these recompilations. Now,
suppose there is a separate function in the same program which is very dynamic
and unsuitable for compilation. Such a function will need to undergo 32
compilations to burst the cache and fallback to eager. These 32 recompilations
are too many and we want to fallback for these compilation-unfriendly functions
sooner.

In the new scenario, we can have (1) recompile_limit = 2, (2)
accumulated_recompile_limit = 32. This means that each ID_MATCH'd object can
have maximum of two cache entries, and the maximum number of cache entries
(irrespective of ID_MATCH obj) is 32. This covers the case of forward code
object which has 32 recompilations. For the other function, the one unsuitable
for recompilation, our limit is 2. So, we will burst the cache in just 2
recompilations. In this manner, these 2 limits help us resolve the tension
mentioned earlier.
"""


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
    num_cache_entries_with_same_id_matched_objs: int = 0

    def will_compilation_exceed(self, limit: int) -> bool:
        # Checks if a compilation will exceed the given limit (that's why >=).
        return (
            self.will_compilation_exceed_accumulated_limit()
            or self.will_compilation_exceed_specific_limit(limit)
        )

    def will_compilation_exceed_accumulated_limit(self) -> bool:
        return self.num_cache_entries >= config.accumulated_recompile_limit

    def will_compilation_exceed_specific_limit(self, limit: int) -> bool:
        return self.num_cache_entries_with_same_id_matched_objs >= limit


def _get_weakref_from_f_locals(
    frame: DynamoFrameType, local_name: str
) -> Optional[weakref.ref[Any]]:
    obj = frame.f_locals.get(local_name, None)
    weak_id = None
    try:
        weak_id = weakref.ref(obj)
    except TypeError:
        pass  # cannot weakref bool object
    return weak_id


def _has_same_id_matched_objs(frame: DynamoFrameType, cache_entry: Any) -> bool:
    """
    Checks if the ID_MATCH'd objects saved on cache_entry are same as the ones
    in frame.f_locals.
    """
    if not cache_entry:
        return False

    for (
        local_name,
        weakref_from_cache_entry,
    ) in cache_entry.guard_manager.id_matched_objs.items():
        if weakref_from_cache_entry() is not None:
            weakref_from_frame = _get_weakref_from_f_locals(frame, local_name)
            if weakref_from_frame is not weakref_from_cache_entry:
                return False

    # Also covers the case where no ID_MATCH objects are saved in frame.f_locals
    return True


def compute_cache_size(
    frame: DynamoFrameType, cache_entry: Any
) -> CacheSizeRelevantForFrame:
    # Walk the linked list to calculate the cache size
    num_cache_entries = 0
    num_cache_entries_with_same_id_matched_objs = 0

    while cache_entry:
        num_cache_entries += 1
        # Track the number of cache entries having same ID_MATCH'd objects as
        # that of frame.f_locals. This will be used later to compare against the
        # recompile_limit.
        if _has_same_id_matched_objs(frame, cache_entry):
            num_cache_entries_with_same_id_matched_objs += 1
        cache_entry = cache_entry.next

    return CacheSizeRelevantForFrame(
        num_cache_entries, num_cache_entries_with_same_id_matched_objs
    )


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


def exceeds_recompile_limit(
    cache_size: CacheSizeRelevantForFrame, compile_id: CompileId
) -> tuple[bool, str]:
    """
    Checks if we are exceeding the cache size limit.
    """
    if cache_size.will_compilation_exceed_accumulated_limit():
        return True, "accumulated_recompile_limit"
    if cache_size.will_compilation_exceed_specific_limit(config.recompile_limit):
        return True, "recompile_limit"
    # NOTE this check is needed in the case that the frame's cache doesn't grow
    # and we keep recompiling. This can happen if the guard guard_manager becomes invalidated,
    # e.g. due to guarded objects being freed. This technically makes the
    # will_compilation_exceed_accumulated_limit check unnecessary, but we will keep the
    # check in case we have a better fix in the future.
    assert compile_id.frame_compile_id is not None
    if compile_id.frame_compile_id >= config.accumulated_recompile_limit:
        return True, "accumulated_recompile_limit"
    return False, ""
