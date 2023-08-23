import logging
import types
import weakref
from collections import defaultdict
from dataclasses import dataclass, field
from typing import DefaultDict, Set
from weakref import ReferenceType

from . import config

log = logging.getLogger(__name__)
"""
[Note on cache size limit]

Background - TorchDynamo cache is a linked list. Each cache entry is a
(check_fn, out_code, next pointer). These are stored on the f_code's co_extra
scratch space. When a frame is invoked, we walk this linked list and run
check_fn in each cache_entry to decide if the frame needs recompilation. If none
of the check_fn's returns True, we recompile and add a new entry. To ensure we
don't end up recompiling infinitely, we put limits on the cache size.

There are two limits
1) cache_size_limit
2) accumulated_cache_size_limit


Earlier we used to have only limit - maximum number of entries in 1 cache line
(which is now represented by (2) above). So, why do we need two limits? Lets try
to understand that.

In general, we want our cache limit value to be a small number (e.g. 8 or even
lower). This ensures that for frames that cause too many recompilation fall to
eager quickly. However, there is another problem that prevents us from lowering
the value of cache_size_limit. This is due to ID_MATCH'd guards. Today, we put
ID_MATCH guards on nn module if there is a graph break. This means we will have
many recompilations for the same code object because the ID_MATCH guard fails
for different instances of the nn module. This is a common pattern in how models
are authored. Therefore, this requires us to keep the cache_size_limit high.

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

In the new scenario, we can have (1) cache_size_limit = 2, (2)
accumulated_cache_size_limit = 32. This means that each ID_MATCH'd object can
have maximum of two cache entries, and the maximum number of cache entries
(irrespective of ID_MATCH obj) is 32. This covers the case of forward code
object which has 32 recompilations. For the other function, the one unsuitable
for recompilation, our limit is 2. So, we will burst the cache in just 2
recompilations. In this manner, these 2 limits help us resolve the tension
mentioned earlier.
"""


@dataclass
class CacheSize:
    """
    Tracks the different cache size limits per code object. See the note at the
    top of the file.
    """

    # Total number of CacheEntry objects in the Dynamo linked list
    total: int = 0

    # Number of CacheEntry objects for an object with ID_MATCH guard.
    per_id_guarded_obj: DefaultDict[ReferenceType, int] = field(
        default_factory=defaultdict(int)
    )

    # local_names of objects with ID_MATCH guards
    id_matched_local_names: Set[str] = field(default_factory=set)


def compute_cache_size(frame: types.FrameType, cache_entry) -> CacheSize:
    # Walk the linked list to calculate the cache size
    cache_size_per_id_matched_obj = defaultdict(int)
    total = 0
    local_names = set()
    while cache_entry:
        total += 1
        for local_name, weak_id in cache_entry.check_fn.id_matched_objs.items():
            if weak_id() is not None:
                cache_size_per_id_matched_obj[weak_id] += 1
                local_names.add(local_name)
        cache_entry = cache_entry.next

    return CacheSize(total, cache_size_per_id_matched_obj, local_names)


def _get_weakref_from_f_locals(frame: types.FrameType, local_name: str):
    obj = frame.f_locals.get(local_name, None)
    weak_id = None
    try:
        weak_id = weakref.ref(obj)
    except TypeError:
        pass  # cannot weakref bool object
    return weak_id


def is_recompilation(frame: types.FrameType, cache_size: CacheSize) -> bool:
    local_names = cache_size.id_matched_local_names

    # If there is no ID_MATCH guard, just check the total cache size
    if not local_names:
        return cache_size.total >= 1

    per_id_guarded_obj = cache_size.per_id_guarded_obj
    for local_name in local_names:
        weak_id = _get_weakref_from_f_locals(frame, local_name)
        if (
            weak_id
            and weak_id in per_id_guarded_obj
            and per_id_guarded_obj[weak_id] >= 1
        ):
            return True
    return False


def exceeds_cache_size(frame: types.FrameType, cache_size: CacheSize) -> bool:
    local_names = cache_size.id_matched_local_names
    per_id_guarded_obj = cache_size.per_id_guarded_obj
    accumulated_limit = config.accumulated_cache_size_limit
    limit = config.cache_size_limit

    # If there are no id_guarded_obj, we want to limit the number of
    # cache_entries.
    if not local_names:
        return cache_size.total >= limit

    for local_name in local_names:
        weak_id = _get_weakref_from_f_locals(frame, local_name)
        if (
            weak_id
            and weak_id in per_id_guarded_obj
            and per_id_guarded_obj[weak_id] >= limit
        ):
            return True

    # Ensure that total number of cache entries are bounded.
    return cache_size.total >= accumulated_limit
