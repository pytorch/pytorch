import logging
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
number of cache entries that have a ID_MATCH'd guard for a particular nn module
instance. And, (2)nd limit just becomes a safeguard mechanism to have a maximum
compilations for a code object. One way to reason about this is thinking of
TorchDynamo cache as 2 level cache where first level is code -> id_matched_objs
and the second level is id_matched_obj -> linked_list_per_id_matched_obj. Note
that this is only for logical reasoning, our cache is still a linked list.

One important question is - what is the limit the check_fn does not have
ID_MATCH'd object? In that case, we choose the lower of the 2 limits - which is
(1). By doing this, we limit the number of recompilations for functions that do
not have ID_MATCH'd guards but still unsuitable for compilation. In this way,
the two limits resolve the tension mentioned earlier.
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

    # Sources of objects with ID_MATCH guards
    id_guarded_sources: Set[ReferenceType] = field(default_factory=set)

    def __str__(self):
        return f"CacheSize(total={self.total}, per_guarded_obj={tuple(self.per_id_guarded_obj.values())})"


def compute_cache_size(frame, cache_entry):
    # Walk the linked list to calculate the cache size
    cache_size_per_id_matched_obj = defaultdict(int)
    total = 0
    sources = set()
    while cache_entry:
        total += 1
        for source, weak_id in cache_entry.check_fn.id_matched_objs.items():
            if weak_id() is not None:
                cache_size_per_id_matched_obj[weak_id] += 1
                sources.add(source)
        cache_entry = cache_entry.next

    return CacheSize(total, cache_size_per_id_matched_obj, sources)


def _get_weakref_from_f_locals(frame, source):
    obj = frame.f_locals.get(source, None)
    weak_id = None
    try:
        weak_id = weakref.ref(obj)
    except TypeError:
        pass  # cannot weakref bool object
    return weak_id


def is_recompilation(frame, cache_size):
    sources = cache_size.id_guarded_sources

    if not sources:
        return cache_size.total >= 1

    per_id_guarded_obj = cache_size.per_id_guarded_obj
    for source in sources:
        weak_id = _get_weakref_from_f_locals(frame, source)
        if (
            weak_id
            and weak_id in per_id_guarded_obj
            and per_id_guarded_obj[weak_id] >= 1
        ):
            return True
    return False


def exceeds_cache_size(frame, cache_size) -> bool:
    sources = cache_size.id_guarded_sources
    per_id_guarded_obj = cache_size.per_id_guarded_obj
    accumulated_limit = config.accumulated_cache_size_limit
    limit = config.cache_size_limit

    # If there are no id_guarded_obj, we want to limit the number of
    # cache_entries.
    if not sources:
        return cache_size.total >= limit

    for source in sources:
        weak_id = _get_weakref_from_f_locals(frame, source)
        if (
            weak_id
            and weak_id in per_id_guarded_obj
            and per_id_guarded_obj[weak_id] >= limit
        ):
            return True

    # Ensure that total number of cache entries are bounded.
    return cache_size.total >= accumulated_limit
