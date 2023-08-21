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
TBD
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
