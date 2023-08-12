import logging
import types
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict

from . import config
from .exc import unimplemented
from .utils import guard_failures, troubleshooting_url

log = logging.getLogger(__name__)
"""
[Note on cache size limit]

TorchDynamo keeps the optimized bytecode resulting from each compilation on the
extra scratch space of the code object. Currently, the cache is implemented as
linked list, where each node contains check_fn and the optimized bytecode.
check_fn is used to find out which code object we are interested in.

This file has logic to track cache size limit. When a cache size limit is
reached, we instruct eval_frame.c to fallback to eager for this frame, as this
frame is friendly to compilation.

There are two different types of cache size limits we are interested in:
1) Total cache size limit per code object - config.cache_size_limit
2) Per-specilized instance cache size limit per code object - config.specialized_instance_cache_size_limit

Lets start with understanding why we need (2). Earlier, we had only one
cache_size_limit, i.e., only (1). But we observed different scenarios where
situations demanded completely different cache size limit. On one hand, we
wanted the total cache size limit to be small because functions that are not
compilation-friendly and recompile on every invocation would take a long time to
burst the cache. But on the other hand, there was a different commonly-occurring
scenario that demanded large cache size limit. Dynamo inserts guards on the
instance of nn module instances (typically seen as a guard on self) to
differentiate between different instances of nn module. This is a quite common
pattern in how models are typically constructed. As a result, this would require
us to have a very large cache_size limit (like 64).

To solve this problem, we introduce a second cache size limit - (2) - which
tracks the cache size limit per specialized instance. Here specialized instance
is represented by a tuple of guarded nn module instances (there could be
multiple guarded nn module instances in the same frame). The tuple could be
empty as well, denoting the absence of any guarded nn module instance.
Therefore, this new limit (2) enables us to effectively reduce the cache size
per specialized instance to a smaller value. With this change, the purpose of
(1) becomes more of a safeguard mechanism where we want to have a max limit of
recompilations per frame.
"""


@dataclass
class CodeObjectCacheSizeTracker:
    """
    Tracks the different cache size limits per code object. See the note at the
    top of the file.
    """

    total_cache_size: int = 0
    specialized_instance_cache_size: defaultdict[int] = field(
        default_factory=lambda: defaultdict(int)
    )


cache_size_tracker: Dict[types.CodeType, CodeObjectCacheSizeTracker] = defaultdict(
    lambda: CodeObjectCacheSizeTracker()
)


def format_func_info(code):
    return f"'{code.co_name}' ({code.co_filename}:{code.co_firstlineno})"


def compute_specialized_instance_cache_key(guarded_nn_modules):
    cache_key = ()
    if guarded_nn_modules:
        # TODO - Ed mentioned about WeakKeyDictionary somewhere, find out why it
        # is needed.
        sorted_nn_modules = [
            guarded_nn_modules[x] for x in sorted(guarded_nn_modules.keys())
        ]
        cache_key = tuple(sorted_nn_modules)
    return cache_key


def erase_cache_size_entry(input_code, tracker, cache_key):
    tracker.specialized_instance_cache_size.clear()
    tracker.total_cache_size = 0
    cache_size_tracker.pop(input_code)



def update_cache_size(guarded_nn_modules, input_code):
    tracker = cache_size_tracker[input_code]
    cache_key = compute_specialized_instance_cache_key(guarded_nn_modules)

    # Check cache size limits
    cache_size_limit = config.cache_size_limit
    if tracker.total_cache_size >= cache_size_limit:
        erase_cache_size_entry(input_code, tracker, cache_key)
        cache_size_limit_reached(
            input_code, "config.cache_size_limit", cache_size_limit
        )

    specialized_instance_cache_size_limit = config.specialized_instance_cache_size_limit
    if (
        tracker.specialized_instance_cache_size[cache_key]
        >= specialized_instance_cache_size_limit
    ):
        erase_cache_size_entry(input_code, tracker, cache_key)
        cache_size_limit_reached(
            input_code,
            "config.specialized_instance_cache_size_limit",
            specialized_instance_cache_size_limit,
        )

    # Update the keys
    tracker.total_cache_size += 1
    tracker.specialized_instance_cache_size[cache_key] += 1


def is_recompilation(guarded_nn_modules, input_code):
    # We can't just rely on total cache size to detect recompilation. If we are
    # compiling the code object for a different specialized instance, we don't
    # want to call it recompilation. Recompilation happens only if we are
    # recompiling for a specialized instance.
    if get_cache_size(input_code) == 0:
        return False

    tracker = cache_size_tracker[input_code]
    cache_key = compute_specialized_instance_cache_key(guarded_nn_modules)

    # cache_key already present in the tracker means we are recompiling
    return cache_key in tracker.specialized_instance_cache_size


def get_cache_size(input_code):
    if input_code in cache_size_tracker:
        return cache_size_tracker[input_code].total_cache_size
    return 0


def cache_size_limit_reached(code, cache_size_limit_type, cache_size_limit):
    def format_guard_failures(code):
        # For the common case, it's sufficient to see just the most recent failure.
        # We could add a verbose mode if needed
        return f"  reasons: {str(guard_failures[code][-1])}\n"

    if config.report_guard_failures:
        assert code in guard_failures, "TODO(whc) any other recompile reasons?"

        log.warning(
            "torch._dynamo hit %s (%s)\n"
            "   function: %s\n"
            "   reasons:  %s\n"
            "to diagnose recompilation issues, see %s.",
            cache_size_limit_type,
            cache_size_limit,
            format_func_info(code),
            format_guard_failures(code),
            troubleshooting_url,
        )
    else:
        log.warning(
            "torch._dynamo hit %s (%s)\n"
            "   function: %s\n"
            "to diagnose recompilation issues, set env variable TORCHDYNAMO_REPORT_GUARD_FAILURES=1"
            " and also see %s.",
            cache_size_limit_type,
            cache_size_limit,
            format_func_info(code),
            troubleshooting_url,
        )
    unimplemented(f"{cache_size_limit_type} reached")
