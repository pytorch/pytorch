# Profile-guided optimization for Dynamo

from __future__ import annotations

import os
import os.path
from typing import TYPE_CHECKING
import functools
import pickle
import logging

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from torch._dynamo.variables.builder import FrameStateSizeEntry

# TODO: an individual cache file per code object may not be optimal from an IO
# perspective

@functools.lru_cache(None)
def _get_code_object_cache_path(filename, firstlineno, name):
    # TODO: fix inductor layering problem
    from torch._inductor.runtime.runtime_utils import cache_dir
    from torch._inductor.codecache import sha256_hash

    # TODO: pickle is too heavy a hammer, do not actually need it
    # TODO: this scheme makes manual inspection of cache entries difficult,
    # consider adding some breadcrumbs in the name for ease of use
    r = os.path.join(cache_dir(), "dynamo", sha256_hash(pickle.dumps((filename, firstlineno, name))))

    log.debug("get_code_object_cache_path %s %s %s = %s", filename, firstlineno, name, r)
    return r

def get_code_object_cache_path(tx):
    return _get_code_object_cache_path(tx.f_code.co_filename, tx.f_code.co_firstlineno, tx.f_code.co_name)

# NB: this INTENTIONALLY does not get updated when you write same process, we
# want to KEEP writing from the same process
@functools.lru_cache(None)
def _get_code_object_cache(filename, firstlineno, name):
    # TODO: this try-catch seems to have scope that's too long
    path = _get_code_object_cache_path(filename, firstlineno, name)
    try:
        with open(path, 'rb') as f:
            try:
                r = pickle.load(f)
            except Exception:
                log.warning("get_code_object_cache failed while reading %s", path)
                raise
            log.info("get_code_object_cache hit len(frame_state)=%s", len(r))
            log.debug("get_code_object_cache %s", r)
            return r
    except OSError:
        return None

def get_code_object_cache(tx):
    return _get_code_object_cache(tx.f_code.co_filename, tx.f_code.co_firstlineno, tx.f_code.co_name)

def get_automatic_dynamic_initial_frame_state(tx, name: str) -> FrameStateSizeEntry:
    if frame_state := get_code_object_cache(tx):
        r = frame_state.get(name)
        if r is not None:
            log.debug("get_automatic_dynamic_initial_frame_state %s = %s", name, r)
        return r
    return None

def put_automatic_dynamic_frame_state(tx, frame_state: object) -> None:
    # Do nothing if profile already exists (policy decision!)
    if get_code_object_cache(tx) is not None:
        return

    path = get_code_object_cache_path(tx)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        log.info("put_code_object_cache len(frame_state)=%s", len(frame_state))
        log.debug("put_code_object_cache %s", frame_state)
        pickle.dump(frame_state, f)
