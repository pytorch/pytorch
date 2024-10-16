# Profile-guided optimization for Dynamo

from __future__ import annotations

import functools
import logging
import os
import os.path
import pickle
from typing import Dict, Optional, TYPE_CHECKING


log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from torch._dynamo.symbolic_convert import InstructionTranslator
    from torch._dynamo.variables.builder import FrameStateSizeEntry

# TODO: an individual cache file per code object may not be optimal from an IO
# perspective


@functools.lru_cache(None)
def _get_code_object_cache_path(filename: str, firstlineno: int, name: str) -> str:
    # TODO: fix inductor layering problem
    from torch._inductor.codecache import sha256_hash
    from torch._inductor.runtime.runtime_utils import cache_dir

    # TODO: pickle is too heavy a hammer, do not actually need it
    # TODO: this scheme makes manual inspection of cache entries difficult,
    # consider adding some breadcrumbs in the name for ease of use
    r = os.path.join(
        cache_dir(), "dynamo", sha256_hash(pickle.dumps((filename, firstlineno, name)))
    )

    log.debug(
        "get_code_object_cache_path %s %s %s = %s", filename, firstlineno, name, r
    )
    return r


def get_code_object_cache_path(tx: InstructionTranslator) -> str:
    return _get_code_object_cache_path(
        tx.f_code.co_filename, tx.f_code.co_firstlineno, tx.f_code.co_name
    )


# NB: this INTENTIONALLY does not get updated when you write same process, we
# want to KEEP updating the file from the same process as we learn more
# dynamic information
@functools.lru_cache(None)
def _get_code_object_cache(
    filename: str, firstlineno: int, name: str
) -> Optional[Dict[str, FrameStateSizeEntry]]:
    # TODO: this try-catch seems to have scope that's too long
    path = _get_code_object_cache_path(filename, firstlineno, name)
    try:
        with open(path, "rb") as f:
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


def get_code_object_cache(
    tx: InstructionTranslator,
) -> Optional[Dict[str, FrameStateSizeEntry]]:
    return _get_code_object_cache(
        tx.f_code.co_filename, tx.f_code.co_firstlineno, tx.f_code.co_name
    )


def get_automatic_dynamic_initial_frame_state(
    tx: InstructionTranslator, name: str
) -> Optional[FrameStateSizeEntry]:
    if frame_state := get_code_object_cache(tx):
        r = frame_state.get(name)
        if r is not None:
            log.debug("get_automatic_dynamic_initial_frame_state %s = %s", name, r)
        return r
    return None


def put_automatic_dynamic_frame_state(
    tx: InstructionTranslator, frame_state: Dict[str, FrameStateSizeEntry]
) -> None:
    # Do nothing if profile already exists (policy decision!)
    # TODO: if a process crashes midway through compilation while profiling, the profile
    # may be irreparably "corrupted" (i.e., unable to force automatic dynamic
    # as appropriate).  This is a downside of not continuously updating the
    # profiles... not sure if it is worth or not.
    if get_code_object_cache(tx) is not None:
        return

    path = get_code_object_cache_path(tx)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        log.info("put_code_object_cache len(frame_state)=%s", len(frame_state))
        log.debug("put_code_object_cache %s", frame_state)
        pickle.dump(frame_state, f)
