"""Guard installation.

``install_guard`` records guards on the current ``TracingContext`` during
tracing. It lives in ``torch._guards`` (rather than ``torch._dynamo``) because it
writes only to package-owned state (``TracingContext.guards_context``) and has no
dynamo dependency; it is the guard write-path used throughout Dynamo's tracer.
"""

from __future__ import annotations

import logging

import torch._logging
from torch._guards import Guard, TracingContext
from torch._guards.source import is_from_skip_guard_source


guards_log = torch._logging.getArtifactLogger(__name__, "guards")
verbose_guards_log = torch._logging.getArtifactLogger(__name__, "verbose_guards")


def install_guard(*guards: Guard, skip: int = 0) -> None:
    """
    Add dynamo guards to the current tracing context.

    Args:
        guards: guard(s) to add
        skip: number of stack frames to ignore for debug stack trace
    """
    guards_context = TracingContext.get().guards_context
    if guards_context.skip_install:
        return

    collect_debug_stack = guards_log.isEnabledFor(
        logging.DEBUG
    ) or verbose_guards_log.isEnabledFor(logging.DEBUG)
    add = guards_context.dynamo_guards.add
    for guard in guards:
        if not isinstance(guard, Guard):
            raise AssertionError(f"Expected Guard, got {type(guard)}")
        if is_from_skip_guard_source(guard.originating_source):
            continue
        add(guard, collect_debug_stack=collect_debug_stack, skip=skip + 1)
