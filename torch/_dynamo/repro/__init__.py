from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Protocol, TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Generator

    from torch._functorch.fx_minifier import MinifierSanityCheckFailed


log = logging.getLogger(__name__)


@dataclass
class _MinifierSanityResult:
    error: MinifierSanityCheckFailed | None = None

    def raise_if_failed(self, original_failure: Exception | None = None) -> None:
        if self.error is None:
            return
        if original_failure is not None:
            raise original_failure
        raise self.error


@contextmanager
def _minifier_sanity_guard() -> Generator[_MinifierSanityResult, None, None]:
    from torch._functorch.fx_minifier import MinifierSanityCheckFailed

    result = _MinifierSanityResult()
    try:
        yield result
    except MinifierSanityCheckFailed as error:
        log.warning(
            "Minifier could not reproduce the original failure. This can happen "
            "when the failure is nondeterministic or the error filter no longer matches."
        )
        result.error = error


class ReproOptions(Protocol):
    """
    Read-only view of the argparse Namespace threaded through the repro_*
    entrypoints in after_aot.py and after_dynamo.py.

    argparse populates the Namespace dynamically, so a Protocol (rather than a
    concrete class) is the pragmatic fit: it documents the attributes those
    functions actually read without asserting how the Namespace is built. Not
    every attribute is populated for every subcommand -- e.g. after_dynamo
    never sets tracing_mode -- but each repro_* function only reads the subset
    that its own subparser defines.
    """

    # Shared across the after_aot and after_dynamo subparsers.
    command: str
    accuracy: str
    save_dir: str | None

    # after_aot-only attributes.
    tracing_mode: str | None
    check_str: str | None
    is_inference: bool
    isolate: bool
    offload_to_disk: bool
    skip_saving_eager_intermediates: bool
    skip_sanity: bool
    max_granularity: int | None
    stable_hash: bool
    skip_saving_inductor_intermediates: bool
    skip_saving_float64_intermediates: bool
    skip_check_deterministic: bool

    # after_dynamo-only attributes.
    backend: str | None
    autocast: bool
    only_fwd: bool
