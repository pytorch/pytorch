from __future__ import annotations

from typing import Protocol


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
