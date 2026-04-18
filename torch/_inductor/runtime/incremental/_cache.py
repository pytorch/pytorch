from __future__ import annotations

import hashlib
import re
from collections.abc import Hashable
from typing import TYPE_CHECKING

from ..runtime_utils import triton_config_to_hashable
from ..triton_heuristics import CachingAutotuner

if TYPE_CHECKING:
    from ._launcher import Launcher

# Per-kernel pool of Launcher objects, keyed by config. Two CachingAutotuner
# instances with the same kernel (source + metadata) but overlapping-not-equal
# config sets share Launchers for the configs they have in common; configs
# unique to one autotuner are added to the same pool. Strong references on
# purpose: a converged Launcher's accumulated timing is exactly what we want
# the next autotuner to inherit.
_caching_autotuner_registry: dict[str, dict[Hashable, Launcher]] = {}


def _caching_autotuner_kernel_key(autotuner: CachingAutotuner) -> str | None:
    """Compute a content-based key for a CachingAutotuner's kernel.

    Returns a SHA-256 of the normalized source, inductor metadata, and Triton
    compilation metadata. The config set is intentionally NOT part of the key
    so that autotuners with overlapping config sets can share launchers.

    NOTE: ``str(sorted(meta.items()))`` is used to canonicalize metadata for
    hashing; values must therefore have a deterministic ``__repr__``. Built-in
    types are fine; custom objects whose repr embeds ``id()`` would silently
    make keys non-deterministic and defeat sharing.
    """
    if not hasattr(autotuner.fn, "src"):
        return None

    name: str = autotuner.fn.__name__
    # Whole-word replace so a kernel named ``foo`` doesn't clobber substrings
    # of unrelated identifiers like ``foobar``.
    name_re = re.compile(rf"\b{re.escape(name)}\b")
    normalized_src = name_re.sub("triton_", autotuner.fn.src)
    normalized_inductor = name_re.sub(
        "triton_", str(sorted(autotuner.inductor_meta.items()))
    )
    normalized_triton = name_re.sub(
        "triton_", str(sorted(autotuner.triton_meta.items()))
    )

    hasher = hashlib.sha256()
    hasher.update(normalized_src.encode("utf-8"))
    hasher.update(normalized_inductor.encode("utf-8"))
    hasher.update(normalized_triton.encode("utf-8"))
    return hasher.hexdigest()


def get_launcher_pool(obj: object) -> dict[Hashable, Launcher] | None:
    """Return the shared per-config Launcher pool for ``obj``'s kernel.

    Returns ``None`` if ``obj`` is not a CachingAutotuner or no key can be
    computed (e.g. ``fn`` has no ``src``). Otherwise returns the existing
    pool dict — or a fresh empty one, registered for future callers.
    """
    if (
        isinstance(obj, CachingAutotuner)
        and (key := _caching_autotuner_kernel_key(obj)) is not None
    ):
        return _caching_autotuner_registry.setdefault(key, {})
    return None


def get_or_create_launcher(
    pool: dict[Hashable, Launcher],
    raw_launcher: object,
    launcher_factory: type[Launcher],
) -> Launcher:
    """Return ``pool[cfg_key]`` if present, else build it via ``launcher_factory``."""
    cfg_key = triton_config_to_hashable(raw_launcher.config)  # pyrefly: ignore
    if (launcher := pool.get(cfg_key)) is None:
        launcher = launcher_factory(
            fn=raw_launcher,
            config=raw_launcher.config,  # pyrefly: ignore
            cache_hash=raw_launcher.cache_hash,  # pyrefly: ignore
        )
        pool[cfg_key] = launcher
    return launcher
