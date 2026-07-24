"""AOT builder for index_add: the scatter_add TMA kernel, re-exported.

``index_add(x, 0, index, source)`` with alpha=1 IS
``scatter_add(x, 0, index.unsqueeze(-1).expand_as(source), source)``,
and the TMA kernel's ABI already takes the index as a 1D tensor (the
expansion is pattern-matched away at the host layer), so index_add's
natural arguments map onto it directly. One kernel body serves two
aten ops; index bounds checks (trap_if_oob) come with it.

Only the artifact prefix changes: it names the exported C symbols, and
two declarations cannot share a prefix (duplicate symbols at link).
"""

from ..scatter_add.tma_kernel import build as _tma_build


def build(spec: dict) -> dict:
    b = _tma_build(spec)
    # A silently un-renamed prefix would export duplicate C symbols and
    # fail at link time, far from this cause.
    assert b["prefix"].startswith("scatter_add_tma"), b["prefix"]  # noqa: S101
    b["prefix"] = b["prefix"].replace("scatter_add_tma", "index_add_tma")
    return b
