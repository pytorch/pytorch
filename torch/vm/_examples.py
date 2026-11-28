"""The traced run's inputs, as a pass is handed them.

``gm.meta["vm_example_inputs"]`` is an :class:`Examples` while the passes
run. What a pass list makes of a graph is shared by every region with the
same graph, so what a pass makes may depend on a weight's VALUE only if the
pipeline knows: an ``Examples`` says so when it hands a weight out, as
``input_types[i].const`` does, and the output is then made once per set of
weights.

Interface
  ``Examples(items, weights, on_weight)``: ``items`` the inputs; ``weights``
  the positions that are constants of the region (None: not known, every
  tensor counts); ``on_weight`` is called when one of those is handed out.
  ``raw(gm)``: the inputs as a list, nothing said (for the VM's own code,
  which reads sizes and dtypes); ``Examples.weights_read()``: a value
  computed from the inputs is handed out; ``Examples.clear()``: the passes
  are done.
"""

from collections.abc import Callable, Iterator
from typing import Any

import torch
import torch.fx


_HOW = (
    "index it, iterate it or take list(examples) for the tensors; a weight's value is "
    "better read through input_types[i].const, or handed to the lowering with bind="
)


class Examples:
    """The inputs of the traced run, in the order of the graph's
    placeholders: ``examples[i]``, ``len(examples)``, iteration and
    ``list(examples)`` work as on a list. It is no list: handing out a weight
    (a constant of the region) makes what the pass returns the output for
    these weights only, one per set of weights instead of one for every
    region with this graph, so each such region runs the pass list again.
    A pass that only needs a weight's value does better to read
    ``input_types[i].const`` or to hand the weight on with ``bind=``."""

    __slots__ = ("_items", "_on_weight", "_weights")

    def __init__(
        self,
        items: list[Any],
        weights: frozenset[int] | None,
        on_weight: Callable[[], None],
    ) -> None:
        self._items = items
        self._weights = weights
        self._on_weight = on_weight

    def _handed(self, positions: range) -> None:
        if any(
            isinstance(self._items[i], torch.Tensor)
            if self._weights is None
            else i in self._weights
            for i in positions
        ):
            self._on_weight()

    def weights_read(self) -> None:
        """A value computed from the inputs is handed to the pass."""
        self._handed(range(len(self._items)))

    def clear(self) -> None:
        del self._items[:]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, i: int | slice) -> Any:
        if not isinstance(i, (int, slice)):
            raise TypeError(f"the traced run's inputs are indexed by position: {_HOW}")
        n = len(self._items)
        self._handed(range(n)[i] if isinstance(i, slice) else range(n)[i : i + 1 or n])
        return self._items[i]

    def __iter__(self) -> Iterator[Any]:
        self._handed(range(len(self._items)))
        return iter(list(self._items))

    def __deepcopy__(self, memo: dict[int, Any]) -> "Examples":
        # One object for every graph a pass makes of the one it was given.
        return self

    def __add__(self, other: Any) -> Any:
        raise TypeError(f"the traced run's inputs are no list: {_HOW}")

    __radd__ = __add__

    def __repr__(self) -> str:
        return f"<the traced run's {len(self._items)} inputs: {_HOW}>"


def raw(gm: torch.fx.GraphModule) -> list[Any]:
    """The traced run's inputs of ``gm`` while its passes run (else empty),
    with nothing said about what is read."""
    examples = gm.meta.get("vm_example_inputs")
    return examples._items if isinstance(examples, Examples) else list(examples or [])
