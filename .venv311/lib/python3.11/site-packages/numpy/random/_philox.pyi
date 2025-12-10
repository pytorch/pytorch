from typing import TypedDict, type_check_only

from numpy import uint64
from numpy._typing import _ArrayLikeInt_co
from numpy.random.bit_generator import BitGenerator, SeedSequence
from numpy.typing import NDArray

@type_check_only
class _PhiloxInternal(TypedDict):
    counter: NDArray[uint64]
    key: NDArray[uint64]

@type_check_only
class _PhiloxState(TypedDict):
    bit_generator: str
    state: _PhiloxInternal
    buffer: NDArray[uint64]
    buffer_pos: int
    has_uint32: int
    uinteger: int

class Philox(BitGenerator):
    def __init__(
        self,
        seed: _ArrayLikeInt_co | SeedSequence | None = ...,
        counter: _ArrayLikeInt_co | None = ...,
        key: _ArrayLikeInt_co | None = ...,
    ) -> None: ...
    @property
    def state(
        self,
    ) -> _PhiloxState: ...
    @state.setter
    def state(
        self,
        value: _PhiloxState,
    ) -> None: ...
    def jumped(self, jumps: int = ...) -> Philox: ...
    def advance(self, delta: int) -> Philox: ...
