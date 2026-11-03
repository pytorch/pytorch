"""Base types for the CUPTI user-defined-record schema: :class:`Ctype` and :class:`Field`.

Kept in a tiny leaf module (no intra-package imports) so both the generated
``_cupti_stubs`` module and the hand-curated ``records`` module can import it
without a cycle.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from typing_extensions import Self


class Ctype(Enum):
    """How to interpret a field's bytes at decode time. The byte *width* always comes
    from CUPTI's captured record layout (``ppRecordLayouts``); ``Ctype`` only says
    whether those bytes are unsigned/signed/float, or a ``const char*`` to dereference.
    Struct/opaque fields need no ctype -- the decoder skips any field whose layout size
    is not 1/2/4/8."""

    UINT = "u"  # unsigned integer (also the default for enums)
    INT = "i"  # signed integer
    FLOAT = "f"  # float / double
    CSTR = "cstr"  # const char* -- dereferenced to a Python str

    def numpy(self, size: int) -> str:
        """Little-endian numpy dtype string for a numeric ctype at ``size`` bytes."""
        return f"<{self.value}{size}"


class Field(int):
    """A CUPTI user-defined-record field. A ``Field`` *is* its ``CUpti_Activity*FieldIds``
    id -- it subclasses ``int`` -- so it can be used directly anywhere the integer id is
    expected (selection element, column key, set/dict member) without ``.id``/``int()``.
    It additionally carries its :class:`Ctype` for decode."""

    ctype: Ctype

    def __new__(cls, id: int, ctype: Ctype) -> Self:
        self = super().__new__(cls, id)
        self.ctype = ctype
        return self

    @property
    def id(self) -> int:
        """The field id (== ``int(self)``); kept for call sites that read ``.id``."""
        return int(self)

    @property
    def string(self) -> bool:
        """True for ``const char*`` fields (dereferenced to str during decode)."""
        return self.ctype is Ctype.CSTR

    def __repr__(self) -> str:
        return f"Field({int(self)}, {self.ctype.name})"
