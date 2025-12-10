from collections.abc import Callable, Mapping
from enum import Enum
from typing import Any, Generic, ParamSpec, Self, TypeAlias, overload
from typing import Literal as L

from typing_extensions import TypeVar

__all__ = ["Expr"]

###

_Tss = ParamSpec("_Tss")
_ExprT = TypeVar("_ExprT", bound=Expr)
_ExprT1 = TypeVar("_ExprT1", bound=Expr)
_ExprT2 = TypeVar("_ExprT2", bound=Expr)
_OpT_co = TypeVar("_OpT_co", bound=Op, default=Op, covariant=True)
_LanguageT_co = TypeVar("_LanguageT_co", bound=Language, default=Language, covariant=True)
_DataT_co = TypeVar("_DataT_co", default=Any, covariant=True)
_LeftT_co = TypeVar("_LeftT_co", default=Any, covariant=True)
_RightT_co = TypeVar("_RightT_co", default=Any, covariant=True)

_RelCOrPy: TypeAlias = L["==", "!=", "<", "<=", ">", ">="]
_RelFortran: TypeAlias = L[".eq.", ".ne.", ".lt.", ".le.", ".gt.", ".ge."]

_ToExpr: TypeAlias = Expr | complex | str
_ToExprN: TypeAlias = _ToExpr | tuple[_ToExprN, ...]
_NestedString: TypeAlias = str | tuple[_NestedString, ...] | list[_NestedString]

###

class OpError(Exception): ...
class ExprWarning(UserWarning): ...

class Language(Enum):
    Python = 0
    Fortran = 1
    C = 2

class Op(Enum):
    INTEGER = 10
    REAL = 12
    COMPLEX = 15
    STRING = 20
    ARRAY = 30
    SYMBOL = 40
    TERNARY = 100
    APPLY = 200
    INDEXING = 210
    CONCAT = 220
    RELATIONAL = 300
    TERMS = 1_000
    FACTORS = 2_000
    REF = 3_000
    DEREF = 3_001

class RelOp(Enum):
    EQ = 1
    NE = 2
    LT = 3
    LE = 4
    GT = 5
    GE = 6

    @overload
    @classmethod
    def fromstring(cls, s: _RelCOrPy, language: L[Language.C, Language.Python] = ...) -> RelOp: ...
    @overload
    @classmethod
    def fromstring(cls, s: _RelFortran, language: L[Language.Fortran]) -> RelOp: ...

    #
    @overload
    def tostring(self, /, language: L[Language.C, Language.Python] = ...) -> _RelCOrPy: ...
    @overload
    def tostring(self, /, language: L[Language.Fortran]) -> _RelFortran: ...

class ArithOp(Enum):
    POS = 1
    NEG = 2
    ADD = 3
    SUB = 4
    MUL = 5
    DIV = 6
    POW = 7

class Precedence(Enum):
    ATOM = 0
    POWER = 1
    UNARY = 2
    PRODUCT = 3
    SUM = 4
    LT = 6
    EQ = 7
    LAND = 11
    LOR = 12
    TERNARY = 13
    ASSIGN = 14
    TUPLE = 15
    NONE = 100

class Expr(Generic[_OpT_co, _DataT_co]):
    op: _OpT_co
    data: _DataT_co

    @staticmethod
    def parse(s: str, language: Language = ...) -> Expr: ...

    #
    def __init__(self, /, op: Op, data: _DataT_co) -> None: ...

    #
    def __lt__(self, other: Expr, /) -> bool: ...
    def __le__(self, other: Expr, /) -> bool: ...
    def __gt__(self, other: Expr, /) -> bool: ...
    def __ge__(self, other: Expr, /) -> bool: ...

    #
    def __pos__(self, /) -> Self: ...
    def __neg__(self, /) -> Expr: ...

    #
    def __add__(self, other: Expr, /) -> Expr: ...
    def __radd__(self, other: Expr, /) -> Expr: ...

    #
    def __sub__(self, other: Expr, /) -> Expr: ...
    def __rsub__(self, other: Expr, /) -> Expr: ...

    #
    def __mul__(self, other: Expr, /) -> Expr: ...
    def __rmul__(self, other: Expr, /) -> Expr: ...

    #
    def __pow__(self, other: Expr, /) -> Expr: ...

    #
    def __truediv__(self, other: Expr, /) -> Expr: ...
    def __rtruediv__(self, other: Expr, /) -> Expr: ...

    #
    def __floordiv__(self, other: Expr, /) -> Expr: ...
    def __rfloordiv__(self, other: Expr, /) -> Expr: ...

    #
    def __call__(
        self,
        /,
        *args: _ToExprN,
        **kwargs: _ToExprN,
    ) -> Expr[L[Op.APPLY], tuple[Self, tuple[Expr, ...], dict[str, Expr]]]: ...

    #
    @overload
    def __getitem__(self, index: _ExprT | tuple[_ExprT], /) -> Expr[L[Op.INDEXING], tuple[Self, _ExprT]]: ...
    @overload
    def __getitem__(self, index: _ToExpr | tuple[_ToExpr], /) -> Expr[L[Op.INDEXING], tuple[Self, Expr]]: ...

    #
    def substitute(self, /, symbols_map: Mapping[Expr, Expr]) -> Expr: ...

    #
    @overload
    def traverse(self, /, visit: Callable[_Tss, None], *args: _Tss.args, **kwargs: _Tss.kwargs) -> Expr: ...
    @overload
    def traverse(self, /, visit: Callable[_Tss, _ExprT], *args: _Tss.args, **kwargs: _Tss.kwargs) -> _ExprT: ...

    #
    def contains(self, /, other: Expr) -> bool: ...

    #
    def symbols(self, /) -> set[Expr]: ...
    def polynomial_atoms(self, /) -> set[Expr]: ...

    #
    def linear_solve(self, /, symbol: Expr) -> tuple[Expr, Expr]: ...

    #
    def tostring(self, /, parent_precedence: Precedence = ..., language: Language = ...) -> str: ...

class _Pair(Generic[_LeftT_co, _RightT_co]):
    left: _LeftT_co
    right: _RightT_co

    def __init__(self, /, left: _LeftT_co, right: _RightT_co) -> None: ...

    #
    @overload
    def substitute(self: _Pair[_ExprT1, _ExprT2], /, symbols_map: Mapping[Expr, Expr]) -> _Pair[Expr, Expr]: ...
    @overload
    def substitute(self: _Pair[_ExprT1, object], /, symbols_map: Mapping[Expr, Expr]) -> _Pair[Expr, Any]: ...
    @overload
    def substitute(self: _Pair[object, _ExprT2], /, symbols_map: Mapping[Expr, Expr]) -> _Pair[Any, Expr]: ...
    @overload
    def substitute(self, /, symbols_map: Mapping[Expr, Expr]) -> _Pair: ...

class _FromStringWorker(Generic[_LanguageT_co]):
    language: _LanguageT_co

    original: str | None
    quotes_map: dict[str, str]

    @overload
    def __init__(self: _FromStringWorker[L[Language.C]], /, language: L[Language.C] = ...) -> None: ...
    @overload
    def __init__(self, /, language: _LanguageT_co) -> None: ...

    #
    def finalize_string(self, /, s: str) -> str: ...

    #
    def parse(self, /, inp: str) -> Expr | _Pair: ...

    #
    @overload
    def process(self, /, s: str, context: str = "expr") -> Expr | _Pair: ...
    @overload
    def process(self, /, s: list[str], context: str = "expr") -> list[Expr | _Pair]: ...
    @overload
    def process(self, /, s: tuple[str, ...], context: str = "expr") -> tuple[Expr | _Pair, ...]: ...
    @overload
    def process(self, /, s: _NestedString, context: str = "expr") -> Any: ...  # noqa: ANN401
