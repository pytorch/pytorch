r"""Evaluate match expressions, as used by `-k` and `-m`.

The grammar is:

expression: expr? EOF
expr:       and_expr ('or' and_expr)*
and_expr:   not_expr ('and' not_expr)*
not_expr:   'not' not_expr | '(' expr ')' | ident kwargs?

ident:      (\w|:|\+|-|\.|\[|\]|\\|/)+
kwargs:     ('(' name '=' value ( ', ' name '=' value )*  ')')
name:       a valid ident, but not a reserved keyword
value:      (unescaped) string literal | (-)?[0-9]+ | 'False' | 'True' | 'None'

The semantics are:

- Empty expression evaluates to False.
- ident evaluates to True or False according to a provided matcher function.
- or/and/not evaluate according to the usual boolean semantics.
- ident with parentheses and keyword arguments evaluates to True or False according to a provided matcher function.
"""

from __future__ import annotations

import ast
import dataclasses
import enum
import keyword
import re
import types
from typing import Iterator
from typing import Literal
from typing import Mapping
from typing import NoReturn
from typing import overload
from typing import Protocol
from typing import Sequence


__all__ = [
    "Expression",
    "ParseError",
]


class TokenType(enum.Enum):
    LPAREN = "left parenthesis"
    RPAREN = "right parenthesis"
    OR = "or"
    AND = "and"
    NOT = "not"
    IDENT = "identifier"
    EOF = "end of input"
    EQUAL = "="
    STRING = "string literal"
    COMMA = ","


@dataclasses.dataclass(frozen=True)
class Token:
    __slots__ = ("type", "value", "pos")
    type: TokenType
    value: str
    pos: int


class ParseError(Exception):
    """The expression contains invalid syntax.

    :param column: The column in the line where the error occurred (1-based).
    :param message: A description of the error.
    """

    def __init__(self, column: int, message: str) -> None:
        self.column = column
        self.message = message

    def __str__(self) -> str:
        return f"at column {self.column}: {self.message}"


class Scanner:
    __slots__ = ("tokens", "current")

    def __init__(self, input: str) -> None:
        self.tokens = self.lex(input)
        self.current = next(self.tokens)

    def lex(self, input: str) -> Iterator[Token]:
        pos = 0
        while pos < len(input):
            if input[pos] in (" ", "\t"):
                pos += 1
            elif input[pos] == "(":
                yield Token(TokenType.LPAREN, "(", pos)
                pos += 1
            elif input[pos] == ")":
                yield Token(TokenType.RPAREN, ")", pos)
                pos += 1
            elif input[pos] == "=":
                yield Token(TokenType.EQUAL, "=", pos)
                pos += 1
            elif input[pos] == ",":
                yield Token(TokenType.COMMA, ",", pos)
                pos += 1
            elif (quote_char := input[pos]) in ("'", '"'):
                end_quote_pos = input.find(quote_char, pos + 1)
                if end_quote_pos == -1:
                    raise ParseError(
                        pos + 1,
                        f'closing quote "{quote_char}" is missing',
                    )
                value = input[pos : end_quote_pos + 1]
                if (backslash_pos := input.find("\\")) != -1:
                    raise ParseError(
                        backslash_pos + 1,
                        r'escaping with "\" not supported in marker expression',
                    )
                yield Token(TokenType.STRING, value, pos)
                pos += len(value)
            else:
                match = re.match(r"(:?\w|:|\+|-|\.|\[|\]|\\|/)+", input[pos:])
                if match:
                    value = match.group(0)
                    if value == "or":
                        yield Token(TokenType.OR, value, pos)
                    elif value == "and":
                        yield Token(TokenType.AND, value, pos)
                    elif value == "not":
                        yield Token(TokenType.NOT, value, pos)
                    else:
                        yield Token(TokenType.IDENT, value, pos)
                    pos += len(value)
                else:
                    raise ParseError(
                        pos + 1,
                        f'unexpected character "{input[pos]}"',
                    )
        yield Token(TokenType.EOF, "", pos)

    @overload
    def accept(self, type: TokenType, *, reject: Literal[True]) -> Token: ...

    @overload
    def accept(
        self, type: TokenType, *, reject: Literal[False] = False
    ) -> Token | None: ...

    def accept(self, type: TokenType, *, reject: bool = False) -> Token | None:
        if self.current.type is type:
            token = self.current
            if token.type is not TokenType.EOF:
                self.current = next(self.tokens)
            return token
        if reject:
            self.reject((type,))
        return None

    def reject(self, expected: Sequence[TokenType]) -> NoReturn:
        raise ParseError(
            self.current.pos + 1,
            "expected {}; got {}".format(
                " OR ".join(type.value for type in expected),
                self.current.type.value,
            ),
        )


# True, False and None are legal match expression identifiers,
# but illegal as Python identifiers. To fix this, this prefix
# is added to identifiers in the conversion to Python AST.
IDENT_PREFIX = "$"


def expression(s: Scanner) -> ast.Expression:
    if s.accept(TokenType.EOF):
        ret: ast.expr = ast.Constant(False)
    else:
        ret = expr(s)
        s.accept(TokenType.EOF, reject=True)
    return ast.fix_missing_locations(ast.Expression(ret))


def expr(s: Scanner) -> ast.expr:
    ret = and_expr(s)
    while s.accept(TokenType.OR):
        rhs = and_expr(s)
        ret = ast.BoolOp(ast.Or(), [ret, rhs])
    return ret


def and_expr(s: Scanner) -> ast.expr:
    ret = not_expr(s)
    while s.accept(TokenType.AND):
        rhs = not_expr(s)
        ret = ast.BoolOp(ast.And(), [ret, rhs])
    return ret


def not_expr(s: Scanner) -> ast.expr:
    if s.accept(TokenType.NOT):
        return ast.UnaryOp(ast.Not(), not_expr(s))
    if s.accept(TokenType.LPAREN):
        ret = expr(s)
        s.accept(TokenType.RPAREN, reject=True)
        return ret
    ident = s.accept(TokenType.IDENT)
    if ident:
        name = ast.Name(IDENT_PREFIX + ident.value, ast.Load())
        if s.accept(TokenType.LPAREN):
            ret = ast.Call(func=name, args=[], keywords=all_kwargs(s))
            s.accept(TokenType.RPAREN, reject=True)
        else:
            ret = name
        return ret

    s.reject((TokenType.NOT, TokenType.LPAREN, TokenType.IDENT))


BUILTIN_MATCHERS = {"True": True, "False": False, "None": None}


def single_kwarg(s: Scanner) -> ast.keyword:
    keyword_name = s.accept(TokenType.IDENT, reject=True)
    if not keyword_name.value.isidentifier():
        raise ParseError(
            keyword_name.pos + 1,
            f"not a valid python identifier {keyword_name.value}",
        )
    if keyword.iskeyword(keyword_name.value):
        raise ParseError(
            keyword_name.pos + 1,
            f"unexpected reserved python keyword `{keyword_name.value}`",
        )
    s.accept(TokenType.EQUAL, reject=True)

    if value_token := s.accept(TokenType.STRING):
        value: str | int | bool | None = value_token.value[1:-1]  # strip quotes
    else:
        value_token = s.accept(TokenType.IDENT, reject=True)
        if (
            (number := value_token.value).isdigit()
            or number.startswith("-")
            and number[1:].isdigit()
        ):
            value = int(number)
        elif value_token.value in BUILTIN_MATCHERS:
            value = BUILTIN_MATCHERS[value_token.value]
        else:
            raise ParseError(
                value_token.pos + 1,
                f'unexpected character/s "{value_token.value}"',
            )

    ret = ast.keyword(keyword_name.value, ast.Constant(value))
    return ret


def all_kwargs(s: Scanner) -> list[ast.keyword]:
    ret = [single_kwarg(s)]
    while s.accept(TokenType.COMMA):
        ret.append(single_kwarg(s))
    return ret


class MatcherCall(Protocol):
    def __call__(self, name: str, /, **kwargs: str | int | bool | None) -> bool: ...


@dataclasses.dataclass
class MatcherNameAdapter:
    matcher: MatcherCall
    name: str

    def __bool__(self) -> bool:
        return self.matcher(self.name)

    def __call__(self, **kwargs: str | int | bool | None) -> bool:
        return self.matcher(self.name, **kwargs)


class MatcherAdapter(Mapping[str, MatcherNameAdapter]):
    """Adapts a matcher function to a locals mapping as required by eval()."""

    def __init__(self, matcher: MatcherCall) -> None:
        self.matcher = matcher

    def __getitem__(self, key: str) -> MatcherNameAdapter:
        return MatcherNameAdapter(matcher=self.matcher, name=key[len(IDENT_PREFIX) :])

    def __iter__(self) -> Iterator[str]:
        raise NotImplementedError()

    def __len__(self) -> int:
        raise NotImplementedError()


class Expression:
    """A compiled match expression as used by -k and -m.

    The expression can be evaluated against different matchers.
    """

    __slots__ = ("code",)

    def __init__(self, code: types.CodeType) -> None:
        self.code = code

    @classmethod
    def compile(self, input: str) -> Expression:
        """Compile a match expression.

        :param input: The input expression - one line.
        """
        astexpr = expression(Scanner(input))
        code: types.CodeType = compile(
            astexpr,
            filename="<pytest match expression>",
            mode="eval",
        )
        return Expression(code)

    def evaluate(self, matcher: MatcherCall) -> bool:
        """Evaluate the match expression.

        :param matcher:
            Given an identifier, should return whether it matches or not.
            Should be prepared to handle arbitrary strings as input.

        :returns: Whether the expression matches or not.
        """
        ret: bool = bool(eval(self.code, {"__builtins__": {}}, MatcherAdapter(matcher)))
        return ret
