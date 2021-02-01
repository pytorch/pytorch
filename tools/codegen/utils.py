import re
from typing import Tuple, List, Iterable, Iterator, Callable, Sequence, TypeVar, Optional
from enum import Enum
import contextlib
import textwrap

# Many of these functions share logic for defining both the definition
# and declaration (for example, the function signature is the same), so
# we organize them into one function that takes a Target to say which
# code we want.
Target = Enum('Target', ('DEFINITION', 'DECLARATION', 'REGISTRATION'))

# Matches "foo" in "foo, bar" but not "foobar". Used to search for the
# occurrence of a parameter in the derivative formula
IDENT_REGEX = r'(^|\W){}($|\W)'

# TODO: Use a real parser here; this will get bamboozled
def split_name_params(schema: str) -> Tuple[str, List[str]]:
    m = re.match(r'(\w+)(\.\w+)?\((.*)\)', schema)
    if m is None:
        raise RuntimeError(f'Unsupported function schema: {schema}')
    name, _, params = m.groups()
    return name, params.split(', ')

T = TypeVar('T')
S = TypeVar('S')

# These two functions purposely return generators in analogy to map()
# so that you don't mix up when you need to list() them

# Map over function that may return None; omit Nones from output sequence
def mapMaybe(func: Callable[[T], Optional[S]], xs: Iterable[T]) -> Iterator[S]:
    for x in xs:
        r = func(x)
        if r is not None:
            yield r

# Map over function that returns sequences and cat them all together
def concatMap(func: Callable[[T], Sequence[S]], xs: Iterable[T]) -> Iterator[S]:
    for x in xs:
        for r in func(x):
            yield r

# Conveniently add error context to exceptions raised.  Lets us
# easily say that an error occurred while processing a specific
# context.
@contextlib.contextmanager
def context(msg: str) -> Iterator[None]:
    try:
        yield
    except Exception as e:
        # TODO: this does the wrong thing with KeyError
        msg = textwrap.indent(msg, '  ')
        msg = f'{e.args[0]}\n{msg}' if e.args else msg
        e.args = (msg,) + e.args[1:]
        raise
