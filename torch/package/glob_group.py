# mypy: allow-untyped-defs
import re
from typing import Iterable, Union

GlobPattern = Union[str, Iterable[str]]


class GlobGroup:
    """A set of patterns that candidate strings will be matched against.

    A candidate is composed of a list of segments separated by ``separator``, e.g. "foo.bar.baz".

    A pattern contains one or more segments. Segments can be:
        - A literal string (e.g. "foo"), which matches exactly.
        - A string containing a wildcard (e.g. "torch*", or "foo*baz*"). The wildcard matches
          any string, including the empty string.
        - A double wildcard ("**"). This matches against zero or more complete segments.

    Examples:
        ``torch.**``: matches ``torch`` and all its submodules, e.g. ``torch.nn`` and ``torch.nn.functional``.
        ``torch.*``: matches ``torch.nn`` or ``torch.functional``, but not ``torch.nn.functional``.
        ``torch*.**``: matches ``torch``, ``torchvision``, and all their submodules.

    A candidates will match the ``GlobGroup`` if it matches any of the ``include`` patterns and
    none of the ``exclude`` patterns.

    Args:
        include (Union[str, Iterable[str]]): A string or list of strings,
            each representing a pattern to be matched against. A candidate
            will match if it matches *any* include pattern
        exclude (Union[str, Iterable[str]]): A string or list of strings,
            each representing a pattern to be matched against. A candidate
            will be excluded from matching if it matches *any* exclude pattern.
        separator (str): A string that delimits segments in candidates and
            patterns. By default this is "." which corresponds to how modules are
            named in Python. Another common value for this is "/", which is
            the Unix path separator.
    """

    def __init__(
        self, include: GlobPattern, *, exclude: GlobPattern = (), separator: str = "."
    ):
        self._dbg = f"GlobGroup(include={include}, exclude={exclude})"
        self.include = GlobGroup._glob_list(include, separator)
        self.exclude = GlobGroup._glob_list(exclude, separator)
        self.separator = separator

    def __str__(self):
        return self._dbg

    def __repr__(self):
        return self._dbg

    def matches(self, candidate: str) -> bool:
        candidate = self.separator + candidate
        return any(p.fullmatch(candidate) for p in self.include) and all(
            not p.fullmatch(candidate) for p in self.exclude
        )

    @staticmethod
    def _glob_list(elems: GlobPattern, separator: str = "."):
        if isinstance(elems, str):
            return [GlobGroup._glob_to_re(elems, separator)]
        else:
            return [GlobGroup._glob_to_re(e, separator) for e in elems]

    @staticmethod
    def _glob_to_re(pattern: str, separator: str = "."):
        # to avoid corner cases for the first component, we prefix the candidate string
        # with '.' so `import torch` will regex against `.torch`, assuming '.' is the separator
        def component_to_re(component):
            if "**" in component:
                if component == "**":
                    return "(" + re.escape(separator) + "[^" + separator + "]+)*"
                else:
                    raise ValueError("** can only appear as an entire path segment")
            else:
                return re.escape(separator) + ("[^" + separator + "]*").join(
                    re.escape(x) for x in component.split("*")
                )

        result = "".join(component_to_re(c) for c in pattern.split(separator))
        return re.compile(result)
