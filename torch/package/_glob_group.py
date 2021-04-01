import re
from typing import Iterable, Union

GlobPattern = Union[str, Iterable[str]]


class _GlobGroup:
    def __init__(
        self, include: "GlobPattern", exclude: "GlobPattern", separator: str = "."
    ):
        self._dbg = f"_GlobGroup(include={include}, exclude={exclude})"
        self.include = _GlobGroup._glob_list(include, separator)
        self.exclude = _GlobGroup._glob_list(exclude, separator)
        self.separator = separator

    def __str__(self):
        return self._dbg

    def matches(self, candidate: str) -> bool:
        candidate = self.separator + candidate
        return any(p.fullmatch(candidate) for p in self.include) and all(
            not p.fullmatch(candidate) for p in self.exclude
        )

    @staticmethod
    def _glob_list(elems: "GlobPattern", separator: str = "."):
        if isinstance(elems, str):
            return [_GlobGroup._glob_to_re(elems, separator)]
        else:
            return [_GlobGroup._glob_to_re(e, separator) for e in elems]

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
