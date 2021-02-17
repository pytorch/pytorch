from typing import Union, Iterable
import re

GlobPattern = Union[str, Iterable[str]]


class _GlobGroup:
    def __init__(self, include: "GlobPattern", exclude: "GlobPattern"):
        self._dbg = f"_GlobGroup(include={include}, exclude={exclude})"
        self.include = _GlobGroup._glob_list(include)
        self.exclude = _GlobGroup._glob_list(exclude)

    def __str__(self):
        return self._dbg

    def matches(self, candidate: str) -> bool:
        if candidate[:5] != ".data":
            candidate = "." + candidate
        return any(p.fullmatch(candidate) for p in self.include) and all(
            not p.fullmatch(candidate) for p in self.exclude
        )

    @staticmethod
    def _glob_list(elems: "GlobPattern"):
        if isinstance(elems, str):
            return [_GlobGroup._glob_to_re(elems)]
        else:
            return [_GlobGroup._glob_to_re(e) for e in elems]

    @staticmethod
    def _glob_to_re(pattern: str):
        # to avoid corner cases for the first component, we prefix the candidate string
        # with '.' so `import torch` will regex against `.torch`
        def component_to_re(component):
            if "**" in component:
                if component == "**":
                    return "(\\.[^.]+)*"
                else:
                    raise ValueError("** can only appear as an entire path segment")
            else:
                return "\\." + "[^.]*".join(re.escape(x) for x in component.split("*"))

        result = "".join(component_to_re(c) for c in pattern.split("."))
        return re.compile(result)
