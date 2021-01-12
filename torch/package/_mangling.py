"""Import mangling.
See mangling.md for details.
"""
import re

_mangle_index = 0


class PackageMangler:
    """
    Used on import, to ensure that all modules imported have a shared mangle parent.
    """

    def __init__(self):
        global _mangle_index
        self._mangle_index = _mangle_index
        # Increment the global index
        _mangle_index += 1
        # Angle brackets are used so that there is almost no chance of
        # confusing this module for a real module. Plus, it is Python's
        # preferred way of denoting special modules.
        self._mangle_parent = f"<torch_package_{self._mangle_index}>"

    def mangle(self, name) -> str:
        return self._mangle_parent + "." + name

    def demangle(self, mangled) -> str:
        """
        Note: This only demangles names that were mangled by this specific
        PackageMangler. It will pass through names created by a different
        PackageMangler instance.
        """
        if mangled.startswith(self._mangle_parent + "."):
            return mangled.partition(".")[2]

        # wasn't a mangled name
        return mangled

    def parent_name(self):
        return self._mangle_parent


def is_mangled(name: str) -> bool:
    return bool(re.match(r"<torch_package_\d+>", name))


def demangle(name: str) -> str:
    """
    Note: Unlike PackageMangler.demangle, this version works on any
    mangled name, irrespective of which PackageMangler created it.
    """
    return name.partition(".")[2] if is_mangled(name) else name
