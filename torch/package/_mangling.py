"""
Mangling Imports
================
Why do we do this?
------------------
To avoid accidental name collisions with modules in `sys.modules`. Consider the following:
::
    from torchvision.models import resnet18
    local_resnet18 = resnet18()

    # a loaded resnet18, potentially with a different implementation than the local one!
    i = torch.PackageImporter('my_resnet_18.pt')
    loaded_resnet18 = i.load_pickle('model', 'model.pkl')

    print(type(local_resnet18).__module__)  # 'torchvision.models.resnet18'
    print(type(loaded_resnet18).__module__)  # ALSO 'torchvision.models.resnet18'

These two model types have the same originating ``__module__`` name set.
While this isn't facially incorrect, there are a number of places in
cpython and elsewhere that assume you can take any module name, look it
up ``sys.modules``, and get the right module back, including:
    - `import_from <https://github.com/python/cpython/blob/5977a7989d49c3e095c7659a58267d87a17b12b1/Python/ceval.c#L5524>`_
    - ``inspect``: used in TorchScript to retrieve source code to compile
    - probably more that we don't know about.

In these cases, we may silently pick up the wrong module for ``loaded_resnet18``
and e.g. TorchScript the wrong source code for our model.

How are names mangled?
----------------------
On import, all modules produced by a given ``PackageImporter`` are given a
new top-level module as their parent. This is called the `mangle parent`. For example:
::
    torchvision.models.resnet18
becomes
::
    __torch_package0__.torchvision.models.resnet18

The mangle parent is made unique to a given ``PackageImporter`` instance by
bumping the ``mangle_index``, i.e. ``__torch__package{mangle_index}__``.

It is important to note that this mangling happens `on import`, and the
results are never saved into a package. Assigning mangle parents on import
means that we can enforce that mangle parents are unique within the
environment doing the importing. It also allows us to avoid serializing (and
maintaining backward compatibility for) this detail.

.. note::
    No mangle parents should ever be saved into a package.

Mangled names in PackageExporter
--------------------------------
Occasionally ``PackageExporter`` may encounter mangled names during export. For
example, the user may be re-packaging an object that was imported (and thus
had its module name mangled).

This means that ``PackageExporter`` must demangle all module names before it
writes them to the package. This is needed in two places:
    - When converting the module name to a filename to for placing source code in the package.
    - When saving the module name in the pickle GLOBAL opcode.
"""
import re
from typing import Dict

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
        self._mangle_parent = f"__torch_package_{self._mangle_index}__"

    def mangle(self, name):
        return self._mangle_parent + "." + name

    def demangle(self, mangled):
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


class PackageDemangler:
    """
    Used on export, to ensure that we are only writing demangled names to the pacakge.
    """

    def __init__(self):
        # Map of demangled name => original name
        # This is to detect name collisions when you are trying to save something like:
        #    foo.bar => foo.bar
        #    __torch0__.foo.bar => foo.bar   # collision!
        #    __torch1__.foo.bar => foo.bar   # collision!
        self.demangled_names: Dict[str, str] = {}

    def demangle(self, name):
        """
        Note: Unlike PackageMangler.demangle, this version work on any
        mangled name, irrespective of which PackageMangler created it.
        """
        if not _was_mangled(name):
            self.demangled_names[name] = name
            return name

        demangled = name.partition(".")[2]
        # Check for whether this demangled name collides with a previously saved module.
        existing_name = self.demangled_names.get(demangled)
        if existing_name is None:
            self.demangled_names[demangled] = name
            return demangled

        if name != existing_name:
            raise RuntimeError(
                "Name collision! Tried to save two different packaged modules:"
                f"'{name}' and '{existing_name}' which resolve to the same name: '{demangled}'."
            )
        return demangled


def _was_mangled(name):
    return re.match(r"__torch_package_\d+__\.", name)


def check_not_mangled(name):
    assert not _was_mangled(name)
