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

Mangling in PackageImporter
---------------------------
On import, all modules produced by a given ``PackageImporter`` are given a
new top-level module as their parent. This is called the `mangle parent`. For example:
::
    torchvision.models.resnet18
becomes
::
    __torch_package0__.torchvision.models.resnet18

The mangle parent is made unique to a given ``PackageImporter`` instance by
bumping the ``mangle_index``, i.e. ``__torch__package{mangle_index}__``.

Additionally, when we want to import a module from the PackageImporter
(either by calling ``import_module`` or through ``__import__``), we need to
make sure to demangle the user-provided name before looking it up in the
package.

It is important to note that this mangling happens `on import`, and the
results are never saved into a package. Assigning mangle parents on import
means that we can enforce that mangle parents are unique within the
environment doing the importing. It also allows us to avoid serializing (and
maintaining backward compatibility for) this detail.

.. note::
    No mangle parents should ever be saved into a package.

Demangling in PackageExporter
--------------------------------
Occasionally ``PackageExporter`` may encounter mangled names during export. For
example, the user may be re-packaging an object that was imported (and thus
had its module name mangled).

This means that all entry points to ``PackageExporter`` must properly
demangle any module names passed to them before doing anything else. That
way, internally ``PackageExporter`` only ever deals with unmangled names.

There are two additional complications with demangling in PackageExporter.

First, name collisions. Consider the following user code:
::
    pe = PackageExporter('package.pt')
    pe.save_module('foo.bar')
    pe.save_module('__torch_package0__.foo.bar')
    pe.save_module('__torch_package1__.foo.bar')

All three packages demangle to the same ``'foo.bar'`` package, which leads to
confusing behavior. To guard against this, ``PackageExporter`` keeps track of
the demanglings it has performed and errors when a collision would have
occurred.

Second, pickled ``GLOBAL`` opcodes. When the pickler is writing out where to
find a class to reconstruct its state, it typically looks at
``obj.__module__`` to record how to retrieve it. We need to ensure that any
global references in the pickle bytecode are always demangled, which is taken
care of in ``CustomImportPickler``.
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


def _is_mangled(name: str) -> bool:
    return bool(re.match(r"__torch_package_\d+__\.", name))


def check_not_mangled(name: str):
    assert not _is_mangled(name)


class DemangledModuleName(str):
    """
    Tracks whether a name has passed through `demangle`. Otherwise behaves like a string.
    """

    pass


def demangle(name: str) -> DemangledModuleName:
    """
    Note: Unlike PackageMangler.demangle, this version works on any
    mangled name, irrespective of which PackageMangler created it.
    """
    demangled = name.partition(".")[2] if _is_mangled(name) else name
    return DemangledModuleName(demangled)
