# mypy: allow-untyped-defs
import _compat_pickle
import pickle

from .importer import Importer


class PackageUnpickler(pickle._Unpickler):  # type: ignore[name-defined]
    """Package-aware unpickler.

    This behaves the same as a normal unpickler, except it uses `importer` to
    find any global names that it encounters while unpickling.
    """

    def __init__(self, importer: Importer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._importer = importer

    def find_class(self, module, name):
        # Subclasses may override this.
        if self.proto < 3 and self.fix_imports:  # type: ignore[attr-defined]
            if (module, name) in _compat_pickle.NAME_MAPPING:
                module, name = _compat_pickle.NAME_MAPPING[(module, name)]
            elif module in _compat_pickle.IMPORT_MAPPING:
                module = _compat_pickle.IMPORT_MAPPING[module]
        mod = self._importer.import_module(module)
        return getattr(mod, name)
