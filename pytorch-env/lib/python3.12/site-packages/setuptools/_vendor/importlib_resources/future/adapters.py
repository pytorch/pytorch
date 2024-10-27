import functools
import pathlib
from contextlib import suppress
from types import SimpleNamespace

from .. import readers, _adapters


def _block_standard(reader_getter):
    """
    Wrap _adapters.TraversableResourcesLoader.get_resource_reader
    and intercept any standard library readers.
    """

    @functools.wraps(reader_getter)
    def wrapper(*args, **kwargs):
        """
        If the reader is from the standard library, return None to allow
        allow likely newer implementations in this library to take precedence.
        """
        try:
            reader = reader_getter(*args, **kwargs)
        except NotADirectoryError:
            # MultiplexedPath may fail on zip subdirectory
            return
        # Python 3.10+
        mod_name = reader.__class__.__module__
        if mod_name.startswith('importlib.') and mod_name.endswith('readers'):
            return
        # Python 3.8, 3.9
        if isinstance(reader, _adapters.CompatibilityFiles) and (
            reader.spec.loader.__class__.__module__.startswith('zipimport')
            or reader.spec.loader.__class__.__module__.startswith(
                '_frozen_importlib_external'
            )
        ):
            return
        return reader

    return wrapper


def _skip_degenerate(reader):
    """
    Mask any degenerate reader. Ref #298.
    """
    is_degenerate = (
        isinstance(reader, _adapters.CompatibilityFiles) and not reader._reader
    )
    return reader if not is_degenerate else None


class TraversableResourcesLoader(_adapters.TraversableResourcesLoader):
    """
    Adapt loaders to provide TraversableResources and other
    compatibility.

    Ensures the readers from importlib_resources are preferred
    over stdlib readers.
    """

    def get_resource_reader(self, name):
        return (
            _skip_degenerate(_block_standard(super().get_resource_reader)(name))
            or self._standard_reader()
            or super().get_resource_reader(name)
        )

    def _standard_reader(self):
        return self._zip_reader() or self._namespace_reader() or self._file_reader()

    def _zip_reader(self):
        with suppress(AttributeError):
            return readers.ZipReader(self.spec.loader, self.spec.name)

    def _namespace_reader(self):
        with suppress(AttributeError, ValueError):
            return readers.NamespaceReader(self.spec.submodule_search_locations)

    def _file_reader(self):
        try:
            path = pathlib.Path(self.spec.origin)
        except TypeError:
            return None
        if path.exists():
            return readers.FileReader(SimpleNamespace(path=path))


def wrap_spec(package):
    """
    Override _adapters.wrap_spec to use TraversableResourcesLoader
    from above. Ensures that future behavior is always available on older
    Pythons.
    """
    return _adapters.SpecLoaderAdapter(package.__spec__, TraversableResourcesLoader)
