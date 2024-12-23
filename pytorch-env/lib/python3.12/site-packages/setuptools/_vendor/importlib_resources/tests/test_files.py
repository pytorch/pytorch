import textwrap
import unittest
import warnings
import importlib
import contextlib

import importlib_resources as resources
from ..abc import Traversable
from . import data01
from . import util
from . import _path
from .compat.py39 import os_helper
from .compat.py312 import import_helper


@contextlib.contextmanager
def suppress_known_deprecation():
    with warnings.catch_warnings(record=True) as ctx:
        warnings.simplefilter('default', category=DeprecationWarning)
        yield ctx


class FilesTests:
    def test_read_bytes(self):
        files = resources.files(self.data)
        actual = files.joinpath('utf-8.file').read_bytes()
        assert actual == b'Hello, UTF-8 world!\n'

    def test_read_text(self):
        files = resources.files(self.data)
        actual = files.joinpath('utf-8.file').read_text(encoding='utf-8')
        assert actual == 'Hello, UTF-8 world!\n'

    def test_traversable(self):
        assert isinstance(resources.files(self.data), Traversable)

    def test_joinpath_with_multiple_args(self):
        files = resources.files(self.data)
        binfile = files.joinpath('subdirectory', 'binary.file')
        self.assertTrue(binfile.is_file())

    def test_old_parameter(self):
        """
        Files used to take a 'package' parameter. Make sure anyone
        passing by name is still supported.
        """
        with suppress_known_deprecation():
            resources.files(package=self.data)


class OpenDiskTests(FilesTests, unittest.TestCase):
    def setUp(self):
        self.data = data01


class OpenZipTests(FilesTests, util.ZipSetup, unittest.TestCase):
    pass


class OpenNamespaceTests(FilesTests, unittest.TestCase):
    def setUp(self):
        from . import namespacedata01

        self.data = namespacedata01


class OpenNamespaceZipTests(FilesTests, util.ZipSetup, unittest.TestCase):
    ZIP_MODULE = 'namespacedata01'


class SiteDir:
    def setUp(self):
        self.fixtures = contextlib.ExitStack()
        self.addCleanup(self.fixtures.close)
        self.site_dir = self.fixtures.enter_context(os_helper.temp_dir())
        self.fixtures.enter_context(import_helper.DirsOnSysPath(self.site_dir))
        self.fixtures.enter_context(import_helper.isolated_modules())


class ModulesFilesTests(SiteDir, unittest.TestCase):
    def test_module_resources(self):
        """
        A module can have resources found adjacent to the module.
        """
        spec = {
            'mod.py': '',
            'res.txt': 'resources are the best',
        }
        _path.build(spec, self.site_dir)
        import mod

        actual = resources.files(mod).joinpath('res.txt').read_text(encoding='utf-8')
        assert actual == spec['res.txt']


class ImplicitContextFilesTests(SiteDir, unittest.TestCase):
    def test_implicit_files(self):
        """
        Without any parameter, files() will infer the location as the caller.
        """
        spec = {
            'somepkg': {
                '__init__.py': textwrap.dedent(
                    """
                    import importlib_resources as res
                    val = res.files().joinpath('res.txt').read_text(encoding='utf-8')
                    """
                ),
                'res.txt': 'resources are the best',
            },
        }
        _path.build(spec, self.site_dir)
        assert importlib.import_module('somepkg').val == 'resources are the best'


if __name__ == '__main__':
    unittest.main()
