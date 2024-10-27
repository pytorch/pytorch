import sys
import unittest
import importlib_resources as resources
import pathlib

from . import data01
from . import util
from importlib import import_module


class ResourceTests:
    # Subclasses are expected to set the `data` attribute.

    def test_is_file_exists(self):
        target = resources.files(self.data) / 'binary.file'
        self.assertTrue(target.is_file())

    def test_is_file_missing(self):
        target = resources.files(self.data) / 'not-a-file'
        self.assertFalse(target.is_file())

    def test_is_dir(self):
        target = resources.files(self.data) / 'subdirectory'
        self.assertFalse(target.is_file())
        self.assertTrue(target.is_dir())


class ResourceDiskTests(ResourceTests, unittest.TestCase):
    def setUp(self):
        self.data = data01


class ResourceZipTests(ResourceTests, util.ZipSetup, unittest.TestCase):
    pass


def names(traversable):
    return {item.name for item in traversable.iterdir()}


class ResourceLoaderTests(unittest.TestCase):
    def test_resource_contents(self):
        package = util.create_package(
            file=data01, path=data01.__file__, contents=['A', 'B', 'C']
        )
        self.assertEqual(names(resources.files(package)), {'A', 'B', 'C'})

    def test_is_file(self):
        package = util.create_package(
            file=data01, path=data01.__file__, contents=['A', 'B', 'C', 'D/E', 'D/F']
        )
        self.assertTrue(resources.files(package).joinpath('B').is_file())

    def test_is_dir(self):
        package = util.create_package(
            file=data01, path=data01.__file__, contents=['A', 'B', 'C', 'D/E', 'D/F']
        )
        self.assertTrue(resources.files(package).joinpath('D').is_dir())

    def test_resource_missing(self):
        package = util.create_package(
            file=data01, path=data01.__file__, contents=['A', 'B', 'C', 'D/E', 'D/F']
        )
        self.assertFalse(resources.files(package).joinpath('Z').is_file())


class ResourceCornerCaseTests(unittest.TestCase):
    def test_package_has_no_reader_fallback(self):
        """
        Test odd ball packages which:
        # 1. Do not have a ResourceReader as a loader
        # 2. Are not on the file system
        # 3. Are not in a zip file
        """
        module = util.create_package(
            file=data01, path=data01.__file__, contents=['A', 'B', 'C']
        )
        # Give the module a dummy loader.
        module.__loader__ = object()
        # Give the module a dummy origin.
        module.__file__ = '/path/which/shall/not/be/named'
        module.__spec__.loader = module.__loader__
        module.__spec__.origin = module.__file__
        self.assertFalse(resources.files(module).joinpath('A').is_file())


class ResourceFromZipsTest01(util.ZipSetupBase, unittest.TestCase):
    ZIP_MODULE = 'data01'

    def test_is_submodule_resource(self):
        submodule = import_module('data01.subdirectory')
        self.assertTrue(resources.files(submodule).joinpath('binary.file').is_file())

    def test_read_submodule_resource_by_name(self):
        self.assertTrue(
            resources.files('data01.subdirectory').joinpath('binary.file').is_file()
        )

    def test_submodule_contents(self):
        submodule = import_module('data01.subdirectory')
        self.assertEqual(
            names(resources.files(submodule)), {'__init__.py', 'binary.file'}
        )

    def test_submodule_contents_by_name(self):
        self.assertEqual(
            names(resources.files('data01.subdirectory')),
            {'__init__.py', 'binary.file'},
        )

    def test_as_file_directory(self):
        with resources.as_file(resources.files('data01')) as data:
            assert data.name == 'data01'
            assert data.is_dir()
            assert data.joinpath('subdirectory').is_dir()
            assert len(list(data.iterdir()))
        assert not data.parent.exists()


class ResourceFromZipsTest02(util.ZipSetupBase, unittest.TestCase):
    ZIP_MODULE = 'data02'

    def test_unrelated_contents(self):
        """
        Test thata zip with two unrelated subpackages return
        distinct resources. Ref python/importlib_resources#44.
        """
        self.assertEqual(
            names(resources.files('data02.one')),
            {'__init__.py', 'resource1.txt'},
        )
        self.assertEqual(
            names(resources.files('data02.two')),
            {'__init__.py', 'resource2.txt'},
        )


class DeletingZipsTest(util.ZipSetupBase, unittest.TestCase):
    """Having accessed resources in a zip file should not keep an open
    reference to the zip.
    """

    def test_iterdir_does_not_keep_open(self):
        [item.name for item in resources.files('data01').iterdir()]

    def test_is_file_does_not_keep_open(self):
        resources.files('data01').joinpath('binary.file').is_file()

    def test_is_file_failure_does_not_keep_open(self):
        resources.files('data01').joinpath('not-present').is_file()

    @unittest.skip("Desired but not supported.")
    def test_as_file_does_not_keep_open(self):  # pragma: no cover
        resources.as_file(resources.files('data01') / 'binary.file')

    def test_entered_path_does_not_keep_open(self):
        """
        Mimic what certifi does on import to make its bundle
        available for the process duration.
        """
        resources.as_file(resources.files('data01') / 'binary.file').__enter__()

    def test_read_binary_does_not_keep_open(self):
        resources.files('data01').joinpath('binary.file').read_bytes()

    def test_read_text_does_not_keep_open(self):
        resources.files('data01').joinpath('utf-8.file').read_text(encoding='utf-8')


class ResourceFromNamespaceTests:
    def test_is_submodule_resource(self):
        self.assertTrue(
            resources.files(import_module('namespacedata01'))
            .joinpath('binary.file')
            .is_file()
        )

    def test_read_submodule_resource_by_name(self):
        self.assertTrue(
            resources.files('namespacedata01').joinpath('binary.file').is_file()
        )

    def test_submodule_contents(self):
        contents = names(resources.files(import_module('namespacedata01')))
        try:
            contents.remove('__pycache__')
        except KeyError:
            pass
        self.assertEqual(
            contents, {'subdirectory', 'binary.file', 'utf-8.file', 'utf-16.file'}
        )

    def test_submodule_contents_by_name(self):
        contents = names(resources.files('namespacedata01'))
        try:
            contents.remove('__pycache__')
        except KeyError:
            pass
        self.assertEqual(
            contents, {'subdirectory', 'binary.file', 'utf-8.file', 'utf-16.file'}
        )

    def test_submodule_sub_contents(self):
        contents = names(resources.files(import_module('namespacedata01.subdirectory')))
        try:
            contents.remove('__pycache__')
        except KeyError:
            pass
        self.assertEqual(contents, {'binary.file'})

    def test_submodule_sub_contents_by_name(self):
        contents = names(resources.files('namespacedata01.subdirectory'))
        try:
            contents.remove('__pycache__')
        except KeyError:
            pass
        self.assertEqual(contents, {'binary.file'})


class ResourceFromNamespaceDiskTests(ResourceFromNamespaceTests, unittest.TestCase):
    site_dir = str(pathlib.Path(__file__).parent)

    @classmethod
    def setUpClass(cls):
        sys.path.append(cls.site_dir)

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(cls.site_dir)


class ResourceFromNamespaceZipTests(
    util.ZipSetupBase,
    ResourceFromNamespaceTests,
    unittest.TestCase,
):
    ZIP_MODULE = 'namespacedata01'


if __name__ == '__main__':
    unittest.main()
