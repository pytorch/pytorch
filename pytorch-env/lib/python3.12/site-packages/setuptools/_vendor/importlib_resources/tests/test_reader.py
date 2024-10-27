import os.path
import sys
import pathlib
import unittest

from importlib import import_module
from importlib_resources.readers import MultiplexedPath, NamespaceReader


class MultiplexedPathTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = pathlib.Path(__file__).parent / 'namespacedata01'

    def test_init_no_paths(self):
        with self.assertRaises(FileNotFoundError):
            MultiplexedPath()

    def test_init_file(self):
        with self.assertRaises(NotADirectoryError):
            MultiplexedPath(self.folder / 'binary.file')

    def test_iterdir(self):
        contents = {path.name for path in MultiplexedPath(self.folder).iterdir()}
        try:
            contents.remove('__pycache__')
        except (KeyError, ValueError):
            pass
        self.assertEqual(
            contents, {'subdirectory', 'binary.file', 'utf-16.file', 'utf-8.file'}
        )

    def test_iterdir_duplicate(self):
        data01 = pathlib.Path(__file__).parent.joinpath('data01')
        contents = {
            path.name for path in MultiplexedPath(self.folder, data01).iterdir()
        }
        for remove in ('__pycache__', '__init__.pyc'):
            try:
                contents.remove(remove)
            except (KeyError, ValueError):
                pass
        self.assertEqual(
            contents,
            {'__init__.py', 'binary.file', 'subdirectory', 'utf-16.file', 'utf-8.file'},
        )

    def test_is_dir(self):
        self.assertEqual(MultiplexedPath(self.folder).is_dir(), True)

    def test_is_file(self):
        self.assertEqual(MultiplexedPath(self.folder).is_file(), False)

    def test_open_file(self):
        path = MultiplexedPath(self.folder)
        with self.assertRaises(FileNotFoundError):
            path.read_bytes()
        with self.assertRaises(FileNotFoundError):
            path.read_text()
        with self.assertRaises(FileNotFoundError):
            path.open()

    def test_join_path(self):
        data01 = pathlib.Path(__file__).parent.joinpath('data01')
        prefix = str(data01.parent)
        path = MultiplexedPath(self.folder, data01)
        self.assertEqual(
            str(path.joinpath('binary.file'))[len(prefix) + 1 :],
            os.path.join('namespacedata01', 'binary.file'),
        )
        sub = path.joinpath('subdirectory')
        assert isinstance(sub, MultiplexedPath)
        assert 'namespacedata01' in str(sub)
        assert 'data01' in str(sub)
        self.assertEqual(
            str(path.joinpath('imaginary'))[len(prefix) + 1 :],
            os.path.join('namespacedata01', 'imaginary'),
        )
        self.assertEqual(path.joinpath(), path)

    def test_join_path_compound(self):
        path = MultiplexedPath(self.folder)
        assert not path.joinpath('imaginary/foo.py').exists()

    def test_join_path_common_subdir(self):
        data01 = pathlib.Path(__file__).parent.joinpath('data01')
        data02 = pathlib.Path(__file__).parent.joinpath('data02')
        prefix = str(data01.parent)
        path = MultiplexedPath(data01, data02)
        self.assertIsInstance(path.joinpath('subdirectory'), MultiplexedPath)
        self.assertEqual(
            str(path.joinpath('subdirectory', 'subsubdir'))[len(prefix) + 1 :],
            os.path.join('data02', 'subdirectory', 'subsubdir'),
        )

    def test_repr(self):
        self.assertEqual(
            repr(MultiplexedPath(self.folder)),
            f"MultiplexedPath('{self.folder}')",
        )

    def test_name(self):
        self.assertEqual(
            MultiplexedPath(self.folder).name,
            os.path.basename(self.folder),
        )


class NamespaceReaderTest(unittest.TestCase):
    site_dir = str(pathlib.Path(__file__).parent)

    @classmethod
    def setUpClass(cls):
        sys.path.append(cls.site_dir)

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(cls.site_dir)

    def test_init_error(self):
        with self.assertRaises(ValueError):
            NamespaceReader(['path1', 'path2'])

    def test_resource_path(self):
        namespacedata01 = import_module('namespacedata01')
        reader = NamespaceReader(namespacedata01.__spec__.submodule_search_locations)

        root = os.path.abspath(os.path.join(__file__, '..', 'namespacedata01'))
        self.assertEqual(
            reader.resource_path('binary.file'), os.path.join(root, 'binary.file')
        )
        self.assertEqual(
            reader.resource_path('imaginary'), os.path.join(root, 'imaginary')
        )

    def test_files(self):
        namespacedata01 = import_module('namespacedata01')
        reader = NamespaceReader(namespacedata01.__spec__.submodule_search_locations)
        root = os.path.abspath(os.path.join(__file__, '..', 'namespacedata01'))
        self.assertIsInstance(reader.files(), MultiplexedPath)
        self.assertEqual(repr(reader.files()), f"MultiplexedPath('{root}')")


if __name__ == '__main__':
    unittest.main()
