import io
import unittest

import importlib_resources as resources

from importlib_resources._adapters import (
    CompatibilityFiles,
    wrap_spec,
)

from . import util


class CompatibilityFilesTests(unittest.TestCase):
    @property
    def package(self):
        bytes_data = io.BytesIO(b'Hello, world!')
        return util.create_package(
            file=bytes_data,
            path='some_path',
            contents=('a', 'b', 'c'),
        )

    @property
    def files(self):
        return resources.files(self.package)

    def test_spec_path_iter(self):
        self.assertEqual(
            sorted(path.name for path in self.files.iterdir()),
            ['a', 'b', 'c'],
        )

    def test_child_path_iter(self):
        self.assertEqual(list((self.files / 'a').iterdir()), [])

    def test_orphan_path_iter(self):
        self.assertEqual(list((self.files / 'a' / 'a').iterdir()), [])
        self.assertEqual(list((self.files / 'a' / 'a' / 'a').iterdir()), [])

    def test_spec_path_is(self):
        self.assertFalse(self.files.is_file())
        self.assertFalse(self.files.is_dir())

    def test_child_path_is(self):
        self.assertTrue((self.files / 'a').is_file())
        self.assertFalse((self.files / 'a').is_dir())

    def test_orphan_path_is(self):
        self.assertFalse((self.files / 'a' / 'a').is_file())
        self.assertFalse((self.files / 'a' / 'a').is_dir())
        self.assertFalse((self.files / 'a' / 'a' / 'a').is_file())
        self.assertFalse((self.files / 'a' / 'a' / 'a').is_dir())

    def test_spec_path_name(self):
        self.assertEqual(self.files.name, 'testingpackage')

    def test_child_path_name(self):
        self.assertEqual((self.files / 'a').name, 'a')

    def test_orphan_path_name(self):
        self.assertEqual((self.files / 'a' / 'b').name, 'b')
        self.assertEqual((self.files / 'a' / 'b' / 'c').name, 'c')

    def test_spec_path_open(self):
        self.assertEqual(self.files.read_bytes(), b'Hello, world!')
        self.assertEqual(self.files.read_text(encoding='utf-8'), 'Hello, world!')

    def test_child_path_open(self):
        self.assertEqual((self.files / 'a').read_bytes(), b'Hello, world!')
        self.assertEqual(
            (self.files / 'a').read_text(encoding='utf-8'), 'Hello, world!'
        )

    def test_orphan_path_open(self):
        with self.assertRaises(FileNotFoundError):
            (self.files / 'a' / 'b').read_bytes()
        with self.assertRaises(FileNotFoundError):
            (self.files / 'a' / 'b' / 'c').read_bytes()

    def test_open_invalid_mode(self):
        with self.assertRaises(ValueError):
            self.files.open('0')

    def test_orphan_path_invalid(self):
        with self.assertRaises(ValueError):
            CompatibilityFiles.OrphanPath()

    def test_wrap_spec(self):
        spec = wrap_spec(self.package)
        self.assertIsInstance(spec.loader.get_resource_reader(None), CompatibilityFiles)


class CompatibilityFilesNoReaderTests(unittest.TestCase):
    @property
    def package(self):
        return util.create_package_from_loader(None)

    @property
    def files(self):
        return resources.files(self.package)

    def test_spec_path_joinpath(self):
        self.assertIsInstance(self.files / 'a', CompatibilityFiles.OrphanPath)
