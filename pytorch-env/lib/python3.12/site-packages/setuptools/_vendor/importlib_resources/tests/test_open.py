import unittest

import importlib_resources as resources
from . import data01
from . import util


class CommonBinaryTests(util.CommonTests, unittest.TestCase):
    def execute(self, package, path):
        target = resources.files(package).joinpath(path)
        with target.open('rb'):
            pass


class CommonTextTests(util.CommonTests, unittest.TestCase):
    def execute(self, package, path):
        target = resources.files(package).joinpath(path)
        with target.open(encoding='utf-8'):
            pass


class OpenTests:
    def test_open_binary(self):
        target = resources.files(self.data) / 'binary.file'
        with target.open('rb') as fp:
            result = fp.read()
            self.assertEqual(result, bytes(range(4)))

    def test_open_text_default_encoding(self):
        target = resources.files(self.data) / 'utf-8.file'
        with target.open(encoding='utf-8') as fp:
            result = fp.read()
            self.assertEqual(result, 'Hello, UTF-8 world!\n')

    def test_open_text_given_encoding(self):
        target = resources.files(self.data) / 'utf-16.file'
        with target.open(encoding='utf-16', errors='strict') as fp:
            result = fp.read()
        self.assertEqual(result, 'Hello, UTF-16 world!\n')

    def test_open_text_with_errors(self):
        """
        Raises UnicodeError without the 'errors' argument.
        """
        target = resources.files(self.data) / 'utf-16.file'
        with target.open(encoding='utf-8', errors='strict') as fp:
            self.assertRaises(UnicodeError, fp.read)
        with target.open(encoding='utf-8', errors='ignore') as fp:
            result = fp.read()
        self.assertEqual(
            result,
            'H\x00e\x00l\x00l\x00o\x00,\x00 '
            '\x00U\x00T\x00F\x00-\x001\x006\x00 '
            '\x00w\x00o\x00r\x00l\x00d\x00!\x00\n\x00',
        )

    def test_open_binary_FileNotFoundError(self):
        target = resources.files(self.data) / 'does-not-exist'
        with self.assertRaises(FileNotFoundError):
            target.open('rb')

    def test_open_text_FileNotFoundError(self):
        target = resources.files(self.data) / 'does-not-exist'
        with self.assertRaises(FileNotFoundError):
            target.open(encoding='utf-8')


class OpenDiskTests(OpenTests, unittest.TestCase):
    def setUp(self):
        self.data = data01


class OpenDiskNamespaceTests(OpenTests, unittest.TestCase):
    def setUp(self):
        from . import namespacedata01

        self.data = namespacedata01


class OpenZipTests(OpenTests, util.ZipSetup, unittest.TestCase):
    pass


class OpenNamespaceZipTests(OpenTests, util.ZipSetup, unittest.TestCase):
    ZIP_MODULE = 'namespacedata01'


if __name__ == '__main__':
    unittest.main()
