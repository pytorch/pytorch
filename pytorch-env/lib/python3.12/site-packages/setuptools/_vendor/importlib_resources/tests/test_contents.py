import unittest
import importlib_resources as resources

from . import data01
from . import util


class ContentsTests:
    expected = {
        '__init__.py',
        'binary.file',
        'subdirectory',
        'utf-16.file',
        'utf-8.file',
    }

    def test_contents(self):
        contents = {path.name for path in resources.files(self.data).iterdir()}
        assert self.expected <= contents


class ContentsDiskTests(ContentsTests, unittest.TestCase):
    def setUp(self):
        self.data = data01


class ContentsZipTests(ContentsTests, util.ZipSetup, unittest.TestCase):
    pass


class ContentsNamespaceTests(ContentsTests, unittest.TestCase):
    expected = {
        # no __init__ because of namespace design
        'binary.file',
        'subdirectory',
        'utf-16.file',
        'utf-8.file',
    }

    def setUp(self):
        from . import namespacedata01

        self.data = namespacedata01
