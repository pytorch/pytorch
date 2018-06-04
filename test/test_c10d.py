import multiprocessing
import tempfile
import unittest

import torch.c10d as c10d

from common import TestCase


TCP_ADDR = '127.0.0.1'
TCP_PORT = 29500


class StoreTestBase(object):
    def _create_store(self, i):
        raise RuntimeError("implement this")

    def _test_set_get(self, fs):
        fs.set("key0", "value0")
        fs.set("key1", "value1")
        fs.set("key2", "value2")
        self.assertEqual(b"value0", fs.get("key0"))
        self.assertEqual(b"value1", fs.get("key1"))
        self.assertEqual(b"value2", fs.get("key2"))

    def test_set_get(self):
        self._test_set_get(self._create_store())


class FileStoreTest(TestCase, StoreTestBase):
    def setUp(self):
        self.file = tempfile.NamedTemporaryFile()

    def _create_store(self):
        return c10d.FileStore(self.file.name)


class TCPStoreTest(TestCase, StoreTestBase):
    def _create_store(self):
        return c10d.TCPStore(TCP_ADDR, TCP_PORT, True)


if __name__ == '__main__':
    unittest.main()
