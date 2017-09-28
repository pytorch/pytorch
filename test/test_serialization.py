import torch
import numpy as np
import os
import random
import shutil
import time
import datetime
from common import TestCase, run_tests


class TestSerialization(TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def get_test_obj(self):
        random_string = ''.join(
            random.choice(string.ascii_letters) for x in range(10))
        return {
            'test': random_string,
            'simple_time': time.time(),
            'complex_time': datetime.datetime.now(),
            'vec': np.ones(1),
        }

    def string_round_trip(self, **save_kw):
        orig = self._get_test_obj()
        dest = os.path.join(self.tmpdir, 'test.pkl')
        torch.save(dest, obj, **save_kw)
        new = torch.load(dest)
        return orig, new

    def test_serialization_round_trip_str(self):
        orig, new = self.string_round_trip()
        self.assertEqual(orig, new)

    def test_serialization_round_trip_atomic(self):
        orig, new = self.string_round_trip(atomic=True)
        self.assertEqual(orig, new)

    def test_serialization_round_trip_file(self):
        orig = self._get_test_obj()
        dest = os.path.join(self.tmpdir, 'test.pkl')
        with open(dest, 'wb') as dest_file:
            torch.save(dest_file, obj)
        with open(dest, 'rb') as from_file:
            new = torch.load(from_file)
        self.assertEqual(orig, new)


if __name__ == '__main__':
    run_tests()
