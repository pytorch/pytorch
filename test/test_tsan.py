# Owner(s): ["module: ci"]

import threading

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestTSan(TestCase):
    def test_interned_strings_race(self):
        # Exercise a known data race in InternedStringsTable
        # (python_dimname.cpp): concurrent THPDimname_parse calls on
        # new dimname strings trigger unsynchronized find/emplace on a
        # shared ska::flat_hash_map.
        barrier = threading.Barrier(4)

        def fn(names):
            barrier.wait()
            for name in names:
                torch.tensor([1], names=[name])

        # Each thread parses unique dimname strings so that addMapping
        # (emplace) races with lookup (find) across threads.
        thread_names = [
            [f"dim_a_{i}" for i in range(100)],
            [f"dim_b_{i}" for i in range(100)],
            [f"dim_c_{i}" for i in range(100)],
            [f"dim_d_{i}" for i in range(100)],
        ]
        threads = [threading.Thread(target=fn, args=(names,)) for names in thread_names]
        for t in threads:
            t.start()
        for t in threads:
            t.join()


if __name__ == "__main__":
    run_tests()
