import torch
from torch.utils.data import IterableDataset
from torch.utils.data.datasets import (CollateIterableDataset, BatchIterableDataset,
        SamplerIterableDataset)
from torch.testing._internal.common_utils import (TestCase, run_tests)


class IterDatasetWithoutLen(IterableDataset):
    def __init__(self, ds):
        super().__init__()
        self.ds = ds

    def __iter__(self):
        for i in self.ds:
            yield i


class IterDatasetWithLen(IterableDataset):
    def __init__(self, ds):
        super().__init__()
        self.ds = ds
        self.length = len(ds)

    def __iter__(self):
        for i in self.ds:
            yield i

    def __len__(self):
        return self.length


class TestFunctionalIterableDataset(TestCase):
    def test_collate_dataset(self):
        arrs = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        ds_len = IterDatasetWithLen(arrs)
        ds_nolen = IterDatasetWithoutLen(arrs)

        def _collate_fn(batch):
            return torch.tensor(sum(batch), dtype=torch.float)

        collate_ds = CollateIterableDataset(ds_len, collate_fn=_collate_fn)
        self.assertEqual(len(ds_len), len(collate_ds))
        ds_iter = iter(ds_len)
        for x in collate_ds:
            y = next(ds_iter)
            self.assertEqual(x, torch.tensor(sum(y), dtype=torch.float))

        collate_ds_nolen = CollateIterableDataset(ds_nolen)
        with self.assertRaises(NotImplementedError):
            len(collate_ds_nolen)
        ds_nolen_iter = iter(ds_nolen)
        for x in collate_ds_nolen:
            y = next(ds_nolen_iter)
            self.assertEqual(x, torch.tensor(y))

    def test_batch_dataset(self):
        arrs = range(10)
        ds = IterDatasetWithLen(arrs)
        with self.assertRaises(AssertionError):
            batch_ds0 = BatchIterableDataset(ds, batch_size=0)

        # Default not drop the last batch
        batch_ds1 = BatchIterableDataset(ds, batch_size=3)
        self.assertEqual(len(batch_ds1), 4)
        batch_iter = iter(batch_ds1)
        value = 0
        for i in range(len(batch_ds1)):
            batch = next(batch_iter)
            if i == 3:
                self.assertEqual(len(batch), 1)
                self.assertEqual(batch, [9])
            else:
                self.assertEqual(len(batch), 3)
                for x in batch:
                    self.assertEqual(x, value)
                    value += 1

        # Drop the last batch
        batch_ds2 = BatchIterableDataset(ds, batch_size=3, drop_last=True)
        self.assertEqual(len(batch_ds2), 3)
        value = 0
        for batch in batch_ds2:
            self.assertEqual(len(batch), 3)
            for x in batch:
                self.assertEqual(x, value)
                value += 1

        batch_ds3 = BatchIterableDataset(ds, batch_size=2)
        self.assertEqual(len(batch_ds3), 5)
        batch_ds4 = BatchIterableDataset(ds, batch_size=2, drop_last=True)
        self.assertEqual(len(batch_ds4), 5)

        ds_nolen = IterDatasetWithoutLen(arrs)
        batch_ds_nolen = BatchIterableDataset(ds_nolen, batch_size=5)
        with self.assertRaises(NotImplementedError):
            len(batch_ds_nolen)

    def test_sampler_dataset(self):
        arrs = range(10)
        ds = IterDatasetWithLen(arrs)
        sampled_ds = SamplerIterableDataset(ds)
        self.assertEqual(len(sampled_ds), 10)
        i = 0
        for x in sampled_ds:
            self.assertEqual(x, i)
            i += 1

        ds_nolen = IterDatasetWithoutLen(arrs)
        sampled_ds = SamplerIterableDataset(ds_nolen)
        with self.assertRaises(NotImplementedError):
            len(sampled_ds)


if __name__ == '__main__':
    run_tests()
