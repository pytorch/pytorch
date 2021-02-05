from bitarray import bitarray

from torch.testing._internal.common_utils import TestCase, run_tests
from torch.utils.data._utils.index_utils import Range, Permutation


class TestRange(TestCase):

    def test_slicing(self):
        self.assertEqual(list(Range(10)[0:20:2]),
                         [0, 2, 4, 6, 8, 0, 2, 4, 6, 8])

        self.assertEqual(list(Range(20)[0:20:2][0:10:2]),
                         [0, 4, 8, 12, 16])

    def test_size_0(self):
        self.assertEqual(list(Range(0)), [])

    def test_size_0_stop_nonzero(self):
        with self.assertRaises(ValueError):
            self.assertEqual(list(Range(0, stop=10)), [])

    def test_size_1(self):
        self.assertEqual(list(Range(1)), [0])
        self.assertEqual(list(Range(1, stop=3)), [0, 0, 0])


class TestPermutation(TestCase):

    def test_slicing(self):
        self.assertEqual(list(Permutation(10, 42, stop=20)[::2]),
                         [7, 4, 9, 8, 0, 7, 4, 9, 8, 0])

        self.assertEqual(list(Permutation(10, 42, stop=20)[::2][0:10:2]),
                         [7, 9, 0, 4, 8])

    def test_size_0(self):
        self._test_exhaustiveness(0, 0)

    def test_size_1(self):
        self._test_exhaustiveness(1, 0)

    def test_small_values(self):
        for i in range(100):
            for s in range(i):
                self._test_exhaustiveness(i, s)

    def test_large_value(self):
        size = 195036
        seed = 42
        seq = Permutation(size, seed)

        visited = bitarray(size)
        visited.setall(False)
        for i in range(len(seq)):
            value = seq[i]
            if visited[value]:
                raise Exception
            visited[value] = True
        assert visited.all()

        visited.setall(False)
        for value in seq:
            if visited[value]:
                raise Exception
            visited[value] = True
        assert visited.all()

        visited.setall(False)
        for value in seq[:size // 2]:
            if visited[value]:
                raise Exception
            visited[value] = True
        assert len(seq[:size // 2]) == size // 2
        for value in seq[size // 2:]:
            if visited[value]:
                raise Exception
            visited[value] = True
        assert visited.all()

        visited.setall(False)
        for value in seq[0:size:2]:
            if visited[value]:
                raise Exception
            visited[value] = True
        assert len(seq[0:size:2]) == size // 2
        for value in seq[1:size:2]:
            if visited[value]:
                raise Exception
            visited[value] = True
        assert visited.all()

    def _test_exhaustiveness(self, size, seed):
        seq = Permutation(size, seed)
        visited = bitarray(size)
        visited.setall(False)
        for value in seq:
            if visited[value]:
                raise Exception
            visited[value] = True
        assert visited.all()


if __name__ == '__main__':
    run_tests()
