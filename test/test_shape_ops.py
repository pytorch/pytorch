import torch
import numpy as np

from itertools import product, combinations, permutations
from functools import partial
import random
import warnings

from torch._six import nan
from torch.testing._internal.common_utils import (
    TestCase, run_tests, make_tensor, torch_to_numpy_dtype_dict)
from torch.testing._internal.common_methods_invocations import shape_funcs
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests, onlyCPU, dtypes, onlyOnCPUAndCUDA,
    dtypesIfCPU, dtypesIfCUDA, ops)

# TODO: replace with make_tensor
def _generate_input(shape, dtype, device, with_extremal):
    if shape == ():
        x = torch.tensor((), dtype=dtype, device=device)
    else:
        if dtype.is_floating_point or dtype.is_complex:
            # work around torch.randn not being implemented for bfloat16
            if dtype == torch.bfloat16:
                x = torch.randn(*shape, device=device) * random.randint(30, 100)
                x = x.to(torch.bfloat16)
            else:
                x = torch.randn(*shape, dtype=dtype, device=device) * random.randint(30, 100)
            x[torch.randn(*shape) > 0.5] = 0
            if with_extremal and dtype.is_floating_point:
                # Use extremal values
                x[torch.randn(*shape) > 0.5] = float('nan')
                x[torch.randn(*shape) > 0.5] = float('inf')
                x[torch.randn(*shape) > 0.5] = float('-inf')
            elif with_extremal and dtype.is_complex:
                x[torch.randn(*shape) > 0.5] = complex('nan')
                x[torch.randn(*shape) > 0.5] = complex('inf')
                x[torch.randn(*shape) > 0.5] = complex('-inf')
        elif dtype == torch.bool:
            x = torch.zeros(shape, dtype=dtype, device=device)
            x[torch.randn(*shape) > 0.5] = True
        else:
            x = torch.randint(15, 100, shape, dtype=dtype, device=device)

    return x

class TestShapeOps(TestCase):

    # TODO: update to work on CUDA, too
    @onlyCPU
    def test_unbind(self, device):
        x = torch.rand(2, 3, 4, 5)
        for dim in range(4):
            res = torch.unbind(x, dim)
            res2 = x.unbind(dim)
            self.assertEqual(x.size(dim), len(res))
            self.assertEqual(x.size(dim), len(res2))
            for i in range(dim):
                self.assertEqual(x.select(dim, i), res[i])
                self.assertEqual(x.select(dim, i), res2[i])

    # TODO: update to work on CUDA, too?
    @onlyCPU
    def test_tolist(self, device):
        list0D = []
        tensor0D = torch.tensor(list0D)
        self.assertEqual(tensor0D.tolist(), list0D)

        table1D = [1., 2., 3.]
        tensor1D = torch.tensor(table1D)
        storage = torch.Storage(table1D)
        self.assertEqual(tensor1D.tolist(), table1D)
        self.assertEqual(storage.tolist(), table1D)
        self.assertEqual(tensor1D.tolist(), table1D)
        self.assertEqual(storage.tolist(), table1D)

        table2D = [[1, 2], [3, 4]]
        tensor2D = torch.tensor(table2D)
        self.assertEqual(tensor2D.tolist(), table2D)

        tensor3D = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        tensorNonContig = tensor3D.select(1, 1)
        self.assertFalse(tensorNonContig.is_contiguous())
        self.assertEqual(tensorNonContig.tolist(), [[3, 4], [7, 8]])

    @dtypes(torch.int64, torch.float, torch.complex128)
    def test_movedim_invalid(self, device, dtype):
        shape = self._rand_shape(4, min_size=5, max_size=10)
        x = _generate_input(shape, dtype, device, False)

        for fn in [torch.movedim, torch.moveaxis]:
            # Invalid `source` and `destination` dimension
            with self.assertRaisesRegex(IndexError, "Dimension out of range"):
                fn(x, 5, 0)

            with self.assertRaisesRegex(IndexError, "Dimension out of range"):
                fn(x, 0, 5)

            # Mismatch in size of `source` and `destination`
            with self.assertRaisesRegex(RuntimeError, "movedim: Invalid source or destination dims:"):
                fn(x, (1, 0), (0, ))

            with self.assertRaisesRegex(RuntimeError, "movedim: repeated dim in `source`"):
                fn(x, (0, 0), (0, 1))

            with self.assertRaisesRegex(RuntimeError, "movedim: repeated dim in `source`"):
                fn(x, (0, 1, 0), (0, 1, 2))

            with self.assertRaisesRegex(RuntimeError, "movedim: repeated dim in `destination`"):
                fn(x, (0, 1), (1, 1))

            with self.assertRaisesRegex(RuntimeError, "movedim: repeated dim in `destination`"):
                fn(x, (0, 1, 2), (1, 0, 1))

    @dtypes(torch.int64, torch.float, torch.complex128)
    def test_movedim(self, device, dtype):
        for fn in [torch.moveaxis, torch.movedim]:
            for nd in range(5):
                shape = self._rand_shape(nd, min_size=5, max_size=10)
                x = _generate_input(shape, dtype, device, with_extremal=False)
                for random_negative in [True, False]:
                    for src_dim, dst_dim in permutations(range(nd), r=2):
                        random_prob = random.random()

                        if random_negative and random_prob > 0.66:
                            src_dim = src_dim - nd
                        elif random_negative and random_prob > 0.33:
                            dst_dim = dst_dim - nd
                        elif random_negative:
                            src_dim = src_dim - nd
                            dst_dim = dst_dim - nd

                        # Integer `source` and `destination`
                        torch_fn = partial(fn, source=src_dim, destination=dst_dim)
                        np_fn = partial(np.moveaxis, source=src_dim, destination=dst_dim)
                        self.compare_with_numpy(torch_fn, np_fn, x, device=None, dtype=None)

                    if nd == 0:
                        continue

                    def make_index_negative(sequence, idx):
                        sequence = list(sequence)
                        sequence[random_idx] = sequence[random_idx] - nd
                        return tuple(src_sequence)

                    for src_sequence in permutations(range(nd), r=random.randint(1, nd)):
                        # Sequence `source` and `destination`
                        dst_sequence = tuple(random.sample(range(nd), len(src_sequence)))

                        # Randomly change a dim to a negative dim representation of itself.
                        random_prob = random.random()
                        if random_negative and random_prob > 0.66:
                            random_idx = random.randint(0, len(src_sequence) - 1)
                            src_sequence = make_index_negative(src_sequence, random_idx)
                        elif random_negative and random_prob > 0.33:
                            random_idx = random.randint(0, len(src_sequence) - 1)
                            dst_sequence = make_index_negative(dst_sequence, random_idx)
                        elif random_negative:
                            random_idx = random.randint(0, len(src_sequence) - 1)
                            dst_sequence = make_index_negative(dst_sequence, random_idx)
                            random_idx = random.randint(0, len(src_sequence) - 1)
                            src_sequence = make_index_negative(src_sequence, random_idx)

                        torch_fn = partial(fn, source=src_sequence, destination=dst_sequence)
                        np_fn = partial(np.moveaxis, source=src_sequence, destination=dst_sequence)
                        self.compare_with_numpy(torch_fn, np_fn, x, device=None, dtype=None)

            # Move dim to same position
            x = torch.randn(2, 3, 5, 7, 11)
            torch_fn = partial(fn, source=(0, 1), destination=(0, 1))
            np_fn = partial(np.moveaxis, source=(0, 1), destination=(0, 1))
            self.compare_with_numpy(torch_fn, np_fn, x, device=None, dtype=None)

            torch_fn = partial(fn, source=1, destination=1)
            np_fn = partial(np.moveaxis, source=1, destination=1)
            self.compare_with_numpy(torch_fn, np_fn, x, device=None, dtype=None)

            # Empty Sequence
            torch_fn = partial(fn, source=(), destination=())
            np_fn = partial(np.moveaxis, source=(), destination=())
            self.compare_with_numpy(torch_fn, np_fn, x, device=None, dtype=None)

    @dtypes(torch.float, torch.bool)
    def test_diag(self, device, dtype):
        if dtype is torch.bool:
            x = torch.rand(100, 100, device=device) >= 0.5
        else:
            x = torch.rand(100, 100, dtype=dtype, device=device)

        res1 = torch.diag(x)
        res2 = torch.tensor((), dtype=dtype, device=device)
        torch.diag(x, out=res2)
        self.assertEqual(res1, res2)

    def test_diagonal(self, device):
        x = torch.randn((100, 100), device=device)
        result = torch.diagonal(x)
        expected = torch.diag(x)
        self.assertEqual(result, expected)

        x = torch.randn((100, 100), device=device)
        result = torch.diagonal(x, 17)
        expected = torch.diag(x, 17)
        self.assertEqual(result, expected)

    @onlyCPU
    @dtypes(torch.float)
    def test_diagonal_multidim(self, device, dtype):
        x = torch.randn(10, 11, 12, 13, dtype=dtype, device=device)
        xn = x.numpy()
        for args in [(2, 2, 3),
                     (2,),
                     (-2, 1, 2),
                     (0, -2, -1)]:
            result = torch.diagonal(x, *args)
            expected = xn.diagonal(*args)
            self.assertEqual(expected.shape, result.shape)
            self.assertEqual(expected, result)
        # test non-continguous
        xp = x.permute(1, 2, 3, 0)
        result = torch.diagonal(xp, 0, -2, -1)
        expected = xp.numpy().diagonal(0, -2, -1)
        self.assertEqual(expected.shape, result.shape)
        self.assertEqual(expected, result)

    @onlyOnCPUAndCUDA
    @dtypesIfCPU(*torch.testing.get_all_dtypes(include_complex=False, include_bool=False, include_half=False,
                                               include_bfloat16=False))
    @dtypesIfCUDA(*torch.testing.get_all_dtypes(include_complex=False, include_bool=False, include_bfloat16=False))
    def test_trace(self, device, dtype):
        def test(shape):
            tensor = make_tensor(shape, device, dtype, low=-9, high=9)
            expected_dtype = tensor.sum().dtype
            expected_dtype = torch_to_numpy_dtype_dict[expected_dtype]

            result = np.trace(tensor.cpu().numpy(), dtype=expected_dtype)
            expected = torch.tensor(result, device=device)
            self.assertEqual(tensor.trace(), expected)

        shapes = (
            [10, 1],
            [1, 10],
            [100, 100],
            [20, 100],
            [100, 20],
        )
        for shape in shapes:
            test(shape)

    def generate_clamp_baseline(self, device, dtype, *, min_vals, max_vals, with_nans):
        """
        Creates a random tensor for a given device and dtype, and computes the expected clamped
        values given the min_vals and/or max_vals.
        If with_nans is provided, then some values are randomly set to nan.
        """
        X = torch.rand(100, device=device).mul(50).add(-25)  # uniform in [-25, 25]
        X = X.to(dtype)
        if with_nans:
            mask = torch.randint(0, 2, X.shape, dtype=torch.bool, device=device)
            X[mask] = nan

        if isinstance(min_vals, torch.Tensor):
            min_vals = min_vals.cpu().numpy()

        if isinstance(max_vals, torch.Tensor):
            max_vals = max_vals.cpu().numpy()

        # Use NumPy implementation as reference
        X_clamped = torch.tensor(np.clip(X.cpu().numpy(), a_min=min_vals, a_max=max_vals), device=device)
        return X, X_clamped

    # Tests clamp and its alias, clip
    @dtypes(torch.int64, torch.float32)
    def test_clamp(self, device, dtype):
        op_list = (torch.clamp, torch.Tensor.clamp, torch.Tensor.clamp_,
                   torch.clip, torch.Tensor.clip, torch.Tensor.clip_)

        # min/max argument product
        args = product((-10, None), (10, None))

        for op in op_list:
            for min_val, max_val in args:
                if min_val is None and max_val is None:
                    continue

                X, Y_expected = self.generate_clamp_baseline(device, dtype,
                                                             min_vals=min_val,
                                                             max_vals=max_val,
                                                             with_nans=False)

                # Test op
                X1 = X.clone()  # So that the in-place ops do not change X
                Y_actual = op(X1, min_val, max_val)
                self.assertEqual(Y_expected, Y_actual)

                # Test op-out behavior (out does not exist for method versions)
                if op in (torch.clamp, torch.clip):
                    Y_out = torch.empty_like(X)
                    op(X, min=min_val, max=max_val, out=Y_out)
                    self.assertEqual(Y_expected, Y_out)

    def test_clamp_propagates_nans(self, device):
        op_list = (torch.clamp, torch.Tensor.clamp, torch.Tensor.clamp_,
                   torch.clip, torch.Tensor.clip, torch.Tensor.clip_)

        # min/max argument product
        args = product((-10, None), (10, None))

        for op in op_list:
            for min_val, max_val in args:
                if min_val is None and max_val is None:
                    continue

                X, Y_expected = self.generate_clamp_baseline(device, torch.float,
                                                             min_vals=min_val,
                                                             max_vals=max_val,
                                                             with_nans=True)
                Y_expected = torch.isnan(Y_expected)

                # Test op
                X1 = X.clone()  # So that the in-place ops do not change X
                Y_actual = op(X1, min_val, max_val)
                self.assertEqual(Y_expected, torch.isnan(Y_actual))

                # Test op-out behavior (out does not exist for method versions)
                if op in (torch.clamp, torch.clip):
                    Y_out = torch.empty_like(X)
                    op(X, min_val, max_val, out=Y_out)
                    self.assertEqual(Y_expected, torch.isnan(Y_out))

    def test_clamp_raises_arg_errors(self, device):
        X = torch.randn(100, dtype=torch.float, device=device)
        error_msg = 'At least one of \'min\' or \'max\' must not be None'
        with self.assertRaisesRegex(RuntimeError, error_msg):
            X.clamp()
        with self.assertRaisesRegex(RuntimeError, error_msg):
            X.clamp_()
        with self.assertRaisesRegex(RuntimeError, error_msg):
            torch.clamp(X)

    def test_flip(self, device):
        data = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], device=device).view(2, 2, 2)

        self.assertEqual(torch.tensor([5, 6, 7, 8, 1, 2, 3, 4]).view(2, 2, 2), data.flip(0))
        self.assertEqual(torch.tensor([3, 4, 1, 2, 7, 8, 5, 6]).view(2, 2, 2), data.flip(1))
        self.assertEqual(torch.tensor([2, 1, 4, 3, 6, 5, 8, 7]).view(2, 2, 2), data.flip(2))
        self.assertEqual(torch.tensor([7, 8, 5, 6, 3, 4, 1, 2]).view(2, 2, 2), data.flip(0, 1))
        self.assertEqual(torch.tensor([8, 7, 6, 5, 4, 3, 2, 1]).view(2, 2, 2), data.flip(0, 1, 2))

        # check for wrap dim
        self.assertEqual(torch.tensor([2, 1, 4, 3, 6, 5, 8, 7]).view(2, 2, 2), data.flip(-1))
        # check for permute
        self.assertEqual(torch.tensor([6, 5, 8, 7, 2, 1, 4, 3]).view(2, 2, 2), data.flip(0, 2))
        self.assertEqual(torch.tensor([6, 5, 8, 7, 2, 1, 4, 3]).view(2, 2, 2), data.flip(2, 0))

        # not allow flip on the same dim more than once
        self.assertRaises(RuntimeError, lambda: data.flip(0, 1, 1))
        # not allow empty list as input
        self.assertRaises(TypeError, lambda: data.flip())

        # not allow size of flip dim > total dims
        self.assertRaises(IndexError, lambda: data.flip(0, 1, 2, 3))
        # not allow dim > max dim
        self.assertRaises(IndexError, lambda: data.flip(3))

        # test for non-contiguous case
        expanded_data = torch.arange(1, 4, device=device).view(3, 1).expand(3, 2)
        transposed_data = torch.arange(1, 9, device=device).view(2, 2, 2).transpose(0, 1)
        self.assertEqual(torch.tensor([3, 3, 2, 2, 1, 1]).view(3, 2), expanded_data.flip(0))
        self.assertEqual(torch.tensor([8, 7, 4, 3, 6, 5, 2, 1]).view(2, 2, 2), transposed_data.flip(0, 1, 2))

        # test for shape
        data = torch.randn(2, 3, 4, device=device)
        size = [2, 3, 4]
        test_dims = []
        for i in range(1, 3):
            test_dims += combinations(range(len(size)), i)

        for ds in test_dims:
            self.assertEqual(size, list(data.flip(ds).size()))

        # test rectangular case
        data = torch.tensor([1, 2, 3, 4, 5, 6], device=device).view(2, 3)
        flip0_result = torch.tensor([[4, 5, 6], [1, 2, 3]], device=device)
        flip1_result = torch.tensor([[3, 2, 1], [6, 5, 4]], device=device)

        self.assertEqual(flip0_result, data.flip(0))
        self.assertEqual(flip1_result, data.flip(1))

        # test empty tensor, should just return an empty tensor of the same shape
        data = torch.tensor((), device=device)
        self.assertEqual(data, data.flip(0))

        # test bool tensor
        a = torch.tensor([False, True], device=device)
        self.assertEqual(a.flip(0), torch.tensor([True, False]))

        # case: dims=()
        a = torch.randn(3, 2, 1, device=device)
        self.assertEqual(a.flip(dims=()), a)

    def _rand_shape(self, dim, min_size, max_size):
        shape = []
        for i in range(dim):
            shape.append(random.randint(min_size, max_size))
        return tuple(shape)

    @dtypes(torch.cfloat, torch.cdouble)
    def test_complex_flip(self, device, dtype):
        rand_dim = random.randint(3, 4)
        shape = self._rand_shape(rand_dim, 5, 10)

        # Axis to sample for given shape.
        for i in range(1, rand_dim):
            # Check all combinations of `i` axis.
            for flip_dim in combinations(range(rand_dim), i):
                data = torch.randn(*shape, device=device, dtype=dtype)
                torch_fn = partial(torch.flip, dims=flip_dim)
                np_fn = partial(np.flip, axis=flip_dim)
                self.compare_with_numpy(torch_fn, np_fn, data)

    def _test_fliplr_flipud(self, torch_fn, np_fn, min_dim, max_dim, device, dtype):
        for dim in range(min_dim, max_dim + 1):
            shape = self._rand_shape(dim, 5, 10)
            # Randomly scale the input
            if dtype.is_floating_point or dtype.is_complex:
                data = torch.randn(*shape, device=device, dtype=dtype)
            else:
                data = torch.randint(0, 10, shape, device=device, dtype=dtype)
            self.compare_with_numpy(torch_fn, np_fn, data)

    @dtypes(torch.int64, torch.double, torch.cdouble)
    def test_fliplr(self, device, dtype):
        self._test_fliplr_flipud(torch.fliplr, np.fliplr, 2, 4, device, dtype)

    @dtypes(torch.int64, torch.double, torch.cdouble)
    def test_fliplr_invalid(self, device, dtype):
        x = torch.randn(42).to(dtype)
        with self.assertRaisesRegex(RuntimeError, "Input must be >= 2-d."):
            torch.fliplr(x)
        with self.assertRaisesRegex(RuntimeError, "Input must be >= 2-d."):
            torch.fliplr(torch.tensor(42, device=device, dtype=dtype))

    @dtypes(torch.int64, torch.double, torch.cdouble)
    def test_flipud(self, device, dtype):
        self._test_fliplr_flipud(torch.flipud, np.flipud, 1, 4, device, dtype)

    @dtypes(torch.int64, torch.double, torch.cdouble)
    def test_flipud_invalid(self, device, dtype):
        with self.assertRaisesRegex(RuntimeError, "Input must be >= 1-d."):
            torch.flipud(torch.tensor(42, device=device, dtype=dtype))

    def test_rot90(self, device):
        data = torch.arange(1, 5, device=device).view(2, 2)
        self.assertEqual(torch.tensor([1, 2, 3, 4]).view(2, 2), data.rot90(0, [0, 1]))
        self.assertEqual(torch.tensor([2, 4, 1, 3]).view(2, 2), data.rot90(1, [0, 1]))
        self.assertEqual(torch.tensor([4, 3, 2, 1]).view(2, 2), data.rot90(2, [0, 1]))
        self.assertEqual(torch.tensor([3, 1, 4, 2]).view(2, 2), data.rot90(3, [0, 1]))

        # test for default args k=1, dims=[0, 1]
        self.assertEqual(data.rot90(), data.rot90(1, [0, 1]))

        # test for reversed order of dims
        self.assertEqual(data.rot90(3, [0, 1]), data.rot90(1, [1, 0]))

        # test for modulo of k
        self.assertEqual(data.rot90(5, [0, 1]), data.rot90(1, [0, 1]))
        self.assertEqual(data.rot90(3, [0, 1]), data.rot90(-1, [0, 1]))
        self.assertEqual(data.rot90(-5, [0, 1]), data.rot90(-1, [0, 1]))

        # test for dims out-of-range error
        self.assertRaises(RuntimeError, lambda: data.rot90(1, [0, -3]))
        self.assertRaises(RuntimeError, lambda: data.rot90(1, [0, 2]))

        # test tensor with more than 2D
        data = torch.arange(1, 9, device=device).view(2, 2, 2)
        self.assertEqual(torch.tensor([2, 4, 1, 3, 6, 8, 5, 7]).view(2, 2, 2), data.rot90(1, [1, 2]))
        self.assertEqual(data.rot90(1, [1, -1]), data.rot90(1, [1, 2]))

        # test for errors
        self.assertRaises(RuntimeError, lambda: data.rot90(1, [0, 3]))
        self.assertRaises(RuntimeError, lambda: data.rot90(1, [1, 1]))
        self.assertRaises(RuntimeError, lambda: data.rot90(1, [0, 1, 2]))
        self.assertRaises(RuntimeError, lambda: data.rot90(1, [0]))

    @dtypes(torch.cfloat, torch.cdouble)
    def test_complex_rot90(self, device, dtype):
        shape = self._rand_shape(random.randint(2, 4), 5, 10)
        for rot_times in range(4):
            data = torch.randn(*shape, device=device, dtype=dtype)
            torch_fn = partial(torch.rot90, k=rot_times, dims=[0, 1])
            np_fn = partial(np.rot90, k=rot_times, axes=[0, 1])
            self.compare_with_numpy(torch_fn, np_fn, data)

    # TODO: update once warning flag is available to always trigger ONCE warnings
    # Ensures nonzero does not throw a warning, even when the as_tuple argument
    #   is not provided
    def test_nonzero_no_warning(self, device):
        t = torch.randn((2, 2), device=device)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            torch.nonzero(t)
            t.nonzero()
            self.assertEqual(len(w), 0)

    @dtypes(*torch.testing.get_all_dtypes(include_complex=False))
    def test_nonzero(self, device, dtype):

        shapes = [
            torch.Size((12,)),
            torch.Size((12, 1)),
            torch.Size((1, 12)),
            torch.Size((6, 2)),
            torch.Size((3, 2, 2)),
            torch.Size((5, 5, 5)),
        ]

        def gen_nontrivial_input(shape, dtype, device):
            if dtype != torch.bfloat16:
                return torch.randint(2, shape, device=device, dtype=dtype)
            else:
                # windows does not work for bfloat16 randing
                return torch.randint(2, shape, device=device, dtype=torch.float).to(dtype)

        for shape in shapes:
            tensor = gen_nontrivial_input(shape, dtype, device)
            dst1 = torch.nonzero(tensor, as_tuple=False)
            dst2 = tensor.nonzero(as_tuple=False)
            dst3 = torch.empty([], dtype=torch.long, device=device)
            torch.nonzero(tensor, out=dst3)
            if self.device_type != 'xla':
                # xla does not raise runtime error
                self.assertRaisesRegex(
                    RuntimeError,
                    "scalar type Long",
                    lambda: torch.nonzero(tensor, out=torch.empty([], dtype=torch.float, device=device))
                )
            if self.device_type == 'cuda':
                self.assertRaisesRegex(
                    RuntimeError,
                    "on the same device",
                    lambda: torch.nonzero(tensor, out=torch.empty([], dtype=torch.long))
                )
            np_array = tensor.cpu().numpy() if dtype != torch.bfloat16 else tensor.float().cpu().numpy()
            np_result = torch.from_numpy(np.stack(np_array.nonzero())).t()
            self.assertEqual(dst1.cpu(), np_result, atol=0, rtol=0)
            self.assertEqual(dst2.cpu(), np_result, atol=0, rtol=0)
            self.assertEqual(dst3.cpu(), np_result, atol=0, rtol=0)
            tup1 = torch.nonzero(tensor, as_tuple=True)
            tup2 = tensor.nonzero(as_tuple=True)
            tup1 = torch.stack(tup1).t().cpu()
            tup2 = torch.stack(tup2).t().cpu()
            self.assertEqual(tup1, np_result, atol=0, rtol=0)
            self.assertEqual(tup2, np_result, atol=0, rtol=0)

    def test_nonzero_astuple_out(self, device):
        t = torch.randn((3, 3, 3), device=device)
        out = torch.empty_like(t, dtype=torch.long)

        with self.assertRaises(RuntimeError):
            torch.nonzero(t, as_tuple=True, out=out)

        self.assertEqual(torch.nonzero(t, as_tuple=False, out=out), torch.nonzero(t, out=out))

        # Verifies that JIT script cannot handle the as_tuple kwarg
        # See Issue https://github.com/pytorch/pytorch/issues/45499.
        def _foo(t):
            tuple_result = torch.nonzero(t, as_tuple=True)
            nontuple_result = torch.nonzero(t, as_tuple=False)
            out = torch.empty_like(nontuple_result)
            torch.nonzero(t, as_tuple=False, out=out)
            return tuple_result, nontuple_result, out

        with self.assertRaises(RuntimeError):
            scripted_foo = torch.jit.script(_foo)

        # Verifies that JIT tracing works fine
        traced_foo = torch.jit.trace(_foo, t)
        traced_tuple, traced_nontuple, traced_out = traced_foo(t)
        expected_tuple = torch.nonzero(t, as_tuple=True)
        expected_nontuple = torch.nonzero(t)

        self.assertEqual(traced_tuple, expected_tuple)
        self.assertEqual(traced_nontuple, expected_nontuple)
        self.assertEqual(traced_out, expected_nontuple)

    @onlyOnCPUAndCUDA
    def test_nonzero_discontiguous(self, device):
        shape = (4, 4)
        tensor = torch.randint(2, shape, device=device)
        tensor_nc = torch.empty(shape[0], shape[1] * 2, device=device)[:, ::2].copy_(tensor)
        dst1 = tensor.nonzero(as_tuple=False)
        dst2 = tensor_nc.nonzero(as_tuple=False)
        self.assertEqual(dst1, dst2, atol=0, rtol=0)
        dst3 = torch.empty_like(dst1)
        data_ptr = dst3.data_ptr()
        # expect dst3 storage to be reused
        torch.nonzero(tensor, out=dst3)
        self.assertEqual(data_ptr, dst3.data_ptr())
        self.assertEqual(dst1, dst3, atol=0, rtol=0)
        # discontiguous out
        dst4 = torch.empty(dst1.size(0), dst1.size(1) * 2, dtype=torch.long, device=device)[:, ::2]
        data_ptr = dst4.data_ptr()
        strides = dst4.stride()
        torch.nonzero(tensor, out=dst4)
        self.assertEqual(data_ptr, dst4.data_ptr())
        self.assertEqual(dst1, dst4, atol=0, rtol=0)
        self.assertEqual(strides, dst4.stride())

    def test_nonzero_non_diff(self, device):
        x = torch.randn(10, requires_grad=True)
        nz = x.nonzero()
        self.assertFalse(nz.requires_grad)

class TestShapeFuncs(TestCase):
    """Test suite for Shape manipulating operators using the ShapeFuncInfo."""

    @dtypes(*(torch.uint8, torch.int64, torch.double, torch.complex128))
    @ops([op for op in shape_funcs if op.name in ['tile', 'repeat']])
    def test_repeat_tile_vs_numpy(self, device, dtype, op):
        samples = op.sample_inputs(device, dtype, requires_grad=False)
        for sample in samples:
            assert isinstance(sample.input, torch.Tensor)
            expected = op.ref(sample.input.cpu().numpy(), *sample.args, **sample.kwargs)
            result = op(sample.input, *sample.args, **sample.kwargs).cpu().numpy()
            self.assertEqual(expected, result)

instantiate_device_type_tests(TestShapeOps, globals())
instantiate_device_type_tests(TestShapeFuncs, globals())

if __name__ == '__main__':
    run_tests()
