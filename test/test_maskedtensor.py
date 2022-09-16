# Owner(s): ["module: masked operators"]

import torch
from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
    make_tensor,
    parametrize,
    instantiate_parametrized_tests,
)
from torch.testing._internal.common_methods_invocations import (
    SampleInput,
)

from torch.masked import MaskedTensor, masked_bmm
from torch.masked.maskedtensor.core import _masks_match, _tensors_match
from torch.masked.maskedtensor.unary import NATIVE_INPLACE_UNARY_FNS, NATIVE_UNARY_FNS
from torch.masked.maskedtensor.binary import NATIVE_BINARY_FNS, NATIVE_INPLACE_BINARY_FNS


def _compare_mt_t(mt_result, t_result):
    mask = mt_result.get_mask()
    mt_result_data = mt_result.get_data()
    if mask.layout in {torch.sparse_coo, torch.sparse_csr}:
        mask = mask.to_dense()
    if mt_result_data.layout in {torch.sparse_coo, torch.sparse_csr}:
        mt_result_data = mt_result_data.to_dense()
    a = mt_result_data.detach().masked_fill_(~mask, 0)
    b = t_result.detach().masked_fill_(~mask, 0)
    if not _tensors_match(a, b, exact=False):
        raise ValueError("The data in MaskedTensor a and Tensor b do not match")

def _compare_mts(mt1, mt2):
    mt_data1 = mt1.get_data()
    mt_data2 = mt2.get_data()
    if mt_data1.layout != mt_data2.layout:
        raise ValueError("mt1's data and mt2's data do not have the same layout. "
                         f"mt1.get_data().layout = {mt_data1.layout} while mt2.get_data().layout = {mt_data2.layout}")

    mask = mt1.get_mask()
    mask2 = mt2.get_mask()
    if not _masks_match(mt1, mt2):
        raise ValueError("mt1 and mt2 must have matching masks")
    if mask.layout != mask2.layout:
        raise ValueError("mt1's mask and mt2's mask do not have the same layout. "
                         f"mt1.get_mask().layout = {mask.layout} while mt2.get_mask().layout = {mask2.layout}")
    if mask.layout in {torch.sparse_coo, torch.sparse_csr}:
        mask = mask.to_dense()

    if mt_data1.layout in {torch.sparse_coo, torch.sparse_csr}:
        mt_data1 = mt_data1.to_dense()
        mt_data2 = mt_data2.to_dense()
    a = mt_data1.detach().masked_fill_(~mask, 0)
    b = mt_data2.detach().masked_fill_(~mask, 0)

    if not _tensors_match(a, b, exact=False):
        raise ValueError("The data in MaskedTensor mt1 and MaskedTensor mt2 do not match")

def _create_random_mask(shape, device):
    return make_tensor(
        shape, device=device, dtype=torch.bool, low=0, high=1, requires_grad=False
    )

def _generate_sample_data(
    device="cpu", dtype=torch.float, requires_grad=True, layout=torch.strided
):
    assert layout in {
        torch.strided,
        torch.sparse_coo,
        torch.sparse_csr,
    }, "Layout must be strided/sparse_coo/sparse_csr"
    shapes = [
        [],
        [2],
        [3, 5],
        [3, 2, 1, 2],
    ]
    inputs = []
    for s in shapes:
        data = make_tensor(s, device=device, dtype=dtype, requires_grad=requires_grad)  # type: ignore[arg-type]
        mask = _create_random_mask(s, device)
        if layout == torch.sparse_coo:
            mask = mask.to_sparse_coo().coalesce()
            data = data.sparse_mask(mask).requires_grad_(requires_grad)
        elif layout == torch.sparse_csr:
            if data.ndim != 2 and mask.ndim != 2:
                continue
            mask = mask.to_sparse_csr()
            data = data.sparse_mask(mask)
        inputs.append(SampleInput(data, kwargs={"mask": mask}))
    return inputs

def _fix_fn_name(fn_name):
    if fn_name[-1] == "_":
        fn_name = fn_name[:-1]
    return fn_name


class TestUnary(TestCase):
    def _get_test_data(self, fn_name):
        data = torch.randn(10, 10)
        mask = torch.rand(10, 10) > 0.5
        fn_name = _fix_fn_name(fn_name)
        if fn_name in ["log", "log10", "log1p", "log2", "sqrt"]:
            data = data.mul(0.5).abs()
        if fn_name in ["rsqrt"]:
            data = data.abs() + 1  # Void division by zero
        if fn_name in ["acos", "arccos", "asin", "arcsin", "logit"]:
            data = data.abs().mul(0.5).clamp(0, 1)
        if fn_name in ["atanh", "arctanh", "erfinv"]:
            data = data.mul(0.5).clamp(-1, 1)
        if fn_name in ["acosh", "arccosh"]:
            data = data.abs() + 1
        if fn_name in ["bitwise_not"]:
            data = data.mul(128).to(torch.int8)
        return data, mask

    def _get_sample_kwargs(self, fn_name):
        fn_name = _fix_fn_name(fn_name)
        kwargs = {}
        if fn_name in ["clamp", "clip"]:
            kwargs["min"] = -0.5
            kwargs["max"] = 0.5
        return kwargs

    def _get_sample_args(self, fn_name, data, mask):
        fn_name = _fix_fn_name(fn_name)
        mt = MaskedTensor(data, mask)
        t_args = [data]
        mt_args = [mt]
        if fn_name in ["pow"]:
            t_args += [2.0]
            mt_args += [2.0]
        return t_args, mt_args

    @parametrize("fn", NATIVE_UNARY_FNS)
    def test_unary(self, fn):
        torch.random.manual_seed(0)
        fn_name = fn.__name__
        data, mask = self._get_test_data(fn_name)
        kwargs = self._get_sample_kwargs(fn_name)

        t_args, mt_args = self._get_sample_args(fn_name, data, mask)

        mt_result = fn(*mt_args, **kwargs)
        t_result = fn(*t_args, **kwargs)
        _compare_mt_t(mt_result, t_result)

    @parametrize("fn", NATIVE_INPLACE_UNARY_FNS)
    def test_inplace_unary(self, fn):
        torch.random.manual_seed(0)
        fn_name = fn.__name__
        data, mask = self._get_test_data(fn_name)
        kwargs = self._get_sample_kwargs(fn_name)

        t_args, mt_args = self._get_sample_args(fn_name, data, mask)

        mt_result = fn(*mt_args, **kwargs)
        t_result = fn(*t_args, **kwargs)
        _compare_mt_t(mt_result, t_result)

class TestBinary(TestCase):
    def _get_test_data(self, fn_name):
        fn_name = _fix_fn_name(fn_name)
        data0 = torch.randn(10, 10)
        data1 = torch.randn(10, 10)
        mask = torch.rand(10, 10) > 0.5
        if fn_name in ["bitwise_and", "bitwise_or", "bitwise_xor"]:
            data0 = data0.mul(128).to(torch.int8)
            data1 = data1.mul(128).to(torch.int8)
        if fn_name in ["bitwise_left_shift", "bitwise_right_shift"]:
            data0 = data0.abs().to(torch.int64)
            data1 = data1.abs().to(torch.int64)
        return data0, data1, mask

    def _get_sample_kwargs(self, fn_name):
        fn_name = _fix_fn_name(fn_name)
        kwargs = {}
        return kwargs

    def _yield_sample_args(self, fn_name, data0, data1, mask):
        """ Returns two sets of Tensor and MaskedTensor args for a binary function to compute.
            Tensor args are all the same (just the two provided data tensors),
            while the MaskedTensor args tests both (MaskedTensor, MaskedTensor) and (MaskedTensor, Tensor)
        """
        fn_name = _fix_fn_name(fn_name)
        mt0 = MaskedTensor(data0, mask)
        mt1 = MaskedTensor(data1, mask)

        t_args = [data0, data1]
        mt_args = [mt0, mt1]
        yield t_args, mt_args

        t_args = [data0, data1]
        mt_args = [mt0, data1]
        yield t_args, mt_args

    @parametrize("fn", NATIVE_BINARY_FNS)
    def test_binary(self, fn):
        torch.random.manual_seed(0)
        fn_name = fn.__name__
        data0, data1, mask = self._get_test_data(fn_name)
        kwargs = self._get_sample_kwargs(fn_name)

        for (t_args, mt_args) in self._yield_sample_args(fn_name, data0, data1, mask):
            mt_result = fn(*mt_args, **kwargs)
            t_result = fn(*t_args, **kwargs)
            _compare_mt_t(mt_result, t_result)

    @parametrize("fn", NATIVE_INPLACE_BINARY_FNS)
    def test_inplace_binary(self, fn):
        torch.random.manual_seed(0)
        fn_name = fn.__name__
        data0, data1, mask = self._get_test_data(fn_name)
        kwargs = self._get_sample_kwargs(fn_name)

        for (t_args, mt_args) in self._yield_sample_args(fn_name, data0, data1, mask):
            mt_result = fn(*mt_args, **kwargs)
            t_result = fn(*t_args, **kwargs)
            _compare_mt_t(mt_result, t_result)

    @parametrize("fn_name", ["add", "add_"])
    def test_masks_match(self, fn_name):
        torch.random.manual_seed(0)
        fn = getattr(torch.ops.aten, fn_name)
        data0, data1, mask = self._get_test_data(fn_name)
        mask0 = mask
        mask1 = torch.rand(mask.size()) > 0.5
        mt0 = MaskedTensor(data0, mask0)
        mt1 = MaskedTensor(data1, mask1)
        try:
            fn(mt0, mt1)
            raise AssertionError()
        except ValueError as e:
            assert (
                "Input masks must match. If you need support for this, please open an issue on Github."
                == str(e)
            )

class TestReductions(TestCase):
    def test_max_not_implemented(self):
        d = torch.tensor([[0, 1, 2], [3, 4, 5.0]])
        m = torch.tensor([[True, False, False], [False, True, False]])
        mt = MaskedTensor(d, m)
        with self.assertRaisesRegex(TypeError, "no implementation found for 'torch._ops.aten.max.default'"):
            mt.max()

    def test_sum(self):
        d = torch.tensor([[0, 1, 2, 6], [3, 4, 5.0, 7]])
        m = torch.tensor([[True, False, False, True], [False, True, False, True]])
        mt = MaskedTensor(d, m)
        _compare_mts(MaskedTensor(torch.tensor(17.0), torch.tensor(True)), mt.sum())
        _compare_mts(
            MaskedTensor(
                torch.tensor([0.0, 4.0, 1.0, 13]),
                torch.tensor([True, True, False, True]),
            ),
            mt.sum(dim=0),
        )

    def test_sum_grad(self):
        d = torch.tensor([[0, 1, 2], [3, 4, 5.0]])
        m = torch.tensor([[True, False, False], [False, True, False]])
        mt = MaskedTensor(d, m, requires_grad=True)
        mt.sum().backward()
        _compare_mts(mt.grad, MaskedTensor(torch.tensor(1.0).expand_as(m), m))

    def test_mean(self):
        d = torch.tensor([[0, 1, 3, 2], [3, 4, 1.0, 4]])
        m = torch.tensor([[True, False, False, True], [False, True, False, True]])
        mt = MaskedTensor(d, m)
        _compare_mts(MaskedTensor(torch.tensor(2.5), torch.tensor(True)), mt.mean())
        _compare_mts(
            MaskedTensor(
                torch.tensor([0.0, 4.0, 1.0, 3]),
                torch.tensor([True, True, False, True]),
            ),
            mt.mean(dim=0),
        )

    """
        The following block of tests "test_mean_grad_case_1[a through e] are used to test the functionality of
        the two different ways of constructing MaskedTensors:
            MaskedTensor(data, mask, requires_grad=True/False) -- NO differentiable constructor and always a leaf
            MaskedTensor.from_values(data, mask) -- differentiable constructor

        Like torch.tensor(data), MaskedTensor(data, mask) will provide a UserWarning if data.requires_grad=True
        MaskedTensor.from_values does not take in requires_grad -- it just takes on the requires_grad from data

        Therefore, there are 6 cases to test and we use `mean` as a proxy to test the different combinations

        Assuming mt.mean().backward() is run after each constructor:

        Case 1a:
            values.requires_grad = True
            mt = MaskedTensor(values, mask, requires_grad=True)
        yields
            - Provide a UserWarning because values.requires_grad=True
            - values.grad = None
            - mt.grad is a MaskedTensor with the correct gradient

        Case 1b:
            values.requires_grad = False
            mt = MaskedTensor(values, mask, requires_grad=True)
        yields
            - values.grad = None
            - mt.grad is a MaskedTensor with the correct gradient

        Case 2a/2b:
            values.requires_grad = True/False
            mt = MaskedTensor(values, mask, requires_grad=False)

            will both yield a RuntimeError of "element 0 of tensors does not require grad and does not have a grad_fn"
            as expected. When values.requires_grad=True, we will also get a UserWarning

        Case 3a:
            values.requires_grad = True
            mt = MaskedTensor.from_values(values, mask)
        yields
            - values.grad is a MaskedTensor with the correct gradient
            - mt.grad is None and gives a UserWarning that
              "The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. Its .grad"

        Case 3b:
            values.requires_grad = False
            mt = MaskedTensor.from_values(values, mask)

            will yield a RuntimeError of "element 0 of tensors does not require grad and does not have a grad_fn"
            as expected.
    """
    def test_mean_grad_case_1a(self):
        """ values.requires_grad = True
            mt = MaskedTensor(values, mask, requires_grad=True)
        """
        d = torch.tensor([[0, 1, 2], [3, 4, 5.0]], requires_grad=True)
        m = torch.tensor([[True, False, False], [False, True, False]])
        with self.assertWarnsRegex(UserWarning, "It is not recommended to create a MaskedTensor"):
            mt = MaskedTensor(d, m, requires_grad=True)
        mt.mean().backward()
        self.assertIsNone(d.grad)
        _compare_mts(mt.grad, MaskedTensor(torch.tensor([[0.5, 0, 0], [0, 0.5, 0]]), m))

    def test_mean_grad_case_1b(self):
        """ values.requires_grad = False
            mt = MaskedTensor(values, mask, requires_grad=True)
        """
        d = torch.tensor([[0, 1, 2], [3, 4, 5.0]])
        m = torch.tensor([[True, False, False], [False, True, False]])
        mt = MaskedTensor(d, m, requires_grad=True)
        mt.mean().backward()
        self.assertIsNone(d.grad)
        _compare_mts(mt.grad, MaskedTensor(torch.tensor([[0.5, 0, 0], [0, 0.5, 0]]), m))

    def test_mean_grad_case_1c(self):
        """ values.requires_grad = True
            mt = MaskedTensor(values, mask, requires_grad=False)
        """
        d = torch.tensor([[0, 1, 2], [3, 4, 5.0]], requires_grad=True)
        m = torch.tensor([[True, False, False], [False, True, False]])
        with self.assertWarnsRegex(UserWarning, "It is not recommended to create a MaskedTensor"):
            mt = MaskedTensor(d, m, requires_grad=False)
        result = mt.mean()
        msg = "element 0 of tensors does not require grad and does not have a grad_fn"
        with self.assertRaisesRegex(RuntimeError, msg):
            result.backward()


    def test_mean_grad_case_1d(self):
        """ values.requires_grad = False
            mt = MaskedTensor(values, mask, requires_grad=False)
        """
        d = torch.tensor([[0, 1, 2], [3, 4, 5.0]])
        m = torch.tensor([[True, False, False], [False, True, False]])
        mt = MaskedTensor(d, m, requires_grad=False)
        result = mt.mean()
        msg = "element 0 of tensors does not require grad and does not have a grad_fn"
        with self.assertRaisesRegex(RuntimeError, msg):
            result.backward()

    def test_mean_grad_case_1e(self):
        """ values.requires_grad = True
            mt = MaskedTensor.from_values(values, mask)
        """
        d = torch.tensor([[0, 1, 2], [3, 4, 5.0]], requires_grad=True)
        m = torch.tensor([[True, False, False], [False, True, False]])
        mt = MaskedTensor.from_values(d, m)
        mt.mean().backward()
        _compare_mts(d.grad, MaskedTensor(torch.tensor([[0.5, 0, 0], [0, 0.5, 0]]), m))
        msg = "The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. Its .grad"
        with self.assertWarnsRegex(UserWarning, msg):
            self.assertIsNone(mt.grad)

    def test_mean_grad_case_1f(self):
        """ values.requires_grad = False
            mt = MaskedTensor.from_values(values, mask)
        """
        d = torch.tensor([[0, 1, 2], [3, 4, 5.0]])
        m = torch.tensor([[True, False, False], [False, True, False]])
        mt = MaskedTensor.from_values(d, m)
        result = mt.mean()
        msg = "element 0 of tensors does not require grad and does not have a grad_fn"
        with self.assertRaisesRegex(RuntimeError, msg):
            result.backward()

    def test_mean_dim_grad(self):
        d = torch.tensor([[0, 1, 2], [3, 4, 5.0]])
        m = torch.tensor([[True, True, False], [False, True, False]])
        mt = MaskedTensor(d, m, requires_grad=True)
        mt.mean(1).sum().backward()
        _compare_mts(mt.grad, MaskedTensor(torch.tensor([[0.5, 0.5, 0], [0, 1, 0]]), m))

    def test_amax(self):
        d = torch.tensor([[0, 1, 3, -3], [3, -4, 1.0, 3]])
        m = torch.tensor([[True, False, False, True], [False, True, False, True]])
        mt = MaskedTensor(d, m)
        _compare_mts(MaskedTensor(torch.tensor(3.0), torch.tensor(True)), mt.amax())
        _compare_mts(
            MaskedTensor(
                torch.tensor([0.0, -4.0, 1.0, 3]),
                torch.tensor([True, True, False, True]),
            ),
            mt.amax(dim=0),
        )

    def test_amax_grad(self):
        d = torch.tensor([[0, 1, 2], [3, 4, 5.0]])
        m = torch.tensor([[True, False, False], [False, True, False]])
        mt = MaskedTensor(d, m, requires_grad=True)
        mt.amax().backward()
        _compare_mts(mt.grad, MaskedTensor(torch.tensor([[0.0, 0, 0], [0, 1, 0]]), m))

    def test_amin(self):
        d = torch.tensor([[0, 1, 3, -3], [3, -4, 1.0, 3]])
        m = torch.tensor([[True, False, False, True], [False, True, False, True]])
        mt = MaskedTensor(d, m)
        _compare_mts(MaskedTensor(torch.tensor(-4.0), torch.tensor(True)), mt.amin())
        _compare_mts(
            MaskedTensor(
                torch.tensor([0.0, -4.0, 1.0, -3]),
                torch.tensor([True, True, False, True]),
            ),
            mt.amin(dim=0),
        )

    def test_amin_grad(self):
        d = torch.tensor([[0, 1, 2], [3, 4, 5.0]])
        m = torch.tensor([[True, False, False], [False, True, False]])
        mt = MaskedTensor(d, m, requires_grad=True)
        mt.amin().backward()
        _compare_mts(mt.grad, MaskedTensor(torch.tensor([[1.0, 0, 0], [0, 0, 0]]), m))

    def test_prod(self):
        d = torch.tensor([[0, 1, 3, 0.0], [float("nan"), 4, 1.0, 5.0]])
        m = torch.tensor([[True, False, False, True], [False, True, False, True]])
        mt = MaskedTensor(d, m)
        _compare_mts(MaskedTensor(torch.tensor(0.0), torch.tensor(True)), mt.prod())
        _compare_mts(
            MaskedTensor(
                torch.tensor([0.0, 4.0, 1.0, 0.0]),
                torch.tensor([True, True, False, True]),
            ),
            mt.prod(dim=0),
        )

    def test_prod_grad(self):
        d = torch.tensor([[2, float("nan"), 2], [3, 4, 5.0]])
        m = torch.tensor([[True, False, False], [False, True, False]])
        mt = MaskedTensor(d, m, requires_grad=True)
        mt.prod().backward()
        _compare_mts(mt.grad, MaskedTensor(torch.tensor([[4.0, 0, 0], [0, 2, 0]]), m))

    def test_all(self):
        d = torch.tensor([[True, True, False, False], [False, True, True, True]])
        m = torch.tensor([[True, False, False, True], [False, True, False, True]])
        mt = MaskedTensor(d, m)
        _compare_mts(MaskedTensor(torch.tensor(False), torch.tensor(True)), mt.all())
        _compare_mts(
            MaskedTensor(
                torch.tensor([True, True, True, False]),
                torch.tensor([True, True, False, True]),
            ),
            mt.all(dim=0),
        )

        m = torch.tensor([[True, False, True, False], [False, True, False, False]])
        mt = MaskedTensor(d, m)
        _compare_mts(
            MaskedTensor(
                torch.tensor([True, True, False, True]),
                torch.tensor([True, True, True, False]),
            ),
            mt.all(dim=0),
        )

    def test_grad_dtype(self):
        d = torch.tensor([[True, True, False], [False, True, True]])
        m = torch.tensor([[True, False, False], [False, True, False]])
        msg = "Only Tensors of floating point and complex dtype can require gradients"
        with self.assertRaisesRegex(RuntimeError, msg):
            MaskedTensor(d, m, requires_grad=True)

class TestMatMul(TestCase):
    def test_bmm(self):
        x = torch.rand(3, 2, 1)
        key_padding_mask = torch.tensor(
            [
                [False, False, False],
                [False, True, True],
            ]
        )
        x_mt = MaskedTensor(x, ~(key_padding_mask.transpose(0, 1).unsqueeze(-1)))
        x = x.masked_fill(~x_mt.get_mask(), 0)
        attn_2 = torch.bmm(x, x.transpose(-2, -1))
        attn_3 = torch.bmm(x_mt, x_mt.transpose(-2, -1))
        self.assertEqual(attn_3.get_data().masked_fill(~attn_3.get_mask(), 0), attn_2)  # type: ignore[attr-defined]

    def test_masked_bmm(self):
        key_padding_mask = torch.tensor(
            [
                [False, False, False, True],
                [False, True, True, True],
                [False, True, False, True],
            ]
        )
        x = torch.arange(4 * 3 * 2).reshape(4, 3, 2).float()
        x_mt = MaskedTensor(
            x,
            ~(key_padding_mask.transpose(0, 1).unsqueeze(-1).expand_as(x)),
            requires_grad=True,
        )
        attn_mask_bool = torch.tensor(
            [
                [False, True, True],
                [False, False, True],
                [True, False, False],
            ]
        )
        attn_mask = attn_mask_bool.float().masked_fill_(attn_mask_bool, float("-inf"))
        v = masked_bmm(x, x_mt.transpose(1, 2), attn_mask)
        v.sum().backward()

    def test_linear(self):
        x = torch.arange(4 * 3 * 2).reshape(4, 3, 2)
        w_x = torch.arange(10).reshape(5, 2) + x.amax()
        linear = torch.nn.functional.linear
        key_padding_mask = torch.tensor(
            [
                [False, False, False, True],
                [False, True, True, True],
                [False, True, False, True],
            ]
        )
        x_mt = MaskedTensor(
            x, ~(key_padding_mask.transpose(0, 1).unsqueeze(-1).expand_as(x))
        )

instantiate_parametrized_tests(TestUnary)
instantiate_parametrized_tests(TestBinary)
instantiate_parametrized_tests(TestReductions)
instantiate_parametrized_tests(TestMatMul)

if __name__ == '__main__':
    run_tests()
