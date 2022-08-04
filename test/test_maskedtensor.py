# Owner(s): ["module: masked operators"]

import torch
from torch.testing._internal.common_utils import (
    TestCase, run_tests, make_tensor, parametrize, instantiate_parametrized_tests,
)
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    ops,
)
from torch.testing._internal.common_methods_invocations import (
    SampleInput,
)

from torch._masked import _combine_input_and_mask
from torch.masked.maskedtensor.core import _masks_match, _tensors_match
from torch.masked.maskedtensor import masked_tensor, masked_bmm
from torch.masked.maskedtensor.unary import NATIVE_INPLACE_UNARY_FNS, NATIVE_UNARY_FNS, UNARY_NAMES
from torch.masked.maskedtensor.binary import NATIVE_BINARY_FNS, NATIVE_INPLACE_BINARY_FNS, BINARY_NAMES

def _compare_mt_t(mt_result, t_result):
    mask = mt_result.masked_mask
    mt_result_data = mt_result.masked_data
    if mask.layout in {torch.sparse_coo, torch.sparse_csr}:
        mask = mask.to_dense()
    if mt_result_data.layout in {torch.sparse_coo, torch.sparse_csr}:
        mt_result_data = mt_result_data.to_dense()
    a = mt_result_data.detach().masked_fill_(~mask, 0)
    b = t_result.detach().masked_fill_(~mask, 0)
    assert _tensors_match(a, b, exact=False)


def _compare_mts(mt1, mt2):
    assert mt1.masked_data.layout == mt2.masked_data.layout
    assert mt1.masked_mask.layout == mt2.masked_mask.layout
    assert _masks_match(mt1, mt2)
    mask = mt1.masked_mask
    mt_data1 = mt1.masked_data
    mt_data2 = mt2.masked_data
    if mask.layout in {torch.sparse_coo, torch.sparse_csr}:
        mask = mask.to_dense()
    if mt_data1.layout in {torch.sparse_coo, torch.sparse_csr}:
        mt_data1 = mt_data1.to_dense()
        mt_data2 = mt_data2.to_dense()
    a = mt_data1.detach().masked_fill_(~mask, 0)
    b = mt_data2.detach().masked_fill_(~mask, 0)
    assert _tensors_match(a, b, exact=False)

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


class TestBasics(TestCase):
    def test_add(self):
        data = torch.arange(5.0)
        mask = torch.tensor([True, True, False, True, False])
        m0 = masked_tensor(data, mask)
        m1 = masked_tensor(data, ~mask)
        self.assertRaises(ValueError, lambda: m0 + m1)

    def test_softmax(self):
        x = torch.randn(3, 4) * 0.1
        m = torch.tensor(
            [
                [True, True, True, False],
                [False, True, False, True],
                [True, True, False, False],
            ]
        )
        mx = masked_tensor(x, m, requires_grad=True)
        ts = torch.softmax(mx, -1)
        ts.sum().backward()
        xinf = x.masked_fill(~m, float("-inf")).detach().clone().requires_grad_()
        torch.softmax(xinf, -1)

    def test_where(self):
        # http://pytorch.org/maskedtensor/main/notebooks/nan_grad.html
        x = torch.tensor(
            [-10.0, -5, 0, 5, 10, 50, 60, 70, 80, 90, 100], requires_grad=True
        )
        mask = x < 0
        mx = masked_tensor(x, mask, requires_grad=True)
        my = masked_tensor(torch.ones_like(x), ~mask, requires_grad=True)
        y = torch.where(mask, torch.exp(mx), my)
        s = y.sum()
        s.backward()

    def test_mha_issue_41508(self):
        # https://github.com/pytorch/pytorch/issues/41508
        import torch

        torch.manual_seed(0)
        attn_nn = torch.nn.MultiheadAttention(1, 1, bias=False)
        attn_mt = torch.nn.MultiheadAttention(1, 1, bias=False)
        for (na, a), (nb, b) in zip(
            attn_nn.named_parameters(), attn_mt.named_parameters()
        ):
            a.data.copy_(b.data)

        x = torch.rand(3, 2, 1)
        key_padding_mask = torch.as_tensor(
            [
                [False, False, False],
                [False, True, True],
            ]
        )
        attn_mask = torch.as_tensor(
            [
                [False, True, True],
                [False, False, True],
                [True, False, False],
            ]
        )
        output, scores = attn_nn(
            x, x, x, key_padding_mask=key_padding_mask, attn_mask=attn_mask
        )
        loss0 = output[0, :].sum()

        x_mt = masked_tensor(
            x, ~(key_padding_mask.transpose(0, 1).unsqueeze(-1).expand_as(x))
        )

        output, scores = attn_mt(x, x_mt, x, attn_mask=attn_mask)
        loss1 = output[0, :].sum()
        self.assertEqual(loss0, loss1.masked_data)

    def test_chunk(self):
        return
        # This breaks because split_backward allocates
        # Tensors using zero and then cats them together.
        # I don't know why the masks are coming into play here.
        # It's an autograd thing.
        k_data = torch.tensor([4.0])
        k_mask = torch.tensor([True])
        k = masked_tensor(k_data[0], k_mask[0], requires_grad=True)
        w = torch.tensor([1.0, 2.0], requires_grad=True)
        w_q, w_k = w.chunk(2)
        o0 = k + w_k
        o0.backward()
        return

    def test_to_sparse(self):
        for sample in _generate_sample_data():
            data = sample.input
            mask = sample.kwargs["mask"]
            mt = masked_tensor(data.clone().detach(), mask, requires_grad=True)

            sparse_mt = mt.to_sparse()
            data.to_sparse().to_dense().sum().backward()
            sparse_mt.to_dense().sum().backward()

            _compare_mt_t(sparse_mt, data)
            _compare_mt_t(mt.grad, data.grad)

    def test_to_dense(self):
        samples = _generate_sample_data(
            layout=torch.sparse_coo
        ) + _generate_sample_data(layout=torch.sparse_csr)
        for sample in samples:
            data = sample.input
            mask = sample.kwargs["mask"]
            mt = masked_tensor(data.clone().detach(), mask, requires_grad=True)

            dense_data = data.to_dense().clone().detach().requires_grad_(True)
            dense_mt = mt.to_dense()
            dense_data.sum().backward()
            dense_mt.sum().backward()

            _compare_mt_t(dense_mt, dense_data)
            _compare_mt_t(mt.grad.to_dense(), dense_data.grad)

    def test_to_dense_and_sparse_coo(self):
        for sample in _generate_sample_data(layout=torch.strided):
            data = sample.input
            mask = sample.kwargs["mask"]
            ms = mask.to_sparse_coo().coalesce()

            t1 = data.clone().detach().requires_grad_(True)
            t1s = data.sparse_mask(ms).clone().detach().requires_grad_(True)
            mt = masked_tensor(t1, mask, requires_grad=True)
            mts = masked_tensor(t1s, ms, requires_grad=True)

            converted = mt.to_sparse().to_dense().requires_grad_(True)
            converted.sum().backward()

            converted2 = mts.to_dense().requires_grad_(True)
            converted2.sum().backward()

            _compare_mts(mt.grad, mts.grad.to_dense())

    def test_to_dense_and_sparse_csr(self):
        for sample in _generate_sample_data(layout=torch.strided):
            data = sample.input
            mask = sample.kwargs["mask"]
            if data.ndim != 2:
                continue
            ms = mask.to_sparse_csr()

            t1 = data.clone().detach().requires_grad_(True)
            t1s = data.sparse_mask(ms).clone().detach().requires_grad_(True)
            mt = masked_tensor(t1, mask, requires_grad=True)
            mts = masked_tensor(t1s, ms, requires_grad=True)

            converted = mt.to_sparse_csr().to_dense()
            converted.sum().backward()

            converted2 = mts.to_dense()
            converted2.sum().backward()

            _compare_mts(mt.grad, mts.grad.to_dense())

    def test_contiguous(self):
        data = torch.randn(3, 3)

        contiguous_data = data.clone()
        mask1 = (contiguous_data > 0).bool()
        not_contiguous_data = torch.as_strided(data.clone(), (2, 2), (1, 2))
        mask2 = (not_contiguous_data > 0).bool()

        contiguous_mt = masked_tensor(contiguous_data, mask1)
        not_contiguous_mt = masked_tensor(not_contiguous_data, mask2)

        contiguous_mt_sparse = masked_tensor(
            contiguous_data.to_sparse_coo(), mask1.to_sparse_coo()
        )
        not_contiguous_mt_sparse = masked_tensor(
            not_contiguous_data.to_sparse_coo(), mask2.to_sparse_coo()
        )

        self.assertEqual(contiguous_data.is_contiguous(), True)
        self.assertEqual(not_contiguous_data.is_contiguous(), False)

        self.assertEqual(contiguous_mt.is_contiguous(), True)
        self.assertEqual(not_contiguous_mt.is_contiguous(), False)

        error_msg = "MaskedTensors with sparse data do not have is_contiguous"
        for t in [contiguous_mt_sparse, not_contiguous_mt_sparse]:
            with self.assertRaisesRegex(ValueError, error_msg):
                t.is_contiguous()
            with self.assertRaisesRegex(ValueError, error_msg):
                t.contiguous()

        now_contiguous_mt = not_contiguous_mt.contiguous()

        self.assertEqual(now_contiguous_mt.is_contiguous(), True)
        self.assertEqual(now_contiguous_mt.masked_data.is_contiguous(), True)
        self.assertEqual(now_contiguous_mt.is_contiguous(), True)

class TestUnary(TestCase):
    def _get_test_data(self, fn_name):
        data = torch.randn(10, 10)
        mask = torch.rand(10, 10) > 0.5
        if fn_name[-1] == "_":
            fn_name = fn_name[:-1]
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
        if fn_name[-1] == "_":
            fn_name = fn_name[:-1]
        kwargs = {}
        if fn_name in ["clamp", "clip"]:
            kwargs["min"] = -0.5
            kwargs["max"] = 0.5
        return kwargs

    def _get_sample_args(self, fn_name, data, mask):
        if fn_name[-1] == "_":
            fn_name = fn_name[:-1]
        mt = masked_tensor(data, mask)
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
        if fn_name[-1] == "_":
            fn_name = fn_name[:-1]
        data0 = torch.randn(10, 10)
        data1 = torch.randn(10, 10)
        mask = torch.rand(10, 10) > 0.5
        if fn_name in ["bitwise_and", "bitwise_or", "bitwise_xor"]:
            data0 = data0.mul(128).to(torch.int8)
            data1 = data1.mul(128).to(torch.int8)
        if fn_name in ["bitwise_left_shift", "bitwise_right_shift"]:
            data0 = data0.to(torch.int64)
            data1 = data1.to(torch.int64)
        return data0, data1, mask

    def _get_sample_kwargs(self, fn_name):
        if fn_name[-1] == "_":
            fn_name = fn_name[:-1]
        kwargs = {}
        return kwargs

    def _yield_sample_args(self, fn_name, data0, data1, mask):
        if fn_name[-1] == "_":
            fn_name = fn_name[:-1]
        mt0 = masked_tensor(data0, mask)
        mt1 = masked_tensor(data1, mask)

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
        mt0 = masked_tensor(data0, mask0)
        mt1 = masked_tensor(data1, mask1)
        try:
            fn(mt0, mt1)
            raise AssertionError()
        except ValueError as e:
            assert (
                "Input masks must match. If you need support for this, please open an issue on Github."
                == str(e)
            )

class TestReductions(TestCase):
    def test_not_implemented(self):
        d = torch.tensor([[0, 1, 2], [3, 4, 5.0]])
        m = torch.tensor([[True, False, False], [False, True, False]])
        mt = masked_tensor(d, m)
        self.assertRaises(TypeError, lambda: mt.max())

    def test_sum(self):
        d = torch.tensor([[0, 1, 2, 6], [3, 4, 5.0, 7]])
        m = torch.tensor([[True, False, False, True], [False, True, False, True]])
        mt = masked_tensor(d, m)
        _compare_mts(masked_tensor(torch.tensor(17.0), torch.tensor(True)), mt.sum())
        _compare_mts(
            masked_tensor(
                torch.tensor([0.0, 4.0, 1.0, 13]),
                torch.tensor([True, True, False, True]),
            ),
            mt.sum(dim=0),
        )

    def test_sum_grad(self):
        d = torch.tensor([[0, 1, 2], [3, 4, 5.0]])
        m = torch.tensor([[True, False, False], [False, True, False]])
        mt = masked_tensor(d, m, requires_grad=True)
        mt.sum().backward()
        _compare_mts(mt.grad, masked_tensor(torch.tensor(1.0).expand_as(m), m))

    def test_mean(self):
        d = torch.tensor([[0, 1, 3, 2], [3, 4, 1.0, 4]])
        m = torch.tensor([[True, False, False, True], [False, True, False, True]])
        mt = masked_tensor(d, m)
        _compare_mts(masked_tensor(torch.tensor(2.5), torch.tensor(True)), mt.mean())
        _compare_mts(
            masked_tensor(
                torch.tensor([0.0, 4.0, 1.0, 3]),
                torch.tensor([True, True, False, True]),
            ),
            mt.mean(dim=0),
        )

    def test_mean_grad(self):
        d = torch.tensor([[0, 1, 2], [3, 4, 5.0]])
        m = torch.tensor([[True, False, False], [False, True, False]])
        mt = masked_tensor(d, m, requires_grad=True)
        mt.mean().backward()
        _compare_mts(mt.grad, masked_tensor(torch.tensor(1.0).expand_as(m), m))

    def test_amax(self):
        d = torch.tensor([[0, 1, 3, -3], [3, -4, 1.0, 3]])
        m = torch.tensor([[True, False, False, True], [False, True, False, True]])
        mt = masked_tensor(d, m)
        _compare_mts(masked_tensor(torch.tensor(3.0), torch.tensor(True)), mt.amax())
        _compare_mts(
            masked_tensor(
                torch.tensor([0.0, -4.0, 1.0, 3]),
                torch.tensor([True, True, False, True]),
            ),
            mt.amax(dim=0),
        )

    def test_amax_grad(self):
        d = torch.tensor([[0, 1, 2], [3, 4, 5.0]])
        m = torch.tensor([[True, False, False], [False, True, False]])
        mt = masked_tensor(d, m, requires_grad=True)
        mt.amax().backward()
        _compare_mts(mt.grad, masked_tensor(torch.tensor(1.0).expand_as(m), m))

    def test_amin(self):
        d = torch.tensor([[0, 1, 3, -3], [3, -4, 1.0, 3]])
        m = torch.tensor([[True, False, False, True], [False, True, False, True]])
        mt = masked_tensor(d, m)
        _compare_mts(masked_tensor(torch.tensor(-4.0), torch.tensor(True)), mt.amin())
        _compare_mts(
            masked_tensor(
                torch.tensor([0.0, -4.0, 1.0, -3]),
                torch.tensor([True, True, False, True]),
            ),
            mt.amin(dim=0),
        )

    def test_amin_grad(self):
        d = torch.tensor([[0, 1, 2], [3, 4, 5.0]])
        m = torch.tensor([[True, False, False], [False, True, False]])
        mt = masked_tensor(d, m, requires_grad=True)
        mt.amin().backward()
        _compare_mts(mt.grad, masked_tensor(torch.tensor(1.0).expand_as(m), m))

    def test_prod(self):
        d = torch.tensor([[0, 1, 3, 0.0], [float("nan"), 4, 1.0, 5.0]])
        m = torch.tensor([[True, False, False, True], [False, True, False, True]])
        mt = masked_tensor(d, m)
        _compare_mts(masked_tensor(torch.tensor(0.0), torch.tensor(True)), mt.prod())
        _compare_mts(
            masked_tensor(
                torch.tensor([0.0, 4.0, 1.0, 0.0]),
                torch.tensor([True, True, False, True]),
            ),
            mt.prod(dim=0),
        )

    def test_prod_grad(self):
        d = torch.tensor([[0, float("nan"), 2], [3, 4, 5.0]])
        m = torch.tensor([[True, False, False], [False, True, False]])
        mt = masked_tensor(d, m, requires_grad=True)
        mt.prod().backward()
        _compare_mts(mt.grad, masked_tensor(torch.tensor(1.0).expand_as(m), m))

    def test_all(self):
        d = torch.tensor([[True, True, False, False], [False, True, True, True]])
        m = torch.tensor([[True, False, False, True], [False, True, False, True]])
        mt = masked_tensor(d, m)
        _compare_mts(masked_tensor(torch.tensor(False), torch.tensor(True)), mt.all())
        _compare_mts(
            masked_tensor(
                torch.tensor([True, True, True, False]),
                torch.tensor([True, True, False, True]),
            ),
            mt.all(dim=0),
        )

        m = torch.tensor([[True, False, True, False], [False, True, False, False]])
        mt = masked_tensor(d, m)
        _compare_mts(
            masked_tensor(
                torch.tensor([True, True, False, True]),
                torch.tensor([True, True, True, False]),
            ),
            mt.all(dim=0),
        )

    def test_all_grad(self):
        d = torch.tensor([[True, True, False], [False, True, True]])
        m = torch.tensor([[True, False, False], [False, True, False]])
        self.assertRaises(RuntimeError, lambda: masked_tensor(d, m, requires_grad=True))

class TestMatMul(TestCase):
    def test_bmm(self):
        x = torch.rand(3, 2, 1)
        key_padding_mask = torch.as_tensor(
            [
                [False, False, False],
                [False, True, True],
            ]
        )
        x_mt = masked_tensor(
            x, ~(key_padding_mask.transpose(0, 1).unsqueeze(-1).expand_as(x))
        )
        x = x.masked_fill(~x_mt.mask(), 0)
        attn_2 = torch.bmm(x, x.transpose(-2, -1))
        attn_3 = torch.bmm(x_mt, x_mt.transpose(-2, -1))
        self.assertEqual(attn_3.masked_data.masked_fill(~attn_3.mask(), 0), attn_2)  # type: ignore[attr-defined]

    def test_bmm_2(self):
        x = torch.arange(3 * 2 * 2).reshape(3, 2, 2).float()
        x_t = x.transpose(-2, -1) + x.sum()
        key_padding_mask = torch.as_tensor(
            [
                [False, False, False],
                [False, True, True],
            ]
        )
        x_mt = masked_tensor(
            x, ~(key_padding_mask.transpose(0, 1).unsqueeze(-1).expand_as(x))
        )
        y = torch.bmm(x, x_t)
        y = torch.bmm(x, x_mt.transpose(-2, -1) + x.sum())

    def test_masked_bmm(self):
        key_padding_mask = torch.as_tensor(
            [
                [False, False, False, True],
                [False, True, True, True],
                [False, True, False, True],
            ]
        )
        x = torch.arange(4 * 3 * 2).reshape(4, 3, 2).float().requires_grad_()
        x_mt = masked_tensor(
            x,
            ~(key_padding_mask.transpose(0, 1).unsqueeze(-1).expand_as(x)),
            requires_grad=True,
        )
        attn_mask_bool = torch.as_tensor(
            [
                [False, True, True],
                [False, False, True],
                [True, False, False],
            ]
        )
        attn_mask = attn_mask_bool.float().masked_fill_(attn_mask_bool, float("-inf"))
        v = masked_bmm(x, x_mt.transpose(1, 2), attn_mask)
        v.sum().backward()
        x = torch.arange(4 * 3 * 2).reshape(4, 3, 2).float().requires_grad_()
        x0 = torch.arange(4 * 3 * 2).reshape(4, 3, 2).float().requires_grad_()
        y = torch.bmm(x, x0.transpose(-2, -1))
        y = y * (~attn_mask_bool).float()
        y.sum().backward()

    def test_linear(self):
        x = torch.arange(4 * 3 * 2).reshape(4, 3, 2)
        w_x = torch.arange(10).reshape(5, 2) + x.amax()
        linear = torch.nn.functional.linear
        key_padding_mask = torch.as_tensor(
            [
                [False, False, False, True],
                [False, True, True, True],
                [False, True, False, True],
            ]
        )
        x_mt = masked_tensor(
            x, ~(key_padding_mask.transpose(0, 1).unsqueeze(-1).expand_as(x))
        )

instantiate_parametrized_tests(TestBasics)
instantiate_parametrized_tests(TestUnary)
instantiate_parametrized_tests(TestBinary)
instantiate_parametrized_tests(TestReductions)
instantiate_parametrized_tests(TestMatMul)

if __name__ == '__main__':
    run_tests()
