# Owner(s): ["module: PrivateUse1"]

import collections
import functools

import torch
from torch.nn.attention import SDPBackend
from torch.testing._internal.common_nn import NNTestCase
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase


SDPAShape = collections.namedtuple(
    "Sdpa_Shape", ["batch", "num_heads", "seq_len", "head_dim"]
)


class TestFactory(TestCase):
    def test_empty(self):
        x = torch.empty(3, device="openreg")
        self.assertEqual(x.device.type, "openreg")
        self.assertEqual(x.shape, torch.Size([3]))

        x = torch.empty([2, 3, 4, 5], device="openreg", names=["N", "C", "H", "W"])
        self.assertEqual(x.device.type, "openreg")
        self.assertEqual(x.shape, torch.Size([2, 3, 4, 5]))

        with torch._subclasses.fake_tensor.FakeTensorMode():
            x = torch.empty(3, 3, device="openreg")
            y = torch.empty(3, 3, device="openreg:0")
            z = x + y
            self.assertEqual(z.device.type, "openreg")
            self.assertEqual(z.shape, torch.Size([3, 3]))

    def test_zeros(self):
        y = torch.zeros(3, device="openreg")
        self.assertEqual(y.device.type, "openreg")
        self.assertEqual(y.shape, torch.Size([3]))

    def test_tensor(self):
        z = torch.tensor((), device="openreg")
        self.assertEqual(z.device.type, "openreg")
        self.assertEqual(z.shape, torch.Size([0]))


class TestCopy(TestCase):
    def test_copy_same_device(self):
        a = torch.ones(10, device="openreg").clone()
        self.assertEqual(a, torch.ones(10, device="openreg"))

    def test_cross_device_copy(self):
        a = torch.rand(10)
        b = a.to(device="openreg").add(2).to(device="cpu")
        self.assertEqual(b, a + 2)

    def test_cross_diff_devices_copy(self):
        a = torch.ones(10, device="openreg:0").to(device="openreg:1").to(device="cpu")
        self.assertEqual(a, torch.ones(10))


class TestOps(TestCase):
    def test_masked_select(self):
        tensor_cpu = torch.randn(10)
        tensor_openreg = tensor_cpu.to(device="openreg")
        mask = tensor_openreg.gt(0)
        out = torch.masked_select(tensor_openreg, mask)

        self.assertEqual(out, tensor_cpu.masked_select(tensor_cpu.gt(0)))

    def test_expand(self):
        x = torch.tensor([[1], [2], [3]], device="openreg")
        y = x.expand(3, 2)
        self.assertEqual(y.to(device="cpu"), torch.tensor([[1, 1], [2, 2], [3, 3]]))
        self.assertEqual(x.data_ptr(), y.data_ptr())

    def test_resize(self):
        tensor_cpu = torch.randn([4, 4])

        tensor_openreg = tensor_cpu.openreg()
        self.assertTrue(tensor_openreg.size() == torch.Size([4, 4]))

        storage_openreg = tensor_openreg.storage()
        self.assertTrue(storage_openreg.size() == 16)

        tensor_openreg.resize_(2, 2, 2, 2)
        self.assertTrue(tensor_openreg.size() == torch.Size([2, 2, 2, 2]))

        storage_openreg = tensor_openreg.storage()
        self.assertTrue(storage_openreg.size() == 16)

    def test_printing(self):
        a = torch.ones(20, device="openreg")
        print(a)


class TestSTUB(TestCase):
    def test_backend_dispatchstub(self):
        x_cpu = torch.randn(2, 2, 3, dtype=torch.float32, device="cpu")
        x_openreg = x_cpu.to("openreg")

        y_cpu = torch.abs(x_cpu)
        y_openreg = torch.abs(x_openreg)
        self.assertEqual(y_cpu, y_openreg.cpu())

        o_cpu = torch.randn(2, 2, 6, dtype=torch.float32, device="cpu")
        o_openreg = o_cpu.to("openreg")
        # output operand with resize flag is False in TensorIterator.
        torch.abs(x_cpu, out=o_cpu[:, :, 0:6:2])
        torch.abs(x_openreg, out=o_openreg[:, :, 0:6:2])
        self.assertEqual(o_cpu, o_openreg.cpu())

        # output operand with resize flag is True in TensorIterator and
        # convert output to contiguous tensor in TensorIterator.
        torch.abs(x_cpu, out=o_cpu[:, :, 0:6:3])
        torch.abs(x_openreg, out=o_openreg[:, :, 0:6:3])
        self.assertEqual(o_cpu, o_openreg.cpu())


class TestQuantization(TestCase):
    def test_quantize(self):
        x = torch.randn(3, 4, 5, dtype=torch.float32, device="openreg")
        quantized_tensor = torch.quantize_per_tensor(x, 0.1, 10, torch.qint8)
        self.assertEqual(quantized_tensor.device, torch.device("openreg:0"))
        self.assertEqual(quantized_tensor.dtype, torch.qint8)


class TestAutogradFunction(TestCase):
    def test_compile_autograd_function_returns_self(self):
        in_ref = torch.randn(4, device="openreg", requires_grad=True)
        out_ref = torch.ops.openreg.custom_autograd_fn_returns_self(in_ref)
        out_ref.sum().backward()

        in_test = in_ref.detach().clone().requires_grad_(True)
        # TODO(FFFrog): Need to support inductor for OpenReg first.
        out_test = torch.compile(backend="aot_eager")(
            torch.ops.openreg.custom_autograd_fn_returns_self
        )(in_test)
        out_test.sum().backward()

        self.assertEqual(out_ref, out_test)
        self.assertEqual(in_ref.grad, in_test.grad)

    @skipIfTorchDynamo("Temporary disabled due to torch._ops.OpOverloadPacket")
    def test_compile_autograd_function_aliasing(self):
        in_ref = torch.randn(4, device="openreg", requires_grad=True)
        out_ref = torch.ops.openreg.custom_autograd_fn_aliasing(in_ref)
        out_ref.sum().backward()

        in_test = in_ref.detach().clone().requires_grad_(True)
        # TODO(FFFrog): Need to support inductor for OpenReg first.
        out_test = torch.compile(backend="aot_eager")(
            torch.ops.openreg.custom_autograd_fn_aliasing
        )(in_test)
        out_test.sum().backward()

        self.assertEqual(out_ref, out_test)
        self.assertEqual(in_ref.grad, in_test.grad)


class TestFallback(TestCase):
    def test_scalar_type_fallback(self):
        x_cpu = torch.Tensor([[0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]]).to(torch.int64)
        x = torch.triu_indices(3, 3, device="openreg")
        self.assertEqual(x_cpu, x)

    def test_tensor_type_fallback(self):
        x = torch.Tensor([[1, 2, 3], [2, 3, 4]]).to("openreg")
        y = torch.Tensor([1, 0, 2]).to("openreg")
        self.assertTrue(x.device.type, "openreg")
        self.assertFalse(x.is_cpu)

        z_cpu = torch.Tensor([[0, 2, 1], [1, 3, 2]])
        # call sub op, which will fallback to cpu
        z = torch.sub(x, y)
        self.assertEqual(z_cpu, z)

        # call index op, which will fallback to cpu
        z_cpu = torch.Tensor([3, 1])
        y = torch.Tensor([1, 0]).long().to("openreg")
        z = x[y, y]
        self.assertEqual(z_cpu, z)

    def test_tensorlist_type_fallback(self):
        # create tensors located in custom device
        v_openreg = torch.Tensor([1, 2, 3]).to("openreg")
        # create result tensor located in cpu
        z_cpu = torch.Tensor([2, 4, 6])
        # create tensorlist for foreach_add op
        x = (v_openreg, v_openreg)
        y = (v_openreg, v_openreg)

        # Check that our device is correct.
        self.assertTrue(v_openreg.device.type == "openreg")
        self.assertFalse(v_openreg.is_cpu)

        # call _foreach_add op, which will fallback to cpu
        z = torch._foreach_add(x, y)
        self.assertEqual(z_cpu, z[0])
        self.assertEqual(z_cpu, z[1])


class TestSDPA(NNTestCase):
    @skipIfTorchDynamo()
    def test_fused_sdp_choice_privateuseone(self):
        batch_size, seq_len, num_heads, head_dim = 4, 256, 2, 128
        make_tensor = functools.partial(torch.rand, device="cpu", dtype=torch.float16)
        shape = SDPAShape(batch_size, num_heads, seq_len, head_dim)
        q_cpu, k_cpu, v_cpu = make_tensor(shape), make_tensor(shape), make_tensor(shape)
        q_privateuse1 = q_cpu.to("openreg")
        k_privateuse1 = k_cpu.to("openreg")
        v_privateuse1 = v_cpu.to("openreg")
        assert (
            torch._fused_sdp_choice(q_privateuse1, k_privateuse1, v_privateuse1)
            == SDPBackend.OVERRIDEABLE.value
        )

    def test_scaled_dot_product_fused_attention_overrideable(self):
        batch_size, seq_len, num_heads, head_dim = 4, 256, 2, 128
        make_tensor = functools.partial(torch.rand, device="cpu", dtype=torch.float16)
        shape = SDPAShape(batch_size, num_heads, seq_len, head_dim)
        q_cpu, k_cpu, v_cpu = make_tensor(shape), make_tensor(shape), make_tensor(shape)
        q_privateuse1 = q_cpu.to("openreg")
        k_privateuse1 = k_cpu.to("openreg")
        v_privateuse1 = v_cpu.to("openreg")
        torch.nn.functional.scaled_dot_product_attention(
            q_privateuse1, k_privateuse1, v_privateuse1, attn_mask=None, dropout_p=0.0
        )

    def test_scaled_dot_product_fused_attention_overrideable_backward(self):
        batch_size, seq_len, num_heads, head_dim = 4, 256, 2, 128
        make_tensor = functools.partial(
            torch.rand, device="cpu", dtype=torch.float16, requires_grad=True
        )
        shape = (batch_size, num_heads, seq_len, head_dim)
        q_cpu, k_cpu, v_cpu = make_tensor(shape), make_tensor(shape), make_tensor(shape)
        attn_mask = make_tensor((batch_size, num_heads, seq_len, seq_len))
        q_privateuse1 = q_cpu.to("openreg")
        k_privateuse1 = k_cpu.to("openreg")
        v_privateuse1 = v_cpu.to("openreg")
        attn_mask_privateuse1 = attn_mask.to("openreg")
        (
            output,
            logsumexp,
            cum_seq_q,
            cum_seq_k,
            max_q,
            max_k,
            philox_seed,
            philox_offset,
            _debug_attn_mask,
        ) = torch.ops.aten._scaled_dot_product_fused_attention_overrideable(
            q_privateuse1, k_privateuse1, v_privateuse1, attn_bias=attn_mask_privateuse1
        )

        rand_upward = torch.rand(
            shape, device="cpu", dtype=torch.float16, requires_grad=False
        )
        rand_upward_privateuse1 = rand_upward.to("openreg")
        grad_input_mask = [True, True, True, True]
        _grad_q, _grad_k, _grad_v, _grad_attn_mask = (
            torch.ops.aten._scaled_dot_product_fused_attention_overrideable_backward(
                rand_upward_privateuse1,
                q_privateuse1,
                k_privateuse1,
                v_privateuse1,
                attn_mask_privateuse1,
                grad_input_mask,
                output,
                logsumexp,
                cum_seq_q,
                cum_seq_k,
                max_q,
                max_k,
                dropout_p=0.0,
                is_causal=False,
                philox_seed=philox_seed,
                philox_offset=philox_offset,
            )
        )


if __name__ == "__main__":
    run_tests()
