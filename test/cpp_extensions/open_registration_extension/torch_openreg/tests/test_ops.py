# Owner(s): ["module: PrivateUse1"]

import collections
import functools
import unittest

import torch
from torch.nn.attention import SDPBackend
from torch.testing._internal.common_nn import NNTestCase
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase


SDPAShape = collections.namedtuple(
    "Sdpa_Shape", ["batch", "num_heads", "seq_len", "head_dim"]
)


class TestFactory(TestCase):
    def test_empty(self):
        """Test empty tensor creation"""
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
        """Test zeros tensor creation"""
        y = torch.zeros(3, device="openreg")
        self.assertEqual(y.device.type, "openreg")
        self.assertEqual(y.shape, torch.Size([3]))

    def test_tensor(self):
        """Test tensor creation from empty tuple"""
        z = torch.tensor((), device="openreg")
        self.assertEqual(z.device.type, "openreg")
        self.assertEqual(z.shape, torch.Size([0]))


class TestCopy(TestCase):
    def test_copy_same_device(self):
        """Test copy operation on same device"""
        a = torch.ones(10, device="openreg").clone()
        self.assertEqual(a, torch.ones(10, device="openreg"))

    def test_cross_device_copy(self):
        """Test copy operation across CPU and openreg"""
        a = torch.rand(10)
        b = a.to(device="openreg").add(2).to(device="cpu")
        self.assertEqual(b, a + 2)

    def test_cross_diff_devices_copy(self):
        """Test copy operation across different openreg devices"""
        a = torch.ones(10, device="openreg:0").to(device="openreg:1").to(device="cpu")
        self.assertEqual(a, torch.ones(10))


class TestOps(TestCase):
    def test_masked_select(self):
        """Test masked_select operation"""
        tensor_cpu = torch.randn(10)
        tensor_openreg = tensor_cpu.to(device="openreg")
        mask = tensor_openreg.gt(0)
        out = torch.masked_select(tensor_openreg, mask)

        self.assertEqual(out, tensor_cpu.masked_select(tensor_cpu.gt(0)))

    def test_expand(self):
        """Test tensor expand operation"""
        x = torch.tensor([[1], [2], [3]], device="openreg")
        y = x.expand(3, 2)
        self.assertEqual(y.to(device="cpu"), torch.tensor([[1, 1], [2, 2], [3, 3]]))
        self.assertEqual(x.data_ptr(), y.data_ptr())

    def test_resize(self):
        """Test tensor resize operation"""
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
        """Test tensor printing"""
        a = torch.ones(20, device="openreg")
        print(a)


class TestSTUB(TestCase):
    def test_backend_dispatchstub(self):
        """Test backend dispatch stub for abs operation"""
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
        """Test quantization per tensor"""
        x = torch.randn(3, 4, 5, dtype=torch.float32, device="openreg")
        quantized_tensor = torch.quantize_per_tensor(x, 0.1, 10, torch.qint8)
        self.assertEqual(quantized_tensor.device, torch.device("openreg:0"))
        self.assertEqual(quantized_tensor.dtype, torch.qint8)


class TestAutogradFunction(TestCase):
    def test_compile_autograd_function_returns_self(self):
        """Test compiled autograd function that returns self"""
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
        """Test compiled autograd function with aliasing"""
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
        """Test scalar type fallback to CPU"""
        x_cpu = torch.Tensor([[0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]]).to(torch.int64)
        x = torch.triu_indices(3, 3, device="openreg")
        self.assertEqual(x_cpu, x)

    def test_tensor_type_fallback(self):
        """Test tensor type fallback to CPU"""
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
        """Test tensor list type fallback to CPU"""
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
        """Test fused SDP choice for privateuse1 backend"""
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
        """Test scaled dot product fused attention overrideable forward"""
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
        """Test scaled dot product fused attention overrideable backward"""
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


class TestFactoryExtended(TestCase):
    def test_empty_with_memory_format(self):
        """Test empty tensor creation with memory format"""
        x = torch.empty(1, 2, 3, 4, device="openreg", memory_format=torch.channels_last)
        self.assertEqual(x.device.type, "openreg")
        self.assertEqual(x.shape, torch.Size([1, 2, 3, 4]))

        x = torch.empty(
            2, 3, 4, device="openreg", memory_format=torch.contiguous_format
        )
        self.assertEqual(x.device.type, "openreg")
        self.assertTrue(x.is_contiguous())

    def test_empty_strided(self):
        """Test empty_strided tensor creation"""
        size = (3, 4)
        stride = (4, 1)
        x = torch.empty_strided(size, stride, device="openreg")
        self.assertEqual(x.device.type, "openreg")
        self.assertEqual(x.shape, torch.Size(size))
        self.assertEqual(x.stride(), stride)

    def test_ones(self):
        """Test ones tensor creation"""
        x = torch.ones(3, 4, device="openreg")
        self.assertEqual(x.device.type, "openreg")
        self.assertEqual(x.shape, torch.Size([3, 4]))
        self.assertTrue(torch.all(x == 1))

    def test_ones_like(self):
        """Test ones_like tensor creation"""
        x = torch.randn(3, 4, device="openreg")
        y = torch.ones_like(x)
        self.assertEqual(y.device.type, "openreg")
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(torch.all(y == 1))

    def test_randn(self):
        """Test randn tensor creation"""
        x = torch.randn(3, 4, device="openreg")
        self.assertEqual(x.device.type, "openreg")
        self.assertEqual(x.shape, torch.Size([3, 4]))

    def test_full(self):
        """Test full tensor creation"""
        x = torch.full((3, 4), 5.0, device="openreg")
        self.assertEqual(x.device.type, "openreg")
        self.assertEqual(x.shape, torch.Size([3, 4]))
        self.assertTrue(torch.all(x == 5.0))


class TestCopyExtended(TestCase):
    def test_copy_different_dtypes(self):
        """Test copy with different dtypes"""
        x = torch.randn(3, 4, dtype=torch.float32, device="openreg")
        y = torch.empty(3, 4, dtype=torch.float64, device="openreg")
        y.copy_(x)
        self.assertEqual(y.dtype, torch.float64)
        self.assertEqual(y.cpu(), x.cpu().double())

    def test_clone(self):
        """Test tensor clone"""
        x = torch.randn(3, 4, device="openreg")
        y = x.clone()
        self.assertEqual(y.device.type, "openreg")
        self.assertEqual(y, x)
        self.assertNotEqual(y.data_ptr(), x.data_ptr())

    def test_copy_non_blocking(self):
        """Test non-blocking copy"""
        x = torch.randn(3, 4, device="openreg")
        y = torch.empty(3, 4, device="openreg")
        y.copy_(x, non_blocking=True)
        self.assertEqual(y, x)


class TestOpsExtended(TestCase):
    def test_view(self):
        """Test tensor view operation"""
        x = torch.randn(2, 3, 4, device="openreg")
        y = x.view(6, 4)
        self.assertEqual(y.device.type, "openreg")
        self.assertEqual(y.shape, torch.Size([6, 4]))
        self.assertEqual(x.data_ptr(), y.data_ptr())

    def test_reshape(self):
        """Test tensor reshape operation"""
        x = torch.randn(2, 3, 4, device="openreg")
        y = x.reshape(6, 4)
        self.assertEqual(y.device.type, "openreg")
        self.assertEqual(y.shape, torch.Size([6, 4]))

    def test_as_strided(self):
        """Test as_strided operation"""
        x = torch.randn(3, 4, device="openreg")
        y = torch.as_strided(x, (2, 2), (4, 1), 1)
        self.assertEqual(y.device.type, "openreg")
        self.assertEqual(y.shape, torch.Size([2, 2]))

    def test_local_scalar_dense(self):
        """Test local scalar dense extraction"""
        x = torch.tensor([5.0], device="openreg")
        scalar = x.item()
        self.assertEqual(scalar, 5.0)

    def test_set_tensor(self):
        """Test set_ operation with tensor source"""
        x = torch.randn(3, 4, device="openreg")
        y = torch.empty(3, 4, device="openreg")
        y.set_(x)
        self.assertEqual(y, x)

    def test_set_storage(self):
        """Test set_ operation with storage source"""
        x = torch.randn(3, 4, device="openreg")
        storage = x.storage()
        y = torch.empty(3, 4, device="openreg")
        y.set_(storage, 0, y.size())
        self.assertEqual(y, x)


class TestSTUBExtended(TestCase):
    def test_abs_contiguous(self):
        """Test abs operation with contiguous tensor"""
        x = torch.randn(2, 3, dtype=torch.float32, device="openreg")
        y = torch.abs(x)
        self.assertEqual(y.device.type, "openreg")
        self.assertTrue(torch.all(y >= 0))
        self.assertEqual(y.shape, x.shape)

    def test_abs_non_contiguous(self):
        """Test abs operation with non-contiguous tensor"""
        x = torch.randn(2, 3, dtype=torch.float32, device="openreg")
        x_t = x.t()  # Transpose makes it non-contiguous
        y = torch.abs(x_t)
        self.assertEqual(y.device.type, "openreg")
        self.assertTrue(torch.all(y >= 0))

    def test_custom_abs(self):
        """Test custom abs operation"""
        x = torch.randn(2, 3, dtype=torch.float32, device="openreg")
        y = torch.ops.openreg.custom_abs(x)
        self.assertEqual(y.device.type, "openreg")
        self.assertTrue(torch.all(y >= 0))
        self.assertEqual(y.shape, x.shape)

    def test_abs_out(self):
        """Test abs with output tensor"""
        x = torch.randn(2, 3, dtype=torch.float32, device="openreg")
        out = torch.empty_like(x)
        torch.abs(x, out=out)
        self.assertEqual(out.device.type, "openreg")
        self.assertTrue(torch.all(out >= 0))
        self.assertEqual(out, torch.abs(x))


@unittest.skip("Skipping all quantization tests for openreg backend")
class TestQuantizationExtended(TestCase):
    def test_quantize_per_tensor_different_scales(self):
        """Test quantization with different scales"""
        x = torch.randn(3, 4, 5, dtype=torch.float32, device="openreg")

        scale = 0.1
        zero_point = 10
        quantized = torch.quantize_per_tensor(x, scale, zero_point, torch.qint8)
        self.assertEqual(quantized.device.type, "openreg")
        self.assertEqual(quantized.dtype, torch.qint8)
        self.assertEqual(quantized.q_scale(), scale)
        self.assertEqual(quantized.q_zero_point(), zero_point)

    def test_quantize_per_tensor_quint8(self):
        """Test quantization with quint8 dtype"""
        x = torch.randn(3, 4, dtype=torch.float32, device="openreg")
        quantized = torch.quantize_per_tensor(x, 0.1, 128, torch.quint8)
        self.assertEqual(quantized.device.type, "openreg")
        self.assertEqual(quantized.dtype, torch.quint8)

    def test_dequantize(self):
        """Test dequantization"""
        x = torch.randn(3, 4, dtype=torch.float32, device="openreg")
        quantized = torch.quantize_per_tensor(x, 0.1, 10, torch.qint8)
        dequantized = quantized.dequantize()
        self.assertEqual(dequantized.device.type, "openreg")
        self.assertEqual(dequantized.dtype, torch.float32)


class TestFallbackExtended(TestCase):
    def test_cpu_fallback_blocklist(self):
        """Test that abs is blocked from CPU fallback"""
        x = torch.randn(2, 3, dtype=torch.float32, device="openreg")
        # abs should work (it's implemented)
        y = torch.abs(x)
        self.assertEqual(y.device.type, "openreg")

        # But abs.out should also work
        out = torch.empty_like(x)
        torch.abs(x, out=out)
        self.assertEqual(out.device.type, "openreg")

    def test_fallback_operations(self):
        """Test various fallback operations"""
        x = torch.randn(3, 4, device="openreg")
        y = torch.randn(3, 4, device="openreg")

        # Operations that should fallback to CPU
        z = torch.add(x, y)
        self.assertEqual(z.device.type, "openreg")

        z = torch.mul(x, y)
        self.assertEqual(z.device.type, "openreg")

    def test_fallback_with_scalars(self):
        """Test fallback with scalar operations"""
        x = torch.randn(3, 4, device="openreg")
        y = x + 1.0
        self.assertEqual(y.device.type, "openreg")

        y = x * 2.0
        self.assertEqual(y.device.type, "openreg")


class TestSDPAExtended(NNTestCase):
    @skipIfTorchDynamo()
    def test_fused_sdp_choice_with_mask(self):
        """Test fused SDP choice with attention mask"""
        batch_size, seq_len, num_heads, head_dim = 4, 256, 2, 128
        make_tensor = functools.partial(torch.rand, device="cpu", dtype=torch.float16)
        shape = SDPAShape(batch_size, num_heads, seq_len, head_dim)
        q_cpu, k_cpu, v_cpu = make_tensor(shape), make_tensor(shape), make_tensor(shape)
        attn_mask = make_tensor((batch_size, num_heads, seq_len, seq_len))

        q_privateuse1 = q_cpu.to("openreg")
        k_privateuse1 = k_cpu.to("openreg")
        v_privateuse1 = v_cpu.to("openreg")
        attn_mask_privateuse1 = attn_mask.to("openreg")

        backend = torch._fused_sdp_choice(
            q_privateuse1, k_privateuse1, v_privateuse1, attn_mask_privateuse1
        )
        self.assertEqual(backend, SDPBackend.OVERRIDEABLE.value)

    @skipIfTorchDynamo()
    def test_scaled_dot_product_attention_with_dropout(self):
        """Test scaled dot product attention with dropout"""
        batch_size, seq_len, num_heads, head_dim = 4, 256, 2, 128
        make_tensor = functools.partial(torch.rand, device="cpu", dtype=torch.float16)
        shape = SDPAShape(batch_size, num_heads, seq_len, head_dim)
        q_cpu, k_cpu, v_cpu = make_tensor(shape), make_tensor(shape), make_tensor(shape)

        q_privateuse1 = q_cpu.to("openreg")
        k_privateuse1 = k_cpu.to("openreg")
        v_privateuse1 = v_cpu.to("openreg")

        output = torch.nn.functional.scaled_dot_product_attention(
            q_privateuse1,
            k_privateuse1,
            v_privateuse1,
            attn_mask=None,
            dropout_p=0.1,
            is_causal=False,
        )
        self.assertEqual(output.device.type, "openreg")
        self.assertEqual(output.shape, shape)

    @skipIfTorchDynamo()
    def test_scaled_dot_product_attention_is_causal(self):
        """Test scaled dot product attention with causal mask"""
        batch_size, seq_len, num_heads, head_dim = 4, 256, 2, 128
        make_tensor = functools.partial(torch.rand, device="cpu", dtype=torch.float16)
        shape = SDPAShape(batch_size, num_heads, seq_len, head_dim)
        q_cpu, k_cpu, v_cpu = make_tensor(shape), make_tensor(shape), make_tensor(shape)

        q_privateuse1 = q_cpu.to("openreg")
        k_privateuse1 = k_cpu.to("openreg")
        v_privateuse1 = v_cpu.to("openreg")

        output = torch.nn.functional.scaled_dot_product_attention(
            q_privateuse1,
            k_privateuse1,
            v_privateuse1,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
        )
        self.assertEqual(output.device.type, "openreg")
        self.assertEqual(output.shape, shape)


class TestCustomAutogradFunctions(TestCase):
    def test_custom_autograd_fn_returns_self_basic(self):
        """Test basic usage of custom_autograd_fn_returns_self"""
        x = torch.randn(4, device="openreg", requires_grad=True)
        y = torch.ops.openreg.custom_autograd_fn_returns_self(x)

        # Should return the same tensor
        self.assertEqual(x, y)
        self.assertTrue(y.requires_grad)

        # Test backward
        loss = y.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        # Gradient should be 0.5 * 1.0 = 0.5
        self.assertTrue(torch.allclose(x.grad, torch.ones_like(x) * 0.5))

    def test_custom_autograd_fn_aliasing_basic(self):
        """Test basic usage of custom_autograd_fn_aliasing"""
        x = torch.randn(4, device="openreg", requires_grad=True)
        y = torch.ops.openreg.custom_autograd_fn_aliasing(x)

        # Should return a view of the same tensor
        self.assertEqual(x.shape, y.shape)
        self.assertTrue(y.requires_grad)

        # Test backward
        loss = y.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        # Gradient should be 0.5 * 1.0 = 0.5
        self.assertTrue(torch.allclose(x.grad, torch.ones_like(x) * 0.5))

    def test_custom_autograd_fn_returns_self_no_grad(self):
        """Test custom_autograd_fn_returns_self without requires_grad"""
        x = torch.randn(4, device="openreg", requires_grad=False)
        y = torch.ops.openreg.custom_autograd_fn_returns_self(x)
        self.assertEqual(x, y)
        self.assertFalse(y.requires_grad)

    def test_custom_autograd_fn_aliasing_no_grad(self):
        """Test custom_autograd_fn_aliasing without requires_grad"""
        x = torch.randn(4, device="openreg", requires_grad=False)
        y = torch.ops.openreg.custom_autograd_fn_aliasing(x)
        self.assertEqual(x.shape, y.shape)
        self.assertFalse(y.requires_grad)


if __name__ == "__main__":
    run_tests()
