import time

import torch
import torch.nn as nn
import torch.utils._pytree as pytree
from torch._inductor.runtime.benchmarking import benchmarker
from torch._inductor.test_case import run_tests, TestCase
from torch.ops import aten

from padded_tensor import *
from utils import *

from transformer_model import *


# class AttentionTest(nn.Module):
#    def __init__(self) -> None:
#        super().__init__()
#
#        # Define model parameters
#        self.n_local_heads = 8
#        self.head_dim = 128
#        self.dim = 4096
#        self.n_head = 32
#        self.vocab_size = 128256
#
#        self.dtype = torch.bfloat16
#
#        # Initialize token embeddings and layers
#        self.tok_embeddings = nn.Embedding(self.vocab_size, self.dim).to(
#            device="cuda", dtype=self.dtype
#        )
#        total_head_dim = (self.n_head + 2 * self.n_local_heads) * self.head_dim
#        self.wqkv = nn.Linear(self.dim, total_head_dim, bias=False).to(
#            device="cuda", dtype=self.dtype
#        )
#        self.wo = nn.Linear(self.dim, self.dim, bias=False).to(
#            device="cuda", dtype=self.dtype
#        )
#
#    def create_kv_cache(self, max_batch_size, max_seq_length):
#        cache_shape = (
#            max_batch_size,
#            self.n_local_heads,
#            max_seq_length,
#            self.head_dim,
#        )
#        self.register_buffer(
#            "k_cache", torch.zeros(cache_shape, dtype=self.dtype, device="cuda")
#        )
#        self.register_buffer(
#            "v_cache", torch.zeros(cache_shape, dtype=self.dtype, device="cuda")
#        )
#
#    def reset_kv_cache(self):
#        self.k_cache.zero_()
#        self.v_cache.zero_()
#
#    def f_1(self, idx):
#        x = self.tok_embeddings(idx)
#
#        kv_size = self.n_local_heads * self.head_dim
#        yy = self.wqkv(x)
#        q, k, v = yy.split([self.dim, kv_size, kv_size], dim=-1)
#
#        return q, k, v
#
#    def f_2(self, q, k, v, bsz, seqlen):
#        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
#        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
#        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)
#
#        return q, k, v
#
#    def apply_rotary_emb(self, x: Tensor, freqs_cis: Tensor) -> Tensor:
#        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
#        freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
#        x_out2 = torch.stack(
#            [
#                xshaped[..., 0] * freqs_cis[..., 0]
#                - xshaped[..., 1] * freqs_cis[..., 1],
#                xshaped[..., 1] * freqs_cis[..., 0]
#                + xshaped[..., 0] * freqs_cis[..., 1],
#            ],
#            -1,
#        )
#
#        x_out2 = x_out2.flatten(3)
#        return x_out2.type_as(x)
#
#    def f_3(self, q, k, v, freqs_cis):
#        q = self.apply_rotary_emb(q, freqs_cis)
#        k = self.apply_rotary_emb(k, freqs_cis)
#
#        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))
#
#        return q, k, v
#
#    def kv_update(self, input_pos, k_val, v_val, k_cache, v_cache):
#        # input_pos: [S], k_val: [B, H, S, D]
#        # assert input_pos.shape[0] == k_val.shape[2]
#
#        k_out = self.k_cache
#        v_out = self.v_cache
#        k_out[:, :, input_pos] = k_val
#        v_out[:, :, input_pos] = v_val
#
#        return k_out, v_out
#
#    def f_4(self, k, v, input_pos, k_cache, v_cache):
#        k, v = self.kv_update(input_pos, k, v, k_cache, v_cache)
#
#        k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
#        v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
#
#        return k, v
#
#    def f_5(self, q, k, v, mask):
#        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
#        return (y,)
#
#    def f_6(self, y, bsz, seqlen):
#        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)
#        return (y,)
#
#    def f_7(self, y):
#        y = self.wo(y)
#        return (y,)
#
#    def f_attention(self, x, freqs_cis, mask, input_pos, k_cache, v_cache):
#        bsz, seqlen = x.shape
#
#        q, k, v = self.f_1(x)
#        q, k, v = self.f_2(q, k, v, bsz, seqlen)
#        q, k, v = self.f_3(q, k, v, freqs_cis)
#        k, v = self.f_4(k, v, input_pos, k_cache, v_cache)
#        (y,) = self.f_5(q, k, v, mask)
#        (y,) = self.f_6(y, bsz, seqlen)
#        (y,) = self.f_7(y)
#
#        return (y,)


class PaddedTensorTestCase(TestCase):
    def setUp(self):
        super().setUp()

    def are_equal(self, outputs, outputs_p):
        outputs = [outputs] if not isinstance(outputs, list) else outputs
        outputs_p = [outputs_p] if not isinstance(outputs_p, list) else outputs_p

        for x, x_p in zip(outputs, outputs_p):
            x_p = x_p.tensor if isinstance(x_p, PaddedTensor) else x_p

            slice_idx = [slice(0, s) for s in x.shape]
            self.assertEqual(x, x_p[tuple(slice_idx)])


#
# class TestAttention(PaddedTensorTestCase):
#    def setUp(self):
#        super().setUp()
#
#    def run_unpadded_padded(self, fn, inputs, inputs_p):
#        # Run the function on unpadded and padded inputs
#        inputs = pytree.tree_map(lambda x: x.clone(), inputs)
#        outputs = fn(*inputs)
#
#        inputs_p = pytree.tree_map(lambda x: x.clone(), inputs_p)
#        outputs_p = fn(*inputs_p)
#
#        # Check the outputs are equal
#        self.are_equal(outputs, outputs_p)
#
#    def run_unpadded_padded_bench(self, model, fn, inputs, inputs_p, outputs_p_shapes):
#        # Precondition: Are inputs and inputs_p equal?
#        self.are_equal(inputs, inputs_p)
#
#        def median(x):
#            return sorted(x)[len(x) // 2]
#
#        def bench(fn, inputs, n_iter=10):
#            times = []
#            outputs = None
#
#            for _ in range(n_iter):
#                model.reset_kv_cache()
#                inps = pytree.tree_map(lambda x: x.clone(), inputs)
#
#                start = time.time()
#                outputs = fn(*inps)
#                times.append(time.time() - start)
#
#            outputs = pytree.tree_map(lambda x: x.clone(), outputs)
#            return median(times), outputs
#
#        # Run eager
#        time_eager, outputs_eager = bench(fn, inputs)
#        time_eager_padded, outputs_eager_padded = bench(fn, inputs_p)
#
#        self.are_equal(outputs_eager, outputs_eager_padded)
#
#        # Run compiled
#        fn_compiled = torch.compile(fn, mode="reduce-overhead")
#
#        time_compiled, outputs_compiled = bench(fn_compiled, inputs)
#        time_compiled_padded, outputs_compiled_padded = bench(fn_compiled, inputs_p)
#
#        self.are_equal(outputs_eager, outputs_compiled)
#        self.are_equal(outputs_eager, outputs_compiled_padded)
#
#        # Report results
#        print(f"Unpadded time (eager): {time_eager}")
#        print(f"Padded time (eager): {time_eager_padded}")
#        print(f"Unpadded time (compiled): {time_compiled}")
#        print(f"Padded time (compiled): {time_compiled_padded}")
#
#    def test_attention_1(self):
#        model = TransformerModel()
#
#        inputs = [torch.ones(4, 2, dtype=torch.int32)]
#        MULTIPLIERS = {0: 128, 1: 128, 2: 1}
#        inputs_p = pytree.tree_map(
#            lambda x: PaddedTensor(x, {0: 128, 1: 128, 2: 1}, None), inputs
#        )
#
#        self.run_unpadded_padded(model.f_1, inputs, inputs_p)
#
#    def test_attention_2(self):
#        model = TransformerModel()
#
#        inputs = [torch.randn(4, 2, 10), torch.randn(4, 2, 10), torch.randn(4, 2, 10)]
#        MULTIPLIERS = {0: 8, 1: 8, 2: 1}
#        inputs_p = pytree.tree_map(lambda x: PaddedTensor(x, MULTIPLIERS, None), inputs)
#
#        self.run_unpadded_padded(model.f_2, inputs + [4, 2], inputs_p + [8, 8])
#
#    def test_attention_3(self):
#        model = TransformerModel()
#
#        inputs = [
#            torch.randn(4, 2, 5, 2),
#            torch.randn(4, 2, 5, 2),
#            torch.randn(4, 2, 5, 2),
#            torch.randn(2, 1, 2),
#        ]
#        MULTIPLIERS = {0: 8, 1: 8, 2: 1}
#        inputs_p = [PaddedTensor(x, MULTIPLIERS, None) for x in inputs[:-1]] + [
#            PaddedTensor(inputs[-1], {0: 8, 1: 1, 2: 1}, None)
#        ]
#
#        self.run_unpadded_padded(model.f_3, inputs, inputs_p)
#
#    def test_attention_4(self):
#        model = TransformerModel()
#
#        k_cache, v_cache = self.create_kv_cache()
#
#        inputs = [
#            torch.randn(4, 5, 2, 2),
#            torch.randn(4, 5, 2, 2),
#            torch.ones(2, dtype=torch.int32),
#            k_cache,
#            v_cache,
#        ]
#        MULTIPLIERS = {0: 1, 1: 1, 2: 1, 3: 1}
#        inputs_p = pytree.tree_map(lambda x: PaddedTensor(x, MULTIPLIERS, None), inputs)
#
#        self.run_unpadded_padded(model.f_4, inputs, inputs_p)
#
#    def test_attention_5(self):
#        model = TransformerModel()
#
#        inputs = [
#            torch.randn([4, 5, 2, 2]),
#            torch.randn([4, 5, 16, 2]),
#            torch.randn([4, 5, 16, 2]),
#            torch.ones([1, 1, 2, 16]),
#        ]
#        MULTIPLIERS = {0: 8, 1: 8, 2: 1}
#        inputs_p = pytree.tree_map(lambda x: PaddedTensor(x, MULTIPLIERS, None), inputs)
#
#        self.run_unpadded_padded(model.f_5, inputs, inputs_p)
#
#    def test_attention_6(self):
#        model = TransformerModel()
#
#        inputs = [torch.randn([4, 5, 2, 2])]
#        MULTIPLIERS = {0: 8, 1: 8, 2: 1}
#        inputs_p = pytree.tree_map(lambda x: PaddedTensor(x, MULTIPLIERS, None), inputs)
#
#        self.run_unpadded_padded(model.f_6, inputs + [4, 2], inputs_p + [4, 2])
#
#    def test_attention_7(self):
#        model = TransformerModel()
#
#        inputs = [torch.randn([4, 2, 10])]
#        MULTIPLIERS = {0: 8, 1: 8, 2: 1}
#        inputs_p = pytree.tree_map(lambda x: PaddedTensor(x, MULTIPLIERS, None), inputs)
#
#        self.run_unpadded_padded(model.f_7, inputs, inputs_p)
#
#    def gen_rand_inputs(self, model, bsz, seqlen):
#        x = torch.randint(0, 128256, (bsz, seqlen)).to(device="cuda")
#        freqs_cis = torch.randn(seqlen, 64, 2).to(device="cuda", dtype=model.dtype)
#        mask = torch.ones([bsz, 1, 1, seqlen], dtype=torch.bool, device="cuda")
#        input_pos = torch.arange(0, seqlen, dtype=torch.int32, device="cuda")
#
#        model.create_kv_cache(bsz, seqlen)
#
#        inputs = [x, freqs_cis, mask, input_pos, model.k_cache, model.v_cache]
#
#        return inputs
#
#    def pad_inputs(self, model, inputs, N, bsz, seqlen):
#        model.reset_kv_cache()
#        model.create_kv_cache(bsz, seqlen)
#
#        x, freqs_cis, mask, input_pos, k_cache, v_cache = inputs
#
#        inputs_p = [
#            PaddedTensor(x, [N, N], None),
#            PaddedTensor(freqs_cis, [N, 1, 1], None),
#            PaddedTensor(mask, [N, 1, 1, N], None, False),
#            PaddedTensor(input_pos, [N], None, -1),
#            PaddedTensor(k_cache, [N, 1, N, 1], None),
#            PaddedTensor(v_cache, [N, 1, N, 1], None),
#        ]
#
#        return inputs_p
#
#    def test_attention_bench(self):
#        model = TransformerModel()
#        fn = model.f_attention
#
#        with torch.no_grad():
#            inputs = self.gen_rand_inputs(model, 15, 1023)
#
#            def median(x):
#                return sorted(x)[len(x) // 2]
#
#            def bench(fn, inputs, n_warmup=5, n_iter=10):
#                times = []
#                outputs = None
#
#                # Warmup
#                inps = pytree.tree_map(lambda x: x.clone(), inputs)
#                for _ in range(n_warmup):
#                    fn(*inps)
#
#                for _ in range(n_iter):
#                    model.reset_kv_cache()
#                    inps = pytree.tree_map(lambda x: x.clone(), inputs)
#
#                    t = benchmarker.benchmark_gpu(lambda: fn(*inps))
#                    torch.cuda.synchronize()
#
#                    # start = time.time()
#                    # outputs = fn(*inps)
#                    # t = time.time() - start
#
#                    times.append(t)
#                torch._dynamo.reset()
#
#                outputs = fn(*inps)
#                outputs = pytree.tree_map(lambda x: x.clone(), outputs)
#                return median(times), outputs
#
#            # Run unpadded
#            time_eager, outputs_eager = bench(fn, inputs, n_warmup=1, n_iter=1)
#
#            # self.are_equal(outputs_eager, outputs_eager_padded)
#
#            # Run unpadded and padded compiled
#            fn_compiled = torch.compile(fn, mode="reduce-overhead")
#
#            time_compiled, outputs_compiled = bench(fn_compiled, inputs)
#
#            inputs_p = self.pad_inputs(model, inputs, 16, 16, 1024)
#            time_compiled_padded, outputs_compiled_padded = bench(fn_compiled, inputs_p)
#
#            print(outputs_compiled_padded[0].shape)
#            print(outputs_compiled_padded[0].orig_shape)
#
#            # # Report results
#            print(f"Unpadded time (eager): {time_eager}")
#            print(f"Unpadded time (compiled): {time_compiled}")
#            print(f"Padded time (compiled): {time_compiled_padded}")
#
#            self.are_equal(outputs_eager, outputs_compiled_padded)


class ModelTests(PaddedTensorTestCase):
    def test_transformer_model(self):
        with torch.no_grad():
            with torch.device("cuda"):
                pad = 4
                bsz, seqlen = 4, 2 + pad

                # Set up transformer
                args = ModelArgs.from_name("stories15M")
                transformer = Transformer(args)
                transformer.setup_caches(bsz, seqlen)

                # Set up inputs
                inputs = (
                    torch.randint(0, 3, (bsz, seqlen - pad)).to(device="cuda"),
                    torch.randint(0, 3, (seqlen - pad,)).to(device="cuda"),
                )

                inputs_p = [
                    PaddedTensor(inputs[0], [bsz, seqlen], None),
                    PaddedTensor(inputs[1], [seqlen], None, -1),
                ]

                # Run
                out = transformer(*inputs)

                transformer = torch.compile(transformer, mode="reduce-overhead")
                out_p = transformer(*inputs_p)
                out_p = pytree.tree_map(lambda x: x.unpad(), out_p)

                # Check
                self.are_equal(out, out_p)

                is_out_equal = pytree.tree_map(
                    lambda o, p: torch.allclose(o, p, atol=1e-5), out, out_p
                )
                self.assertTrue(is_out_equal)


class AtenOpTests(PaddedTensorTestCase):
    def setUp(self):
        super().setUp()

    def assert_padded_dims(self, z: PaddedTensor, padded_dim_idxs: List[int]):
        padded_dim_idxs_set = set(padded_dim_idxs)

        for dim_idx in range(len(z.shape)):
            if dim_idx in padded_dim_idxs_set:
                self.assertTrue(z.orig_shape[dim_idx].is_padded)
            else:
                self.assertFalse(z.orig_shape[dim_idx].is_padded)

    def test_elementwise_unary(self):
        for op in [aten.tril, aten.sin, aten.rsqrt, aten.silu]:
            a = PaddedTensor(torch.randn(3, 3), [4, 4])
            z = op(a)

            self.assertEqual(z.shape, torch.Size([4, 4]))
            self.assertEqual(z.unpad().shape, torch.Size([3, 3]))
            self.assert_padded_dims(z, [0, 1])

    def test_elementwise_binary(self):
        for op in [aten.add, aten.sub, aten.mul, aten.div]:
            a = PaddedTensor(torch.randn(3, 5), [4, 6])
            b = PaddedTensor(torch.randn(3, 5), [4, 6])
            z = op(a, b)

            self.assertEqual(z.shape, torch.Size([4, 6]))
            self.assertEqual(z.unpad().shape, torch.Size([3, 5]))
            self.assert_padded_dims(z, [0, 1])

    def test_view(self):
        # Collapse
        # ############
        # Collapse start
        x = PaddedTensor(torch.randn(3, 5, 7), [4, 6, 1])
        z = aten.view(x, [24, 7])

        self.assertEqual(z.unpad().shape, torch.Size([15, 7]))
        self.assert_padded_dims(z, [0])

        # Collapse end
        x = PaddedTensor(torch.randn(3, 5, 7), [4, 6, 1])
        z = aten.view(x, [4, 42])

        self.assertEqual(z.unpad().shape, torch.Size([3, 35]))
        self.assert_padded_dims(z, [0, 1])

        # Collapse middle
        x = PaddedTensor(torch.randn(3, 5, 7, 9), [4, 6, 1, 1])
        z = aten.view(x, [4, 42, 9])

        self.assertEqual(z.unpad().shape, torch.Size([3, 35, 9]))
        self.assert_padded_dims(z, [0, 1])

        # Collapse multiple
        x = PaddedTensor(torch.randn(3, 5, 7, 9, 11), [4, 6, 1, 1, 1])
        z = aten.view(x, [24, 7, 99])

        self.assertEqual(z.unpad().shape, torch.Size([15, 7, 99]))
        self.assert_padded_dims(z, [0])

        # Expand
        # ############
        # Expand start
        x = PaddedTensor(torch.randn(3, 5, 7), [1, 6, 1])
        z = aten.view(x, [18, 7])

        self.assertEqual(z.unpad().shape, torch.Size([15, 7]))
        self.assert_padded_dims(z, [0])

        # Expand end
        x = PaddedTensor(torch.randn(3, 5, 7), [1, 6, 1])
        z = aten.view(x, [3, 42])
        self.assertEqual(z.unpad().shape, torch.Size([3, 35]))
        self.assert_padded_dims(z, [1])

        # Test that unpad throws an exception, when we can't infer the dim.
        x = PaddedTensor(torch.randn(15, 7), [4, 1, 1])
        z = aten.view(x, [4, 4, 7])
        with self.assertRaisesRegex(
            Exception,
            "PaddedTensor couldn't figure out a shape, likely due to an expansion.",
        ):
            z.unpad()


if __name__ == "__main__":
    run_tests()
