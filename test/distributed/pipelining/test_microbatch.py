# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]
from model_registry import ModelWithKwargs

import torch
from torch.distributed.pipelining import pipeline
from torch.distributed.pipelining.microbatch import (
    merge_chunks,
    split_args_kwargs_into_chunks,
    TensorChunkSpec,
)

from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    skipXPUIf,
)
from torch.testing._internal.common_utils import run_tests, TestCase


d_hid = 512
torch.manual_seed(0)


class MicrobatchTests(TestCase):
    def test_split_and_merge(self):
        x0 = torch.randn(128, d_hid)
        x1 = torch.randn(256, d_hid)
        x2 = torch.randn(512, d_hid)

        args = (x0, x1, x2)
        kwargs = {"x0": x0, "x1": x1, "x2": x2}

        # Default chunking: dim 0
        arg_chunks, kwarg_chunks = split_args_kwargs_into_chunks(args, kwargs, 2)
        assert len(arg_chunks) == 2
        assert len(kwarg_chunks) == 2
        assert arg_chunks[0][0].shape == torch.Size([64, d_hid])
        assert arg_chunks[1][0].shape == torch.Size([64, d_hid])
        assert arg_chunks[0][1].shape == torch.Size([128, d_hid])
        assert arg_chunks[0][2].shape == torch.Size([256, d_hid])
        assert kwarg_chunks[0]["x0"].shape == torch.Size([64, d_hid])
        assert kwarg_chunks[0]["x1"].shape == torch.Size([128, d_hid])
        assert kwarg_chunks[1]["x2"].shape == torch.Size([256, d_hid])

        # Merge chunks back together
        merged_args = merge_chunks(
            arg_chunks,
            (TensorChunkSpec(0), TensorChunkSpec(0), TensorChunkSpec(0)),
        )
        torch.testing.assert_close(merged_args, args)

        merged_kwargs = merge_chunks(
            kwarg_chunks,
            {
                "x0": TensorChunkSpec(0),
                "x1": TensorChunkSpec(0),
                "x2": TensorChunkSpec(0),
            },
        )
        torch.testing.assert_close(merged_kwargs, kwargs)
        print("Microbatch test passed")

    def test_split_block_mask(self, device):
        B = 4
        H = 1
        Q_LEN = 32
        KV_LEN = 32
        DIM = 32

        def create_block_causal_mask(batch, eos_id: int):
            mask = batch == eos_id
            mask[:, -1] = True
            acc_mask = torch.cumsum(torch.where(mask, 1, 0), dim=1)
            seq_idx = torch.zeros_like(acc_mask, dtype=torch.int32)
            seq_idx[:, 1:] = acc_mask[:, :-1]

            def block_causal_mask(
                b: torch.Tensor,
                h: torch.Tensor,
                q_idx: torch.Tensor,
                kv_idx: torch.Tensor,
            ):
                return (seq_idx[b, q_idx] == seq_idx[b, kv_idx]) & (q_idx >= kv_idx)

            return block_causal_mask

        batch = list(range(30)) * 5
        batch = torch.tensor(batch[: B * Q_LEN], device=device).reshape(B, Q_LEN)
        q = torch.randn(B, H, Q_LEN, DIM, device=device)
        k = torch.randn(B, H, KV_LEN, DIM, device=device)
        v = torch.randn(B, H, KV_LEN, DIM, device=device)
        # block_mask_fn = torch.compile(create_block_mask)
        block_mask_fn = create_block_mask
        block_mask = block_mask_fn(
            create_block_causal_mask(batch, 29),
            B=B,
            H=H,
            Q_LEN=Q_LEN,
            KV_LEN=KV_LEN,
            device=device,
        )
        # flex_fn = torch.compile(flex_attention)
        flex_fn = flex_attention
        out = flex_fn(q, k, v, block_mask=block_mask)

        arg_split, _ = split_args_kwargs_into_chunks(
            (q, k, v, block_mask),
            {},
            chunks=B,
            args_chunk_spec=None,
            kwargs_chunk_spec=None,
        )

        for i in range(B):
            q_chunk, k_chunk, v_chunk, block_mask_chunk = arg_split[i]
            out_chunk = flex_fn(q_chunk, k_chunk, v_chunk, block_mask=block_mask_chunk)
            self.assertEqual(q_chunk.squeeze(0), q[i])
            self.assertEqual(k_chunk.squeeze(0), k[i])
            self.assertEqual(v_chunk.squeeze(0), v[i])
            self.assertEqual(
                block_mask_chunk.kv_indices.squeeze(0), block_mask.kv_indices[i]
            )
            self.assertEqual(
                block_mask_chunk.kv_num_blocks.squeeze(0), block_mask.kv_num_blocks[i]
            )
            self.assertEqual(
                block_mask_chunk.full_kv_num_blocks.squeeze(0),
                block_mask.full_kv_num_blocks[i],
            )
            self.assertEqual(
                block_mask_chunk.full_kv_indices.squeeze(0),
                block_mask.full_kv_indices[i],
            )
            self.assertEqual(
                block_mask_chunk.q_indices.squeeze(0), block_mask.q_indices[i]
            )
            self.assertEqual(out_chunk.squeeze(0), out[i])

    @skipXPUIf(True, "https://github.com/intel/torch-xpu-ops/issues/1682")
    def test_chunk_spec(self, device):
        mod = ModelWithKwargs().to(device)
        batch_size = ModelWithKwargs.DEFAULT_BATCH_SIZE

        x = torch.randn(batch_size, d_hid, device=device)
        y = torch.randn(batch_size, d_hid, device=device)

        num_chunks = 4

        args_chunk_spec = TensorChunkSpec.from_tuple((0,))
        kwargs_chunk_spec = TensorChunkSpec.from_dict({"y": 0})

        args_split, kwargs_split = split_args_kwargs_into_chunks(
            (x,),
            {"y": y},
            num_chunks,
            args_chunk_spec,
            kwargs_chunk_spec,
        )

        pipe = pipeline(
            mod,
            mb_args=args_split[0],
            mb_kwargs=kwargs_split[0],
        ).to(device)

        ref = mod(x, y)
        out = pipe(x, y)[0]

        torch.testing.assert_close(out, ref)
        print(f"equivalence test passed {torch.sum(out)} ref {torch.sum(ref)}")


devices = ["cpu", "cuda", "hpu", "xpu"]
instantiate_device_type_tests(
    MicrobatchTests, globals(), only_for=devices, allow_xpu=True
)

if __name__ == "__main__":
    run_tests()
