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
        B = 6
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

        batch = list(range(30)) * 8
        batch = torch.tensor(batch[: B * Q_LEN], device=device).reshape(B, Q_LEN)
        q = torch.randn(B, H, Q_LEN, DIM, device=device, requires_grad=True)
        k = torch.randn(B, H, KV_LEN, DIM, device=device, requires_grad=True)
        v = torch.randn(B, H, KV_LEN, DIM, device=device, requires_grad=True)
        block_mask_fn = torch.compile(create_block_mask)
        block_mask = block_mask_fn(
            create_block_causal_mask(batch, 29),
            B=B,
            H=H,
            Q_LEN=Q_LEN,
            KV_LEN=KV_LEN,
            device=device,
        )
        if device == "cuda":
            flex_fn = torch.compile(flex_attention)
        else:
            # It's unclear why CPU + torch.compile + flex_attention can cause an issue.
            flex_fn = flex_attention
        out = flex_fn(q, k, v, block_mask=block_mask)
        out.sum().backward()

        q_clone = q.clone().detach()
        k_clone = k.clone().detach()
        v_clone = v.clone().detach()
        arg_split, _ = split_args_kwargs_into_chunks(
            (q_clone, k_clone, v_clone, {"block_mask": block_mask}),
            {},
            chunks=B,
            args_chunk_spec=None,
            kwargs_chunk_spec=None,
        )

        q_chunks = []
        dq_chunks = []
        k_chunks = []
        dk_chunks = []
        v_chunks = []
        dv_chunks = []
        block_mask_chunks = []
        out_chunks = []
        for i in range(len(arg_split)):
            q_chunk, k_chunk, v_chunk, block_mask_chunk = arg_split[i]
            q_chunk.requires_grad = True
            k_chunk.requires_grad = True
            v_chunk.requires_grad = True
            out_chunk = flex_fn(
                q_chunk, k_chunk, v_chunk, block_mask=block_mask_chunk["block_mask"]
            )
            out_chunk.sum().backward()
            q_chunks.append(q_chunk)
            dq_chunks.append(q_chunk.grad)
            k_chunks.append(k_chunk)
            dk_chunks.append(k_chunk.grad)
            v_chunks.append(v_chunk)
            dv_chunks.append(v_chunk.grad)
            block_mask_chunks.append(block_mask_chunk["block_mask"])
            out_chunks.append(out_chunk)

        concat_q = torch.cat(q_chunks, dim=0)
        concat_dq = torch.cat(dq_chunks, dim=0)
        concat_k = torch.cat(k_chunks, dim=0)
        concat_dk = torch.cat(dk_chunks, dim=0)
        concat_v = torch.cat(v_chunks, dim=0)
        concat_dv = torch.cat(dv_chunks, dim=0)
        concat_kv_indices = torch.cat(
            [bm.kv_indices for bm in block_mask_chunks], dim=0
        )
        concat_kv_num_blocks = torch.cat(
            [bm.kv_num_blocks for bm in block_mask_chunks], dim=0
        )
        concat_kv_full_num_blocks = torch.cat(
            [bm.full_kv_num_blocks for bm in block_mask_chunks], dim=0
        )
        concat_kv_full_indices = torch.cat(
            [bm.full_kv_indices for bm in block_mask_chunks], dim=0
        )
        concat_out = torch.cat(out_chunks, dim=0)
        self.assertEqual(concat_q, q)
        self.assertEqual(concat_dq, q.grad)
        self.assertEqual(concat_k, k)
        self.assertEqual(concat_dk, k.grad)
        self.assertEqual(concat_v, v)
        self.assertEqual(concat_dv, v.grad)
        self.assertEqual(concat_kv_indices, block_mask.kv_indices)
        self.assertEqual(concat_kv_num_blocks, block_mask.kv_num_blocks)
        self.assertEqual(concat_kv_full_num_blocks, block_mask.full_kv_num_blocks)
        self.assertEqual(concat_kv_full_indices, block_mask.full_kv_indices)
        self.assertEqual(concat_out, out)

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
