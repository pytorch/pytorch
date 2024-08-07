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

    def test_chunk_spec(self):
        mod = ModelWithKwargs()
        batch_size = ModelWithKwargs.DEFAULT_BATCH_SIZE

        x = torch.randn(batch_size, d_hid)
        y = torch.randn(batch_size, d_hid)

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
        )

        ref = mod(x, y)
        out = pipe(x, y)[0]
        torch.testing.assert_close(out, ref)
        print(f"equivalence test passed {torch.sum(out)} ref {torch.sum(ref)}")


if __name__ == "__main__":
    run_tests()
