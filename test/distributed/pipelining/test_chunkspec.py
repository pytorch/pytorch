# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]
import torch
from torch.distributed.pipelining import pipe_split, pipeline
from torch.distributed.pipelining.microbatch import (
    ArgsChunkSpec,
    KwargsChunkSpec,
    split_args_kwargs_into_chunks,
)
from torch.testing._internal.common_utils import run_tests, TestCase


d_hid = 512
batch_size = 256

torch.manual_seed(0)


class ModelWithKwargs(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mm_param0 = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.mm_param1 = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.mm_param2 = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.lin1 = torch.nn.Linear(d_hid, d_hid)
        self.lin2 = torch.nn.Linear(d_hid, d_hid)

    def forward(self, x, y, z=torch.zeros(batch_size, d_hid)):
        x = torch.mm(x, self.mm_param0)
        x = x + y
        x = torch.relu(x)
        x = x + z
        pipe_split()
        x = torch.mm(x, self.mm_param1)
        x = self.lin1(x)
        pipe_split()
        x = torch.relu(x)
        x = torch.mm(x, self.mm_param2)
        pipe_split()
        x = self.lin2(x)
        x = torch.relu(x)
        return x


class ChunkSpecTests(TestCase):
    def test_chunk_spec(self):
        mod = ModelWithKwargs()

        x = torch.randn(batch_size, d_hid)
        y = torch.randn(batch_size, d_hid)
        z = torch.randn(batch_size, d_hid)

        num_chunks = 4

        args_chunk_spec = ArgsChunkSpec.create((0, 0))
        kwargs_chunk_spec = KwargsChunkSpec.create({"z": 0})

        args_split, kwargs_split = split_args_kwargs_into_chunks(
            (x, y),
            {"z": z},
            num_chunks,
            args_chunk_spec,
            kwargs_chunk_spec,
        )

        pipe = pipeline(
            mod,
            mb_args=args_split[0],
            mb_kwargs=kwargs_split[0],
        )

        assert pipe.num_stages == 4

        ref = mod(x, y, z)
        out = pipe(x, y, z)[0]
        torch.testing.assert_close(out, ref)
        print(f"equivalence test passed {torch.sum(out)} ref {torch.sum(ref)}")


if __name__ == "__main__":
    run_tests()
