# Owner(s): ["module: dynamo"]
import sys
import unittest
from typing import Dict, List

import torch
import torch._dynamo.config
import torch._dynamo.test_case
from torch import nn
from torch._dynamo.test_case import TestCase
from torch._dynamo.testing import CompileCounter
from torch.testing._internal.common_distributed import (
    _dynamo_dist_per_rank_init,
    at_least_x_gpu,
    DynamoDistributedMultiProcTestCase,
    requires_nccl,
)
from torch.testing._internal.common_utils import NoTest


try:
    from torchrec.datasets.random import RandomRecDataset
    from torchrec.distributed import comm_ops
    from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor

    HAS_TORCHREC = True
except ImportError:
    HAS_TORCHREC = False


@torch._dynamo.config.patch(force_unspec_int_unbacked_size_like_on_torchrec_kjt=True)
class BucketizeMod(torch.nn.Module):
    def __init__(self, feature_boundaries: Dict[str, List[float]]):
        super().__init__()
        self.bucket_w = torch.nn.ParameterDict()
        self.boundaries_dict = {}
        for key, boundaries in feature_boundaries.items():
            self.bucket_w[key] = torch.nn.Parameter(
                torch.empty([len(boundaries) + 1]).fill_(1.0),
                requires_grad=True,
            )
            buf = torch.tensor(boundaries, requires_grad=False)
            self.register_buffer(
                f"{key}_boundaries",
                buf,
                persistent=False,
            )
            self.boundaries_dict[key] = buf

    def forward(self, features: "KeyedJaggedTensor") -> "KeyedJaggedTensor":
        weights_list = []
        for key, boundaries in self.boundaries_dict.items():
            jt = features[key]
            bucketized = torch.bucketize(jt.weights(), boundaries)
            # doesn't super matter I guess
            # hashed = torch.ops.fb.index_hash(bucketized, seed=0, modulo=len(boundaries))
            hashed = bucketized
            weights = torch.gather(self.bucket_w[key], dim=0, index=hashed)
            weights_list.append(weights)
        return KeyedJaggedTensor(
            keys=features.keys(),
            values=features.values(),
            weights=torch.cat(weights_list),
            lengths=features.lengths(),
            offsets=features.offsets(),
            stride=features.stride(),
            length_per_key=features.length_per_key(),
        )


if not HAS_TORCHREC:
    print("torchrec not available, skipping tests", file=sys.stderr)
    TestCase = NoTest  # noqa: F811


@unittest.skipIf(not HAS_TORCHREC, "these tests require torchrec")
class TorchRecTests(TestCase):
    def test_pooled(self):
        tables = [
            (nn.EmbeddingBag(2000, 8), ["a0", "b0"]),
            (nn.EmbeddingBag(2000, 8), ["a1", "b1"]),
            (nn.EmbeddingBag(2000, 8), ["b2"]),
        ]

        embedding_groups = {
            "a": ["a0", "a1"],
            "b": ["b0", "b1", "b2"],
        }

        counter = CompileCounter()

        @torch.compile(backend=counter, fullgraph=True, dynamic=True)
        def f(id_list_features: KeyedJaggedTensor):
            id_list_jt_dict: Dict[str, JaggedTensor] = id_list_features.to_dict()
            pooled_embeddings = {}
            # TODO: run feature processor
            for emb_module, feature_names in tables:
                features_dict = id_list_jt_dict
                for feature_name in feature_names:
                    f = features_dict[feature_name]
                    pooled_embeddings[feature_name] = emb_module(
                        f.values(), f.offsets()
                    )

            pooled_embeddings_by_group = {}
            for group_name, group_embedding_names in embedding_groups.items():
                group_embeddings = [
                    pooled_embeddings[name] for name in group_embedding_names
                ]
                pooled_embeddings_by_group[group_name] = torch.cat(
                    group_embeddings, dim=1
                )

            return pooled_embeddings_by_group

        dataset = RandomRecDataset(
            keys=["a0", "a1", "b0", "b1", "b2"],
            batch_size=4,
            hash_size=2000,
            ids_per_feature=3,
            num_dense=0,
        )
        di = iter(dataset)

        # unsync should work

        d1 = next(di).sparse_features.unsync()
        d2 = next(di).sparse_features.unsync()
        d3 = next(di).sparse_features.unsync()

        r1 = f(d1)
        r2 = f(d2)
        r3 = f(d3)

        self.assertEqual(counter.frame_count, 1)
        counter.frame_count = 0

        # sync should work too

        d1 = next(di).sparse_features.sync()
        d2 = next(di).sparse_features.sync()
        d3 = next(di).sparse_features.sync()

        r1 = f(d1)
        r2 = f(d2)
        r3 = f(d3)

        self.assertEqual(counter.frame_count, 1)

        # export only works with unsync

        gm = torch._dynamo.export(f)(next(di).sparse_features.unsync()).graph_module
        gm.print_readable()

        self.assertEqual(gm(d1), r1)
        self.assertEqual(gm(d2), r2)
        self.assertEqual(gm(d3), r3)

    def test_bucketize(self):
        mod = BucketizeMod({"f1": [0.0, 0.5, 1.0]})
        features = KeyedJaggedTensor.from_lengths_sync(
            keys=["f1"],
            values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
            lengths=torch.tensor([2, 0, 1, 1, 1, 3]),
            weights=torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
        ).unsync()

        def f(x):
            # This is a trick to populate the computed cache and instruct
            # ShapeEnv that they're all sizey
            x.to_dict()
            return mod(x)

        torch._dynamo.export(f, aten_graph=True)(features).graph_module.print_readable()

    @unittest.expectedFailure
    def test_simple(self):
        jag_tensor1 = KeyedJaggedTensor(
            values=torch.tensor([3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
            keys=["index_0", "index_1"],
            lengths=torch.tensor([0, 0, 1, 1, 1, 3]),
        ).sync()

        # ordinarily, this would trigger one specialization
        self.assertEqual(jag_tensor1.length_per_key(), [1, 5])

        counter = CompileCounter()

        @torch._dynamo.optimize(counter, nopython=True)
        def f(jag_tensor):
            # The indexing here requires more symbolic reasoning
            # and doesn't work right now
            return jag_tensor["index_0"].values().sum()

        f(jag_tensor1)

        self.assertEqual(counter.frame_count, 1)

        jag_tensor2 = KeyedJaggedTensor(
            values=torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
            keys=["index_0", "index_1"],
            lengths=torch.tensor([2, 0, 1, 1, 1, 3]),
        ).sync()

        f(jag_tensor2)

        self.assertEqual(counter.frame_count, 1)


# TODO(yf225): double-check that our CI has torchrec
@unittest.skipIf(not HAS_TORCHREC, "these tests require torchrec")
@requires_nccl()
class TorchRecMultiProcTests(DynamoDistributedMultiProcTestCase):
    def test_awaitable(self):
        with _dynamo_dist_per_rank_init(
            self.rank, self.world_size, fake_pg=not at_least_x_gpu(2)
        ):
            device = torch.device(f"cuda:{self.rank}")

            torch.cuda.set_device(device)

            B_global = 2
            D0 = 8
            D1 = 9

            input_embedding0 = torch.rand(
                (B_global, D0),
                device=device,
                requires_grad=True,
            )
            input_embedding0 = torch.tensor(
                [[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16]],
                device=device,
                requires_grad=True,
                dtype=torch.float32,
            )
            input_embedding1 = torch.rand(
                (B_global, D1),
                device=device,
                requires_grad=True,
            )
            input_embedding1 = torch.tensor(
                [
                    [11, 12, 13, 14, 15, 16, 17, 18, 19],
                    [20, 21, 22, 23, 24, 25, 26, 27, 28],
                ],
                device=device,
                requires_grad=True,
                dtype=torch.float32,
            )

            input_embeddings = [input_embedding0, input_embedding1]
            out_split = [17, 17]

            def wait_fn(awaitable: comm_ops.Request):
                res = awaitable.wait()
                res1 = res[0] + res[1]
                return res1

            def func(compile_wait_fn=False):
                v_embs_out_awaitable = comm_ops.alltoallv(
                    input_embeddings, out_split=out_split
                )
                assert isinstance(v_embs_out_awaitable, comm_ops.Request)
                if compile_wait_fn:
                    with torch._dynamo.config.patch(
                        skip_torchrec=False,
                    ):
                        v_embs_out = torch.compile(
                            wait_fn, backend="inductor", fullgraph=True
                        )(v_embs_out_awaitable)
                else:
                    v_embs_out = wait_fn(v_embs_out_awaitable)
                return torch.sum(v_embs_out)

            out_ref = func(compile_wait_fn=False)
            out_compiled = func(compile_wait_fn=True)
            self.assertEqual(out_ref, out_compiled)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
