import unittest
from typing import Dict

import torch
import torch._dynamo.test_case
from torch import nn
from torch._dynamo.testing import CompileCounter

try:
    from torchrec.datasets.random import RandomRecDataset
    from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor

    HAS_TORCHREC = True
except ImportError:
    HAS_TORCHREC = False


@unittest.skipIf(not HAS_TORCHREC, "these tests require torchrec")
class TorchRecTests(torch._dynamo.test_case.TestCase):
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

        # NB: this MUST be sync'ed, it currently doesn't work with unsync
        d1 = next(di).sparse_features
        d2 = next(di).sparse_features
        d3 = next(di).sparse_features

        r1 = f(d1)
        r2 = f(d2)
        r3 = f(d3)

        self.assertEqual(counter.frame_count, 1)

        """
        # TODO: this doesn't work, export specializes too much
        gm = torch._dynamo.export(f)(next(di).sparse_features).graph_module
        gm.print_readable()

        self.assertEqual(gm(d1), r1)
        self.assertEqual(gm(d2), r1)
        self.assertEqual(gm(d3), r1)
        """

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


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
