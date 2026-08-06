# Owner(s): ["module: inductor"]

from types import SimpleNamespace

from torch._inductor.runtime.triton_heuristics import _get_binary_cta_args
from torch._inductor.test_case import run_tests, TestCase


class TestTritonClusterDims(TestCase):
    def test_cluster_dims_from_binary(self):
        binary = SimpleNamespace(num_ctas=4, cluster_dims=(1, 2, 3))

        self.assertEqual(_get_binary_cta_args(binary), (4, 1, 2, 3))

    def test_cluster_dims_from_metadata(self):
        binary = SimpleNamespace(
            num_ctas=4,
            metadata=SimpleNamespace(cluster_dims=(1, 2, 3)),
        )

        self.assertEqual(_get_binary_cta_args(binary), (4, 1, 2, 3))

    def test_cluster_dims_from_mixed_binary_and_metadata_layouts(self):
        binary = SimpleNamespace(
            cluster_dims=(1, 2, 3),
            metadata=SimpleNamespace(num_ctas=4),
        )

        self.assertEqual(_get_binary_cta_args(binary), (4, 1, 2, 3))

    def test_cluster_dims_from_legacy_metadata_name(self):
        binary = SimpleNamespace(
            num_ctas=4,
            metadata=SimpleNamespace(clusterDims=(1, 2, 3)),
        )

        self.assertEqual(_get_binary_cta_args(binary), (4, 1, 2, 3))

    def test_binary_cluster_dims_take_precedence_over_metadata(self):
        binary = SimpleNamespace(
            num_ctas=4,
            cluster_dims=(1, 2, 3),
            metadata=SimpleNamespace(cluster_dims=(4, 5, 6)),
        )

        self.assertEqual(_get_binary_cta_args(binary), (4, 1, 2, 3))

    def test_cluster_dims_returns_empty_tuple_without_cluster_info(self):
        self.assertEqual(_get_binary_cta_args(SimpleNamespace()), ())

    def test_num_ctas_without_cluster_dims_returns_empty_tuple(self):
        binary = SimpleNamespace(num_ctas=4)

        self.assertEqual(_get_binary_cta_args(binary), ())


if __name__ == "__main__":
    run_tests()
