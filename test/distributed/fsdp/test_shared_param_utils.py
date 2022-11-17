# Owner(s): ["oncall: distributed"]

import torch
import torch.nn as nn
from torch.distributed.fsdp._shared_param_utils import (
    get_lca_query,
    get_shared_param_info_to_lca,
    SharedParamInfo,
)
from torch.testing._internal.common_utils import run_tests, TestCase


class Root(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inner1 = Inner()
        self.inner2 = Inner()
        self.inner3 = Inner()
        self.inner4 = Inner()
        self.inner5 = Inner()
        self.leaf = Leaf()
        # (1) Same depth cousin sharing: LCA = `root`
        self.inner1.child1.p = self.inner2.child2.p
        # (2) Diferent depth cousin sharing: LCA = `root`
        self.inner1.p = self.inner2.child1.p
        # (3) Sibling (same depth) sharing: LCA = `inner3`
        self.inner3.child1.p = self.inner3.child2.p
        # (4, 5, 6) Three-way sharing: LCA = `root`
        # Only two of (4, 5, 6) should be present since order does not matter,
        # where in general for k-way sharing, we expect k-1 entries
        self.inner2.p = self.leaf.p
        self.inner3.p = self.leaf.p
        # (7) Parent-child sharing: LCA = `inner4`
        self.inner4.child1.p = self.inner4.p
        # Parent-child sharing to be ignored
        self.inner5.child1.p = self.inner5.p


class Inner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.randn((1, 1)))
        self.child1 = Leaf()
        self.child2 = Leaf()


class Leaf(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.randn((1, 1)))


class TestSharedParamUtils(TestCase):
    def test_get_shared_param_info_to_lca(self):
        """
        Tests ``get_shared_param_info_to_lca()``, which computes the lowest
        common ancestor for shared parameters.
        """
        root = Root()
        ignored_params = set(root.inner5.parameters())
        shared_param_info_to_lca = get_shared_param_info_to_lca(root, ignored_params)
        # Expect 6 since (1) through (7) with only two out of (4, 5, 6) and
        # `inner5` ignored
        self.assertEqual(len(shared_param_info_to_lca), 6)
        # (1)
        query1 = get_lca_query(root.inner1.child1, root.inner2.child2)
        spi1 = SharedParamInfo(*query1, root.inner1.child1.p)
        lca1 = root
        # (2)
        query2 = get_lca_query(root.inner1, root.inner2.child1)
        spi2 = SharedParamInfo(*query2, root.inner1.p)
        lca2 = root
        # (3)
        query3 = get_lca_query(root.inner3.child1, root.inner3.child2)
        spi3 = SharedParamInfo(*query3, root.inner3.child1.p)
        lca3 = root.inner3
        # (4, 5, 6)
        query4 = get_lca_query(root.inner2, root.leaf)
        spi4 = SharedParamInfo(*query4, root.inner2.p)
        query5 = get_lca_query(root.inner3, root.leaf)
        spi5 = SharedParamInfo(*query5, root.inner2.p)
        query6 = get_lca_query(root.inner2, root.inner3)
        spi6 = SharedParamInfo(*query6, root.inner2.p)
        lca4 = lca5 = lca6 = root
        # (7)
        query7 = get_lca_query(root.inner4.child1, root.inner4)
        spi7 = SharedParamInfo(*query7, root.inner4.p)
        lca7 = root.inner4

        # Check that the keys are present and map to the expected LCA values
        for spi, lca in ((spi1, lca1), (spi2, lca2), (spi3, lca3), (spi7, lca7)):
            self.assertIn(spi, shared_param_info_to_lca)
            self.assertEqual(shared_param_info_to_lca[spi], lca)
        # (4, 5, 6) should only contribute 2 entries, where it does not matter
        # which pair (4, 5), (5, 6), (4, 6) is excluded
        check_in_bools = []
        check_equal_bools = []
        for spi, lca in ((spi4, lca4), (spi5, lca5), (spi6, lca6)):
            has_entry = spi in shared_param_info_to_lca
            check_in_bools.append(has_entry)
            check_equal_bools.append(
                shared_param_info_to_lca[spi] == lca if has_entry else False
            )
        self.assertGreaterEqual(sum(check_in_bools), 2)
        self.assertGreaterEqual(sum(check_equal_bools), 2)


if __name__ == "__main__":
    run_tests()
