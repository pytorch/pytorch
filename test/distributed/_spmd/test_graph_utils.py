# Owner(s): ["oncall: distributed"]
import os

from torch.distributed._spmd.graph_utils import dump_graphs_to_files
from torch.testing._internal.common_utils import run_tests  # noqa: TCH001
from torch.testing._internal.distributed._tensor.common_dtensor import DTensorTestBase


class GraphUtilsTest(DTensorTestBase):
    @property
    def world_size(self):
        return 1

    def test_dump_graphs(self):
        class FakeGraph:
            def __init__(self, postfix):
                self.graph = f"fake graph str {postfix}"

            def __str__(self) -> str:
                return self.graph

        fake_graph1 = {"fake_graph1": FakeGraph(1)}
        folder = dump_graphs_to_files(fake_graph1)
        fake_graph2 = {"fake_graph2": FakeGraph(1)}
        new_folder = dump_graphs_to_files(fake_graph2, folder)
        self.assertEqual(folder, new_folder)

        for i in (1, 2):
            path = os.path.join(folder, f"fake_graph{i}.graph")
            self.assertTrue(os.path.exists(path))
            with open(path) as fp:
                fake_graph = fake_graph1 if i == 1 else fake_graph2
                self.assertEqual(fp.readline(), fake_graph[f"fake_graph{i}"].graph)
            os.remove(path)

        os.rmdir(folder)


if __name__ == "__main__":
    if False:
        run_tests()
