from torch.fx.interpreter import Interpreter


class FakeTensorProp(Interpreter):
    def run(self):
        inp = tuple(
            node.meta["val"]
            for node in self.module.graph.nodes
            if node.op == "placeholder"
        )
        super().run(*inp)

    def run_node(self, node):
        res = super().run_node(node)
        # split_cat fx passes expect "example_value" metadata on the nodes
        node.meta["example_value"] = res
        node.meta["val"] = res
        return res
