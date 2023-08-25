from collections import defaultdict

import torch
import torch.fx
from torch.utils._pytree import tree_map
from .virtualized import V


# Check the pattern: (nn.module, F.function/torch.Tensor.method) matched.
# Works for length 2 patterns with 1 module and 1 function/method.
def matches_module_function_pattern(pattern, node, modules):
    if len(node.args) == 0:
        return False
    if not isinstance(node.args[0], torch.fx.Node) or not isinstance(
        node, torch.fx.Node
    ):
        return False
    # the first node is call_module
    if node.args[0].op != "call_module":
        return False
    if not isinstance(node.args[0].target, str):
        return False
    if node.args[0].target not in modules:
        return False
    if type(modules[node.args[0].target]) is not pattern[0]:
        return False
    # the second node is call_function or call_method
    if node.op != "call_function" and node.op != "call_method":
        return False
    if node.target != pattern[1]:
        return False
    # make sure node.args[0] output is only used by current node.
    if len(node.args[0].users) > 1:
        return False
    return True


class FakeTensorUpdater:
    """
    The main idea here is that it's difficult to maintain accurate fake
    tensors (our primary form of metadata) for each node in our graph as we
    transform it.

    The most reliable way to obtain this information is by rerunning
    faketensor propagation. However, in general, faketensor propagation is
    fairly expensive. So, instead we'd like to only rerun faketensor
    propagation on nodes that have changed.

    In order to detect which nodes have changed, we first hash its node,
    target, and argument lists (which are immutable in FX).

    Then, whenever we call incremental_update, we check which FX nodes have a
    new hash, and recompute the faketensor metadata for that node. Then, we
    continue to recursively compute the faketensors for all users until the
    fake tensors stop changing.
    """

    def __init__(self, graph: torch.fx.Graph):
        self.processed_hashes = set()
        self.graph = graph

        for node in self.graph.nodes:
            self.processed_hashes.add(self.hash_node(node))

    def hash_node(self, node: torch.fx.Node):
        return (node, node.target, node.args, node.kwargs)

    def incremental_update(self):
        processed = set()
        existing_storages = defaultdict(int)
        for node in self.graph.nodes:
            existing_storages[get_node_storage(node)] += 1

        def is_fake_tensor_same(new, old):
            if type(new) != type(old):
                return False
            if isinstance(new, (list, tuple)):
                if len(new) != len(old):
                    return False
                return all(
                    is_fake_tensor_same(new_i, old_i) for new_i, old_i in zip(new, old)
                )
            assert isinstance(new, torch.Tensor)
            if new.shape != old.shape or new.layout != old.layout:
                return False
            if new.layout == torch.strided and new.stride() != old.stride():
                return False
            if get_storage(new) == get_storage(old):
                return True

            # This is the case where it returns a completely fresh storage that's used nowhere else.
            if (
                existing_storages[get_storage(old)] == 1
                and get_storage(new) not in existing_storages
            ):
                return True
            return False

        for node in self.graph.nodes:
            if self.hash_node(node) in self.processed_hashes:
                continue

            def get_fake_tensor(x):
                if isinstance(x, torch.fx.Node):
                    return x.meta["val"]
                return x

            def is_aten_node(node):
                return node.op == "call_function" and isinstance(
                    node.target, torch._ops.OpOverload
                )

            if not is_aten_node(node):
                continue

            processing = [node]
            while len(processing) > 0:
                updating_node = processing.pop()
                if updating_node in processed:
                    continue
                if is_aten_node(updating_node):
                    continue

                args, kwargs = tree_map(
                    get_fake_tensor, (updating_node.args, updating_node.kwargs)
                )
                with V.fake_mode:
                    new_fake_tensor = updating_node.target(*args, **kwargs)
                if "val" in updating_node.meta and is_fake_tensor_same(
                    new_fake_tensor, updating_node.meta["val"]
                ):
                    continue
                updating_node.meta["val"] = new_fake_tensor
                existing_storages.add(get_node_storage(new_fake_tensor))
                processed.add(updating_node)
                for user in updating_node.users:
                    processing.append(user)

                self.processed_hashes.add(self.hash_node(updating_node))


def get_storage(t):
    return t.untyped_storage()._cdata


def get_node_storage(node):
    if "val" not in node.meta:
        return None
    if not isinstance(node.meta["val"], torch.Tensor):
        return None
    if not torch._C._has_storage(node.meta["val"]):
        return None
    return get_storage(node.meta["val"])
