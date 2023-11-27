from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple

import torch
import torch._ops
import torch._subclasses.fake_tensor
import torch.fx
import torch.utils._pytree as pytree

from torch.fx.passes.infra.pass_base import PassBase, PassResult


__all__ = [
    "ZeroOrMultipleDevicesError",
    "ConstructorMoverPass",
]


@torch.fx._compatibility.compatibility(is_backward_compatible=False)
class ZeroOrMultipleDevicesError(RuntimeError):
    def __init__(self, target: str, devices: Iterable[torch.device]):
        self.target = target
        self.devices = list(devices)

        super().__init__(
            f"expected a single device of type {self.target} to be used "
            f"in the whole graph. Got: {self.devices}."
        )


@torch.fx._compatibility.compatibility(is_backward_compatible=False)
class ConstructorMoverPass(PassBase):
    def __init__(
        self, target: str, inplace: bool = False, allow_outputs: bool = False
    ) -> None:
        """
        Move constructors from cpu to the target_device.

        Sweeps through the module, looking for constructor nodes that can be moved
        to the target_device.

        A constructor node can be moved to the target_device iff all of its users
        can also be moved (tested by cannot_be_moved). Otherwise, all dependant
        constructor nodes won't be moved.

        - target: target device type
        - inplace: if True, do not create a new GraphModule. Modify the given graph, instead.
        - allow_outputs: allow outputs to be moved
        """

        self.target = target
        self.inplace = inplace
        self.allow_outputs = allow_outputs

        assert isinstance(target, str), (
            "target should be a string representing the device type. "
            f"Got: {type(target).__name__}"
        )

    def allow_cpu_device(self, node: torch.fx.Node) -> bool:
        """
        Returns whether a node that returns a tensor on the target device may have
        cpu tensors as input.
        """
        return node.target in (
            torch.ops.aten.index.Tensor,
            torch.ops.aten.index_put.default,
            torch.ops.aten.index_put_.default,
            torch.ops.aten.copy.default,
            torch.ops.aten.copy_.default,
            torch.ops.aten.slice_scatter.default,
        )

    def cannot_be_moved(self, node: torch.fx.Node) -> bool:
        """
        Returns whether a node can be moved to the target device.

        If this function returns False, it means that this node and all of its users
        won't be moved into the target device.
        """
        if node.target == "output":
            return not self.allow_outputs

        if not (
            isinstance(node.target, torch._ops.OpOverload)
            and node.target.namespace in ("prims", "aten")
        ):
            return True

        return False

    def get_node_device(self, node: torch.fx.Node) -> Optional[torch.device]:
        """
        Get the device of a node.
        """
        ten = node.meta.get("val")
        return None if not isinstance(ten, torch.Tensor) else ten.device

    def get_cpu_indeg_count(self, graph: torch.fx.Graph) -> Dict[torch.fx.Node, int]:
        """
        Get the number of cpu inputs to a node
        """
        cpu_indeg: Dict[torch.fx.Node, int] = defaultdict(int)

        for node in graph.nodes:
            cpu_count = 0

            def add_cpu_inp(node):
                nonlocal cpu_count
                device = self.get_node_device(node)
                cpu_count += device is not None and device.type == "cpu"

            pytree.tree_map_only(torch.fx.Node, add_cpu_inp, (node.args, node.kwargs))

            if cpu_count:
                cpu_indeg[node] = cpu_count

        return cpu_indeg

    def gather_constructors_and_target_devices(
        self, graph: torch.fx.Graph
    ) -> Tuple[List[torch.fx.Node], Set[torch.device]]:
        target_devices = set()
        constructors = []

        for node in graph.nodes:
            device = self.get_node_device(node)
            if device and device.type == self.target:
                target_devices.add(device)

            # output nodes are not constructors.
            if node.target == "output":
                continue

            if not (
                isinstance(node.target, torch._ops.OpOverload)
                and node.target.namespace in ("prims", "aten")
            ):
                continue

            if not torch._subclasses.fake_tensor._is_tensor_constructor(node.target):
                continue

            if not node.kwargs.get("device") == torch.device("cpu"):
                continue

            constructors.append(node)

        return constructors, target_devices

    def find_movable_constructors(
        self, graph: torch.fx.Graph, constructors: List[torch.fx.Node]
    ) -> List[torch.fx.Node]:
        """
        Starting from the cpu constructors, iterate through the graph and test that all of their
        downstream uses can safely be moved to cpu.
        """
        cpu_indeg: Dict[torch.fx.Node, int] = self.get_cpu_indeg_count(graph)

        # which constructors cannot be moved to cuda
        cannot_move: Set[torch.fx.Node] = set()

        # For any node in the graph, which constructors does it have a dependency on
        constructor_dependencies: Dict[torch.fx.Node, Set[torch.fx.Node]] = defaultdict(
            set
        )

        # if a cpu node has a dependency on two different cpu constructors,
        # then if either constructor cannot be moved to cuda, the other cannot as well.
        # In this case any node with a dependency on one will have a dependency on the other
        equal_constructor_sets: Dict[torch.fx.Node, Set[torch.fx.Node]] = {
            c: {c} for c in constructors
        }

        def make_dependencies_equivalent(
            set1: Set[torch.fx.Node], set2: Set[torch.fx.Node]
        ) -> Set[torch.fx.Node]:
            # could use union find but not worth complexity here
            set1.update(set2)
            for obj in set1:
                equal_constructor_sets[obj] = set1
            return set1

        queue: List[torch.fx.Node] = list(constructors)

        for c in queue:
            constructor_dependencies[c].add(c)

        while queue:
            node = queue.pop()
            dependencies = constructor_dependencies[node]

            for user in node.users:
                if self.cannot_be_moved(user):
                    cannot_move.update(dependencies)
                    break

                # this node was used on a op which takes in multiple devices and output a cuda
                # tensor. we can convert its cpu input to cuda without making further changes
                node_device = self.get_node_device(user)
                if (
                    self.allow_cpu_device(user)
                    and node_device
                    and node_device.type == self.target
                ):
                    # cpu_indeg is a defaultdict.
                    # The line below is needed so as to create, if not existent, the entry in
                    # the dictionary. Then, we can delete it. Otherwise, we get a KeyError.
                    cpu_indeg[user]
                    del cpu_indeg[user]
                else:
                    # otherwise, we should continue look at its downstream uses
                    cpu_indeg[user] -= 1
                    if cpu_indeg[user] == 0:
                        del cpu_indeg[user]
                        queue.append(user)

                unioned_set = make_dependencies_equivalent(
                    dependencies, constructor_dependencies[user]
                )
                constructor_dependencies[user] = unioned_set

        for node in cpu_indeg:
            if constructor_dependencies[node]:
                cannot_move.update(constructor_dependencies[node])

        all_cannot_move = cannot_move.copy()
        for constructor in cannot_move:
            all_cannot_move.update(equal_constructor_sets[constructor])

        return list(set(constructors) - all_cannot_move)

    def move_constructors_to_device(
        self, constructors: Iterable[torch.fx.Node], device: torch.device
    ) -> None:
        """
        Replaces the device keyword-argument of each of the constructors by the
        provided device.
        """
        for node in constructors:
            kwargs = node.kwargs.copy()
            kwargs["device"] = device
            node.kwargs = kwargs

    def call(self, graph_module: torch.fx.GraphModule) -> Optional[PassResult]:
        graph = graph_module.graph
        constructors, target_devices = self.gather_constructors_and_target_devices(
            graph
        )
        movable_constructors = self.find_movable_constructors(graph, constructors)

        if len(movable_constructors) == 0:
            return PassResult(graph_module, False)

        if len(target_devices) != 1:
            raise ZeroOrMultipleDevicesError(self.target, target_devices)

        target_device = next(iter(target_devices))

        if not self.inplace:
            env: Dict[torch.fx.Node, torch.fx.Node] = {}

            new_graph = torch.fx.Graph()
            new_graph.graph_copy(graph, val_map=env)

            # Update movable_constructors with the nodes of the new graph.
            movable_constructors = [env[node] for node in movable_constructors]
            # Create a new GraphModule for the newly created graph.
            graph_module = torch.fx.GraphModule(graph_module, new_graph)

        for node in movable_constructors:
            kwargs = node.kwargs.copy()
            kwargs["device"] = target_device
            node.kwargs = kwargs

        return PassResult(graph_module, True)
