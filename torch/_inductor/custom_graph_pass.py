import hashlib
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from functools import lru_cache
from typing import Any, TYPE_CHECKING, TypeAlias

import torch.fx.graph


if TYPE_CHECKING:
    from torch._functorch.partitioners import NodeInfo


class CustomGraphPass(ABC):
    """
    Implement this interface for custom Graph passes:

    1) The __call__() method contains the implementation of the custom pass.

    2) The uuid() method enables inductor to cache compiled graphs when your custom
    passes are applied. This method can return any identifier as long as it uniquely
    identifies your implementation (and can be pickled). The caching logic includes this
    identifier in its key calculation, i.e., any new value will effectively invalidate
    existing entries. We expect custom passes would typically depend purely on the
    textual representation of the implementation. In that case, we recommend using the
    'get_hash_for_files' helper below to compute a unique hash from the contents of a
    static list of source files, i.e., the source(s) containing the custom pass
    implementation. That approach ensures that any change to the implementation will
    mean a new uuid.

    ** IMPORTANT ** If your custom pass's behavior depends on some external state, then
    you'll need to implement something more complicated (or disable caching).

    EXAMPLE:

    class MyCustomGraphPass(CustomGraphPass):
        def __call__(self, graph: torch.fx.graph.Graph) -> None:
            # my custom graph optimization pass
            #     ...

        def uuid(self) -> Optional[Any]:
            return get_hash_for_files((__file__,))

    """

    @abstractmethod
    def __call__(self, graph: torch.fx.graph.Graph) -> None:
        """
        Implementation of the custom pass.
        """

    @abstractmethod
    def uuid(self) -> Any | None:
        """
        Return an ID to uniquely identify your custom pass implementation. Return None
        to skip inductor code caching entirely.
        """


class CustomGraphModulePass(ABC):
    """
    Implement this interface for custom Graph passes:

    1) The __call__() method contains the implementation of the custom pass.

    2) The uuid() method enables inductor to cache compiled graphs when your custom
    passes are applied. This method can return any identifier as long as it uniquely
    identifies your implementation (and can be pickled). The caching logic includes this
    identifier in its key calculation, i.e., any new value will effectively invalidate
    existing entries. We expect custom passes would typically depend purely on the
    textual representation of the implementation. In that case, we recommend using the
    'get_hash_for_files' helper below to compute a unique hash from the contents of a
    static list of source files, i.e., the source(s) containing the custom pass
    implementation. That approach ensures that any change to the implementation will
    mean a new uuid.
    """

    @abstractmethod
    def __call__(self, gm: torch.fx.GraphModule) -> None:
        """
        Implementation of the custom pass.
        """

    @abstractmethod
    def uuid(self) -> Any | None:
        """
        Return an ID to uniquely identify your custom pass implementation. Return None
        to skip inductor code caching entirely.
        """


class CustomInferenceAwareGraphPass(CustomGraphPass):
    """
    Implement this interface for custom inference aware Graph passes.

    """

    @abstractmethod
    def __call__(self, graph: torch.fx.graph.Graph, is_inference: bool) -> None:
        """
        Implementation of the custom pass.
        """


CustomGraphPassType: TypeAlias = (
    CustomGraphPass | Callable[[torch.fx.graph.Graph], None] | None
)


@lru_cache(1)
def get_hash_for_files(paths: tuple[str, ...], extra: str = "") -> bytes:
    """
    Helper to compute a unique string by hashing the contents of a list of files.
    """
    hasher = hashlib.sha256()
    hasher.update(extra.encode("utf-8"))
    for path in paths:
        with open(path, "rb") as f:
            hasher.update(f.read())
    return hasher.digest()


class CustomPartitionerFn(ABC):
    """
    Implement this interface for custom partitioner:

    1) The __call__() method contains the implementation of the custom partitioner.

    2) The uuid() method enables inductor to cache compiled graphs when your custom
    partitioner are applied. This method can return any identifier as long as it uniquely
    identifies your implementation (and can be pickled). The caching logic includes this
    identifier in its key calculation, i.e., any new value will effectively invalidate
    existing entries. We expect custom partitioner would typically depend purely on the
    textual representation of the implementation. In that case, we recommend using the
    'get_hash_for_files' helper below to compute a unique hash from the contents of a
    static list of source files, i.e., the source(s) containing the custom partitioner
    implementation. That approach ensures that any change to the implementation will
    mean a new uuid.

    EXAMPLE:

    from torch._inductor.custom_graph_pass import get_hash_for_files

    class MyCustomPartitionerFn(CustomPartitionerFn):
        def __call__(
            self,
            gm: torch.fx.GraphModule,
            joint_inputs: Sequence[object],
            **kwargs: Any
        ) -> tuple[torch.fx.GraphModule, torch.fx.GraphModule]:
            # my custom partitioner implementation
            #     ...

        def uuid(self) -> Optional[Any]:
            return get_hash_for_files((__file__,))

    """

    @abstractmethod
    def __call__(
        self, gm: torch.fx.GraphModule, joint_inputs: Sequence[object], **kwargs: Any
    ) -> tuple[torch.fx.GraphModule, torch.fx.GraphModule]:
        """
        Implementation of the custom partitioner.
        """

    @abstractmethod
    def uuid(self) -> Any | None:
        """
        Return an ID to uniquely identify your custom partitioner implementation.
        Return None to skip inductor code caching entirely.
        """


CustomPartitionerFnType: TypeAlias = CustomPartitionerFn | None


class CustomRuntimeEstimator(ABC):
    """
    Implement this interface for custom runtime estimators:

    1) The __call__() method contains the implementation of the runtime estimation.

    2) The uuid() method enables AOTAutograd to cache compiled graphs when your custom
    runtime estimator is used. This method can return any identifier as long as it uniquely
    identifies your implementation (and can be pickled). The caching logic includes this
    identifier in its key calculation, i.e., any new value will effectively invalidate
    existing entries. We expect custom runtime estimators would typically depend purely on the
    textual representation of the implementation. In that case, we recommend using the
    'get_hash_for_files' helper below to compute a unique hash from the contents of a
    static list of source files, i.e., the source(s) containing the custom runtime estimator
    implementation. That approach ensures that any change to the implementation will
    mean a new uuid.

    ** IMPORTANT ** If your custom runtime estimator's behavior depends on some external state,
    then you'll need to implement something more complicated (or disable caching).

    EXAMPLE:

    class MyCustomRuntimeEstimator(CustomRuntimeEstimator):
        def __call__(self, node: fx.Node) -> float:
            # my custom runtime estimation logic
            return estimated_runtime

        def uuid(self) -> Optional[Any]:
            return get_hash_for_files((__file__,))
    """

    @abstractmethod
    def __call__(self, node: "torch.fx.Node") -> float:
        """
        Implementation of the custom runtime estimator.

        Args:
            node: An fx.Node object whose runtime is to be estimated.

        Returns:
            float: The estimated runtime for the node.
        """

    @abstractmethod
    def uuid(self) -> Any | None:
        """
        Return an ID to uniquely identify your custom runtime estimator implementation.
        Return None to skip AOTAutograd caching entirely.
        """


class CustomKnapsackSolver(ABC):
    """
    Implement this interface for custom knapsack solvers:

    1) The __call__() method contains the implementation of the knapsack solver.

    2) The uuid() method enables AOTAutograd to cache compiled graphs when your custom
    knapsack solver is used. This method can return any identifier as long as it uniquely
    identifies your implementation (and can be pickled). The caching logic includes this
    identifier in its key calculation, i.e., any new value will effectively invalidate
    existing entries. We expect custom knapsack solvers would typically depend purely on the
    textual representation of the implementation. In that case, we recommend using the
    'get_hash_for_files' helper below to compute a unique hash from the contents of a
    static list of source files, i.e., the source(s) containing the custom knapsack solver
    implementation. That approach ensures that any change to the implementation will
    mean a new uuid.

    ** IMPORTANT ** If your custom knapsack solver's behavior depends on some external state,
    then you'll need to implement something more complicated (or disable caching).

    EXAMPLE:

    class MyCustomKnapsackSolver(CustomKnapsackSolver):
        def __call__(
            self,
            memory: list[float],
            joint_graph: fx.Graph,
            max_memory: float,
            node_info: NodeInfo,
            all_recomputable_banned_nodes: list[fx.Node],
        ) -> tuple[list[int], list[int]]:
            # my custom knapsack solver logic
            return saved_node_idx, recomp_node_idx

        def uuid(self) -> Optional[Any]:
            return get_hash_for_files((__file__,))
    """

    @abstractmethod
    def __call__(
        self,
        memory: list[float],
        joint_graph: "torch.fx.Graph",
        max_memory: float,
        node_info: "NodeInfo",
        all_recomputable_banned_nodes: list["torch.fx.Node"],
    ) -> tuple[list[int], list[int]]:
        """
        Implementation of the custom knapsack solver.
        """

    @abstractmethod
    def uuid(self) -> Any | None:
        """
        Return an ID to uniquely identify your custom knapsack solver implementation.
        Return None to skip AOTAutograd caching entirely.
        """
