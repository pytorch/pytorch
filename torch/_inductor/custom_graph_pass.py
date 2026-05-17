import hashlib
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import lru_cache
from threading import Lock
from typing import Any, TYPE_CHECKING, TypeAlias

import torch.fx.graph
from torch.utils._ordered_set import OrderedSet


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
CustomGraphModulePassType: TypeAlias = (
    CustomGraphModulePass | Callable[[torch.fx.GraphModule], None] | None
)
SchedulerCustomPassType: TypeAlias = (
    CustomGraphPass | Callable[[list[Any]], list[Any]] | None
)


@dataclass(frozen=True)
class ActiveCustomPasses:
    pre_grad_passes: tuple[
        CustomGraphPass | Callable[[torch.fx.graph.Graph], None], ...
    ] = ()
    joint_pre_passes: tuple[
        CustomGraphPass | Callable[[torch.fx.graph.Graph], None], ...
    ] = ()
    joint_post_passes: tuple[
        CustomGraphPass | Callable[[torch.fx.graph.Graph], None], ...
    ] = ()
    post_grad_pre_passes: tuple[
        CustomGraphPass | Callable[[torch.fx.graph.Graph], None], ...
    ] = ()
    post_grad_post_passes: tuple[
        CustomGraphPass | Callable[[torch.fx.graph.Graph], None], ...
    ] = ()
    custom_backend_passes: tuple[
        tuple[str, CustomGraphModulePass | Callable[[torch.fx.GraphModule], None]], ...
    ] = ()
    pre_fusion_custom_passes: tuple[
        CustomGraphPass | Callable[[list[Any]], list[Any]], ...
    ] = ()
    post_fusion_custom_passes: tuple[
        CustomGraphPass | Callable[[list[Any]], list[Any]], ...
    ] = ()


_active_custom_passes_stack: ContextVar[tuple[ActiveCustomPasses, ...]] = ContextVar(
    "_active_custom_passes_stack", default=()
)

_CONFIG_TO_CONTEXT_FIELD: dict[str, str] = {
    "pre_grad_custom_pass": "pre_grad_passes",
    "joint_custom_pre_pass": "joint_pre_passes",
    "joint_custom_post_pass": "joint_post_passes",
    "post_grad_custom_pre_pass": "post_grad_pre_passes",
    "post_grad_custom_post_pass": "post_grad_post_passes",
    "_pre_fusion_custom_pass": "pre_fusion_custom_passes",
    "_post_fusion_custom_pass": "post_fusion_custom_passes",
}


@dataclass
class _BackendContextState:
    scheduling: Any
    python_wrapper: Any
    cpp_wrapper: Any
    fx_wrapper: Any
    custom_pass: CustomGraphModulePass | None
    custom_config: Any
    refcount: int = 0


_backend_context_lock = Lock()
_backend_context_states: dict[str, _BackendContextState] = {}


def _return_none() -> None:
    return None


def _normalize_graph_passes(
    passes: Sequence[CustomGraphPassType] | None,
) -> tuple[CustomGraphPass | Callable[[torch.fx.graph.Graph], None], ...]:
    if passes is None:
        return ()
    return tuple(p for p in passes if p is not None)


def _normalize_scheduler_passes(
    passes: Sequence[SchedulerCustomPassType] | None,
) -> tuple[CustomGraphPass | Callable[[list[Any]], list[Any]], ...]:
    if passes is None:
        return ()
    return tuple(p for p in passes if p is not None)


def _normalize_backend_passes(
    custom_backend_passes: (
        Mapping[str, CustomGraphModulePassType | Sequence[CustomGraphModulePassType]]
        | Sequence[tuple[str, CustomGraphModulePassType]]
        | None
    ),
) -> tuple[
    tuple[str, CustomGraphModulePass | Callable[[torch.fx.GraphModule], None]], ...
]:
    if custom_backend_passes is None:
        return ()

    normalized: list[
        tuple[str, CustomGraphModulePass | Callable[[torch.fx.GraphModule], None]]
    ] = []
    items = (
        custom_backend_passes.items()
        if isinstance(custom_backend_passes, Mapping)
        else custom_backend_passes
    )
    for device, custom_pass in items:
        if custom_pass is None:
            continue
        if isinstance(custom_pass, Sequence) and not isinstance(
            custom_pass, (str, bytes)
        ):
            for p in custom_pass:
                if p is not None:
                    normalized.append((device, p))
        else:
            normalized.append((device, custom_pass))
    return tuple(normalized)


def _iter_active_passes(field: str):
    for active_passes in reversed(_active_custom_passes_stack.get()):
        yield from getattr(active_passes, field)


def get_active_passes() -> ActiveCustomPasses:
    pre_grad_passes: list[CustomGraphPass | Callable[[torch.fx.graph.Graph], None]] = []
    joint_pre_passes: list[
        CustomGraphPass | Callable[[torch.fx.graph.Graph], None]
    ] = []
    joint_post_passes: list[
        CustomGraphPass | Callable[[torch.fx.graph.Graph], None]
    ] = []
    post_grad_pre_passes: list[
        CustomGraphPass | Callable[[torch.fx.graph.Graph], None]
    ] = []
    post_grad_post_passes: list[
        CustomGraphPass | Callable[[torch.fx.graph.Graph], None]
    ] = []
    custom_backend_passes: list[
        tuple[str, CustomGraphModulePass | Callable[[torch.fx.GraphModule], None]]
    ] = []
    pre_fusion_custom_passes: list[
        CustomGraphPass | Callable[[list[Any]], list[Any]]
    ] = []
    post_fusion_custom_passes: list[
        CustomGraphPass | Callable[[list[Any]], list[Any]]
    ] = []

    for active_passes in reversed(_active_custom_passes_stack.get()):
        pre_grad_passes.extend(active_passes.pre_grad_passes)
        joint_pre_passes.extend(active_passes.joint_pre_passes)
        joint_post_passes.extend(active_passes.joint_post_passes)
        post_grad_pre_passes.extend(active_passes.post_grad_pre_passes)
        post_grad_post_passes.extend(active_passes.post_grad_post_passes)
        custom_backend_passes.extend(active_passes.custom_backend_passes)
        pre_fusion_custom_passes.extend(active_passes.pre_fusion_custom_passes)
        post_fusion_custom_passes.extend(active_passes.post_fusion_custom_passes)

    return ActiveCustomPasses(
        pre_grad_passes=tuple(pre_grad_passes),
        joint_pre_passes=tuple(joint_pre_passes),
        joint_post_passes=tuple(joint_post_passes),
        post_grad_pre_passes=tuple(post_grad_pre_passes),
        post_grad_post_passes=tuple(post_grad_post_passes),
        custom_backend_passes=tuple(custom_backend_passes),
        pre_fusion_custom_passes=tuple(pre_fusion_custom_passes),
        post_fusion_custom_passes=tuple(post_fusion_custom_passes),
    )


def has_active_custom_passes() -> bool:
    return any(
        bool(field_value)
        for active_passes in _active_custom_passes_stack.get()
        for field_value in active_passes.__dict__.values()
    )


def _get_pass_uuid(custom_pass: Any) -> Any | None:
    if isinstance(custom_pass, (CustomGraphPass, CustomGraphModulePass)):
        return custom_pass.uuid()
    return None


def _get_active_pass_uuids(field: str) -> tuple[Any, ...] | None:
    uuids = []
    for custom_pass in _iter_active_passes(field):
        uuid = _get_pass_uuid(custom_pass)
        if uuid is None:
            return None
        uuids.append(uuid)
    return tuple(uuids)


class _CustomPassContextWrapper:
    def __init__(self, field: str) -> None:
        self.field = field

    def __bool__(self) -> bool:
        return any(True for _ in _iter_active_passes(self.field))

    def __reduce__(self) -> tuple[Callable[[], None], tuple[Any, ...]]:
        return _return_none, ()


class _CustomPassContextGraphPass(_CustomPassContextWrapper, CustomGraphPass):
    def __call__(self, graph: torch.fx.graph.Graph) -> None:
        for custom_pass in _iter_active_passes(self.field):
            custom_pass(graph)

    def uuid(self) -> tuple[Any, ...] | None:
        return _get_active_pass_uuids(self.field)


class _CustomPassContextPostGradPass(
    _CustomPassContextWrapper, CustomInferenceAwareGraphPass
):
    def __call__(self, graph: torch.fx.graph.Graph, is_inference: bool = False) -> None:
        for custom_pass in _iter_active_passes(self.field):
            if isinstance(custom_pass, CustomInferenceAwareGraphPass):
                custom_pass(graph, is_inference=is_inference)
            else:
                custom_pass(graph)

    def uuid(self) -> tuple[Any, ...] | None:
        return _get_active_pass_uuids(self.field)


class _CustomPassContextSchedulerPass(_CustomPassContextWrapper, CustomGraphPass):
    def __call__(self, nodes: list[Any]) -> list[Any]:
        for custom_pass in _iter_active_passes(self.field):
            nodes = custom_pass(nodes)
        return nodes

    def uuid(self) -> tuple[Any, ...] | None:
        return _get_active_pass_uuids(self.field)


class _CustomPassContextGraphModulePass(CustomGraphModulePass):
    def __init__(self, device: str) -> None:
        self.device = device

    def __reduce__(self) -> tuple[Callable[[], None], tuple[Any, ...]]:
        return _return_none, ()

    def __bool__(self) -> bool:
        return any(
            device == self.device
            for device, _ in _iter_active_passes("custom_backend_passes")
        )

    def __call__(self, gm: torch.fx.GraphModule) -> None:
        for device, custom_pass in _iter_active_passes("custom_backend_passes"):
            if device == self.device:
                custom_pass(gm)

    def uuid(self) -> tuple[Any, ...] | None:
        uuids = []
        for device, custom_pass in _iter_active_passes("custom_backend_passes"):
            if device != self.device:
                continue
            uuid = _get_pass_uuid(custom_pass)
            if uuid is None:
                return None
            uuids.append(uuid)
        return tuple(uuids) if uuids else None


_CONTEXT_CONFIG_WRAPPERS: dict[str, CustomGraphPass] = {
    "pre_grad_custom_pass": _CustomPassContextGraphPass("pre_grad_passes"),
    "joint_custom_pre_pass": _CustomPassContextGraphPass("joint_pre_passes"),
    "joint_custom_post_pass": _CustomPassContextGraphPass("joint_post_passes"),
    "post_grad_custom_pre_pass": _CustomPassContextPostGradPass("post_grad_pre_passes"),
    "post_grad_custom_post_pass": _CustomPassContextPostGradPass(
        "post_grad_post_passes"
    ),
    "_pre_fusion_custom_pass": _CustomPassContextSchedulerPass(
        "pre_fusion_custom_passes"
    ),
    "_post_fusion_custom_pass": _CustomPassContextSchedulerPass(
        "post_fusion_custom_passes"
    ),
}


def _is_custom_pass_context_wrapper(value: object) -> bool:
    return isinstance(
        value, (_CustomPassContextWrapper, _CustomPassContextGraphModulePass)
    )


def _check_config_hook_conflicts() -> None:
    from . import config as inductor_config

    for config_key in _CONFIG_TO_CONTEXT_FIELD:
        value = getattr(inductor_config, config_key)
        if value is not None and not _is_custom_pass_context_wrapper(value):
            raise RuntimeError(
                "custom_pass_context cannot be used while "
                f"torch._inductor.config.{config_key} is set. "
                "Pass custom passes to custom_pass_context instead."
            )


@contextmanager
def _patch_config_hooks(active_passes: ActiveCustomPasses):
    from . import config as inductor_config

    patches = {}
    for config_key, field in _CONFIG_TO_CONTEXT_FIELD.items():
        if not getattr(active_passes, field):
            continue
        value = getattr(inductor_config, config_key)
        if value is None:
            patches[config_key] = _CONTEXT_CONFIG_WRAPPERS[config_key]

    with inductor_config.patch(patches):
        yield


@contextmanager
def _patch_custom_backend_passes(active_passes: ActiveCustomPasses):
    devices = sorted(
        OrderedSet([device for device, _ in active_passes.custom_backend_passes])
    )
    if not devices:
        yield
        return

    from .codegen.common import (
        get_custom_backend_config_for_device,
        get_custom_backend_pass_for_device,
        get_scheduling_for_device,
        get_wrapper_codegen_for_device,
        init_backend_registration,
        register_backend_for_device,
    )

    def release_device(device: str) -> None:
        state = _backend_context_states[device]
        state.refcount -= 1
        if state.refcount > 0:
            return
        register_backend_for_device(
            device,
            state.scheduling,
            state.python_wrapper,
            state.cpp_wrapper,
            state.fx_wrapper,
            state.custom_pass,
            state.custom_config,
        )
        del _backend_context_states[device]

    init_backend_registration()
    acquired_devices: list[str] = []
    with _backend_context_lock:
        try:
            for device in devices:
                state = _backend_context_states.get(device)
                if state is not None:
                    current_custom_pass = get_custom_backend_pass_for_device(device)
                    if not _is_custom_pass_context_wrapper(current_custom_pass):
                        raise RuntimeError(
                            "custom_pass_context backend state was changed while "
                            f"device {device!r} already had an active context"
                        )
                    state.refcount += 1
                    acquired_devices.append(device)
                    continue

                original_custom_pass = get_custom_backend_pass_for_device(device)
                if original_custom_pass is not None:
                    raise RuntimeError(
                        "custom_pass_context cannot be used while a custom backend "
                        f"pass is registered for device {device!r}. Pass custom "
                        "backend passes to custom_pass_context instead."
                    )

                original_scheduling = get_scheduling_for_device(device)
                if original_scheduling is None:
                    raise RuntimeError(
                        f"custom_pass_context received custom_backend_passes for "
                        f"unregistered device {device!r}"
                    )
                original_python_wrapper = get_wrapper_codegen_for_device(device, False)
                assert original_python_wrapper is not None
                original_cpp_wrapper = get_wrapper_codegen_for_device(device, True)
                original_fx_wrapper = get_wrapper_codegen_for_device(
                    device, fx_wrapper=True
                )
                original_custom_backend_config = get_custom_backend_config_for_device(
                    device
                )
                _backend_context_states[device] = _BackendContextState(
                    original_scheduling,
                    original_python_wrapper,
                    original_cpp_wrapper,
                    original_fx_wrapper,
                    original_custom_pass,
                    original_custom_backend_config,
                    refcount=1,
                )
                register_backend_for_device(
                    device,
                    original_scheduling,
                    original_python_wrapper,
                    original_cpp_wrapper,
                    original_fx_wrapper,
                    _CustomPassContextGraphModulePass(device),
                    original_custom_backend_config,
                )
                acquired_devices.append(device)
        except Exception:
            for device in reversed(acquired_devices):
                release_device(device)
            raise

    try:
        yield
    finally:
        with _backend_context_lock:
            for device in reversed(acquired_devices):
                release_device(device)


def split_custom_passes_from_config_patch(
    config_patch: Mapping[str, Any],
) -> tuple[dict[str, list[Any]], dict[str, Any]]:
    custom_pass_context_kwargs: dict[str, list[Any]] = {}
    remaining_config_patch: dict[str, Any] = {}

    for field_name, value in config_patch.items():
        context_field = _CONFIG_TO_CONTEXT_FIELD.get(field_name)
        if context_field is None:
            remaining_config_patch[field_name] = value
            continue
        if value is None:
            remaining_config_patch[field_name] = value
            continue
        custom_pass_context_kwargs.setdefault(context_field, []).append(value)

    return custom_pass_context_kwargs, remaining_config_patch


@contextmanager
def patched_inductor_config_with_custom_pass_context(
    config_patch: Mapping[str, Any],
):
    custom_pass_context_kwargs, remaining_config_patch = (
        split_custom_passes_from_config_patch(config_patch)
    )

    from . import config as inductor_config

    with (
        inductor_config.patch(remaining_config_patch),
        custom_pass_context(**custom_pass_context_kwargs),
    ):
        yield


@contextmanager
def custom_pass_context(
    *,
    pre_grad_passes: Sequence[CustomGraphPassType] | None = None,
    joint_pre_passes: Sequence[CustomGraphPassType] | None = None,
    joint_post_passes: Sequence[CustomGraphPassType] | None = None,
    post_grad_pre_passes: Sequence[CustomGraphPassType] | None = None,
    post_grad_post_passes: Sequence[CustomGraphPassType] | None = None,
    custom_backend_passes: (
        Mapping[str, CustomGraphModulePassType | Sequence[CustomGraphModulePassType]]
        | Sequence[tuple[str, CustomGraphModulePassType]]
        | None
    ) = None,
    pre_fusion_custom_passes: Sequence[SchedulerCustomPassType] | None = None,
    post_fusion_custom_passes: Sequence[SchedulerCustomPassType] | None = None,
):
    active_passes = ActiveCustomPasses(
        pre_grad_passes=_normalize_graph_passes(pre_grad_passes),
        joint_pre_passes=_normalize_graph_passes(joint_pre_passes),
        joint_post_passes=_normalize_graph_passes(joint_post_passes),
        post_grad_pre_passes=_normalize_graph_passes(post_grad_pre_passes),
        post_grad_post_passes=_normalize_graph_passes(post_grad_post_passes),
        custom_backend_passes=_normalize_backend_passes(custom_backend_passes),
        pre_fusion_custom_passes=_normalize_scheduler_passes(pre_fusion_custom_passes),
        post_fusion_custom_passes=_normalize_scheduler_passes(
            post_fusion_custom_passes
        ),
    )
    _check_config_hook_conflicts()
    token = _active_custom_passes_stack.set(
        _active_custom_passes_stack.get() + (active_passes,)
    )
    try:
        with (
            _patch_config_hooks(active_passes),
            _patch_custom_backend_passes(active_passes),
        ):
            yield active_passes
    finally:
        _active_custom_passes_stack.reset(token)


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
