import copy
import math
import os
import re
import warnings
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Type, TYPE_CHECKING, Union

from typing_extensions import Self

import torch
from torch import nn, optim
from torch.distributed._tools.mod_tracker import ModTracker
from torch.optim.optimizer import (
    register_optimizer_step_post_hook,
    register_optimizer_step_pre_hook,
)
from torch.utils._python_dispatch import (
    is_traceable_wrapper_subclass,
    TorchDispatchMode,
)
from torch.utils._pytree import tree_flatten, tree_map_only

from torch.utils.weak import WeakIdKeyDictionary, weakref

if TYPE_CHECKING:
    from torch.utils.hooks import RemovableHandle


# This value is hard-coded here: https://github.com/pytorch/pytorch/blob/main/c10/cuda/CUDACachingAllocator.cpp#L117
_PYTORCH_MIN_ALLOCATE = (
    2**9 if int(os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING", 0)) == 0 else 1
)
_TOTAL_KEY = "Total"

__all__ = ["MemTracker"]


class _RefType(str, Enum):
    """Base Class for defining memory reference types, categorizing tensors based on their usage within a model."""

    pass


class _State(str, Enum):
    """Base Class for defining module state to capture snapshots ."""

    pass


class _MemRefType(_RefType):
    """
    An enum to define memory reference types, categorizing tensors based on their usage within a model.

        - PARAM: Tensors registered as nn.Parameter within modules.
        - BUFFER: Tensors registered as nn.Buffer within modules.
        - GRAD: Gradients associated with parameters.
        - ACT: Tensors produced during the forward pass and recomputation in activation checkpointing.
        - TMP: Temporary memory used during the backward pass, including gradients of activations.
        - OPT: Tensors holding optimizer states.
        - OTH: Tensors registered via `track_external` that do not fit the above categories.
    """

    PARAM = "Parameter"
    BUFFER = "Buffer"
    GRAD = "Gradient"
    ACT = "Activation"
    TEMP = "Temp"
    OPT = "Optstate"
    OTH = "Other"


class _ModState(_State):
    """
    An enum to define the state of a module.

        - PRE_FW: The module is about to run the forward pass.
        - POST_FW: The module has finished running the forward pass.
        - PEAK_FW: The module has reached the peak memory usage during the forward pass.
        - PRE_BW: The module is about to run the backward pass.
        - PRE_FW_AC: The module is about to run the forward pass with activation checkpointing.
        - POST_FW_AC: The module has finished running the forward pass with activation checkpointing.
        - POST_BW: The module has finished running the backward pass.
        - PEAK_BW: The module has reached the peak memory usage during the backward pass.
    """

    PRE_FW = "Pre-Forward"
    POST_FW = "Post-Forward"
    PEAK_FW = "Peak-Forward"
    PRE_BW = "Pre-Backward"
    PRE_FW_AC = "Pre-Forward-AC"
    POST_FW_AC = "Post-Forward-AC"
    POST_BW = "Post-Backward"
    PEAK_BW = "Peak-Backward"


class _ModMemStats:
    """
    A class to store the memory statistics of a module.

    Args:
        mod_fqn (str): The fully qualified name of the module.
    Attributes:
        mod_fqn (str): The fully qualified name of the module.
        parameter_mem (int): The memory usage of the parameters of the module.
        buffer_mem (int): The memory usage of the buffers of the module.
        input_mem (int): The memory usage of the inputs to the module.
        output_mem (int): The memory usage of the outputs from the module.
        snapshots (Dict[_ModState, Dict[torch.device, Dict[str, int]]]): A dictionary of memory snapshots
        of the module at different states defined by ``_ModState``.
    Note:
        The memory snapshot is stored as a dictionary - Dict[torch.device, Dict[str, int]], where each key is a device,
         and each value is another dictionary with keys as memory reference types defined by `_MemRefType` and
         values as the memory consumed in bytes.
    """

    def __init__(self, mod_fqn: str):
        self.mod_fqn = mod_fqn
        self.parameter_mem: int
        self.buffer_mem: int
        self.input_mem: int
        self.output_mem: int
        self.local_peak: Dict[torch.device, int] = {}
        self.snapshots: Dict[_ModState, List[Dict[torch.device, Dict[str, int]]]] = {}


class _WeakRefInfo:
    """
    Manages memory statistics and device attributes for tensor storages.
    """

    def __init__(
        self, size: int, element_size: int, device: torch.device, reftype: _RefType
    ) -> None:
        """
        Initializes the _WeakRefInfo object with tensor storage properties.

        Args:
        size (int): The number of elements in the tensor storage.
        element_size (int): The size of each element in the tensor storage.
        device (torch.device): The device on which the tensor is allocated.
        reftype (_RefType): The reference type of the tensor.
        """
        self.size = size
        self.element_size = element_size
        self.reftype = reftype
        self.device = device
        self.mem_consumed = self._calculate_mem_consumed()

    def _calculate_mem_consumed(self) -> int:
        """
        Calculates the memory consumed by the tensor storage, considering device-specific allocation rules.

        Returns:
        int: The memory consumed in bytes.
        """
        mem = self.size * self.element_size
        if self.device.type == "cuda":
            return math.ceil((mem) / _PYTORCH_MIN_ALLOCATE) * _PYTORCH_MIN_ALLOCATE
        return mem

    def get_mem_consumed(self, st: torch.UntypedStorage) -> int:
        """
        Updates and returns the memory consumed if the storage size has changed.

        Args:
        st (torch.UntypedStorage): The tensor storage to check for size updates.

        Returns:
        int: The updated memory consumed in bytes.
        """
        if st.size() != self.size:
            self.size = st.size()
            self.mem_consumed = self._calculate_mem_consumed()
        return self.mem_consumed

    @staticmethod
    def get_untyped_storages(t: torch.Tensor) -> Set[torch.UntypedStorage]:
        """
        Recursively extracts untyped storages from a tensor or its subclasses.

        Args:
        t (torch.Tensor): The tensor to extract storages from.

        Returns:
        Set[torch.UntypedStorage]: A set of untyped storages.
        """
        unflattend_tensors = [t]
        flattened_tensor_storages = set()
        while len(unflattend_tensors) > 0:
            obj = unflattend_tensors.pop()
            if is_traceable_wrapper_subclass(obj):
                attrs, _ = obj.__tensor_flatten__()  # type: ignore[attr-defined]
                unflattend_tensors.extend([getattr(obj, attr) for attr in attrs])
            else:
                flattened_tensor_storages.add(obj.untyped_storage())
        return flattened_tensor_storages

    @classmethod
    def create_winfo(
        cls,
        st: torch.UntypedStorage,
        device: torch.device,
        reftype: _RefType,
        WINFO: WeakIdKeyDictionary,
    ) -> Self:
        """
        Creates a new _WeakRefInfo instance and stores it in a dictionary.

        Args:
        st (torch.UntypedStorage): The storage for which to create the info.
        device (torch.device): The device of the tensor.
        reftype (_RefType): The reference type.
        WINFO (WeakIdKeyDictionary): The dictionary to store the new _WeakRefInfo.

        Returns:
        Self: The newly created _WeakRefInfo instance.
        """
        winfo = cls(st.size(), st.element_size(), device, reftype)
        WINFO[st] = winfo
        return winfo

    @classmethod
    def update_and_maybe_create_winfos(
        cls,
        t: torch.Tensor,
        reftype: _RefType,
        WINFO: WeakIdKeyDictionary,
        update_existing: bool = False,
    ) -> Set[Self]:
        """
        Updates or creates _WeakRefInfo instances for the tensor's storages.

        Args:
        t (torch.Tensor): The tensor to process.
        reftype (_RefType): The reference type to apply.
        WINFO (WeakIdKeyDictionary): The dictionary of _WeakRefInfo instances.
        update_existing (bool): If True, raises an error if no existing info is found.

        Returns:
        Set[Self]: A set of updated or newly created _WeakRefInfo instances.
        """
        sts = cls.get_untyped_storages(t)
        winfos = set()
        for st in sts:
            if winfo := WINFO.get(st, None):
                winfo.reftype = reftype
                winfos.add(winfo)
            elif update_existing:
                raise KeyError("No existing winfo found")
            else:
                winfo = cls.create_winfo(st, t.device, reftype, WINFO)
                winfos.add(winfo)
        return winfos


def _get_mem_divisor(units: str) -> int:
    return {"B": 1, "KB": 2**10, "MB": 2**20, "GB": 2**30}.get(units, 1)


def _rounding_fn(value: int, divisor: int, precision: int) -> Union[float, int]:
    return value if divisor == 1 else round(value / divisor, precision)


def _print_snapshot(snapshot: Dict[torch.device, Dict[str, int]], units: str) -> None:
    if len(snapshot) == 0:
        print("No memory tracked.")
        return
    divisor = _get_mem_divisor(units)
    for dev, dev_snap in snapshot.items():
        if _rounding_fn(dev_snap[_TOTAL_KEY], divisor, 2) <= 0:
            continue
        print(
            f"Device: {dev}",
            *(
                f"\t{k}: {_rounding_fn(v, divisor, 2)} {units}"
                for k, v in dev_snap.items()
            ),
            sep="\n",
        )


def _print_snapshot_tabular(
    snapshot: Dict[torch.device, Dict[str, int]], units: str
) -> None:
    if len(snapshot) == 0:
        print("No memory tracked.")
        return
    try:
        from tabulate import tabulate

        divisor = _get_mem_divisor(units)
        table_data = []
        key_list = list(next(iter(snapshot.values())).keys())
        headers = ["Device"] + [f"{key}" for key in key_list]

        for dev, dev_snap in snapshot.items():
            if _rounding_fn(dev_snap[_TOTAL_KEY], divisor, 2) <= 0:
                continue
            row = [str(dev)]
            row.extend(
                f"{_rounding_fn(v, divisor, 2)} {units}" for v in dev_snap.values()
            )
            table_data.append(row)

        print(tabulate(table_data, headers=headers, tablefmt="rst"))
    except ImportError as err:
        raise ImportError(
            "Please install tabulate to use the tabulate option."
        ) from err


def _print_state_snapshots(
    snapshots: Dict[_State, List[Dict[torch.device, Dict[str, int]]]], units: str
) -> None:
    for state, snapshot_list in snapshots.items():
        print(f"{state}")
        for i, snapshot in enumerate(snapshot_list):
            print(f"# {i + 1}:")
            _print_snapshot(snapshot, units)
    print()


def _print_state_snapshots_tabular(
    snapshots: Dict[_State, List[Dict[torch.device, Dict[str, int]]]], units: str
) -> None:
    try:
        from tabulate import tabulate

        table_data = []
        last_state_call = None
        divisor = _get_mem_divisor(units)
        for state, snapshot_list in snapshots.items():
            for i, snapshot in enumerate(snapshot_list):
                state_call = f"{state} # {i + 1}"
                for dev, dev_snap in snapshot.items():
                    if _rounding_fn(dev_snap[_TOTAL_KEY], divisor, 2) <= 0:
                        continue
                    row = {
                        "State & Call": (
                            state_call if state_call != last_state_call else ""
                        ),
                        "Device": str(dev),
                    }
                    last_state_call = state_call
                    for k, v in dev_snap.items():
                        row[f"{k}"] = f"{_rounding_fn(v, divisor, 2)} {units}"
                    table_data.append(row)
        print(tabulate(table_data, headers="keys", tablefmt="rst"))

    except ImportError as err:
        raise ImportError(
            "Please install tabulate to use the tabulate option."
        ) from err


class MemTracker(TorchDispatchMode):
    """
    A TorchDispatchMode to track, categorize and attribute the tensor memory created or accessed within its context.

    It categorizes the tracked tensors as parameters, buffers, activations, gradients, temporary memory and optimizer states
    as defined by ``_MemRefType`` within its context. It captures memory `snapshots` for the modules, called within its context,
    at various states defined by ``_ModState``.

    Attributes:
        memory_tracking: A weakref key dictionary to store the memory statistics of each module. Each key
        is a reference to a module, and each value is a ``_ModMemStats`` object that stores the memory
        statistics of the module.

    Note:
        The MemTracker should be used as a context manager. The modules, optimizers, and any other tensors created within
        the context of MemTracker will be tracked by default. Any tensors or stateful objects such as modules, optimizers etc.
        that need to be tracked but are created outside the MemTracker should be registered using the `track_external` method.
        The `track_external` method should be called before the MemTracker is used. Any tensors created outside the ``MemTracker``
        and not supplied to the `track_external` method will not be tracked by the ``MemTracker``.

    Example usage:

        .. code-block:: python

            module = ...
            optimizer = ...
            inp = ...
            mem_tracker = MemTracker()
            mem_tracker.track_external(module, optimizer, inp)
            with mem_tracker as mt:
                loss = module(inp)
                print("After Forward:")
                mt.display_snapshot("current")
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            mt.display_snapshot("peak")
            mt.display_modulewise_snapshots(depth = 3, units = "MB")
    """

    def __init__(self) -> None:
        self.memory_tracking = WeakIdKeyDictionary()
        self._peak_mem: Dict[torch.device, int] = {}
        self._peak_mem_snap: Dict[torch.device, Dict[str, int]] = {}
        self._param_to_grad_hook_handles = WeakIdKeyDictionary()
        self._optimizer_hook_handles: Optional[
            Tuple[RemovableHandle, RemovableHandle]
        ] = None
        # Dictionary to store the ``_WeakRefInfo`` instances corresponding to each tensor's storage.
        self._WINFO = WeakIdKeyDictionary()
        self._mod_tracker = ModTracker()
        # This is a general memory tracker which can be used with any ``_RefType`` subclass
        # We specify the default reference types for forward, backward and optimizer states
        self._ref_class: Type[_RefType] = _MemRefType
        self._def_fw_ref: _RefType = _MemRefType.ACT
        self._def_bw_ref: _RefType = _MemRefType.TEMP
        self._def_opt_ref: _RefType = _MemRefType.OPT
        # Flags to track if we are in the AC region or optimizer step region
        self._in_opt: bool = False
        self._in_ac: bool = False
        # Weak references to the topmost AC module currently active
        self._ac_mod: Optional[weakref.ref] = None

    def _update_peak_stats(self, peak_state: _State) -> None:
        # We first capture the current memory snapshot of the current tracker state then,
        # We step through each of the modules we have tracked so far in ``memory_tracking``
        #  and check if it is currently active by querying ``_mod_tracker.parents``
        # If it is active, we update the per device peak memory usage for the module
        #  corresponding to the ``_State`` which can be ``PEAK_FW`` or ``PEAK_BW``.
        curr_snap = self.get_tracker_snapshot()

        for mod_stats in self.memory_tracking.values():
            if mod_stats.mod_fqn in self._mod_tracker.parents:
                for dev, dev_snap in curr_snap.items():
                    if mod_stats.local_peak.get(dev, 0) < dev_snap[_TOTAL_KEY]:
                        mod_stats.local_peak[dev] = dev_snap[_TOTAL_KEY]
                        if mod_stats.snapshots.get(peak_state, None) is None:
                            mod_stats.snapshots.setdefault(peak_state, []).append(
                                copy.deepcopy(dev_snap)
                            )
                        else:
                            mod_stats.snapshots[peak_state][-1][dev] = dev_snap

        for dev, dev_snap in curr_snap.items():
            if self._peak_mem.get(dev, 0) < dev_snap[_TOTAL_KEY]:
                self._peak_mem[dev] = dev_snap[_TOTAL_KEY]
                self._peak_mem_snap[dev] = dev_snap

    def _track(self, t: torch.Tensor) -> None:
        # Get the storages of the tensor and check if we have already tracked them.
        # If not, create a new _WeakRefInfo instance and add it to the dictionary.
        # If we are tracking an optimizer state, we use the default optimizer reference type.
        # If we are in backward region and not in AC region, we use the default backward reference type.
        # Else we use the default forward reference type.
        sts = _WeakRefInfo.get_untyped_storages(t)
        for st in sts:
            if self._WINFO.get(st, None):
                return
            elif self._in_opt:
                _WeakRefInfo.create_winfo(st, t.device, self._def_opt_ref, self._WINFO)
            elif self._mod_tracker.is_bw and not self._in_ac:
                _WeakRefInfo.create_winfo(st, t.device, self._def_bw_ref, self._WINFO)
            else:
                _WeakRefInfo.create_winfo(st, t.device, self._def_fw_ref, self._WINFO)

    def get_tracker_snapshot(
        self, type: str = "current"
    ) -> Dict[torch.device, Dict[str, int]]:
        """
        Capture a snapshot of the memory usage breakdown per device, based on the specified type.

        Args:
            type (str): The type of snapshot to capture. Can be "current" for the current memory usage or "peak" for the
                        peak memory usage. Defaults to "current".
        Returns:
            Dict[torch.device, Dict[str, int]]: A dictionary where each key is a torch.device, and each value is another
                                                dictionary. This inner dictionary has keys representing memory reference
                                                types as defined in ``_MemRefType`` and values representing the amount of
                                                memory consumed in bytes.
        Raises:
            ValueError: If an invalid type is specified.
        """
        if type == "current":
            snap: Dict[torch.device, Dict[str, int]] = {}
            for st, winfo in self._WINFO.items():
                dev_snap = snap.setdefault(
                    winfo.device, {reftype: 0 for reftype in self._ref_class}
                )
                dev_snap[winfo.reftype] += winfo.get_mem_consumed(st)

            for dev_snap in snap.values():
                dev_snap[_TOTAL_KEY] = sum(dev_snap.values())

            return snap
        elif type == "peak":
            return self._peak_mem_snap
        else:
            raise ValueError(f"Invalid type {type}")

    def _track_module_params_and_buffers(
        self, module: nn.Module, install_grad_hooks: bool = True
    ) -> Tuple[int, int]:
        # Track the parameters and buffers of the module if not already tracked.
        # If the parameters have gradients, track the gradients as well.
        # If install_grad_hooks is True, install a gradient hook on the parameters
        #  to track the gradients, if it has not already been installed.
        # Return the total memory consumed by the parameters and buffers.
        def _grad_hook(param: nn.Parameter) -> None:
            if param.grad is not None:
                _WeakRefInfo.update_and_maybe_create_winfos(
                    param.grad,
                    _MemRefType.GRAD,
                    self._WINFO,
                )

        param_memory = 0
        for param in module.parameters():
            winfos = _WeakRefInfo.update_and_maybe_create_winfos(
                param, _MemRefType.PARAM, self._WINFO
            )
            param_memory += sum(winfo.mem_consumed for winfo in winfos)
            if param.grad is not None:
                _WeakRefInfo.update_and_maybe_create_winfos(
                    param.grad, _MemRefType.GRAD, self._WINFO
                )
            if (
                self._param_to_grad_hook_handles.get(param, None) is None
                and install_grad_hooks
            ):
                grad_hook_handle = param.register_post_accumulate_grad_hook(_grad_hook)
                self._param_to_grad_hook_handles[param] = grad_hook_handle
        buffer_memory = 0
        for buffer in module.buffers():
            winfos = _WeakRefInfo.update_and_maybe_create_winfos(
                buffer, _MemRefType.BUFFER, self._WINFO
            )
            buffer_memory += sum(winfo.mem_consumed for winfo in winfos)
        return (param_memory, buffer_memory)

    def _track_inputs_or_outputs(self, args: Any) -> int:
        # Calculate the memory consumed by the inputs or outputs of the module.
        input_or_output_memory = 0

        def add_inps_or_outs(t: torch.Tensor) -> None:
            nonlocal input_or_output_memory
            sts = _WeakRefInfo.get_untyped_storages(t)
            for st in sts:
                if winfo := self._WINFO.get(st, None):
                    input_or_output_memory += winfo.mem_consumed

        tree_map_only(torch.Tensor, add_inps_or_outs, args)
        return input_or_output_memory

    def _pre_fw_hook(self, module: nn.Module, inputs: Any) -> None:
        # This is installed as a pre-fwd user hook with ``ModTracker.`` Based on the following cases we
        # set the state and capture the memory snapshot for the module.
        # Case 1: If the module is not in the ``memory_tracking`` dictionary, we track the parameters, buffers,
        #         input and output memory of the module. Create a new ``_ModMemStats`` instance for the module
        #         and add it to the ``memory_tracking`` dictionary.
        # Case 2: If the module is already in the ``memory_tracking`` dictionary and we are in backward, this means
        #         we are in the AC region. We check if this is the top most module in the AC region. If it is,
        #         we store a weak reference and set the flag ``_in_ac`` to True.
        # Case 3: If the module is already in the ``memory_tracking`` dictionary and we are in forward, this means
        #         this module is called for the second time. If it is a root module, that means we are in the next
        #         iteration and we error out. If it is not a root module, that means it's a submodule that is being
        #         used multiple times in the same iteration, which we allow and track.
        # For Case 1 and 3, we also initialiaze the ``local_peak`` and ``PEAK_FW`` snapshot for the module.
        mod_name = self._mod_tracker.get_known_fqn(module)
        assert mod_name is not None
        if module not in self.memory_tracking:
            mod_stats = _ModMemStats(mod_name)
            param_mem, buffer_mem = self._track_module_params_and_buffers(
                module, install_grad_hooks=True
            )
            input_mem = self._track_inputs_or_outputs(inputs)
            mod_stats.parameter_mem = param_mem
            mod_stats.buffer_mem = buffer_mem
            mod_stats.input_mem = input_mem
            self.memory_tracking[module] = mod_stats
            state = _ModState.PRE_FW

        elif self._mod_tracker.is_bw:
            mod_stats = self.memory_tracking[module]
            state = _ModState.PRE_FW_AC
            if self._ac_mod is None:
                self._ac_mod = weakref.ref(module)
                self._in_ac = True
        else:
            parents = set(self._mod_tracker.parents) - {mod_name}
            if len(parents) == 1 and "Global" in parents:
                raise NotImplementedError(
                    "MemTracker does not support memory tracking for multiple iterative calls."
                    " Either use ``reset_mod_stats`` to clear module memory stats for the previous iteration"
                    " or file a github issue if you need this feature."
                )
            mod_stats = self.memory_tracking[module]
            state = _ModState.PRE_FW
            input_mem = self._track_inputs_or_outputs(inputs)
            mod_stats.input_mem = input_mem

        mem_snapshot = self.get_tracker_snapshot()
        if state == _ModState.PRE_FW:
            mod_stats.local_peak = {
                dev: dev_snap[_TOTAL_KEY] for dev, dev_snap in mem_snapshot.items()
            }
            mod_stats.snapshots.setdefault(_ModState.PEAK_FW, []).append(mem_snapshot)
        mod_stats.snapshots.setdefault(state, []).append(copy.deepcopy(mem_snapshot))

    def _post_fw_hook(self, module: nn.Module, inputs: Any, outputs: Any) -> None:
        # This is installed as a post-fwd user hook with ``ModTracker``. Based on the following cases we
        # set the state and capture the memory snapshot for the module.
        # Case 1: This is called in backward, which means we are in the AC region. If this is the top most module
        #         in the AC region, we set the flag ``_in_ac`` to False.
        # Case 2: This is called in forward so we calculate the output memory
        #         of the module and update its mod_stats.
        mod_stats = self.memory_tracking[module]
        if self._mod_tracker.is_bw:
            state = _ModState.POST_FW_AC
            if self._ac_mod is not None and self._ac_mod() is module:
                self._ac_mod = None
                self._in_ac = False
        else:
            state = _ModState.POST_FW
            output_mem = self._track_inputs_or_outputs(outputs)
            mod_stats.output_mem = output_mem
        mod_stats.snapshots.setdefault(state, []).append(self.get_tracker_snapshot())

    def _pre_bw_hook(self, module: nn.Module, args: Any) -> None:
        # This is installed as a pre-bwd user hook with ``ModTracker``. We set the state and capture the
        # snapshot for the module. We also initialize the ``local_peak`` and ``PEAK_BW`` snapshot for it.
        # If the module is None, we skip the hook.
        # This can happen since this installed inside a multi-grad hook on the module's output tensors
        # and the module itself may not be alive during backward.
        if module is None:
            warnings.warn("Module is None. Skipping PRE_BW hook.")
            return
        mod_stats = self.memory_tracking[module]
        mem_snapshot = self.get_tracker_snapshot()
        mod_stats.local_peak = {
            dev: dev_snap[_TOTAL_KEY] for dev, dev_snap in mem_snapshot.items()
        }
        mod_stats.snapshots.setdefault(_ModState.PEAK_BW, []).append(mem_snapshot)
        mod_stats.snapshots.setdefault(_ModState.PRE_BW, []).append(
            copy.deepcopy(mem_snapshot)
        )

    def _post_bw_hook(self, module: nn.Module, args: Any) -> None:
        # This is installed as a post-bwd user hook with ``ModTracker``. We set the state and capture the
        # snapshot for the module if it is not None.
        # This can happen since this installed inside a multi-grad hook on the module's input tensors
        # and the module itself may not be alive during backward.
        if module is None:
            warnings.warn("Module is None. Skipping POST_BW hook.")
            return
        mod_stats = self.memory_tracking[module]
        mod_stats.snapshots.setdefault(_ModState.POST_BW, []).append(
            self.get_tracker_snapshot()
        )

    def _track_optimizer_states(self, optimizer: optim.Optimizer) -> None:
        for states in optimizer.state.values():
            for val in states.values():
                if isinstance(val, torch.Tensor):
                    _WeakRefInfo.update_and_maybe_create_winfos(
                        val, self._def_opt_ref, self._WINFO
                    )

    def _register_global_optimizer_hook(self) -> None:
        # Register a hook on the optimizer step to track the optimizer states.
        # The pre-hook is to set the flag ``_in_opt`` to True. The post-hook unsets the flag,
        # and also tracks any optimizer states that are created during the optimizer step.
        def _opt_step_pre_hook(
            optimizer: optim.Optimizer, args: Any, kwargs: Any
        ) -> None:
            self._in_opt = True

        def _opt_step_post_hook(
            optimizer: optim.Optimizer, args: Any, kwargs: Any
        ) -> None:
            self._track_optimizer_states(optimizer)
            self._in_opt = False

        self._optimizer_hook_handles = (
            register_optimizer_step_pre_hook(_opt_step_pre_hook),
            register_optimizer_step_post_hook(_opt_step_post_hook),
        )

    def _deregister_param_and_optimizer_hooks(self) -> None:
        for grad_hook_handle in self._param_to_grad_hook_handles.values():
            grad_hook_handle.remove()
        self._param_to_grad_hook_handles.clear()

        if self._optimizer_hook_handles is not None:
            for handle in self._optimizer_hook_handles:
                handle.remove()
            self._optimizer_hook_handles = None

    def track_external(
        self, *external: Union[nn.Module, optim.Optimizer, torch.Tensor]
    ) -> None:
        """
        Track tensors and stateful objects like modules, optimizers etc. that are created outside the MemTracker.

        This method should be called before the ``MemTracker`` is used. Any tensors that are not module parameters, buffers,
        gradients activations, or optimizer states will be categorized as ``Other``. If you want them categorized with a
        custom name, please file a GitHub issue. Any tensors created outside the MemTracker and not supplied to this
        method will not be be tracked by ``MemTracker``.

        Args:
            *external (Union[nn.Module, optim.Optimizer, torch.Tensor]): The external modules, optimizers, and
                                                                         tensors to be tracked.
        """
        flat_external, _ = tree_flatten(external)
        for obj in flat_external:
            if isinstance(obj, torch.Tensor):
                _WeakRefInfo.update_and_maybe_create_winfos(
                    obj, _MemRefType.OTH, self._WINFO
                )
            elif isinstance(obj, torch.nn.Module):
                self._track_module_params_and_buffers(obj, install_grad_hooks=False)
            elif isinstance(obj, optim.Optimizer):
                self._track_optimizer_states(obj)
            else:
                raise TypeError(
                    f"Object of type {type(obj)} is not supported for tracking. "
                    f"Only stateful objects like modules, optimizers, and tensors are supported."
                )

    def display_snapshot(
        self, type: str = "current", units: str = "B", tabulate: bool = False
    ) -> None:
        """
        Display the memory usage breakdown snapshot of the tracker based on the specified type and units.

        Keyword args:
            type (str): The type of snapshot to display. Can be "current" for the current memory usage or "peak" for the
                        peak memory usage. Defaults to "current".
            units (str): The units to use for displaying memory usage. Defaults to "B". Supports ["B", "KB", "MB", "GB"].
            tabulate (bool): Whether to display the snapshot in a tabular format. Defaults to False.
        """
        snapshot = self.get_tracker_snapshot(type)
        if tabulate:
            _print_snapshot_tabular(snapshot, units)
        else:
            _print_snapshot(snapshot, units)

    def display_modulewise_snapshots(
        self, depth: int = 2, units: str = "B", tabulate: bool = False
    ) -> None:
        """
        Print per device memory breakdown snapshot for each module called within MemTracker.

        Snapshots are displayed for the states defined by ``_ModState``.
        The module hierarchy is displayed up to the specified depth.

        Keyword Args:
            depth (int, optional): The depth of the module hierarchy to display. Defaults to 2.
            units (str, optional): The units to use for memory tracking. Defaults to "B". Supports ["B", "KB", "MB", "GB"].
            tabulate (bool, optional): Whether to display the snapshot in a tabular format. Defaults to False.
        """

        def natural_sort_key(s: str) -> List[Union[int, str]]:
            return [
                int(text) if text.isdigit() else text.lower()
                for text in re.split("([0-9]+)", s)
            ]

        for mod_stats in sorted(
            self.memory_tracking.values(),
            key=lambda m_stats: natural_sort_key(m_stats.mod_fqn),
        ):
            mod_fqn = mod_stats.mod_fqn
            mod_depth = mod_fqn.count(".") + 1
            if mod_depth > depth:
                continue
            print(f"Module:  {mod_fqn}")
            if tabulate:
                _print_state_snapshots_tabular(mod_stats.snapshots, units)
            else:
                _print_state_snapshots(mod_stats.snapshots, units)

    def reset_mod_stats(self) -> None:
        """
        Reset all the module memory stats. Clears ``memory_tracking`` dictionary.
        """
        self.memory_tracking.clear()

    def __enter__(self) -> "MemTracker":
        self._register_global_optimizer_hook()
        self._mod_tracker.register_user_hooks(
            self._pre_fw_hook,
            self._post_fw_hook,
            self._pre_bw_hook,
            self._post_bw_hook,
        )
        self._peak_mem_snap = self.get_tracker_snapshot()
        self._peak_mem = {
            dev: dev_snap[_TOTAL_KEY] for dev, dev_snap in self._peak_mem_snap.items()
        }
        self._mod_tracker.__enter__()
        super().__enter__()
        return self

    def __exit__(self, *args: Any) -> None:
        self._deregister_param_and_optimizer_hooks()
        self._mod_tracker.clear_user_hooks()
        super().__exit__(*args)
        self._mod_tracker.__exit__(*args)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # type: ignore[no-untyped-def]
        res = func(*args, **kwargs or {})
        tree_map_only(torch.Tensor, self._track, res)
        peak_state = _ModState.PEAK_BW if self._mod_tracker.is_bw else _ModState.PEAK_FW
        self._update_peak_stats(peak_state)
        return res
