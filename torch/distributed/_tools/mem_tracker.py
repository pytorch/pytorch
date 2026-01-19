import math
import os
import re
import warnings
from collections.abc import Callable
from copy import deepcopy
from enum import auto, Enum
from functools import partial, wraps
from typing import Any, TYPE_CHECKING
from typing_extensions import Self

import torch
import torch.distributed._tools.fake_collectives
from torch import nn, optim
from torch._guards import active_fake_mode
from torch.distributed._tools.common_utils import get_untyped_storages
from torch.distributed._tools.mod_tracker import ModTracker
from torch.distributed.tensor import DTensor
from torch.optim.optimizer import (
    register_optimizer_step_post_hook,
    register_optimizer_step_pre_hook,
)
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten, tree_map_only
from torch.utils.weak import WeakIdKeyDictionary, weakref


if TYPE_CHECKING:
    from torch.utils.hooks import RemovableHandle

# This value is hard-coded here:
# https://github.com/pytorch/pytorch/blob/5fba5d83f0703ff8077ab65448a998e9ad6598fd/c10/cuda/CUDACachingAllocator.cpp#L117
_PYTORCH_MIN_ALLOCATE = (
    2**9 if int(os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING", 0)) == 0 else 1
)
_TOTAL_KEY = "Total"

__all__ = ["MemTracker"]


class _RefType(str, Enum):
    """Base Class for defining memory reference types, categorizing tensors based on their usage within a model."""


class _State(str, Enum):
    """Base Class for defining module state to capture snapshots ."""


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
        self.local_peak: dict[torch.device, int] = {}
        self.snapshots: dict[_ModState, list[dict[torch.device, dict[str, int]]]] = {}


class _WeakRefInfo:
    """
    Manages memory statistics and device attributes for tensor storages.
    """

    def __init__(
        self, size: int, element_size: int, device: torch.device, reftype: _RefType
    ) -> None:
        """
        Initializes the ``_WeakRefInfo`` object with tensor storage properties.

        Args:
            size (int): The number of elements in the tensor storage.
            element_size (int): The size of each element in the tensor storage.
            device (torch.device): The device on which the tensor is allocated.
            reftype (_RefType): The reference type of the tensor.
        """
        self.size = size
        self.element_size = element_size
        self.reftype = reftype
        # pyrefly: ignore [read-only]
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

    def update_mem_consumed(self, st: torch.UntypedStorage) -> int:
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

    @classmethod
    def create_winfo(
        cls,
        st: torch.UntypedStorage,
        device: torch.device,
        reftype: _RefType,
        callback: Callable[[Self, weakref.ref], Any] | None = None,
    ) -> tuple[Self, weakref.ref]:
        """
        Creates a new ``_WeakRefInfo`` instance and a weak reference to a ``torch.UntypedStorage`` object,
        optionally attaching a callback to the weak reference.

        Args:
            st (torch.UntypedStorage): The storage object for which to create the weak reference info.
            device (torch.device): The device associated with the storage object.
            reftype (_RefType): The type of reference, used to categorize the storage.
            callback (Optional[Callable[[Self, weakref.ref]]]): A callback function that is called when
                the storage object is about to be finalized (garbage collected). The callback function
                should accept two arguments: the ``_WeakRefInfo`` instance and the weak reference to the storage.
        Returns:
            Tuple[Self, weakref.ref]: A tuple containing the newly created ``_WeakRefInfo`` instance and the
            weak reference to the storage object. The weak reference may have an attached callback if provided.
        """

        winfo = cls(st.size(), st.element_size(), device, reftype)
        w_st = weakref.ref(st, partial(callback, winfo) if callback else None)
        return winfo, w_st


def _get_mem_divisor(units: str) -> int:
    unit_dict = {"B": 1, "KiB": 2**10, "MiB": 2**20, "GiB": 2**30}
    if units in unit_dict:
        return unit_dict[units]
    else:
        raise ValueError(
            f"Unsupported unit: {units}. Supported units are: {', '.join(unit_dict.keys())}"
        )


def _rounding_fn(value: int, divisor: int, precision: int) -> float | int:
    return value if divisor == 1 else round(value / divisor, precision)


def _print_snapshot(snapshot: dict[torch.device, dict[str, int]], units: str) -> None:
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
                f"\t{k.value}: {_rounding_fn(v, divisor, 2)} {units}"
                if isinstance(k, _RefType)
                else f"\t{k}: {_rounding_fn(v, divisor, 2)} {units}"
                for k, v in dev_snap.items()
            ),
            sep="\n",
        )


def _print_snapshot_tabular(
    snapshot: dict[torch.device, dict[str, int]], units: str
) -> None:
    if len(snapshot) == 0:
        print("No memory tracked.")
        return
    try:
        from tabulate import tabulate
    except ImportError as err:
        raise ImportError(
            "Please install tabulate to use the tabulate option."
        ) from err
    divisor = _get_mem_divisor(units)
    table_data = []
    key_list = list(next(iter(snapshot.values())).keys())
    headers = ["Device"] + [
        f"{key.value}" if isinstance(key, _RefType) else f"{key}" for key in key_list
    ]

    for dev, dev_snap in snapshot.items():
        if _rounding_fn(dev_snap[_TOTAL_KEY], divisor, 2) <= 0:
            continue
        row = [str(dev)]
        row.extend(f"{_rounding_fn(v, divisor, 2)} {units}" for v in dev_snap.values())
        table_data.append(row)
    print(tabulate(table_data, headers=headers, tablefmt="rst"))


def _print_state_snapshots(
    snapshots: dict[_State, list[dict[torch.device, dict[str, int]]]], units: str
) -> None:
    for state, snapshot_list in snapshots.items():
        print(f"{state.value}")
        for i, snapshot in enumerate(snapshot_list):
            print(f"# {i + 1}:")
            _print_snapshot(snapshot, units)
    print()


def _print_state_snapshots_tabular(
    snapshots: dict[_State, list[dict[torch.device, dict[str, int]]]], units: str
) -> None:
    try:
        from tabulate import tabulate
    except ImportError as err:
        raise ImportError(
            "Please install tabulate to use the tabulate option."
        ) from err

    table_data = []
    last_state_call = None
    divisor = _get_mem_divisor(units)
    for state, snapshot_list in snapshots.items():
        for i, snapshot in enumerate(snapshot_list):
            state_call = f"{state.value} # {i + 1}"
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
                    row[f"{k.value}" if isinstance(k, _RefType) else f"{k}"] = (
                        f"{_rounding_fn(v, divisor, 2)} {units}"
                    )
                table_data.append(row)
    print(tabulate(table_data, headers="keys", tablefmt="rst"))


class _UpdateType(Enum):
    # These are used for tracking updates to the continuouly maintained memory snapshot.
    # ADD - When a new tensor storage is tracked
    # DEL - When a tensor storage is about to be finalized (garbage collected).
    # REF - When a tensor reference is updated, for instance, the gradients are marked as
    #       generic backward reference types until the grad_hook categorizes them as gradients.
    # SIZE - When a tensor's storage is resized.
    ADD = auto()
    DEL = auto()
    REF = auto()
    SIZE = auto()


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
            mt.display_modulewise_snapshots(depth=3, units="MiB")

    Known Limitations:
        - The ``MemTracker`` does not track memory for tensors that bypass the ``TorchDispatchMode`` ex. under ``no_dispatch``.
        - Resizing tensor storages directly by using non-Tensor methods other than using ``torch.Untyped_Storage.resize_``
          is not tracked. File a Github issue if you have use-cases for this.
        - If the tensors are not traceable or wrappable subclasses of ``torch.Tensor``, then the tracker does not know how to
            track their storages. File a Github issue if you have use-cases for this.
        - During AC in the backward pass there might be misattribution between activation and temp memory, but the peak memory
          will be tracked accurately. This will be fixed in the next update by hooking intricately with ``torch.uitls.checkpoint``.
    """

    def __init__(self) -> None:
        self.memory_tracking = WeakIdKeyDictionary()
        self._curr_mem_snap: dict[torch.device, dict[str, int]] = {}
        self._peak_mem: dict[torch.device, int] = {}
        self._peak_mem_snap: dict[torch.device, dict[str, int]] = {}
        self._param_to_grad_hook_handles = WeakIdKeyDictionary()
        self._optimizer_hook_handles: tuple[RemovableHandle, RemovableHandle] | None = (
            None
        )
        # Dictionary to store the ``_WeakRefInfo`` instances corresponding to each tensor's storage.
        self._WINFO = WeakIdKeyDictionary()
        self._mod_tracker = ModTracker()
        # This is a general memory tracker which can be used with any ``_RefType`` subclass
        self._ref_class: type[_RefType] = _MemRefType
        # Flags to track if we are in the AC region or optimizer step region
        self._in_opt: bool = False
        self._in_ac: bool = False
        # Weak references to the topmost AC module currently active
        self._ac_mod: weakref.ref | None = None
        self._orig_resize = torch.UntypedStorage.resize_
        self._depth = 0

    def _update_snap(
        self,
        u_type: _UpdateType,
        winfo: _WeakRefInfo,
        old_mem_consumed: int | None = None,
        old_reftype: _RefType | None = None,
    ) -> None:
        # Initialize a flag to track if the total memory might drop to zero after updates.
        maybe_zero = False
        # Ensure the device entry exists in the current memory snapshot, initializing if necessary.
        # pyrefly: ignore [no-matching-overload]
        dev_snap = self._curr_mem_snap.setdefault(
            winfo.device, dict.fromkeys(self._ref_class, 0)
        )
        dev_snap.setdefault(_TOTAL_KEY, 0)
        # Handle different types of updates based on the update type (`u_type`).
        if u_type == _UpdateType.ADD:
            # Increase the memory consumed for the specific reference type and update the total.
            dev_snap[winfo.reftype] += winfo.mem_consumed
            dev_snap[_TOTAL_KEY] += winfo.mem_consumed
        elif u_type == _UpdateType.DEL:
            # Decrease the memory consumed for the specific reference type and reduce the total.
            dev_snap[winfo.reftype] -= winfo.mem_consumed
            dev_snap[_TOTAL_KEY] -= winfo.mem_consumed
            maybe_zero = True
        elif u_type == _UpdateType.REF:
            assert old_reftype is not None
            # Adjust memory consumption between two reference types within the same device.
            dev_snap[old_reftype] -= winfo.mem_consumed
            dev_snap[winfo.reftype] += winfo.mem_consumed
        elif u_type == _UpdateType.SIZE:
            assert old_mem_consumed is not None
            # Adjust the memory consumed for a reference type due to a change in size.
            change = winfo.mem_consumed - old_mem_consumed
            dev_snap[winfo.reftype] += change
            dev_snap[_TOTAL_KEY] += change
            maybe_zero = True
        else:
            raise ValueError(f"Invalid update type: {u_type}")
        # Check if the total memory for the device has dropped to zero.
        if maybe_zero:
            if self._curr_mem_snap[winfo.device][_TOTAL_KEY] == 0:
                # Remove the device entry from the memory snapshot if the total memory is zero.
                del self._curr_mem_snap[winfo.device]

    def _update_and_maybe_create_winfos(
        self,
        t: torch.Tensor,
        reftype: _RefType,
        update_existing: bool = False,
    ) -> set[_WeakRefInfo]:
        sts = get_untyped_storages(t)
        winfos = set()
        for st in sts:
            # Attempt to retrieve existing ``_WeakRefInfo`` and its weak reference from the tracking dictionary.
            winfo, _ = self._WINFO.get(st, (None, None))
            if winfo is not None:
                # If ``_WeakRefInfo`` exists, check if the reference type needs to be updated.
                old_reftype = winfo.reftype
                if old_reftype != reftype:
                    # Update the reference type and apply changes via ``_update_snap``.
                    winfo.reftype = reftype
                    self._update_snap(_UpdateType.REF, winfo, old_reftype=old_reftype)
                winfos.add(winfo)
            elif update_existing:
                # If no existing ``_WeakRefInfo`` is found and update_existing is True, raise an error.
                raise KeyError("No existing winfo found")
            else:
                # If no existing _WeakRefInfo is found and update_existing is False, create a new ``_WeakRefInfo``.
                winfo, w_st = _WeakRefInfo.create_winfo(
                    st, t.device, reftype, self._delete_callback
                )
                # Store the new ``_WeakRefInfo`` and its weak reference in the tracking dictionary.
                self._WINFO[st] = (winfo, w_st)
                # Update the snapshot for the newly added ``_WeakRefInfo``.
                if winfo.mem_consumed > 0:
                    self._update_snap(_UpdateType.ADD, winfo)
                winfos.add(winfo)
        return winfos

    def _delete_callback(self, winfo: _WeakRefInfo, w_st: weakref.ref) -> None:
        # Callback to be called when the storage object corresponding to the  ``_WeakRefInfo``
        # instance is about to be finalized.
        if winfo.mem_consumed > 0:
            self._update_snap(_UpdateType.DEL, winfo)

    def _track_resize(self) -> None:
        # Need to monkey-patch this because ``torch.UntypedStorage.resize_`` is not captured
        # by ``TorchDispatchMode``.
        @wraps(self._orig_resize)
        def resize_(st: torch.UntypedStorage, size: int) -> None:
            self._orig_resize(st, size)
            winfo, _ = self._WINFO.get(st, (None, None))
            if winfo is not None and winfo.size != st.size():
                old_mem_consumed = winfo.mem_consumed
                winfo.update_mem_consumed(st)
                self._update_snap(
                    _UpdateType.SIZE, winfo, old_mem_consumed=old_mem_consumed
                )

        torch.UntypedStorage.resize_ = resize_  # type: ignore[method-assign, assignment]

    def _restore_resize(self) -> None:
        torch.UntypedStorage.resize_ = self._orig_resize  # type: ignore[method-assign]

    def _update_peak_stats(self, peak_state: _State) -> None:
        # We first capture the current memory snapshot of the current tracker state then,
        # We step through each of the modules we have tracked so far in ``memory_tracking``
        #  and check if it is currently active by querying ``_mod_tracker.parents``
        # If it is active, we update the per device peak memory usage for the module
        #  corresponding to the ``_State`` which can be ``PEAK_FW`` or ``PEAK_BW``.
        curr_snap = self._curr_mem_snap

        for mod_stats in self.memory_tracking.values():
            if mod_stats.mod_fqn in self._mod_tracker.parents:
                if peak_state in mod_stats.snapshots:
                    for dev, dev_snap in curr_snap.items():
                        if mod_stats.local_peak.get(dev, 0) < dev_snap[_TOTAL_KEY]:
                            mod_stats.local_peak[dev] = dev_snap[_TOTAL_KEY]
                            mod_stats.snapshots[peak_state][-1][dev] = deepcopy(
                                dev_snap
                            )

        for dev, dev_snap in curr_snap.items():
            if self._peak_mem.get(dev, 0) < dev_snap[_TOTAL_KEY]:
                self._peak_mem[dev] = dev_snap[_TOTAL_KEY]
                self._peak_mem_snap[dev] = deepcopy(dev_snap)

    def _track(self, reftype: _RefType, t: torch.Tensor) -> None:
        # Get the storages of the tensor and check if we have already tracked them.
        # If yes, then check if the storage size has changed and update the current snapshot.
        # Else create a new ``_WeakRefInfo`` instance and add it to the dictionary.
        sts = get_untyped_storages(t)
        for st in sts:
            winfo, _ = self._WINFO.get(st, (None, None))
            if winfo is not None:
                if winfo.size != st.size():
                    old_mem_consumed = winfo.mem_consumed
                    winfo.update_mem_consumed(st)
                    self._update_snap(
                        _UpdateType.SIZE, winfo, old_mem_consumed=old_mem_consumed
                    )
                return
            else:
                winfo, w_st = _WeakRefInfo.create_winfo(
                    st, t.device, reftype, self._delete_callback
                )
                self._WINFO[st] = (winfo, w_st)
                # Update the current snapshot for the newly added ``_WeakRefInfo``.
                if winfo.mem_consumed > 0:
                    self._update_snap(_UpdateType.ADD, winfo)

    def get_tracker_snapshot(
        self, type: str = "current"
    ) -> dict[torch.device, dict[str, int]]:
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
            return deepcopy(self._curr_mem_snap)
        elif type == "peak":
            return deepcopy(self._peak_mem_snap)
        else:
            raise ValueError(f"Invalid type {type}")

    def _track_module_params_and_buffers(
        self, module: nn.Module, install_grad_hooks: bool = True
    ) -> tuple[int, int]:
        # Track the parameters and buffers of the module if not already tracked.
        # If the parameters have gradients, track the gradients as well.
        # If install_grad_hooks is True, install a gradient hook on the parameters
        #  to track the gradients, if it has not already been installed.
        # Return the total memory consumed by the parameters and buffers.
        def _grad_hook(grad: torch.Tensor) -> None:
            self._update_and_maybe_create_winfos(
                grad,
                _MemRefType.GRAD,
            )

        param_memory = 0
        for param in module.parameters():
            winfos = self._update_and_maybe_create_winfos(
                param,
                _MemRefType.PARAM,
            )
            param_memory += sum(winfo.mem_consumed for winfo in winfos)
            if param.grad is not None:
                self._update_and_maybe_create_winfos(
                    param.grad,
                    _MemRefType.GRAD,
                )
            if (
                self._param_to_grad_hook_handles.get(param, None) is None
                and install_grad_hooks
            ):
                grad_hook_handle = param.register_hook(_grad_hook)
                post_acc_grad_hook_handle = param.register_post_accumulate_grad_hook(
                    lambda p: (_grad_hook(p.grad))
                )
                self._param_to_grad_hook_handles[param] = (
                    grad_hook_handle,
                    post_acc_grad_hook_handle,
                )
        buffer_memory = 0
        for buffer in module.buffers():
            winfos = self._update_and_maybe_create_winfos(
                buffer,
                _MemRefType.BUFFER,
            )
            buffer_memory += sum(winfo.mem_consumed for winfo in winfos)
        return (param_memory, buffer_memory)

    def _track_inputs_or_outputs(self, args: Any) -> int:
        # Calculate the memory consumed by the inputs or outputs of the module.
        input_or_output_memory = 0

        def add_inps_or_outs(t: torch.Tensor) -> None:
            nonlocal input_or_output_memory
            sts = get_untyped_storages(t)
            for st in sts:
                winfo, _ = self._WINFO.get(st, (None, None))
                if winfo is not None:
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
        # For Case 1 and 3, we also initialize the ``local_peak`` and ``PEAK_FW`` snapshot for the module.
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
            mod_stats.mod_fqn = mod_name
            mod_stats.input_mem = input_mem

        mem_snapshot = self.get_tracker_snapshot()
        if state == _ModState.PRE_FW:
            mod_stats.local_peak = {
                dev: dev_snap[_TOTAL_KEY] for dev, dev_snap in mem_snapshot.items()
            }
            mod_stats.snapshots.setdefault(_ModState.PEAK_FW, []).append(mem_snapshot)
        mod_stats.snapshots.setdefault(state, []).append(deepcopy(mem_snapshot))

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
            warnings.warn("Module is None. Skipping PRE_BW hook.", stacklevel=2)
            return
        mod_stats = self.memory_tracking[module]
        mem_snapshot = self.get_tracker_snapshot()
        mod_stats.local_peak = {
            dev: dev_snap[_TOTAL_KEY] for dev, dev_snap in mem_snapshot.items()
        }
        mod_stats.snapshots.setdefault(_ModState.PEAK_BW, []).append(mem_snapshot)
        mod_stats.snapshots.setdefault(_ModState.PRE_BW, []).append(
            deepcopy(mem_snapshot)
        )

    def _post_bw_hook(self, module: nn.Module, args: Any) -> None:
        # This is installed as a post-bwd user hook with ``ModTracker``. We set the state and capture the
        # snapshot for the module if it is not None.
        # This can happen since this installed inside a multi-grad hook on the module's input tensors
        # and the module itself may not be alive during backward.
        if module is None:
            warnings.warn("Module is None. Skipping POST_BW hook.", stacklevel=2)
            return
        mod_stats = self.memory_tracking[module]
        mod_stats.snapshots.setdefault(_ModState.POST_BW, []).append(
            self.get_tracker_snapshot()
        )

    def _track_optimizer_states(
        self, reftype: _RefType, optimizer: optim.Optimizer
    ) -> None:
        for states in optimizer.state.values():
            for val in states.values():
                if isinstance(val, torch.Tensor):
                    self._update_and_maybe_create_winfos(
                        val,
                        reftype,
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
            self._track_optimizer_states(_MemRefType.OPT, optimizer)
            self._in_opt = False

        self._optimizer_hook_handles = (
            register_optimizer_step_pre_hook(_opt_step_pre_hook),
            register_optimizer_step_post_hook(_opt_step_post_hook),
        )

    def _deregister_param_and_optimizer_hooks(self) -> None:
        for (
            grad_hook_handle,
            post_acc_grad_hook_handle,
        ) in self._param_to_grad_hook_handles.values():
            grad_hook_handle.remove()
            post_acc_grad_hook_handle.remove()
        self._param_to_grad_hook_handles.clear()

        if self._optimizer_hook_handles is not None:
            for handle in self._optimizer_hook_handles:
                handle.remove()
            self._optimizer_hook_handles = None

    def track_external(
        self, *external: nn.Module | optim.Optimizer | torch.Tensor
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
                self._update_and_maybe_create_winfos(
                    obj,
                    _MemRefType.OTH,
                )
            elif isinstance(obj, torch.nn.Module):
                self._track_module_params_and_buffers(obj, install_grad_hooks=False)
            elif isinstance(obj, optim.Optimizer):
                self._track_optimizer_states(_MemRefType.OPT, obj)
            elif obj is None:
                continue
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
            units (str): The units to use for displaying memory usage. Defaults to "B". Supports ["B", "KiB", "MiB", "GiB"].
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
            units (str, optional): The units to use for memory tracking. Defaults to "B". Supports ["B", "KiB", "MiB", "GiB"].
            tabulate (bool, optional): Whether to display the snapshot in a tabular format. Defaults to False.
        """

        def natural_sort_key(s: str) -> list[int | str]:
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
        if self._depth == 0:
            self._register_global_optimizer_hook()
            self._mod_tracker.register_user_hooks(
                self._pre_fw_hook,
                self._post_fw_hook,
                self._pre_bw_hook,
                self._post_bw_hook,
            )
            self._track_resize()
            self._peak_mem_snap = self.get_tracker_snapshot()
            self._peak_mem = {
                dev: dev_snap[_TOTAL_KEY]
                for dev, dev_snap in self._peak_mem_snap.items()
            }
            self._mod_tracker.__enter__()
        super().__enter__()
        self._depth += 1
        return self

    # pyrefly: ignore [bad-override]
    def __exit__(self, *args: Any) -> None:
        self._depth -= 1
        if self._depth == 0:
            self._deregister_param_and_optimizer_hooks()
            self._mod_tracker.clear_user_hooks()
            self._restore_resize()
            self._mod_tracker.__exit__(*args)
        super().__exit__(*args)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # type: ignore[no-untyped-def]
        # When running this mode with DTensor, ordinarily all modes will
        # run **before** subclasses get a chance to run.
        # Returning NotImplemented here gives us a chance to let DTensor
        # run and desugar into local tensor ops, before `MemTracker` sees them.
        if any(t == DTensor for t in types):
            return NotImplemented
        if (
            func is torch.ops._c10d_functional.wait_tensor.default
            and active_fake_mode()
        ):
            # N.B: This is a hacky way to override the Meta IMPL of wait_tensor. The original impl returns
            # a new tensor which does not happen in eager mode, when a wait_tensor is called.
            # pyrefly: ignore [bad-index, index-error]
            res = args[0]
        else:
            res = func(*args, **kwargs or {})
        # If we are tracking an optimizer state, we use the optimizer reference type.
        # If we are in backward region and not in AC region, we use the backward reference type.
        # Else we use the forward reference type.
        if self._in_opt:
            reftype = _MemRefType.OPT
        elif self._mod_tracker.is_bw and not self._in_ac:
            reftype = _MemRefType.TEMP
        else:
            reftype = _MemRefType.ACT
        tree_map_only(torch.Tensor, partial(self._track, reftype), res)
        peak_state = _ModState.PEAK_BW if self._mod_tracker.is_bw else _ModState.PEAK_FW
        self._update_peak_stats(peak_state)
        return res
