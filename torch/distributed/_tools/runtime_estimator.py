# Owner(s): ["module: unknown"]
import math
import os
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from typing_extensions import Self

import torch
import torch.utils._pytree as pytree
from torch._guards import active_fake_mode
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed._tools.mod_tracker import ModTracker
from torch.distributed._tools.runest_utils import (
    CREATE_OPS,
    get_estimation_configs,
    get_flattened_tensor,
    REDUCTION_OPS,
    resolve_gpu_type,
    VIEW_OPS,
)
from torch.utils._mode_utils import no_dispatch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.flop_counter import flop_registry


aten = torch.ops.aten

# This value is hard-coded here:
# https://github.com/pytorch/pytorch/blob/5fba5d83f0703ff8077ab65448a998e9ad6598fd/c10/cuda/CUDACachingAllocator.cpp#L117
_PYTORCH_MIN_ALLOCATE = (
    2**9 if int(os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING", 0)) == 0 else 1
)


_IGNORE_OPS = VIEW_OPS | CREATE_OPS

__all__ = ["RuntimeEstimator"]


class RuntimeEstimator(TorchDispatchMode):
    """
    Estimates the GPU runtime in milliseconds using various estimation methods under the ``FakeTensorMode``.

    This class provides a ``TorchDispatchMode`` based context manager that can be used to estimate the eager
    runtime of PyTorch functions. It supports two estimation modes, benchmarking (`operator-level-benchmark`) and
    roofline cost modeling (`operator-level-cost-model`).
    For modules executed under this context manager, it agggregates the forward and backward operation runtimes
    and also records their execution orders.

    Attributes:
        mod_runtimes (Dict[str, Dict[str, float]]): A dictionary of module runtimes. The key to the outer dictionary
            is the fully qualified name (FQN) of the module. For each module the forward and backward runtimes of the
            operations are aggregated in the inner dictionary keyed by 'fw' and 'bw'.
        mod_fw_pre_order (List[str]): List of module FQNs in pre-forward execution order.
        mod_bw_pre_order (List[str]): List of module FQNs in pre-backward execution order.
        mod_fw_post_order (List[str]): List of module FQNs in post-forward execution order.
        mod_bw_post_order (List[str]): List of module FQNs in post-backward execution order.
        total_compute_time (float): The total estimated compute time in milliseconds.

    Note:
        1) The benchmarking estimate mode will execute kernels on GPU and assumes that every operation can run in
            isolation without causing an OOM error. It is also designed to be used only under ``FakeTensorMode``.
        2) We only estimate the compute time, if your code has communication, it will not be considered. Again, we will
            support this in future PRs.

    Example usage:

        .. code-block:: python

            runtime_estimator = RuntimeEstimator()
            with FakeTensorMode():
                module = ...
                optimizer = ...
                inp = ...
                with runtime_estimator(estimate_mode_type="operator-level-cost-model"):
                    loss = module(inp)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                runtime_estimator.display_modulewise_stats()
    """

    _float_types: Set[torch.dtype] = {
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    }
    _peak_flops_reg: Dict[torch.dtype, float]
    _peak_flops_factors: Dict[torch.dtype, float]
    _peak_bandwidth: float
    _no_fallback_kernel: Set[torch._ops._OpNamespace] = set()
    _gpu_type: str = ""
    fake_mode: FakeTensorMode

    def __init__(self) -> None:
        super().__init__()
        self._estimate: Callable
        self._estimate_mode_type: str
        self._mod_tracker = ModTracker()
        self.mod_runtimes: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(lambda: 0.0)
        )
        self.mod_fw_pre_order: List[str] = []
        self.mod_bw_pre_order: List[str] = []
        self.mod_fw_post_order: List[str] = []
        self.mod_bw_post_order: List[str] = []
        self.total_compute_time: float = 0.0

    @classmethod
    def init_configs(
        cls,
        gpu_type: str = "",
        custom_config: Optional[
            Tuple[Dict[torch.dtype, float], Dict[torch.dtype, float], float]
        ] = None,
    ) -> None:
        """
        Initialize the configuration for the GPU type, including peak FLOPS, FLOPS factors, and bandwidth.

        Args:
            gpu_type (str, optional):
                The type of GPU to configure specific settings (e.g., "H100_SXM_80GB").
                Defaults to an empty string, which triggers automatic configuration based on the available GPU.
            custom_config (Optional[Tuple[Dict[torch.dtype, float], Dict[torch.dtype, float], float]], optional):
                A tuple containing:
                    - A dictionary mapping `torch.dtype` to peak FLOPS (in GFLOPS/s).
                    - A dictionary mapping `torch.dtype` to peak FLOPS factors.
                    - The peak bandwidth (in GB/s).
                If provided, this overrides the default estimation based on the GPU type.

        Returns:
            None
        Raises:
            TypeError: If `runtime_kwargs` contains invalid types for any of the supported keys.
        """
        if gpu_type and not isinstance(gpu_type, str):
            raise TypeError(f"`gpu_type` must be a str, got {type(gpu_type).__name__}")
        if custom_config:
            if not (isinstance(custom_config, tuple) and len(custom_config) == 3):
                raise TypeError("`custom_config` must be a tuple of length 3")
            if not all(isinstance(custom_config[i], dict) for i in range(2)):
                raise TypeError(
                    "The first two elements of `custom_config` must be dictionaries"
                )
            if not isinstance(custom_config[2], float):
                raise TypeError("The third element of `custom_config` must be a float")
        cls._gpu_type = resolve_gpu_type(gpu_type)
        (
            cls._peak_flops_reg,
            cls._peak_flops_factors,
            cls._peak_bandwidth,
        ) = (
            get_estimation_configs(cls._gpu_type)
            if not custom_config
            else custom_config
        )

    # Adapted from: https://github.com/pytorch/pytorch/blob/9b902b3ee3bd608a19543362b66bf06c373dd374/torch/_subclasses/fake_tensor.py#L1969  # noqa: PGH004,B950
    # NB: returns fake tensors
    @classmethod
    def _maybe_run_and_benchmark_fallback_kernel(  # type: ignore[no-untyped-def]
        cls,
        func,
        args,
        kwargs,
        orig_not_implemented_exception,
    ):
        """
        Runs and benchmarks a fallback kernel for a given function.

        Args:
            func (Callable): The function to benchmark.
            args (Tuple): The arguments to pass to the function.
            kwargs (Dict[str, Any]): The keyword arguments to pass to the function.
            orig_not_implemented_exception (Exception): The original exception to raise if the fallback kernel
                is not implemented.

        Returns:
            Tuple[Any, float]: A tuple containing the result of the function and
                the mean operation time in milliseconds.
        """
        # these should all be supported, just to be safe
        # avoid fallback for operators which inplace modify metadata
        # because the input fake tensors would be umodified
        if torch.Tag.inplace_view in func.tags:  # type: ignore[attr-defined]
            raise orig_not_implemented_exception

        inp_impls = {}
        flat_args, args_spec = pytree.tree_flatten((args, kwargs))
        # Don't use in_kernel_invocation_manager(fake_mode) as we want to do
        # REAL compute (not with meta device)
        with no_dispatch():

            def to_real_tensor(e):  # type: ignore[no-untyped-def]
                if cls.fake_mode.is_our_fake(e):
                    if e.dtype in cls._float_types:
                        out = torch.rand_like(e, device=e.fake_device)
                    else:
                        out = torch.ones_like(e, device=e.fake_device)
                    if e.is_sparse:
                        out._coalesced_(e.is_coalesced())
                    inp_impls[id(out)] = e
                    return out
                return e

            flat_args = [to_real_tensor(a) for a in flat_args]
            args, kwargs = pytree.tree_unflatten(flat_args, args_spec)
            r = func(*args, **kwargs)
            warmup_iters, actual_iters = 2, 3
            for _ in range(warmup_iters):
                func(*args, **kwargs)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record(torch.cuda.current_stream())
            for _ in range(actual_iters):
                func(*args, **kwargs)
            end_event.record(torch.cuda.current_stream())
            torch.cuda.synchronize()
            cuda_time = start_event.elapsed_time(end_event)
            mean_op_time = cuda_time / actual_iters

        storages = set()

        for e in flat_args:
            if isinstance(e, torch.Tensor):
                if not e.is_sparse:
                    storages.add(e._typed_storage()._cdata)

        # TODO: also check metadata change on inputs
        # proper aliasing/metadata relationship between outputs and inputs will
        # not be set up, bc of conversion to device, unless we can reuse an
        # input impl

        def map_out(e):  # type: ignore[no-untyped-def]
            if id(e) not in inp_impls and (
                isinstance(e, torch.Tensor)
                and not e.is_sparse
                and e._typed_storage()._cdata in storages
            ):
                raise orig_not_implemented_exception

            if isinstance(e, torch.Tensor):
                if id(e) in inp_impls:
                    return inp_impls[id(e)]
                else:
                    return cls.fake_mode.fake_tensor_converter.from_real_tensor(
                        cls.fake_mode, e
                    )
            else:
                return e

        return (pytree.tree_map(map_out, r), mean_op_time)

    @classmethod
    def _benchmark_estimate(cls, func, args, kwargs) -> Tuple[Any, float]:  # type: ignore[no-untyped-def]
        """
        Estimates the runtime of a function using benchmarking.

        Args:
            func: The function to estimate.
            args: The arguments to pass to the function.
            kwargs: The keyword arguments to pass to the function.
            res: The result of the function.

        Returns:
            Tuple[Any, float]: A tuple containing the result of the function and
                the mean operation time in milliseconds.
        """
        assert isinstance(
            cls.fake_mode, FakeTensorMode
        ), "Initialize/Assign FakeTensorMode before using this function"
        mean_op_time = 0.0
        if func._overloadpacket not in _IGNORE_OPS:
            try:
                res, mean_op_time = cls._maybe_run_and_benchmark_fallback_kernel(
                    func,
                    args,
                    kwargs,
                    NotImplementedError,
                )
                return (res, mean_op_time)
            except NotImplementedError:
                cls._no_fallback_kernel.add(func._overloadpacket)
        res = func(*args, **kwargs or {})
        return (res, mean_op_time)

    # Adapted from: https://github.com/pytorch/pytorch/blob/9b902b3ee3bd608a19543362b66bf06c373dd374/torch/_inductor/scheduler.py#L589  # noqa: PGH004,B950
    @classmethod
    def _roofline_estimate(cls, func, args, kwargs) -> Tuple[Any, float]:  # type: ignore[no-untyped-def]
        """
        Estimates the runtime of a function using a roofline cost model.

        Args:
            func: The function to estimate.
            args: The arguments to pass to the function.
            kwargs: The keyword arguments to pass to the function.
            out: The output of the function.

        Returns:
            Tuple[Any, float]: A tuple containing the result of the function and
                the mean operation time in milliseconds.
        """
        assert (
            torch.cuda.is_available()
        ), "Roofline estimation needs to access CUDA capabilities to make estimations"

        def get_num_bytes(t: torch.Tensor) -> int:
            """
            Calculates the memory consumption of a tensor.

            Args:
                t (torch.Tensor): The input tensor.

            Returns:
                int: The memory consumption of the tensor in bytes.
            """
            num_bytes = t.untyped_storage().nbytes()
            mem_consumed = (
                math.ceil(num_bytes / _PYTORCH_MIN_ALLOCATE) * _PYTORCH_MIN_ALLOCATE
            )
            return mem_consumed

        def get_compute_time(func_packet, args, kwargs, out, out_dtypes) -> float:  # type: ignore[no-untyped-def]
            """
            Estimates the compute time of an aten operator.

            Args:
                func_packet: The operator overload packet.
                args: The arguments to the operator.
                kwargs: The keyword arguments to the operator.
                out: The output of the operator.
                out_dtypes: The output data types.

            Returns:
                float: The estimated compute time in nanoseconds.
            """
            if func_packet in flop_registry:
                float_dtypes = out_dtypes & cls._float_types
                dtype = min(float_dtypes, key=lambda x: x.itemsize)
                # This gives GFLOPS/sec for the given dtype
                peak_gpu_flops = cls._peak_flops_reg[dtype]
                # factor determines the peak flops that are empirically attained by compute ops
                factor = cls._peak_flops_factors[dtype]
                peak_empirical_flops = factor * peak_gpu_flops
                flop_count_func = flop_registry[func_packet]
                # We divide by a factor of 2 to get the MACs (multiply and accumulate)
                flop_count = flop_count_func(*args, **kwargs, out_val=out) / 2
                # FLOPS/(GFLOPS/sec) gives us time in nanoseconds
                compute_time = flop_count / peak_empirical_flops
                return compute_time
            return 0.0

        def get_transfer_time(func_packet, flat_args_kwargs, flat_outs) -> float:  # type: ignore[no-untyped-def]
            """
            Estimates the memory transfer time of input and output tensors.

            Args:
                flat_args_kwargs (List[torch.Tensor]): The flat list of arguments and keyword arguments.
                flat_outs (List[torch.Tensor]): The flat list of outputs.

            Returns:
                float: The estimated memory transfer time in nanoseconds.
            """
            # The GPU memory bandwidth is in GB/s
            gpu_memory_bandwidth = cls._peak_bandwidth
            read_bytes = sum(
                get_num_bytes(t)
                for t in flat_args_kwargs
                if isinstance(t, torch.Tensor)
            )
            write_bytes = sum(
                get_num_bytes(t) for t in flat_outs if isinstance(t, torch.Tensor)
            )
            counted_bytes = read_bytes + write_bytes
            # The GPU memory bandwidth is in GB/s so the transfer time is in nanoseconds
            transfer_time = counted_bytes / gpu_memory_bandwidth
            if func_packet in REDUCTION_OPS:
                transfer_time *= 2
            return transfer_time

        # Roofline Cost Model Explanation

        # The roofline cost model estimates the execution time of an operator based on
        # the device's empirical maximum FLOPs/sec (pi) and device DRAM bandwidth (beta).

        # Variables:
        # - pi: Maximum empirical FLOPs/sec of the device
        # - beta: Maximum empirical device DRAM bandwidth (bytes/sec) of the device
        # - I: Arithmetic intensity of the operator (FLOPs/bytes)
        # - op_flops: FLOPs required by the operator
        # - op_bytes: Bytes transferred to and from DRAM for the operator

        # Calculation Steps:
        # 1. Calculate arithmetic intensity: I = op_flops / op_bytes
        # 2. Calculate estimated FLOPs/sec: est_flops_sec = min(pi, beta * I)
        # 3. Calculate estimated operator time: estimated_op_time = op_flops / est_flops_sec
        #    This simplifies to: estimated_op_time = max(op_flops / pi, op_flops / (beta * I))
        #    Further simplifying: estimated_op_time = max(op_flops / pi, op_bytes / beta)

        # Simplified Formulas:
        # - compute_time = op_flops / pi
        # - transfer_time = op_bytes / beta
        # - estimated_op_time = max(compute_time, transfer_time)

        kwargs = kwargs if kwargs else {}
        out = func(*args, **kwargs)
        op_time = 0.0
        func_packet = func._overloadpacket
        if func_packet not in _IGNORE_OPS:
            desugared_args = pytree.tree_map_only(
                torch.Tensor, get_flattened_tensor, args
            )
            desugared_kwargs = pytree.tree_map_only(
                torch.Tensor, get_flattened_tensor, kwargs
            )
            desugared_out = pytree.tree_map_only(
                torch.Tensor, get_flattened_tensor, out
            )

            flat_args_kwargs, _ = pytree.tree_flatten(
                (desugared_args, desugared_kwargs)
            )
            flat_outs, _ = pytree.tree_flatten(desugared_out)
            transfer_time = (
                get_transfer_time(func_packet, flat_args_kwargs, flat_outs) / 1.5
            )
            out_dtypes = {t.dtype for t in flat_outs if isinstance(t, torch.Tensor)}
            compute_time = get_compute_time(
                func_packet, desugared_args, desugared_kwargs, desugared_out, out_dtypes
            )
            # We get the estimated time as the max of the transfer time and
            # compute time. We divide by 1e6 to get the time in ms
            op_time = max(transfer_time, compute_time) / 1e6

        return (out, op_time)

    def display_modulewise_stats(self, depth: int = 2) -> None:
        """
        Displays module-wise statistics collected by ``RuntimeEstimator``.

        Prints the pre-forward and pre-backward execution orders.
        Displays the module-wise forward and backward runtimes in milliseconds.

        Args:
            depth (int): The maximum depth of module hierarchy to display (default to 2).
        """
        print("Pre-Forward Execution Order: ")
        for mod_fqn in self.mod_fw_pre_order:
            mod_depth = mod_fqn.count(".") + 1
            if mod_depth > depth:
                continue
            print(mod_fqn)
        print("Pre-Backward Execution Order: ")
        for mod_fqn in self.mod_bw_pre_order:
            mod_depth = mod_fqn.count(".") + 1
            if mod_depth > depth:
                continue
            print(mod_fqn)
        for mod_fqn, runtimes in self.mod_runtimes.items():
            mod_depth = mod_fqn.count(".") + 1
            if mod_depth > depth:
                continue
            print(
                f"{mod_fqn} fw: {runtimes.get('fw', 0.0):.3f}ms bw: {runtimes.get('bw', 0.0):.3f}ms"
            )

    def __torch_dispatch__(self, func, types, args=..., kwargs=None):  # type: ignore[no-untyped-def]
        # TODO: @sanketpurandare: Add logic for incorporating communication time
        res, op_time = self._estimate(func, args, kwargs)
        # if func._overloadpacket not in OPS_TO_ALWAYS_SKIP:
        #     print(f"{func._overloadpacket}: {op_time:.3f}")
        for par in self._mod_tracker.parents:
            if self._mod_tracker.is_bw:
                self.mod_runtimes[par]["bw"] += op_time
            else:
                self.mod_runtimes[par]["fw"] += op_time
        self.total_compute_time += op_time
        return res

    def __call__(
        self,
        estimate_mode_type: str,
        gpu_type: str = "",
        custom_config: Optional[
            Tuple[Dict[torch.dtype, float], Dict[torch.dtype, float], float]
        ] = None,
    ) -> Self:
        """
        Configures the runtime estimation mode and initializes GPU-specific configurations.

        Supported Modes:
            - `"operator-level-benchmark"`: Estimates runtime using operator benchmarking.
            - `"operator-level-cost-model"`: Estimates runtime using a roofline cost model.

        Args:
            estimate_mode_type (str):
                The runtime estimation mode to use. Must be one of the supported modes.
            gpu_type (str, optional):
                The GPU type to configure specific settings (e.g., `"H100_SXM_80GB"`).
                Defaults to an empty string, which triggers automatic configuration based on the available GPU.
            custom_config (Optional[Tuple[Dict[torch.dtype, float], Dict[torch.dtype, float], float]], optional):
                A tuple containing:
                    - A dictionary mapping `torch.dtype` to peak FLOPS (in GFLOPS/s).
                    - A dictionary mapping `torch.dtype` to peak FLOPS factors.
                    - The peak bandwidth (in GB/s).
                If provided, this overrides the default estimation based on the GPU type.

        Returns:
            Self:
                The current instance of `RuntimeEstimator` with the configured estimation mode.

        Raises:
            NotImplementedError:
                If `estimate_mode_type` is not a supported runtime estimation mode.
        """
        if estimate_mode_type == "operator-level-benchmark":
            self._estimate = RuntimeEstimator._benchmark_estimate
        elif estimate_mode_type == "operator-level-cost-model":
            self._estimate = RuntimeEstimator._roofline_estimate
        else:
            raise NotImplementedError(
                f"estimate_mode_type {estimate_mode_type} not supported"
            )
        self._estimate_mode_type = estimate_mode_type
        RuntimeEstimator.init_configs(gpu_type, custom_config)
        return self

    def __enter__(self) -> Self:
        fake_mode = active_fake_mode()
        assert isinstance(
            fake_mode, FakeTensorMode
        ), "No FakeTensorMode found, designed to used under FakeTensorMode"
        RuntimeEstimator.fake_mode = fake_mode
        self.total_compute_time = 0.0
        self.mod_runtimes = defaultdict(lambda: defaultdict(lambda: 0.0))
        self.mod_fw_pre_order.clear()
        self.mod_bw_pre_order.clear()
        self.mod_fw_post_order.clear()
        self.mod_bw_post_order.clear()
        self._mod_tracker.register_user_hooks(
            pre_fw_hook=lambda mod, inp: self.mod_fw_pre_order.append(
                self._mod_tracker.get_known_fqn(mod)
            ),
            pre_bw_hook=lambda mod, g_out: self.mod_bw_pre_order.append(
                self._mod_tracker.get_known_fqn(mod) if mod is not None else ""
            ),
            post_fw_hook=lambda mod, inp, out: self.mod_fw_post_order.append(
                self._mod_tracker.get_known_fqn(mod)
            ),
            post_bw_hook=lambda mod, g_inp: self.mod_bw_post_order.append(
                self._mod_tracker.get_known_fqn(mod) if mod is not None else ""
            ),
        )
        self._mod_tracker.__enter__()
        super().__enter__()
        return self

    def __exit__(self, *args: Any) -> None:
        if len(self._no_fallback_kernel) > 0:
            print("no_fallback_kernel: ", list(self._no_fallback_kernel))
        super().__exit__(*args)
        self._mod_tracker.clear_user_hooks()
        self._mod_tracker.__exit__()
