# mypy: allow-untyped-defs
from __future__ import annotations

import builtins
import copy
import dataclasses
import functools
import hashlib
import inspect
import itertools
import logging
import math
import operator
import os
import os.path
import re
import sys
import threading
import time
from collections import namedtuple
from typing import Any, Callable, Literal, Optional, TYPE_CHECKING, Union

import torch
from torch._prims_common import compute_required_storage_length
from torch.utils._ordered_set import OrderedSet

from ..triton_bundler import TritonBundler
from ..utils import prefix_is_reduction, triton_version_uses_attrs_dict
from . import triton_helpers
from .autotune_cache import AutotuneCache
from .benchmarking import benchmarker
from .coordinate_descent_tuner import CoordescTuner
from .hints import (
    _NUM_THREADS_PER_WARP,
    AutotuneHint,
    DeviceProperties,
    HeuristicType,
    ReductionHint,
    TileHint,
    TRITON_MAX_BLOCK,
    TRITON_MAX_RSPLIT,
)
from .runtime_utils import (
    ceildiv,
    conditional_product,
    create_bandwidth_info_str,
    dynamo_timed,
    get_first_attr,
    get_max_y_grid,
    get_num_bytes,
    next_power_of_2,
    triton_cache_dir,
    triton_config_to_hashable,
    triton_hash_to_path_key,
    validate_triton_config,
)
from .triton_compat import (
    ASTSource,
    autograd_profiler,
    cc_warp_size,
    CompiledKernel,
    Config,
    GPUTarget,
    KernelInterface,
    OutOfResources,
    PTXASError,
    triton,
)


class NoTritonConfigsError(RuntimeError):
    pass


if TYPE_CHECKING:
    from collections.abc import Container, Hashable, Sequence

    LauncherType = Any


log = logging.getLogger(__name__)


def get_total_reduction_numel(numels: dict[str, int]) -> int:
    return conditional_product(
        *[numel for prefix, numel in numels.items() if prefix_is_reduction(prefix)]
    )


def autotune_hints_to_configs(
    hints: OrderedSet[AutotuneHint],
    size_hints,
    block_size: int,
    device_props: DeviceProperties,
) -> list[Config]:
    """
    AutotuneHints can be attached to the metadata of triton kernels for providing
    suggestions about what to try for autotuning. One reason to do this is if there are
    some configs that are only useful in specific scenarios, in which case we can avoid
    wasting compile time on autotuning unless we know we are in one of those scenarios.

    Based on those hints, this function will generate a list of additional autotuning
    configs to try.
    """
    xyz_options: tuple[tuple[int, Optional[int], Optional[int]], ...]
    configs: list[Config] = []
    for hint in hints:
        if hint == AutotuneHint.ONE_ELEMENT_PER_THREAD:
            if len(size_hints) == 1:
                xyz_options = ((block_size // 4, None, None),)
            elif len(size_hints) == 2:
                xyz_options = ((block_size // 4, 1, None), (1, block_size // 4, None))
            elif len(size_hints) == 3:
                xyz_options = (
                    (block_size // 4, 1, 1),
                    (1, block_size // 4, 1),
                    (1, 1, block_size // 4),
                )
            configs.extend(
                triton_config(
                    size_hints,
                    *xyz,
                    num_elements_per_warp=(
                        device_props.warp_size if device_props.warp_size else 32
                    ),
                )
                for xyz in xyz_options
            )

    return configs


def disable_pointwise_autotuning(inductor_meta):
    # Autotuning can give different benchmarking results from run to run, and
    # therefore we disable autotuning when use_deterministic flag is on.
    if inductor_meta.get("are_deterministic_algorithms_enabled"):
        return True
    return not inductor_meta.get("autotune_pointwise", True)


def _dump_launch_params(args, kwargs, launcher, kernel_name, grid):
    call_args = []
    call_kwargs = {}
    for arg in args:
        if isinstance(arg, (int, bool)):
            call_args.append(str(arg))
        else:
            call_args.append("T")
    for k, v in kwargs.items():
        if isinstance(arg, (int, bool)):
            call_kwargs[k] = v
        else:
            call_kwargs[k] = v
    if not triton_version_uses_attrs_dict():
        for k, v in launcher.config.kwargs.items():
            call_kwargs[k] = v
    call_kwargs["num_warps"] = launcher.config.num_warps
    call_kwargs["num_stages"] = launcher.config.num_stages
    args_str = [*call_args]
    args_str.extend(f"{k}={v}" for k, v in call_kwargs.items())
    args_str = ", ".join(args_str)
    abs_path = os.path.abspath(sys.argv[0])
    with open(f"{abs_path}.launch_params", "a") as f:
        f.write(f"{kernel_name} | {args_str} | {grid!r}\n")


class CachingAutotuner(KernelInterface):
    """
    Simplified version of Triton autotuner that has no invalidation
    key and caches the best config to disk to improve cold start times.
    Unlike the main triton Autotuner, this version can precompile all
    configs, and does not rely on the Triton JIT.
    """

    def __init__(
        self,
        fn,
        triton_meta,  # passed directly to triton
        configs,
        save_cache_hook,
        mutated_arg_names: list[str],  # see [Note: clone mutated buffers]
        optimize_mem,
        heuristic_type,
        size_hints=None,
        inductor_meta=None,  # metadata not relevant to triton
        custom_kernel=False,  # whether the kernel is inductor-generated or custom
        filename: Optional[str] = None,
        reset_to_zero_arg_names: Optional[list[str]] = None,
    ):
        super().__init__()

        assert len(configs) > 0, "Non-empty TritonConfig list required for compiling"
        # makes sure there are no pre-hooks on any of the triton configs
        for cfg in configs:
            validate_triton_config(cfg)

        self.fn = fn
        self.device_props: DeviceProperties = triton_meta["device"]
        self.triton_meta = {
            **triton_meta,
            "device": self.device_props.index,
            "device_type": self.device_props.type,
        }
        self.inductor_meta = {} if inductor_meta is None else inductor_meta
        self.save_cache_hook = save_cache_hook
        self.mutated_arg_names = mutated_arg_names
        self.reset_to_zero_arg_names = (
            [] if reset_to_zero_arg_names is None else reset_to_zero_arg_names
        )
        self.optimize_mem = optimize_mem
        self.configs = configs
        self.heuristic_type = heuristic_type
        self.custom_kernel = custom_kernel
        self.cuda_kernel_saved = False
        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                "CachingAutotuner gets %d configs for %s",
                len(self.configs),
                self.fn.__name__,
            )
            for c in self.configs:
                log.debug(c)

        self.compile_results: list[TritonCompileResult] = []
        self.launchers: list[LauncherType] = []
        self.lock = threading.Lock()
        if os.getenv("TRITON_CACHE_DIR") is None:
            os.environ["TRITON_CACHE_DIR"] = triton_cache_dir(
                self.triton_meta.get("device", 0)
            )
        log.debug("Triton cache dir: %s", os.environ["TRITON_CACHE_DIR"])

        self.size_hints = size_hints
        self.coordesc_tuner = CoordescTuner(
            is_mm=False,
            name=self.fn.__name__,
            size_hints=size_hints,
            inductor_meta=self.inductor_meta,
        )
        self.filename = filename

        # used for profiling
        self.kernel_hash: str = ""

        # Kernels are stored in the codecache with the filename as a hash of the code.
        # We rely on this to obtain the kernel hash
        if self.filename is not None:
            base_name = os.path.basename(self.filename)
            if ".py" in base_name:
                self.kernel_hash = os.path.splitext(base_name)[0]

        self.precompile_time_taken_ns = 0
        self.autotune_time_taken_ns = 0
        # Dumps the launch configs after autotuning.
        self.dump_launch_params = (
            os.environ.get("TORCHINDUCTOR_DUMP_LAUNCH_PARAMS", "0") == "1"
        )

        self.triton_interpret = os.environ.get("TRITON_INTERPRET", "0") == "1"

    def precompile(
        self,
        warm_cache_only=False,
        reload_kernel: Optional[Callable[[], CachingAutotuner]] = None,
    ):
        if warm_cache_only:
            self._precompile_worker()
            return
        with self.lock:
            # Helper function for reloading a kernel generated in a worker
            # in the parent class. Normally we don't need to reload the kernel
            # in the parent process, but in certain cases (coordesc tuning, dynamic_scale_rblock),
            # we need to actually run compilation on the parent process
            if reload_kernel is not None:
                self._reload_kernel = reload_kernel
            self._precompile_worker()
            self._make_launchers()
            self._dynamic_scale_rblock()

    def _precompile_worker(self):
        if self.compile_results:
            for result in self.compile_results:
                TritonBundler.put(
                    triton_hash_to_path_key(result.kernel.hash),
                    self.triton_meta.get("device", 0),
                )
            return
        assert not self.launchers
        if not self.configs:
            raise NoTritonConfigsError("No triton configs are available")

        compile_results = []
        exc = None
        for c in self.configs:
            try:
                compile_results.append(self._precompile_config(c))
            except (OutOfResources, PTXASError) as e:
                exc = e
        if len(compile_results) == 0:
            raise NoTritonConfigsError(
                f"No valid triton configs. {type(exc).__name__}: {exc}"
            )
        self.compile_results = compile_results
        self.configs = None

    def _dynamic_scale_rblock(self):
        # TODO(jansel): we should find a way to move this extra compile into the worker process
        # Currently it relies on _make_launchers(), which requires a cuda context, to populate nreg.
        device_prop = self.device_props
        if (
            self.inductor_meta.get("dynamic_scale_rblock", True)
            and not self.inductor_meta.get("persistent_reduction")
            and self.heuristic_type == HeuristicType.REDUCTION
            and self.size_hints is not None
            # Disable for Intel as Triton is not ready to return n_regs for a compiled_binary.
            and device_prop.type in ["cuda", "hip"]
            and device_prop.major
            and (device_prop.major >= 8 or torch.version.hip)
            and device_prop.regs_per_multiprocessor is not None
        ):
            assert device_prop.regs_per_multiprocessor
            assert device_prop.max_threads_per_multi_processor
            assert device_prop.multi_processor_count
            seen_config_hashes: Optional[OrderedSet[Hashable]] = None
            warp_size = device_prop.warp_size or 32
            for result in self.compile_results:
                triton_config = result.config
                compiled_binary = result.kernel
                assert len(self.size_hints) >= 2
                xblock = triton_config.kwargs.get("XBLOCK", 1)
                reduction_kwargs = [
                    kwarg for kwarg in triton_config.kwargs if kwarg.startswith("R")
                ]
                rblocks = [triton_config.kwargs[kwarg] for kwarg in reduction_kwargs]
                total_block = (self.size_hints["x"] + xblock - 1) // xblock
                nreg = getattr(compiled_binary, "n_regs", None)
                if nreg is None:
                    continue

                # make sure rblocks are not too small
                if conditional_product(*rblocks) <= 64:
                    continue

                # each SM of A100 has 65536 32-bit registers. To maximize
                # the theoretical occupancy, we need run 2048 threads on each
                # SM. So each thread should use no more than 65536 / 2048
                # = 32 registers. In cases where occupancy matters, and each
                # thread uses too many registers, reduce R0_BLOCK to reduce
                # the register usage.
                # For kernel https://gist.github.com/shunting314/e4cccc031fe30d378b9b23c08c238cbd
                # from PLBartForCausalLM, latency improve from
                # 7.795ms to 4.883ms.
                #
                if (
                    nreg
                    <= device_prop.regs_per_multiprocessor
                    // device_prop.max_threads_per_multi_processor
                ):
                    continue

                nreg_per_warp = nreg * warp_size
                nreg_per_block = nreg_per_warp * triton_config.num_warps

                # Previously we set max_blocks_per_sm to 'max_threads_per_multi_processo / (32 * num_warps)'
                # The formula below is a tighter upper bound since we have the assumption that
                #   nreg > device_prop.regs_per_multiprocessor // device_prop.max_threads_per_multi_processor
                # due to the if condition above and:
                #   regs_per_multiprocessor / nreg_per_block
                #   = regs_per_multiprocessor / (nreg * 32 * num_warps)
                #   < regs_per_multiprocessor / ((regs_per_multiprocessor / max_threads_per_multi_processor) * 32 * num_warps)
                #   = max_threads_per_multi_processor / (32 * num_warps)
                # Using a tigher upper bound can reveal more optimization opportunities.
                max_blocks_per_sm = max(
                    device_prop.regs_per_multiprocessor // nreg_per_block, 1
                )

                if total_block <= max_blocks_per_sm * device_prop.multi_processor_count:
                    # no need to improve occupancy
                    continue
                new_config = copy.deepcopy(triton_config)

                # Reduce the largest Rn_BLOCK by a factor of 2.
                largest_rkwarg: str = max(
                    reduction_kwargs, key=triton_config.kwargs.__getitem__
                )
                new_config.kwargs[largest_rkwarg] //= 2

                if seen_config_hashes is None:
                    seen_config_hashes = OrderedSet(
                        [
                            triton_config_to_hashable(x.config)
                            for x in self.compile_results
                        ]
                    )
                new_config_hash = triton_config_to_hashable(new_config)
                if new_config_hash in seen_config_hashes:
                    continue
                seen_config_hashes.add(new_config_hash)
                log.debug(
                    "Dynamically scale down %s from TritonConfig(%s) and get a new TritonConfig(%s)",
                    largest_rkwarg,
                    triton_config,
                    new_config,
                )
                if self.fn.fn is None:
                    """
                    We are in the parent process, while this program was compiled in a worker
                    and the fn was dropped in prepare_for_pickle().  We haven't loaded the module
                    containing the real fn yet.
                    """
                    assert hasattr(self, "_reload_kernel")
                    assert callable(self._reload_kernel)
                    self.fn = self._reload_kernel().fn
                self.compile_results.append(self._precompile_config(new_config))

            self._make_launchers()

    def _make_launchers(self):
        if len(self.launchers) == len(self.compile_results):
            return

        from torch._dynamo.device_interface import DeviceGuard

        device_interface = self.get_device_interface()

        # load binary to the correct device
        with DeviceGuard(device_interface, self.triton_meta["device"]):
            # need to initialize context
            device_interface.synchronize(device_interface.current_device())
            launchers = []
            exc = None
            for result in self.compile_results:
                try:
                    launchers.append(result.make_launcher())

                except (OutOfResources, PTXASError) as e:
                    exc = e
        if len(launchers) == 0:
            raise RuntimeError(f"No valid triton configs. {type(exc).__name__}: {exc}")
        self.launchers = launchers

    def prepare_for_pickle(self):
        """Drop stuff from triton.JITFunction that does not pickle.
        This must be called after precompile so that these things are no longer needed.
        """
        self.fn.fn = None
        self.fn.__globals__ = None
        self.fn.used_global_vals = None
        self.fn.repr = _ConstRepr(self.fn.repr(self.fn))
        self.launchers = []

    def __getstate__(self) -> dict[str, Any]:
        assert not self.launchers, (
            "pickle should not be called with after make_launchers()"
        )
        return {
            **self.__dict__,
            "lock": None,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.lock = threading.Lock()

    def get_device_interface(self):
        # this code cannot run in compile workers, because it imports from torch
        from torch._dynamo.device_interface import get_interface_for_device

        return get_interface_for_device(self.device_props.type.replace("hip", "cuda"))

    def _precompile_config(self, cfg: Config) -> TritonCompileResult:
        """Ahead of time compile a given autotuner config."""
        compile_meta = copy.deepcopy(self.triton_meta)
        cfg_kwargs = cfg.kwargs
        if self.device_props.type == "hip":
            cfg_kwargs = {**cfg_kwargs}
            for k in ("matrix_instr_nonkdim", "waves_per_eu", "kpack"):
                if k in cfg_kwargs:
                    compile_meta[k] = cfg_kwargs.pop(k)
        compile_meta["constants"].update(cfg_kwargs)
        for i in self.fn.constexprs:
            arg_name = self.fn.arg_names[i]
            if arg_name not in compile_meta["constants"] and (
                arg_name == "num_warps" or arg_name == "num_stages"
            ):
                compile_meta["constants"][arg_name] = getattr(cfg, arg_name)
        compile_meta["num_warps"] = cfg.num_warps
        compile_meta["num_stages"] = cfg.num_stages
        compile_meta["debug"] = self.inductor_meta.get(
            "assert_indirect_indexing", True
        ) and not self.inductor_meta.get("is_hip", False)

        # device type will be "hip" rather than "cuda" here
        compile_meta["device_type"] = self.device_props.type
        compile_meta["cc"] = self.device_props.cc

        if self.device_props.type == "cpu":
            triton_helpers.set_driver_to_cpu()
        else:
            triton_helpers.set_driver_to_gpu()

        if not ASTSource:
            raise RuntimeError("Installed triton version too old, please upgrade")

        compile_args = (
            ASTSource(
                self.fn,
                compile_meta["signature"],
                compile_meta["constants"],
                compile_meta["configs"][0],
            ),
        )

        target = GPUTarget(
            compile_meta["device_type"],
            compile_meta["cc"],
            cc_warp_size(compile_meta["cc"]),
        )

        options = {
            "num_warps": compile_meta["num_warps"],
            "num_stages": compile_meta["num_stages"],
            "debug": compile_meta["debug"],
            "sanitize_overflow": False,  # turn off additional asserts added for overflow checks
        }
        if self.device_props.type == "hip":
            if "waves_per_eu" in compile_meta:
                options["waves_per_eu"] = compile_meta["waves_per_eu"]
            if "matrix_instr_nonkdim" in compile_meta:
                options["matrix_instr_nonkdim"] = compile_meta["matrix_instr_nonkdim"]
        compile_kwargs = {
            "target": target,
            "options": options,
        }

        try:
            binary = triton.compile(*compile_args, **compile_kwargs)
        except Exception:
            log.exception(
                "Triton compilation failed: %s\n%s\nmetadata: %s",
                self.inductor_meta.get("kernel_name", "triton_"),
                self.fn.src,
                compile_meta,
            )
            raise
        TritonBundler.put(
            triton_hash_to_path_key(binary.hash), self.triton_meta.get("device", 0)
        )
        return TritonCompileResult(binary, cfg, compile_meta, self.inductor_meta)

    def _get_args_with_constexprs(self, args, launcher):
        """
        `args` is passed in with only the non-constexpr args (because the constexpr arg values
        depend on the config). However, in later triton versions, the constexpr args need to be
        added into the args list.
        """
        if triton_version_uses_attrs_dict():
            # first: aggregate the constexpr args in (index, val) pairs
            # so we can sort them by index.
            constexpr_args: list[tuple[int, Any]] = []
            for arg_name, arg_val in launcher.config.kwargs.items():
                constexpr_args.append((self.fn.arg_names.index(arg_name), arg_val))

            constexpr_args.sort()
            new_args = [*args]
            for arg_idx, arg_val in constexpr_args:
                new_args.insert(arg_idx, arg_val)

            return new_args
        return args

    def bench(self, launcher, *args, with_profiler=False, **kwargs):
        """Measure the performance of a given launcher"""
        # we don't skip configs with spilled registers when auto-tuning custom
        # (user-written) Triton kernels, as (i) we don't have any knowledge or
        # control over the kernel code; (ii) there is empirical evidence that
        # for some (complicated) custom Triton kernels, a register-spilling
        # config may yield the best latency.
        if not self.custom_kernel and launcher.n_spills > self.inductor_meta.get(
            "spill_threshold", 16
        ):
            log.debug(
                "Skip config %s because of register spilling: %d",
                launcher.config,
                launcher.n_spills,
            )
            return float("inf")

        device_interface = self.get_device_interface()
        stream = device_interface.get_raw_stream(device_interface.current_device())

        cpu_copies = self.copy_args_to_cpu_if_needed(*args, **kwargs)

        def kernel_call():
            cloned_args, cloned_kwargs = self.maybe_clone_args(
                cpu_copies, *args, **kwargs
            )
            # reset to zero before evaluating any config
            self.reset_to_zero_args(*args, **kwargs)
            args_with_constexprs = self._get_args_with_constexprs(cloned_args, launcher)
            launcher(
                *args_with_constexprs,
                **cloned_kwargs,
                stream=stream,
            )
            self.restore_args_from_cpu(cpu_copies)

        if with_profiler:
            from torch._inductor.utils import do_bench_using_profiling

            return do_bench_using_profiling(kernel_call, warmup=10, rep=40)

        if self.device_props.type == "cpu":
            return benchmarker.benchmark_cpu(kernel_call)

        return benchmarker.benchmark_gpu(kernel_call, rep=40)

    def copy_args_to_cpu_if_needed(self, *args, **kwargs):
        """
        To support benchmarking in the presence of mutated args, we need to avoid
        autotuning contanminating them. We try to pass cloned args to the kernel.
        If those clones would increase the peak memory usage, however, we instead
        copy to cpu and restore them after each iteration. Figure out the args
        to be copied and do the copying.
        """
        if not self.optimize_mem:
            return {}

        copies = {}
        budget = torch.cuda.max_memory_allocated() - torch.cuda.memory_allocated()

        def maybe_copy(name, arg):
            if name in self.mutated_arg_names and arg.is_cuda:
                nonlocal budget
                assert isinstance(arg, torch.Tensor)
                required_storage_length = compute_required_storage_length(
                    arg.size(),
                    arg.stride(),
                    0,
                )
                size = required_storage_length * arg.element_size()
                if size > budget:
                    cpu_arg = torch.empty_strided(
                        (required_storage_length,),
                        (1,),
                        dtype=arg.dtype,
                        device="cpu",
                        pin_memory=True,
                    )
                    cpu_arg.copy_(
                        arg.as_strided((required_storage_length,), (1,)),
                        non_blocking=True,
                    )
                    copies[name] = (arg, cpu_arg)
                else:
                    budget -= size

        for name, arg in zip(self.fn.arg_names, args):
            maybe_copy(name, arg)

        for name, arg in kwargs.items():
            maybe_copy(name, arg)

        return copies

    def restore_args_from_cpu(self, cpu_copies):
        for pair in cpu_copies.values():
            arg, cpu_arg = pair
            required_storage_length = compute_required_storage_length(
                arg.size(),
                arg.stride(),
                0,
            )
            arg.as_strided((required_storage_length,), (1,)).copy_(
                cpu_arg, non_blocking=True
            )

    def reset_to_zero_args(self, *args, **kwargs):
        if not self.reset_to_zero_arg_names:
            return
        for i, arg in enumerate(args):
            if self.fn.arg_names[i] in self.reset_to_zero_arg_names:
                assert isinstance(
                    arg,
                    torch.Tensor,
                ), (
                    "self.reset_to_zero_arg_names should only contain valid argument names"
                )
                arg.zero_()

        for name, arg in kwargs.items():
            if name in self.reset_to_zero_arg_names:
                assert isinstance(
                    arg,
                    torch.Tensor,
                ), (
                    "self.reset_to_zero_arg_names should only contain valid argument names"
                )
                arg.zero_()

    def maybe_clone_args(
        self, exclude: Container[str], *args, **kwargs
    ) -> tuple[list[Any], dict[str, Any]]:
        """
        Prepare new args and kwargs by cloning any in-place buffers
        (that are not in the provided exclusion list), to avoid autotune
        contaminating them. Avoid cloning the other buffers because it
        leads to increased memory usage.
        """
        from ..compile_fx import clone_preserve_strides

        def prepare_arg(name, arg):
            if name in self.mutated_arg_names and name not in exclude:
                assert isinstance(arg, torch.Tensor)
                return clone_preserve_strides(arg)
            else:
                return arg

        cloned_args = [
            prepare_arg(name, arg)
            for name, arg in itertools.zip_longest(self.fn.arg_names[: len(args)], args)
        ]
        cloned_kwargs = {name: prepare_arg(name, arg) for name, arg in kwargs.items()}
        return cloned_args, cloned_kwargs

    def clone_args(self, *args, **kwargs) -> tuple[list[Any], dict[str, Any]]:
        return self.maybe_clone_args(OrderedSet(), *args, **kwargs)

    def benchmark_all_configs(self, *args, **kwargs):
        with dynamo_timed(
            "CachingAutotuner.benchmark_all_configs",
            log_pt2_compile_event=True,
            metadata={"kernel_name": self.inductor_meta.get("kernel_name")},
            # TODO(masnesral): Enable this when we figure out how to get the CompileId:
            # dynamo_compile_runtime_column_us="runtime_triton_autotune_time_us",
        ):
            timings = {
                launcher: self.bench(launcher, *args, **kwargs)
                for launcher in self.launchers
            }

            for k, v in timings.items():
                self.coordesc_tuner.cache_benchmark_result(k.config, v)

            if log.isEnabledFor(logging.DEBUG):
                log.debug("Benchmark all input configs for %s, get:", self.fn.__name__)
                for k, v in timings.items():
                    log.debug(
                        "%s: %f, nreg %d, nspill %d, #shared-mem %s",
                        k.config,
                        v,
                        k.n_regs,
                        k.n_spills,
                        k.shared,
                    )

            self.reset_to_zero_args(*args, **kwargs)
            return timings

    def autotune_to_one_config(self, *args, **kwargs):
        """Do the actual autotuning"""
        start_time = time.time_ns()
        timings = self.benchmark_all_configs(*args, **kwargs)
        benchmark_time_taken_ns = time.time_ns() - start_time
        self.launchers = [builtins.min(timings, key=timings.get)]
        self.autotune_time_taken_ns = (
            self.precompile_time_taken_ns + benchmark_time_taken_ns
        )

        # log the best config
        launcher = self.launchers[0]
        log.debug(
            "Best config for %s: %s: %f, nreg %d, nspill %d, #shared-mem %s",
            self.fn.__name__,
            launcher.config,
            timings[launcher],
            launcher.n_regs,
            launcher.n_spills,
            launcher.shared,
        )

        if self.save_cache_hook:
            self.save_cache_hook(launcher.config, self.autotune_time_taken_ns)

    def save_gpu_kernel(self, stream, launcher):
        key = self.inductor_meta.get("kernel_name", None)  # unique kernel name
        assert key is not None, "kernel_name can not be None"
        params = {
            "mangled_name": (
                launcher.bin.metadata.name
                if hasattr(launcher.bin.metadata, "name")
                else launcher.bin.metadata["name"]
            ),
            "num_warps": (
                launcher.bin.num_warps
                if hasattr(launcher.bin, "num_warps")
                else launcher.bin.metadata.num_warps
            ),
            "shared_mem": (
                launcher.bin.shared
                if hasattr(launcher.bin, "shared")
                else launcher.bin.metadata.shared
            ),
            "stream": stream,
            # User defined triton kernels will have arbitrary kwarg names
            "config": config_to_dict(launcher.config),
            "inductor_meta": self.inductor_meta,
            "triton_meta": self.triton_meta,
            "def_args": launcher.def_args,
            "call_args": launcher.call_args,
        }
        from torch._inductor.codecache import CudaKernelParamCache

        bin_type = {"hip": "hsaco", "xpu": "spv"}.get(self.device_props.type, "cubin")
        binary = launcher.bin.asm[bin_type]
        CudaKernelParamCache.set(key, params, binary, bin_type)

        self.cuda_kernel_saved = True

    def coordinate_descent_tuning(self, launcher, *args, **kwargs):
        """
        Coordinate descent tuning can be run with or without max-autotune.

        The only difference between these two is the starting config for coordinate_descent tuning.
        E.g., assuming regular autotune only get one config C1; while max-autotune get 4 configs C1, C2, C3, C4
        and max-autotune figure out C3 is the best.

        Then if coordinate desecnt tuning is run with max-autotune disabled, it will start from C1;
        while if coordinate descent tuning is run with max-autotune enabled, it will start from C3.
        """
        if (
            self.heuristic_type == HeuristicType.TEMPLATE
            or self.heuristic_type == HeuristicType.USER_AUTOTUNE
        ):
            # skip triton template
            return launcher

        config2launcher = {launcher.config: launcher}

        # TODO: should we just load the kernels ahead of time if we know we're going to call this?
        if self.fn.fn is None:
            """
            We are in the parent process, while this program was compiled in a worker
            and the fn was dropped in prepare_for_pickle().  We haven't loaded the module
            containing the real fn yet.
            """
            assert hasattr(self, "_reload_kernel")
            assert callable(self._reload_kernel)
            self.fn = self._reload_kernel().fn

        def benchmark_one_config(config):
            with self.lock:
                launcher = self._precompile_config(config).make_launcher()
            config2launcher[config] = launcher

            out = self.bench(launcher, *args, **kwargs)
            log.debug(
                "COORDESC: %s: %f, nreg %d, nspill %d, #shared-mem %d",
                launcher.config,
                out,
                launcher.n_regs,
                launcher.n_spills,
                launcher.shared,
            )
            return out

        assert not (
            self.heuristic_type == HeuristicType.PERSISTENT_REDUCTION
            and "R0_BLOCK" in launcher.config.kwargs
        ), (
            "Coordinate descent tuner relies on the assumption that persistent reduction's triton config does not have R0_BLOCK"
        )
        start_time = time.time_ns()
        best_config = self.coordesc_tuner.autotune(
            benchmark_one_config, launcher.config, None
        )
        coordesc_time_taken_ns = time.time_ns() - start_time
        best_config.found_by_coordesc = True

        if self.save_cache_hook:
            self.save_cache_hook(
                best_config,
                self.autotune_time_taken_ns + coordesc_time_taken_ns,
                found_by_coordesc=True,
            )
        return config2launcher.get(best_config)

    def run(
        self,
        *args,
        stream,
        benchmark_run=False,
        **kwargs,
    ):  # type:ignore[override]
        if self.triton_interpret:
            args, grid = self._interpret_args_grid(args, self.configs[0])
            return self.fn[grid](
                *args,
                **kwargs,
                **self.configs[0].kwargs,
            )

        if len(self.launchers) != 1:
            if len(self.launchers) == 0:
                start_time = time.time_ns()
                self.precompile()
                self.precompile_time_taken_ns = time.time_ns() - start_time
            if len(self.launchers) > 1:
                self.autotune_to_one_config(*args, **kwargs)

        if not getattr(
            self.launchers[0].config, "found_by_coordesc", False
        ) and self.inductor_meta.get("coordinate_descent_tuning", False):
            self.launchers = [
                self.coordinate_descent_tuning(self.launchers[0], *args, **kwargs)
            ]

        (launcher,) = self.launchers
        if launcher.store_cubin and (not benchmark_run or not self.cuda_kernel_saved):
            self.save_gpu_kernel(stream, launcher)

        args = self._get_args_with_constexprs(args, launcher)

        if self.dump_launch_params:
            new_args, grid = self._interpret_args_grid(args, launcher.config)
            _dump_launch_params(new_args, kwargs, launcher, self.fn.__name__, grid)

        # it is faster than entering and exiting a context manager, even if the context
        # manager is a nullcontext.
        if autograd_profiler._is_profiler_enabled:
            kernel_kwargs_str = ",".join(
                f"{k}={v}" for (k, v) in launcher.config.kwargs.items()
            )

            profiler_kwargs = {
                "kernel_file": (self.filename or ""),
                "kernel_hash": self.kernel_hash,
                "kernel_backend": "triton",
                "stream": stream,
                "num_warps": launcher.config.num_warps,
                "num_stages": launcher.config.num_stages,
                "kernel_kwargs": kernel_kwargs_str,
            }

            with torch._C._profiler._RecordFunctionFast(
                self.inductor_meta.get("kernel_name", "triton kernel"),
                args,
                profiler_kwargs,
            ):
                return launcher(
                    *args,
                    **kwargs,
                    stream=stream,
                )
        else:
            return launcher(
                *args,
                **kwargs,
                stream=stream,
            )

    def _interpret_args_grid(
        self, args: tuple[Any, ...], cfg: Config
    ) -> tuple[tuple[Any, ...], tuple[int, int, int]]:
        grid = GridExpr.from_meta(self.inductor_meta, cfg).eval_slow(
            dict(
                zip(
                    [
                        *self.fn.arg_names,
                        *self.inductor_meta.get("extra_launcher_args", ()),
                    ],
                    args,
                )
            )
        )
        if self.inductor_meta.get("extra_launcher_args"):
            args = args[: -len(self.inductor_meta["extra_launcher_args"])]
        return args, grid


class _ConstRepr:
    def __init__(self, value: str):
        self.value = value

    def __call__(self, _=None) -> str:
        return self.value


class TritonCompileResult:
    """
    Upstream Triton CompileKernel can not be pickled.  This is a wrapper
    to support serialization and generate the launcher function.
    """

    @staticmethod
    @functools.lru_cache(32)
    def _kernel_metadata_cls(fields: tuple[str, ...]) -> Any:
        return namedtuple("KernelMetadata", sorted(fields))

    def __init__(
        self,
        kernel: CompiledKernel,
        config: Config,
        compile_meta: dict[str, Any],
        inductor_meta: dict[str, Any],
    ) -> None:
        super().__init__()
        self.kernel = kernel
        self.config = config
        self.compile_meta = compile_meta
        self.inductor_meta = inductor_meta

    @staticmethod
    def _serialize_metadata(metadata):
        """
        Triton uses a nested class called KernelMetadata to store metadata information.
        Pickle does not work well with nested namedtuples, as the namedtuple doesn't appear
        in the toplevel namespace of the module. So these serialization/deser functions
        are used to convert the namedtuples to a dict and back.

        As for packed_metadata, depending on the triton backend, KernelMetadata can be
        a namedtuple, or a regular tuple! So the serialization function branches on whether
        the metadata to be serialized is a namedtuple or regular, serializable one.
        """

        def is_namedtuple(obj) -> bool:
            return (
                isinstance(obj, tuple)
                and hasattr(obj, "_asdict")
                and hasattr(obj, "_fields")
            )

        if is_namedtuple(metadata):
            return metadata._asdict()
        else:
            return metadata

    @staticmethod
    def _deserialize_metadata(metadata):
        if isinstance(metadata, dict):
            return TritonCompileResult._kernel_metadata_cls(tuple(metadata.keys()))(
                **metadata
            )
        else:
            return metadata

    def __getstate__(self) -> dict[str, Any]:
        kernel = self.kernel
        # replace the fields that don't pickle nicely
        kernel_state = {
            **kernel.__dict__,
            # See doc about serializing metadata above
            "metadata": self._serialize_metadata(kernel.metadata),
            "packed_metadata": self._serialize_metadata(
                getattr(kernel, "packed_metadata", None)
            ),
            "module": None,  # regenerated by kernel._init_handles()
            "function": None,  # regenerated by kernel._init_handles()
            "run": None,  # regenerated by kernel._init_handles()
        }
        return {**self.__dict__, "kernel": kernel_state}  # type: ignore[dict-item]

    def __setstate__(self, state: dict[str, Any]) -> None:
        # src = ASTSource.__new__(ASTSource)
        # src.__setstate__(state["kernel"]["src"])
        # TODO(jansel): need to fixup src.fn which is now None
        kernel = CompiledKernel.__new__(CompiledKernel)
        metadata = state["kernel"]["metadata"]
        packed_metadata = state["kernel"]["packed_metadata"]
        kernel.__dict__.update(
            {
                **state["kernel"],
                # "src": src,
                "metadata": self._deserialize_metadata(metadata),
                "packed_metadata": self._deserialize_metadata(packed_metadata),
            }
        )
        self.__dict__.update(state)
        self.kernel = kernel

    def make_launcher(self) -> LauncherType:
        """
        Launching triton kernels is performance sensitive, we compile
        a custom Python function get the grid() and reorder the args to
        the underlying wrapper.
        """
        cfg = self.config
        compile_meta = self.compile_meta
        binary = self.kernel
        fn = binary.src.fn
        binary._init_handles()
        """
        https://github.com/pytorch/pytorch/issues/115344

        self.fn.constexprs doesn't properly deal with None args, so when we filter out
        an arg in UserDefinedTritonKernel.codegen, we need to filter it here as well.
        We also don't want to modify self.fn.

        We know that we removed something from the signature if:
            1. It's in compile_meta["constants"]
            2. It isn't a constant we already know about
                Note: The value of interest has already been added to compile_meta['constants'],
                    so we use self.fn.constexprs instead.
            3. It isn't in the compile_meta signature
        """
        known_constants = OrderedSet(
            arg for i, arg in enumerate(fn.arg_names) if i in fn.constexprs
        )
        none_args = OrderedSet(
            k
            for k, v in compile_meta["constants"].items()
            if v is None and k not in known_constants
        )
        none_args = none_args.difference(OrderedSet(compile_meta["signature"].keys()))

        if triton_version_uses_attrs_dict():
            call_args = fn.arg_names
            def_args = fn.arg_names
            if (
                "num_warps" in compile_meta["constants"]
                or "num_stages" in compile_meta["constants"]
            ):
                # num_warps/num_stages are special implicit args that are not in the signature
                # see test_triton_kernel_special_params
                def_args = [
                    arg for arg in def_args if arg not in ("num_warps", "num_stages")
                ]
                repl = {
                    k: str(compile_meta["constants"].get(k))
                    for k in ("num_warps", "num_stages")
                }
                call_args = [repl.get(arg, arg) for arg in call_args]
        else:
            call_args = [
                arg
                for i, arg in enumerate(fn.arg_names)
                if i not in fn.constexprs and arg not in none_args
            ]
            cfg_dict = config_to_dict(cfg)
            def_args = [
                name
                for name in fn.arg_names
                if name not in cfg_dict and name not in none_args
            ]

        binary_shared = (
            binary.shared if hasattr(binary, "shared") else binary.metadata.shared
        )

        scope = {
            "grid_meta": cfg.kwargs,
            "bin": binary,
            "launch_enter_hook": binary.__class__.launch_enter_hook,
            "launch_exit_hook": binary.__class__.launch_exit_hook,
            "metadata": (
                binary.packed_metadata
                if hasattr(binary, "packed_metadata")
                else binary.metadata
            ),
            "shared": binary_shared,
            "num_warps": (
                binary.num_warps
                if hasattr(binary, "num_warps")
                else binary.metadata.num_warps
            ),
            "cta_args": (
                (
                    binary.num_ctas,
                    *get_first_attr(binary, "cluster_dims", "clusterDims"),
                )
                if hasattr(binary, "num_ctas")
                else (
                    (binary.metadata.num_ctas, *binary.metadata.cluster_dims)
                    if hasattr(binary, "metadata")
                    else ()
                )
            ),
            "function": get_first_attr(binary, "function", "cu_function"),
            "runner": get_first_attr(binary, "run", "c_wrapper"),
        }

        if not hasattr(binary, "launch_metadata"):
            # launch args before CompiledKernel.launch_metadata is added.
            # TODO(jansel): delete this branch in mid-2025
            runner_args = [
                "grid_0",
                "grid_1",
                "grid_2",
                "num_warps",
                "*cta_args",
                "shared",
                "stream",
                "function",
                "launch_enter_hook",
                "launch_exit_hook",
                "metadata",
                *call_args,
            ]
        else:  # args after CompiledKernel.launch_metadata: https://github.com/openai/triton/pull/3492
            # Getting the kernel launch args is extremely perf-sensitive.  Evaluating
            # `bin.launch_metadata` is relatively expensive, and returns None unless a
            # `launch_enter_hook` is installed.  So if we don't have that hook installed,
            # we want to burn None in to the launch args with zero overhead.
            # See https://github.com/pytorch/pytorch/issues/123597
            if binary.__class__.launch_enter_hook:
                launch_metadata = f"bin.launch_metadata((grid_0, grid_1, grid_2), stream, {', '.join(call_args)})"
            else:
                launch_metadata = "None"
            runner_args = [
                "grid_0",
                "grid_1",
                "grid_2",
                "stream",
                "function",
                "metadata",
                launch_metadata,
                "launch_enter_hook",
                "launch_exit_hook",
                *call_args,
            ]

        if "extra_launcher_args" in self.inductor_meta:
            def_args = [*def_args, *self.inductor_meta["extra_launcher_args"]]

        grid = GridExpr.from_meta(self.inductor_meta, cfg)
        # grid.prefix is usually empty, grid.x_grid is something like `-(xnumel//-1024)`
        lines = [
            f"def launcher({', '.join(def_args)}, stream):",
            *[f"    {line}" for line in grid.prefix],
            f"    grid_0 = {grid.x_grid}",
            f"    grid_1 = {grid.y_grid}",
            f"    grid_2 = {grid.z_grid}",
            f"    runner({', '.join(runner_args)})",
        ]
        exec("\n".join(lines), scope)

        launcher = scope["launcher"]
        launcher.config = cfg
        launcher.n_regs = getattr(binary, "n_regs", None)
        launcher.n_spills = getattr(binary, "n_spills", None)
        launcher.shared = binary_shared
        launcher.store_cubin = self.inductor_meta.get("store_cubin", False)
        # store this global variable to avoid the high overhead of reading it when calling run
        if launcher.store_cubin:
            launcher.fn = fn
            launcher.bin = binary
            if triton_version_uses_attrs_dict():
                # arg filtering wasn't done above
                cfg_dict = config_to_dict(cfg)
                def_args = [x for x in def_args if x not in cfg_dict]
                call_args = [
                    x
                    for x in call_args
                    if compile_meta["signature"].get(x, "constexpr") != "constexpr"
                    and x not in none_args
                ]
            launcher.def_args = def_args
            launcher.call_args = call_args
        return launcher


def _find_names(obj):
    import gc
    import inspect

    frame = inspect.currentframe()
    while frame is not None:
        frame.f_locals
        frame = frame.f_back
    obj_names = []
    for referrer in gc.get_referrers(obj):
        if isinstance(referrer, dict):
            for k, v in referrer.items():
                if v is obj:
                    obj_names.append(k)
    return obj_names


collected_calls: list[Any] = []


def start_graph():
    collected_calls.clear()


def end_graph(output_file):
    if len(collected_calls) == 0:
        return
    overall_time = sum(call[0] for call in collected_calls)
    overall_gb = sum(call[1] for call in collected_calls)
    cur_file = inspect.stack()[1].filename
    summary_str = (
        f"SUMMARY ({cur_file})\n"
        f"{overall_time:.2f}ms   \t {overall_gb:.2f} GB\t {overall_gb / (overall_time / 1e3):.2f}GB/s"
    )
    log.info(
        "%s",
        summary_str,
    )
    if output_file is not None:
        # sort perf numbers in descending order, i.e. placing the
        # most runtime-heavy kernels at the top of the list
        sorted_calls = sorted(collected_calls, key=lambda c: float(c[0]), reverse=True)
        try:
            with open(output_file, "a") as file:
                log.info(
                    "Save profile bandwidth results to %s",
                    output_file,
                )
                file.write("====================\n")
                file.write(f"TRITON KERNELS BANDWIDTH INFO ({cur_file})\n")
                for ms, num_gb, gb_per_s, kernel_name in sorted_calls:
                    # also display the runtime percentage for each kernel
                    percentage = f"{ms / overall_time * 100:.2f}%"
                    suffix = f" \t {percentage} \t {kernel_name}"
                    bw_info_str = create_bandwidth_info_str(
                        ms,
                        num_gb,
                        gb_per_s,
                        suffix=suffix,
                        color=False,
                    )
                    file.write(bw_info_str + "\n")
                file.write(f"{summary_str}\n\n")
        except Exception as e:
            log.warning(
                "failed to write profile bandwidth result into %s: %s",
                output_file,
                e,
            )


class DebugAutotuner(CachingAutotuner):
    def __init__(
        self,
        *args,
        regex_filter="",
        with_profiler=False,
        with_bandwidth_info=True,
        **kwargs,
    ):
        self.regex_filter = regex_filter
        self.with_profiler = with_profiler
        self.with_bandwidth_info = with_bandwidth_info
        super().__init__(*args, **kwargs)
        self.cached = None

    def run(self, *args, stream, **kwargs):
        if not self.with_bandwidth_info:
            super().run(*args, stream=stream, **kwargs, benchmark_run=True)
            return
        else:
            possible_names = _find_names(self)
            kernel_name = f"{max(possible_names, key=len)}"
            if not re.match(self.regex_filter, kernel_name):
                return

            if len(self.launchers) != 1:
                if len(self.launchers) == 0:
                    start_time = time.time_ns()
                    self.precompile()
                    self.precompile_time_taken_ns = time.time_ns() - start_time
                if len(self.launchers) > 1:
                    self.autotune_to_one_config(*args, **kwargs)
            (launcher,) = self.launchers

            if launcher.store_cubin:
                self.save_gpu_kernel(stream, launcher)

            if self.cached is None:
                ms = self.bench(launcher, *args, with_profiler=self.with_profiler)
                num_in_out_ptrs = len(
                    [
                        arg_name
                        for arg_name in self.fn.arg_names
                        if arg_name.startswith("in_out_ptr")
                    ]
                )
                num_gb = self.inductor_meta.get("kernel_num_gb", None)
                if num_gb is None:
                    num_gb = get_num_bytes(*args, num_in_out_args=num_in_out_ptrs) / 1e9
                gb_per_s = num_gb / (ms / 1e3)
                self.cached = ms, num_gb, gb_per_s, kernel_name
                collected_calls.append((ms, num_gb, gb_per_s, kernel_name))
                log.info(
                    "%s",
                    create_bandwidth_info_str(
                        ms, num_gb, gb_per_s, suffix=f" \t {kernel_name}"
                    ),
                )
            else:
                # in AOTI, we will call the kernel and its timing info has been cached already
                collected_calls.append(self.cached)


def hash_configs(configs: list[Config]):
    """
    Hash used to check for changes in configurations
    """
    hasher = hashlib.sha256()
    for cfg in configs:
        hasher.update(
            f"{sorted(cfg.kwargs.items())} {cfg.num_warps} {cfg.num_stages}\n".encode()
        )
    return hasher.hexdigest()


def cached_autotune(
    size_hints: Optional[list[int]],
    configs: list[Config],
    triton_meta,
    heuristic_type,
    filename=None,
    inductor_meta=None,
    custom_kernel=False,
):
    """
    A copy of triton.autotune that calls our subclass.  Our subclass
    has additional debugging, error handling, and on-disk caching.
    """
    configs = unique_configs(configs)
    assert len(configs) == 1 or filename
    inductor_meta = {} if inductor_meta is None else inductor_meta

    disabled = inductor_meta.get("force_disable_caches", False)

    # on disk caching logic and/or remote caching
    autotune_cache = None
    if (
        not disabled
        and filename is not None
        and (len(configs) > 1 or inductor_meta.get("coordinate_descent_tuning"))
        and not os.environ.get("TRITON_INTERPRET", "0") == "1"
    ):
        configs_hash = hash_configs(configs)

        autotune_cache = AutotuneCache.create(inductor_meta, filename, configs_hash)
        if autotune_cache:
            if best_config := autotune_cache.read_best(inductor_meta, configs):
                configs = [best_config]

    else:
        if disabled:
            log.debug("autotune caching is disabled by config.force_disable_caches")

    mutated_arg_names = inductor_meta.pop("mutated_arg_names", ())
    optimize_mem = inductor_meta.pop("optimize_mem", True)

    if "restore_value" in triton_meta:
        mutated_arg_names += triton_meta.pop("restore_value")

    reset_to_zero_arg_names: list[str] = []
    if "reset_to_zero" in triton_meta:
        reset_to_zero_arg_names.extend(triton_meta.pop("reset_to_zero"))

    def decorator(fn):
        # Remove XBLOCK from config if it's not a function argument.
        # This way, coordinate descent tuning will not try to tune it.
        #
        # Context: When TritonKernel.no_x_dim is True, we hardcode XBLOCK to 1.
        import inspect

        if "XBLOCK" not in inspect.signature(fn.fn).parameters:
            for tconfig in configs:
                if "XBLOCK" in tconfig.kwargs:
                    assert tconfig.kwargs["XBLOCK"] == 1
                    tconfig.kwargs.pop("XBLOCK")

        if inductor_meta.get("profile_bandwidth"):
            return DebugAutotuner(
                fn,
                triton_meta=triton_meta,
                inductor_meta=inductor_meta,
                regex_filter=inductor_meta["profile_bandwidth_regex"],
                with_profiler=inductor_meta[
                    "profile_bandwidth_with_do_bench_using_profiling"
                ],
                configs=configs,
                save_cache_hook=autotune_cache and autotune_cache.save,
                mutated_arg_names=mutated_arg_names,
                reset_to_zero_arg_names=reset_to_zero_arg_names,
                optimize_mem=optimize_mem,
                heuristic_type=heuristic_type,
                size_hints=size_hints,
                custom_kernel=custom_kernel,
                filename=filename,
                with_bandwidth_info=True,
            )
        return CachingAutotuner(
            fn,
            triton_meta=triton_meta,
            inductor_meta=inductor_meta,
            configs=configs,
            save_cache_hook=autotune_cache and autotune_cache.save,
            mutated_arg_names=mutated_arg_names,
            reset_to_zero_arg_names=reset_to_zero_arg_names,
            optimize_mem=optimize_mem,
            heuristic_type=heuristic_type,
            size_hints=size_hints,
            custom_kernel=custom_kernel,
            filename=filename,
        )

    return decorator


def unique_configs(configs: list[Config]):
    """Remove duplicate configurations"""
    seen: OrderedSet[Hashable] = OrderedSet()
    pruned_configs = []

    for cfg in configs:
        key = triton_config_to_hashable(cfg)
        if key not in seen:
            seen.add(key)
            pruned_configs.append(cfg)
    return pruned_configs


def check_config(cfg, *, xnumel=None, ynumel=None, znumel=None):
    for numel, label in zip((xnumel, ynumel, znumel), "XYZ"):
        if numel is None:
            continue
        block = cfg[f"{label}BLOCK"]
        if numel == 1:
            assert block == 1, (
                f"TritonKernel.indexing assumes numel == 1 => BLOCK == 1"
                f" but {label.lower()}numel=={numel} and {label}BLOCK={block} (cfg={cfg})."
            )
        max_block = TRITON_MAX_BLOCK[label]
        max_block_str = f'config.triton.max_block["{label}"]'
        assert max_block % block == 0, (
            f"TritonKernel.indexing assumes {label}BLOCK divides {max_block_str}"
            f" but {label}BLOCK={block} and {max_block_str}={max_block} (cfg={cfg})."
        )


def check_max_block(cfg: dict[str, int]):
    """
    Check that block sizes are within the maximum allowed.
    """
    for var, val in cfg.items():
        block_suffix = "BLOCK"
        if block_suffix in var:
            prefix = var.removesuffix(block_suffix)
            max_block = TRITON_MAX_BLOCK[prefix]
            assert val <= max_block, (
                f"'{var}' too large. Maximum: {max_block}. Actual: {val}."
            )


def _num_warps(num_warps, max_num_warps=8, min_num_warps=2, register_intensive=False):
    # On AMD GPU each warp has 64 lanes which is double the size on NV GPU,
    # therefore using half the number of warps here correspondingly.
    if torch.version.hip:
        max_num_warps = (max_num_warps + 1) // 2
        min_num_warps = (min_num_warps + 1) // 2
    # persistent reduction is register intensive
    if register_intensive:
        max_num_warps = max_num_warps // 2
    return next_power_of_2(min(max(num_warps, min_num_warps), max_num_warps))


def _check_max_grid_x(size_hints, x, num_warps):
    # Check if maxGridSize is exceeded - if so then must scale XBLOCK further
    max_grid_x = 2147483647
    warp_size = (
        64 if torch.version.hip else 32
    )  # TODO: query warp size once #129663 is merged
    num_blocks = (size_hints["x"] + x - 1) // x

    while (num_blocks * num_warps * warp_size) > max_grid_x and x < size_hints["x"]:
        x *= 2  # Scale up XBLOCK if grid exceeds limits
        num_blocks = num_blocks // 2
    if (num_blocks * num_warps * warp_size) > max_grid_x:
        raise AssertionError(
            "Reduction config exceeds cudaDeviceProp maxGridSize. Please raise a pytorch issue"
        )
    return x, num_blocks


def triton_config(
    size_hints,
    x,
    y=None,
    z=None,
    num_stages=1,
    num_elements_per_warp=256,
    min_elem_per_thread=0,
) -> Config:
    """
    Construct a pointwise triton config with some adjustment heuristics
    based on size_hints. Size_hints is a tuple of numels in each tile
    dimension and will be rounded up to the nearest power of 2.

    num_elements_per_warp is a suggestion for controlling how many warps
    the triton config should contain. e.g.: if x=16, y=8, z=4 then
    num_elements = 16*8*4 = 512. Then if we set num_elements_per_warp=128,
    we'll launch 512 (elem) / 128 (elem/warp) = 4 warps. Note that it's
    just a suggestion, and sometimes other adjustment heuristics will
    override the num_elements_per_warp.

    min_elem_per_thread controls the minimum number of elements
    processed by each thread. It's always enforced.
    """
    # Ideally we want to read this from some device config

    maxGridSize = [2147483647, 65535, 65535]

    target = conditional_product(x, y, z)
    if conditional_product(*size_hints.values()) < target:
        target //= 8

    # shrink sizes to size hints
    x = min(x, size_hints["x"])
    if y:
        y = min(y, size_hints["y"])
    if z:
        z = min(z, size_hints["z"])

    # if we are below original block size, scale up where we can;
    # or if the calculated grid size is larger than the limit, we bump up the corresponding dimension
    while x < min(size_hints["x"], TRITON_MAX_BLOCK["X"]) and (
        x * maxGridSize[0] < size_hints["x"] or conditional_product(x, y, z) < target
    ):
        x *= 2
    while (
        y
        and y < min(size_hints["y"], TRITON_MAX_BLOCK["Y"])
        and (
            y * maxGridSize[1] < size_hints["y"]
            or conditional_product(x, y, z) < target
        )
    ):
        y *= 2
    while (
        z
        and z < min(size_hints["z"], TRITON_MAX_BLOCK["Z"])
        and (
            z * maxGridSize[2] < size_hints["z"]
            or conditional_product(x, y, z) < target
        )
    ):
        z *= 2

    num_warps = _num_warps(
        conditional_product(x, y, z) // num_elements_per_warp, min_num_warps=1
    )
    # we are going to arrive at 2 warps only if bs was too small due to
    # numel being too small. However to workaround some ptx bugs we still
    # want at least 4 warps if there's enough elements per thread
    # given that this is a rare situation, don't expect this to affect perf
    # in general
    # see https://github.com/pytorch/pytorch/pull/97950
    if conditional_product(x, y, z) >= 128 and not torch.version.hip:
        num_warps = max(num_warps, 4)
    xnumel = size_hints["x"]
    ynumel = size_hints.get("y")
    znumel = size_hints.get("z")

    # Increase x to satisfy min_elem_per_thread requirements.
    block_size = max(
        conditional_product(x, y, z),
        min_elem_per_thread * _NUM_THREADS_PER_WARP * num_warps,
    )
    x *= math.ceil(block_size / conditional_product(x, y, z))

    x, _num_blocks = _check_max_grid_x(size_hints, x, num_warps)
    x = min(x, size_hints["x"])

    cfg = {"XBLOCK": x}
    if y:
        cfg["YBLOCK"] = y
    if z:
        cfg["ZBLOCK"] = z
    check_max_block(cfg)
    check_config(cfg, xnumel=xnumel, ynumel=ynumel, znumel=znumel)
    return Config(cfg, num_warps=num_warps, num_stages=num_stages)


def _get_nd_reduction_numels(r: int, size_hints: dict[str, int]) -> dict[str, int]:
    """
    Converts a linear reduction numel to ND, in row major order.
    This order is often desirable as it presents opportunities to coalesce memory
    accesses.
    For example, if r = 64 and size_hints = [32,32], this function returns [32, 2].
    This unraveling works because both r and size_hints are powers of 2.
    """
    # Shrink r to size_hints.
    r = min(r, get_total_reduction_numel(size_hints))
    num_reduction_dims = len(
        [prefix for prefix in size_hints if prefix_is_reduction(prefix)]
    )

    remaining = r
    rnumels = {}
    for idx in range(num_reduction_dims - 1, -1, -1):
        prefix = f"r{idx}_"
        max_size = min(size_hints[prefix], TRITON_MAX_BLOCK[prefix.upper()])
        dim = min(max_size, remaining)
        assert remaining % dim == 0, (
            f"Expected dimension '{dim}' to divide remaining size '{remaining}'"
        )
        rnumels[prefix] = dim
        remaining //= dim

    # Sanity check the results.
    final_numel = conditional_product(*rnumels.values())
    assert r == final_numel, (
        f"Expected ND reduction size ({rnumels}) to have {r} elements."
    )
    assert all(rnumels[prefix] <= size_hints[prefix] for prefix in rnumels), (
        f"rnumels exceed size_hints. {rnumels} > {size_hints}"
    )

    return rnumels


def triton_config_reduction(
    size_hints,
    x: int,
    r: int,
    num_stages=1,
    num_warps=None,
    register_intensive=False,
) -> Config:
    """
    Construct a reduction triton config with some adjustment heuristics
    based on size_hints. Size_hints is a tuple of numels in each tile
    dimension and will be rounded up to the nearest power of 2.
    """
    # Convert the linear reduction numel into a multi-dimensional block.
    rnumels = _get_nd_reduction_numels(r, size_hints)

    # shrink sizes to size hints
    x = min(x, size_hints["x"])

    def total_numel() -> int:
        return conditional_product(x, *rnumels.values())

    target = total_numel()
    if conditional_product(*size_hints.values()) < target:
        target //= 8

    # if we are below original block size, scale up where we can
    while x < size_hints["x"] and total_numel() < target:
        x *= 2
    for prefix in sorted(rnumels):
        while rnumels[prefix] < size_hints[prefix] and total_numel() < target:
            rnumels[prefix] *= 2

    if num_warps is None:
        num_warps = total_numel() // 128
    num_warps = _num_warps(
        num_warps, max_num_warps=16, register_intensive=register_intensive
    )

    x, _num_blocks = _check_max_grid_x(size_hints, x, num_warps)

    for prefix in sorted(rnumels):
        while total_numel() > target:
            if rnumels[prefix] == 1:
                break
            rnumels[prefix] //= 2

    cfg = _get_config({"x": x, **rnumels})
    check_max_block(cfg)
    check_config(cfg, xnumel=size_hints["x"])
    return Config(cfg, num_warps=num_warps, num_stages=num_stages)


def _get_config(numels: dict[str, int]) -> dict[str, int]:
    """
    Convert numels ("x", "r0_", etc.) to block sizes ("XBLOCK", "R0_BLOCK"), etc.
    """

    return {prefix.upper() + "BLOCK": numel for prefix, numel in numels.items()}


def triton_config_tiled_reduction(size_hints, x, y, r, num_stages=1):
    """
    Construct a tile reduction triton config with some adjustment
    heuristics based on size_hints. Size_hints is a tuple of numels in
    each tile dimension and will be rounded up to the nearest power of 2.
    """
    # Convert the linear reduction numel into a multi-dimensional block.
    rnumels = _get_nd_reduction_numels(r, size_hints)

    # shrink sizes to size hints
    x = min(x, size_hints["x"])
    y = min(y, size_hints["y"])

    def total_numel() -> int:
        return conditional_product(x, y, *rnumels.values())

    target = total_numel()
    if conditional_product(*size_hints.values()) < target:
        target //= 8

    # if we are below original block size, scale up where we can
    while x < size_hints["x"] and total_numel() < target:
        x *= 2
    for prefix in sorted(rnumels):
        while rnumels[prefix] < size_hints[prefix] and total_numel() < target:
            rnumels[prefix] *= 2
    while y < size_hints[1] and total_numel() < target:
        y *= 2

    cfg = _get_config({"x": x, "y": y, **rnumels})
    num_warps = _num_warps(total_numel() // 256, min_num_warps=1)
    check_config(cfg, xnumel=size_hints[0], ynumel=size_hints[1])
    check_max_block(cfg)
    return Config(cfg, num_warps=num_warps, num_stages=num_stages)


def pointwise(
    size_hints,
    triton_meta,
    tile_hint=None,
    filename=None,
    min_elem_per_thread=0,
    inductor_meta=None,
):
    """
    Construct @triton.heuristics() based on size_hints.
    """
    inductor_meta = {} if inductor_meta is None else inductor_meta
    assert not inductor_meta.get("no_x_dim")

    numel = functools.reduce(operator.mul, size_hints.values())
    bs = max(256, min(numel // 128, 1024))

    hinted_configs = autotune_hints_to_configs(
        inductor_meta.get("autotune_hints", OrderedSet()),
        size_hints,
        bs,
        triton_meta["device"],
    )

    triton_config_with_settings = functools.partial(
        triton_config, min_elem_per_thread=min_elem_per_thread
    )

    configs = None
    if len(size_hints) == 1:
        if disable_pointwise_autotuning(inductor_meta) and not (
            inductor_meta.get("max_autotune")
            or inductor_meta.get("max_autotune_pointwise")
        ):
            configs = [triton_config_with_settings(size_hints, bs)]
        else:
            configs = [
                triton_config_with_settings(size_hints, bs, num_elements_per_warp=256),
                triton_config_with_settings(
                    size_hints, bs // 2, num_elements_per_warp=64
                ),
                *hinted_configs,
            ]
    if len(size_hints) == 2:
        if (
            disable_pointwise_autotuning(inductor_meta) or tile_hint == TileHint.SQUARE
        ) and not (
            inductor_meta.get("max_autotune")
            or inductor_meta.get("max_autotune_pointwise")
        ):
            configs = [triton_config_with_settings(size_hints, 32, 32)]
        else:
            configs = [
                triton_config_with_settings(size_hints, 32, 32),
                triton_config_with_settings(size_hints, 64, 64),  # ~8% better for fp16
                triton_config_with_settings(size_hints, 256, 16),
                triton_config_with_settings(size_hints, 16, 256),
                triton_config_with_settings(size_hints, bs, 1),
                triton_config_with_settings(size_hints, 1, bs),
                *hinted_configs,
            ]
    if len(size_hints) == 3:
        if disable_pointwise_autotuning(inductor_meta):
            configs = [triton_config_with_settings(size_hints, 16, 16, 16)]
        else:
            configs = [
                triton_config_with_settings(size_hints, 16, 16, 16),
                triton_config_with_settings(size_hints, 64, 8, 8),
                triton_config_with_settings(size_hints, 8, 64, 8),
                triton_config_with_settings(size_hints, 8, 8, 64),
                triton_config_with_settings(size_hints, bs, 1, 1),
                triton_config_with_settings(size_hints, 1, bs, 1),
                triton_config_with_settings(size_hints, 1, 1, bs),
                *hinted_configs,
            ]

    if not configs:
        raise NotImplementedError(f"size_hints: {size_hints}")
    return cached_autotune(
        size_hints,
        configs,
        triton_meta=triton_meta,
        inductor_meta=inductor_meta,
        heuristic_type=HeuristicType.POINTWISE,
        filename=filename,
    )


def _reduction_configs(
    *, size_hints: dict[str, int], inductor_meta: dict[str, Any]
) -> list[Config]:
    reduction_hint = inductor_meta.get("reduction_hint", None)

    # Convert reductions to 1D, to simplify heuristics.
    rnumel = get_total_reduction_numel(size_hints)

    register_intensive = False
    MAX_R0_BLOCK = 2048
    if (
        size_hints["x"] >= 1024
        and inductor_meta.get("num_load", 0) + inductor_meta.get("num_reduction", 0)
        >= 10
    ):
        # A heuristics to reduce R0_BLOCK if a kernel potentially need many registers.
        # Consider load and reduction since load need move data into registers and
        # reduction needs an accumulator.
        #
        # The magic numbers are a bit arbitrary.
        #
        # We cannot rely on dynamically scaling down R0_BLOCK later, since sometimes
        # triton makes it to use less registers with worse perf. Check:
        # https://github.com/pytorch/pytorch/issues/126463
        #
        # The heuristic is a very simple one since registers can be reused. But
        # hopefully it can be a good enough indicator.
        MAX_R0_BLOCK = 1024
        register_intensive = True

    contiguous_config = triton_config_reduction(
        size_hints,
        1,
        rnumel if 256 <= rnumel < MAX_R0_BLOCK else MAX_R0_BLOCK,
        register_intensive=register_intensive,
    )
    outer_config = triton_config_reduction(
        size_hints, 64, 8, register_intensive=register_intensive
    )
    tiny_config = triton_config_reduction(
        size_hints,
        2 * (256 // rnumel) if rnumel <= 256 else 1,
        min(rnumel, MAX_R0_BLOCK),
        register_intensive=register_intensive,
    )
    if inductor_meta.get("max_autotune") or inductor_meta.get("max_autotune_pointwise"):
        pass  # skip all these cases
    elif reduction_hint == ReductionHint.INNER:
        return [contiguous_config]
    elif reduction_hint == ReductionHint.OUTER:
        return [outer_config]
    elif reduction_hint == ReductionHint.OUTER_TINY:
        return [tiny_config]
    if disable_pointwise_autotuning(inductor_meta):
        return [triton_config_reduction(size_hints, 32, 128)]
    return [
        contiguous_config,
        outer_config,
        tiny_config,
        triton_config_reduction(size_hints, 64, 64),
        triton_config_reduction(size_hints, 8, 512),
        # halve the XBLOCK/Rn_BLOCK compared to outer_config
        # TODO: this may only be beneficial when each iteration of the reduction
        # is quite heavy. E.g. https://gist.github.com/shunting314/189a8ef69f90db9d614a823385147a72
        triton_config_reduction(size_hints, 64, 4, num_warps=8),
    ]


def reduction(
    size_hints,
    reduction_hint=False,
    triton_meta=None,
    filename=None,
    inductor_meta=None,
):
    """args to @triton.heuristics()"""
    inductor_meta = {} if inductor_meta is None else inductor_meta
    inductor_meta["reduction_hint"] = reduction_hint
    if inductor_meta.get("no_x_dim"):
        size_hints["x"] = 1

    assert triton_meta is not None

    configs = _reduction_configs(size_hints=size_hints, inductor_meta=inductor_meta)
    return cached_autotune(
        size_hints,
        configs=configs,
        triton_meta=triton_meta,
        inductor_meta=inductor_meta,
        heuristic_type=HeuristicType.REDUCTION,
        filename=filename,
    )


def cooperative_reduction(
    size_hints,
    reduction_hint,
    triton_meta,
    filename,
    inductor_meta,
):
    inductor_meta = {} if inductor_meta is None else inductor_meta
    inductor_meta["reduction_hint"] = reduction_hint
    if inductor_meta.get("no_x_dim"):
        size_hints["x"] = 1

    # Cooperative reductions currently only support a single reduction dimension.
    assert len(size_hints) == 2, (
        "Cooperative reductions don't support tiling reduction dims"
    )
    xnumel, rnumel = size_hints["x"], size_hints["r0_"]

    # TODO(jansel): we should base target on the SM count of the local GPU
    target = 64
    split = max(1, min(target // xnumel, TRITON_MAX_RSPLIT))
    assert rnumel >= split
    assert split <= TRITON_MAX_RSPLIT
    if inductor_meta["persistent_reduction"]:
        configs = _persistent_reduction_configs(
            {"x": xnumel, "r0_": rnumel // split}, reduction_hint, inductor_meta
        )
    else:
        configs = _reduction_configs(
            size_hints={"x": xnumel, "r0_": rnumel // split},
            inductor_meta=inductor_meta,
        )
    for config in configs:
        config.kwargs["RSPLIT"] = split
    # TODO(jansel): add more configs in max_autotune

    return cached_autotune(
        size_hints,
        configs=configs,
        triton_meta=triton_meta,
        inductor_meta=inductor_meta,
        heuristic_type=HeuristicType.REDUCTION,
        filename=filename,
    )


def _persistent_reduction_configs(
    size_hints,
    reduction_hint=False,
    inductor_meta=None,
):
    xnumel = size_hints["x"]
    rnumel = get_total_reduction_numel(size_hints)

    configs = [
        triton_config_reduction(size_hints, xblock, rnumel, register_intensive=True)
        for xblock in (1, 8, 32, 128)
        if xblock == 1 or (rnumel * xblock <= 4096 and xblock <= xnumel)
    ]

    # TODO(jansel): we should be able to improve these heuristics
    if reduction_hint == ReductionHint.INNER and rnumel >= 256:
        configs = configs[:1]
    elif reduction_hint == ReductionHint.OUTER:
        configs = configs[-1:]
    elif reduction_hint == ReductionHint.OUTER_TINY:
        configs = [
            triton_config_reduction(
                size_hints,
                2 * (256 // rnumel) if rnumel <= 256 else 1,
                rnumel,
            )
        ]
    for c in configs:
        # we don't need Rn_BLOCK for persistent reduction
        for prefix in size_hints:
            if prefix_is_reduction(prefix):
                c.kwargs.pop(f"{prefix.upper()}BLOCK")

    if disable_pointwise_autotuning(inductor_meta):
        configs = configs[:1]

    return configs


def persistent_reduction(
    size_hints,
    reduction_hint=False,
    triton_meta=None,
    filename=None,
    inductor_meta=None,
):
    inductor_meta = {} if inductor_meta is None else inductor_meta
    inductor_meta["reduction_hint"] = reduction_hint
    if inductor_meta.get("no_x_dim"):
        size_hints["x"] = 1

    configs = _persistent_reduction_configs(size_hints, reduction_hint, inductor_meta)

    return cached_autotune(
        size_hints,
        configs,
        triton_meta=triton_meta,
        inductor_meta=inductor_meta,
        filename=filename,
        heuristic_type=HeuristicType.PERSISTENT_REDUCTION,
    )


def split_scan(
    size_hints,
    reduction_hint=False,
    triton_meta=None,
    filename=None,
    inductor_meta=None,
):
    """Heuristic for TritonSplitScanKernel"""
    inductor_meta = {} if inductor_meta is None else inductor_meta
    inductor_meta["reduction_hint"] = reduction_hint
    if inductor_meta.get("no_x_dim"):
        size_hints["x"] = 1

    assert triton_meta is not None
    if len(size_hints) != 2:
        raise NotImplementedError(f"size_hints: {size_hints}")

    configs = _reduction_configs(size_hints=size_hints, inductor_meta=inductor_meta)

    # Fixup configs to enforce the minimum Rn_BLOCK size
    min_rblock = inductor_meta.get("min_split_scan_rblock", 256)
    for cfg in configs:
        for var in list(cfg.kwargs.keys()):
            if var.startswith("R") and cfg.kwargs[var] < min_rblock:
                cfg.kwargs[var] = min_rblock

    return cached_autotune(
        size_hints,
        configs=configs,
        triton_meta=triton_meta,
        inductor_meta=inductor_meta,
        heuristic_type=HeuristicType.SPLIT_SCAN,
        filename=filename,
    )


def template(num_stages, num_warps, triton_meta, filename=None, inductor_meta=None):
    """
    Compile a triton template
    """
    return cached_autotune(
        None,
        [triton.Config({}, num_stages=num_stages, num_warps=num_warps)],
        triton_meta=triton_meta,
        inductor_meta=inductor_meta,
        heuristic_type=HeuristicType.TEMPLATE,
        filename=filename,
    )


def _pop_config_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    """Extract triton.Config options that should become kwargs"""
    popped = {}
    for key in ("num_warps", "num_stages", "num_ctas", "maxnreg"):
        val = config.pop(key, None)
        if val is not None:
            popped[key] = val
    return popped


def config_to_dict(config: Config) -> dict[str, Any]:
    return {
        **config.kwargs,
        "num_warps": config.num_warps,
        "num_stages": config.num_stages,
    }


def config_from_dict(config: dict[str, Any]) -> Config:
    config = {**config}
    return Config(config, **_pop_config_kwargs(config))


def fixed_config(config, filename, triton_meta, inductor_meta):
    """
    Used when the configuration is already decided at compile time
    """
    config = {**config}
    return cached_autotune(
        None,
        [triton.Config(config, **_pop_config_kwargs(config))],
        triton_meta=triton_meta,
        inductor_meta=inductor_meta,
        heuristic_type=HeuristicType.FIXED,
        filename=filename,
    )


def user_autotune(
    configs, triton_meta, filename=None, inductor_meta=None, custom_kernel=False
):
    """
    Compile a user defined triton kernel
    """
    if len(configs) == 0:
        configs = [triton.Config({})]
    else:
        configs = [*map(config_from_dict, configs)]
    return cached_autotune(
        None,
        configs,
        triton_meta=triton_meta,
        heuristic_type=HeuristicType.USER_AUTOTUNE,
        filename=filename,
        inductor_meta=inductor_meta,
        custom_kernel=custom_kernel,
    )


def foreach(triton_meta, num_warps, filename=None, inductor_meta=None):
    """
    Compile a triton foreach kernel
    """
    return cached_autotune(
        None,
        [triton.Config({}, num_stages=1, num_warps=num_warps)],
        triton_meta=triton_meta,
        inductor_meta=inductor_meta,
        heuristic_type=HeuristicType.TEMPLATE,
        filename=filename,
    )


@dataclasses.dataclass
class GridExpr:
    """Generate code for grid size expressions in launcher"""

    inductor_meta: dict[str, Any]
    mode: Literal["python", "cpp"] = "python"
    prefix: Sequence[str] = ()
    x_grid: Union[str, int] = 1
    y_grid: Union[str, int] = 1
    z_grid: Union[str, int] = 1

    def __post_init__(self) -> None:
        assert self.mode in ("python", "cpp")

    def generate(self, meta: dict[str, int]) -> None:
        raise NotImplementedError

    def ceildiv(
        self, numel: Union[str, int], block: Union[None, int, str]
    ) -> Union[str, int]:
        if block is None or block == 1:
            return numel
        if isinstance(numel, int) and isinstance(block, int):
            return ceildiv(numel, block)  # constant fold
        if self.mode == "python":
            return f"-(({numel}) // -({block}))"
        # trick above doesn't work in C++ due to rounding differences
        return f"(({numel} + ({block} - 1)) / ({block}))"

    def maximum(self, seq: list[Union[int, str]]) -> Union[int, str]:
        """Codegen for max function with constant folding, constants are represented as int"""
        items = self._constant_fold(max, seq)
        if len(items) <= 1:
            return items[0]
        if self.mode == "python":
            return f"max({', '.join(map(str, items))})"
        return functools.reduce(lambda x, y: f"std::max({x}, {y})", items)

    def summation(self, seq: list[Union[int, str]]) -> Union[int, str]:
        """Codegen for sum function with constant folding, constants are represented as int"""
        items = self._constant_fold(sum, seq)
        if len(items) <= 1:
            return items[0]
        return " + ".join(map(str, items))

    def _constant_fold(
        self, fn: Callable[[list[int]], int], seq: list[Union[int, str]]
    ) -> list[Union[int, str]]:
        """Constant fold through a commutative fn where ints are constants"""
        items: list[Union[int, str]] = [x for x in seq if not isinstance(x, int)]
        const_items = [x for x in seq if isinstance(x, int)]
        if const_items:
            items.append(fn(const_items))
        return items

    def assign_tmp(self, name: str, expr: Union[str, int]) -> str:
        # Grid functions are one per kernel, so name collisions are fine
        if self.mode == "python":
            return f"{name} = {expr}"
        if self.mode == "cpp":
            return f"uint32_t {name} = {expr};"
        raise AssertionError(f"invalid mode {self.mode}")

    @staticmethod
    def from_meta(
        inductor_meta: dict[str, Any],
        cfg: Union[Config, dict[str, int]],
        mode: Literal["python", "cpp"] = "python",
    ) -> GridExpr:
        grid_cls = globals()[inductor_meta["grid_type"]]
        assert issubclass(grid_cls, GridExpr)
        grid = grid_cls(inductor_meta=inductor_meta, mode=mode)
        if isinstance(cfg, Config):
            cfg = config_to_dict(cfg)
        grid.generate(cfg)
        return grid

    def eval_slow(self, meta: dict[str, int]) -> tuple[int, int, int]:
        scope = {**meta}
        for line in self.prefix:
            exec(line, scope)
        exec(f"grid_0 = {self.x_grid}", scope)
        exec(f"grid_1 = {self.y_grid}", scope)
        exec(f"grid_2 = {self.z_grid}", scope)
        return scope["grid_0"], scope["grid_1"], scope["grid_2"]


class Grid1D(GridExpr):
    def generate(self, meta: dict[str, int]) -> None:
        self.x_grid = self.ceildiv("xnumel", meta.get("XBLOCK"))


class Grid2D(GridExpr):
    def generate(self, meta: dict[str, int]) -> None:
        self.x_grid = self.ceildiv("xnumel", meta.get("XBLOCK"))
        self.y_grid = self.ceildiv("ynumel", meta.get("YBLOCK"))


class Grid3D(GridExpr):
    def generate(self, meta: dict[str, int]) -> None:
        self.x_grid = self.ceildiv("xnumel", meta.get("XBLOCK"))
        self.y_grid = self.ceildiv("ynumel", meta.get("YBLOCK"))
        self.z_grid = self.ceildiv("znumel", meta.get("ZBLOCK"))


class Grid2DWithYZOverflow(GridExpr):
    def generate(self, meta: dict[str, int]) -> None:
        self.x_grid = self.ceildiv("xnumel", meta.get("XBLOCK"))
        self.prefix = [
            self.assign_tmp("y_grid_raw_", self.ceildiv("ynumel", meta.get("YBLOCK"))),
            self.assign_tmp(
                "y_grid_div_", self.ceildiv("y_grid_raw_", get_max_y_grid())
            ),
        ]
        self.y_grid = self.ceildiv("y_grid_raw_", "y_grid_div_")
        self.z_grid = "y_grid_div_"


class CooperativeReductionGrid(GridExpr):
    def generate(self, meta: dict[str, int]) -> None:
        self.x_grid = str(meta["RSPLIT"])
        self.y_grid = self.ceildiv("xnumel", meta.get("XBLOCK"))


class SplitScanGrid(GridExpr):
    def generate(self, meta: dict[str, int]) -> None:
        assert meta.get("XBLOCK", 1) == 1
        self.x_grid = self.ceildiv("r0_numel", meta.get("R0_BLOCK"))
        self.y_grid = "xnumel"


class FixedGrid(GridExpr):
    @staticmethod
    def setup_grid_as_args() -> dict[str, Any]:
        """Inductor meta so the launcher takes three extra grid arguments"""
        return {
            "grid_type": FixedGrid.__name__,
            "fixed_grid": ["_grid_0", "_grid_1", "_grid_2"],
            "extra_launcher_args": ["_grid_0", "_grid_1", "_grid_2"],
        }

    def generate(self, meta: dict[str, int]) -> None:
        self.x_grid, self.y_grid, self.z_grid = self.inductor_meta["fixed_grid"]


class PrecomputedGrid(GridExpr):
    def generate(self, meta: dict[str, int]) -> None:
        for candidate in self.inductor_meta["precomputed_grids"]:
            if all(meta.get(k) == v for k, v in candidate["config"].items()):
                self.x_grid, self.y_grid, self.z_grid = candidate[self.mode]
                return
        raise AssertionError(
            f"Precomputed grid not found for {meta} in {self.inductor_meta['precomputed_grids']}"
        )


class ComboKernelGrid(GridExpr):
    def generate(self, meta: dict[str, int]):
        combo_meta = self.inductor_meta["combo_grid_meta"]
        if combo_meta["default_config"]:
            meta = {**combo_meta["default_config"], **meta}
        no_x_dims = []
        xnumels = []
        ynumels = []
        znumels = []
        for num in range(combo_meta["num_kernels"]):
            assert (
                combo_meta[f"xnumel_{num}"] is None or combo_meta[f"xnumel_{num}"] > 0
            )
            no_x_dims.append(combo_meta[f"no_x_dim_{num}"])
            xnumels.append(combo_meta[f"xnumel_{num}"] or f"xnumel_{num}")
            if f"ynumel_{num}" in combo_meta:
                ynumels.append(combo_meta[f"ynumel_{num}"] or f"ynumel_{num}")
            if f"znumel_{num}" in combo_meta:
                znumels.append(combo_meta[f"znumel_{num}"] or f"znumel_{num}")

        self.x_grid = self.combo_x_grid(xnumels, no_x_dims, meta)
        if combo_meta["min_blocks"]:
            self.x_grid = self.maximum([self.x_grid, combo_meta["min_blocks"]])
        if ynumels:
            self.y_grid = self.ceildiv(self.maximum(ynumels), meta.get("YBLOCK"))
        if znumels:
            self.z_grid = self.ceildiv(self.maximum(znumels), meta.get("ZBLOCK"))

    def combo_x_grid(
        self,
        xnumels: list[Union[int, str]],
        no_x_dims: list[bool],
        meta: dict[str, int],
    ) -> Union[str, int]:
        raise NotImplementedError


class SequentialComboKernelGrid(ComboKernelGrid):
    def combo_x_grid(
        self,
        xnumels: list[Union[int, str]],
        no_x_dims: list[bool],
        meta: dict[str, int],
    ) -> Union[str, int]:
        assert len(xnumels) == len(no_x_dims)
        return self.summation(
            [
                self.ceildiv(x, 1 if no_x_dim else meta.get("XBLOCK"))
                for x, no_x_dim in zip(xnumels, no_x_dims)
            ]
        )


class RoundRobinComboKernelGrid(ComboKernelGrid):
    def combo_x_grid(
        self,
        xnumels: list[Union[int, str]],
        no_x_dims: list[bool],
        meta: dict[str, int],
    ) -> str:
        assert len(xnumels) == len(no_x_dims)
        num_kernels = self.inductor_meta["combo_grid_meta"]["num_kernels"]
        exprs = [x for x, no_x_dim in zip(xnumels, no_x_dims) if no_x_dim]
        xnumels_x_dim = [x for x, no_x_dim in zip(xnumels, no_x_dims) if not no_x_dim]
        if xnumels_x_dim:
            exprs.append(self.ceildiv(self.maximum(xnumels_x_dim), meta.get("XBLOCK")))
        return f"({self.maximum(exprs)}) * {num_kernels}"
