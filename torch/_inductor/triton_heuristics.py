import builtins
import copy
import functools
import hashlib
import inspect
import json
import logging
import operator
import os
import os.path
import re
import threading
from enum import auto, Enum
from typing import Any, Callable, List, Optional, Set, Tuple

import torch
from torch._dynamo.utils import dynamo_timed

from . import config
from .codecache import cache_dir, CudaKernelParamCache
from .coordinate_descent_tuner import CoordescTuner

from .ir import ReductionHint, TileHint
from .utils import (
    ceildiv,
    conditional_product,
    create_bandwidth_info_str,
    do_bench,
    get_num_bytes,
    has_triton,
    next_power_of_2,
    triton_config_to_hashable,
)


log = logging.getLogger(__name__)

if has_triton():
    import triton
    from triton import Config
    from triton.runtime.jit import get_cuda_stream, KernelInterface
else:
    Config = object
    get_cuda_stream = None
    KernelInterface = object
    triton = None


class HeuristicType(Enum):
    POINTWISE = auto()
    REDUCTION = auto()
    PERSISTENT_REDUCTION = auto()
    TEMPLATE = auto()


class AutotuneHint(Enum):
    ELEMENTS_PER_WARP_32 = 0

    # Triton codegen tries to codegen set of AutotuneHints.
    # Enum.__repr__ looks like "<AutotuneHint.ELEMENTS_PER_WARP_32: 0>""
    # which isn't valid python.
    # Enum.__str__ will just return "AutotuneHint.ELEMENTS_PER_WARP_32".
    __repr__ = Enum.__str__


def autotune_hints_to_configs(
    hints: Set[AutotuneHint], size_hints, block_size
) -> List[Config]:
    """
    AutotuneHints can be attached to the metadata of triton kernels for providing
    suggestions about what to try for autotuning. One reason to do this is if there are
    some configs that are only useful in specific scenarios, in which case we can avoid
    wasting compile time on autotuning unless we know we are in one of those scenarios.

    Based on those hints, this function will generate a list of additional autotuning
    configs to try.
    """
    xyz_options: Tuple[Tuple[Any, ...], ...]
    configs = []

    for hint in hints:
        if hint == AutotuneHint.ELEMENTS_PER_WARP_32:
            if len(size_hints) == 1:
                xyz_options = ((block_size // 4,),)
            elif len(size_hints) == 2:
                xyz_options = ((block_size // 4, 1), (1, block_size // 4))
            elif len(size_hints) == 3:
                xyz_options = (
                    (block_size // 4, 1, 1),
                    (1, block_size // 4, 1),
                    (1, 1, block_size // 4),
                )
            for xyz in xyz_options:
                configs.append(
                    triton_config(  # type: ignore[misc]
                        size_hints,
                        *xyz,
                        num_elements_per_warp=32,
                    )
                )

    return configs


def disable_pointwise_autotuning():
    # Autotuning can give different benchmarking results from run to run, and
    # therefore we disable autotuning when use_deterministic flag is on.
    if torch.are_deterministic_algorithms_enabled():
        return True
    return not config.triton.autotune_pointwise


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
        meta,
        configs,
        save_cache_hook,
        mutated_arg_names,
        heuristic_type,
        size_hints=None,
    ):
        super().__init__()
        self.fn = fn
        self.meta = meta
        self.save_cache_hook = save_cache_hook
        self.mutated_arg_names = mutated_arg_names
        self.configs = configs
        self.heuristic_type = heuristic_type

        if log.isEnabledFor(logging.DEBUG):
            log.debug("CachingAutotuner gets %d configs", len(self.configs))
            for c in self.configs:
                log.debug(c)

        self.launchers = []
        self.lock = threading.Lock()
        if os.getenv("TRITON_CACHE_DIR") is None:
            os.environ["TRITON_CACHE_DIR"] = os.path.join(
                cache_dir(),
                "triton",
                str(self.meta.get("device", 0)),
            )

        self.coordesc_tuner = CoordescTuner(
            is_mm=False, name=self.fn.__name__, size_hints=size_hints
        )

    def precompile(self, warm_cache_only_with_cc=None):
        with self.lock:
            if self.launchers:
                return
            self.launchers = [
                self._precompile_config(c, warm_cache_only_with_cc)
                for c in self.configs
            ]
            self.configs = None

    def _precompile_config(self, cfg: Config, warm_cache_only_with_cc: Optional[int]):
        """Ahead of time compile a given autotuner config."""
        compile_meta = copy.deepcopy(self.meta)
        for k, v in cfg.kwargs.items():
            compile_meta["constants"][self.fn.arg_names.index(k)] = v
        compile_meta["num_warps"] = cfg.num_warps
        compile_meta["num_stages"] = cfg.num_stages
        compile_meta["debug"] = (
            config.triton.assert_indirect_indexing and torch.version.hip is None
        )

        if warm_cache_only_with_cc:
            triton.compile(
                self.fn,
                warm_cache_only=True,
                cc=warm_cache_only_with_cc,
                **compile_meta,
            )
            return

        # load binary to the correct device
        with torch.cuda.device(compile_meta["device"]):
            # need to initialize context
            torch.cuda.synchronize(torch.cuda.current_device())
            binary = triton.compile(
                self.fn,
                **compile_meta,
            )
            binary._init_handles()

        call_args = [
            arg
            for i, arg in enumerate(self.fn.arg_names)
            if i not in self.fn.constexprs
        ]
        def_args = list(self.fn.arg_names)
        while def_args and def_args[-1] in cfg.kwargs:
            def_args.pop()

        scope = {
            "grid_meta": cfg.kwargs,
            "bin": binary,
            "torch": torch,
            "set_device": torch.cuda.set_device,
            "current_device": torch.cuda.current_device,
        }
        exec(
            f"""
            def launcher({', '.join(def_args)}, grid, stream):
                if callable(grid):
                    grid_0, grid_1, grid_2 = grid(grid_meta)
                else:
                    grid_0, grid_1, grid_2 = grid
                bin.c_wrapper(grid_0, grid_1, grid_2, bin.num_warps, bin.shared,
                            stream, bin.cu_function, None, None, None,
                            {', '.join(call_args)})
            """.lstrip(),
            scope,
        )

        launcher = scope["launcher"]
        launcher.config = cfg
        launcher.n_regs = getattr(binary, "n_regs", None)
        launcher.n_spills = getattr(binary, "n_spills", None)
        launcher.shared = getattr(binary, "shared", None)
        launcher.store_cubin = config.triton.store_cubin
        # store this global varible to avoid the high overhead of reading it when calling run
        if launcher.store_cubin:
            launcher.fn = self.fn
            launcher.bin = binary

        return launcher

    def bench(self, launcher, *args, grid):
        """Measure the performance of a given launcher"""
        if launcher.n_spills > config.triton.spill_threshold:
            log.debug(
                "Skip config %s because of register spilling: %d",
                launcher.config,
                launcher.n_spills,
            )
            return float("inf")

        stream = get_cuda_stream(torch.cuda.current_device())

        def kernel_call():
            if launcher.config.pre_hook is not None:
                launcher.config.pre_hook(
                    {**dict(zip(self.arg_names, args)), **launcher.config.kwargs}
                )

            cloned_args = self.clone_args(*args)
            launcher(
                *cloned_args,
                grid=grid,
                stream=stream,
            )

        return do_bench(kernel_call, rep=40, fast_flush=True)

    def clone_args(self, *args):
        from .compile_fx import clone_preserve_strides

        # clone inplace buffers to avoid autotune contaminating them if
        # the kernel does in-place stores. avoid cloning other buffers because
        # it leads to increase memory use
        cloned_args = []
        for i, arg in enumerate(args):
            if self.fn.arg_names[i] in self.mutated_arg_names:
                assert isinstance(arg, torch.Tensor)
                cloned_args.append(clone_preserve_strides(arg))
            else:
                cloned_args.append(arg)

        return cloned_args

    @dynamo_timed
    def benchmark_all_configs(self, *args, **kwargs):
        timings = {
            launcher: self.bench(launcher, *args, **kwargs)
            for launcher in self.launchers
        }

        for k, v in timings.items():
            self.coordesc_tuner.cache_benchmark_result(k.config, v)

        if log.isEnabledFor(logging.DEBUG):
            log.debug("Benchmark all input configs get:")
            for k, v in timings.items():
                log.debug(
                    "%s: %f, nreg %d, nspill %d, #shared-mem %d",
                    k.config,
                    v,
                    k.n_regs,
                    k.n_spills,
                    k.shared,
                )

        return timings

    def autotune_to_one_config(self, *args, **kwargs):
        """Do the actual autotuning"""
        timings = self.benchmark_all_configs(*args, **kwargs)
        self.launchers = [builtins.min(timings, key=timings.get)]
        if self.save_cache_hook:
            self.save_cache_hook(self.launchers[0].config)

    def save_cuda_kernel(self, grid, stream, launcher):
        if callable(grid):
            grid_x, grid_y, grid_z = grid(launcher.config.kwargs)
        else:
            grid_x, grid_y, grid_z = grid

        key = launcher.fn.fn.__qualname__  # unique kernel name
        params = {
            "mangled_name": launcher.bin.metadata["name"],
            "grid_x": grid_x,
            "grid_y": grid_y,
            "grid_z": grid_z,
            "num_warps": launcher.bin.num_warps,
            "shared_mem": launcher.bin.shared,
            "stream": stream,
        }
        CudaKernelParamCache.set(key, params, launcher.bin.asm["cubin"])

    def coordinate_descent_tuning(self, launcher, *args, **kwargs):
        """
        Coordinate descent tuning can be run with or without max-autotune.

        The only difference between these two is the starting config for coordinate_descent tuning.
        E.g., assuming regular autotune only get one config C1; while max-autotune get 4 configs C1, C2, C3, C4
        and max-autotune figure out C3 is the best.

        Then if coordinate descnt tuning is run with max-autotune disabled, it will start from C1;
        while if coordinate descent tuning is run with max-autotune enabled, it will start from C3.
        """
        if self.heuristic_type == HeuristicType.TEMPLATE:
            # skip triton template
            return launcher

        cloned_args = self.clone_args(*args)
        config2launcher = {launcher.config: launcher}

        def benchmark_one_config(config):
            with self.lock:
                launcher = self._precompile_config(config, None)
            config2launcher[config] = launcher

            out = self.bench(launcher, *cloned_args, **kwargs)
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
            and "RBLOCK" in launcher.config.kwargs
        ), "Coordinate descent tuner relies on the assumption that persistent reduction's triton config does not have RBLOCK"
        best_config = self.coordesc_tuner.autotune(
            benchmark_one_config, launcher.config, None
        )
        best_config.found_by_coordesc = True

        if self.save_cache_hook:
            self.save_cache_hook(best_config, found_by_coordesc=True)
        return config2launcher.get(best_config)

    def run(self, *args, grid, stream):
        if len(self.launchers) != 1:
            if len(self.launchers) == 0:
                self.precompile()
            if len(self.launchers) > 1:
                self.autotune_to_one_config(*args, grid=grid)

        if (
            not getattr(self.launchers[0].config, "found_by_coordesc", False)
            and config.coordinate_descent_tuning
        ):
            self.launchers = [
                self.coordinate_descent_tuning(self.launchers[0], *args, grid=grid)
            ]

        (launcher,) = self.launchers
        if launcher.store_cubin:
            self.save_cuda_kernel(grid, stream, launcher)

        if launcher.config.pre_hook is not None:
            launcher.config.pre_hook(
                {**dict(zip(self.arg_names, args)), **launcher.config.kwargs}
            )
        return launcher(
            *args,
            grid=grid,
            stream=stream,
        )


def _find_names(obj):
    import gc
    import inspect

    frame = inspect.currentframe()
    for frame in iter(lambda: frame.f_back, None):  # type: ignore[union-attr]
        frame.f_locals
    obj_names = []
    for referrer in gc.get_referrers(obj):
        if isinstance(referrer, dict):
            for k, v in referrer.items():
                if v is obj:
                    obj_names.append(k)
    return obj_names


collected_calls: List[Any] = []


def start_graph():
    collected_calls.clear()


def end_graph():
    if len(collected_calls) == 0:
        return
    overall_time = sum(call[0] for call in collected_calls)
    overall_gb = sum(call[1] for call in collected_calls)
    cur_file = inspect.stack()[1].filename
    print(f"SUMMARY ({cur_file})")
    print(
        f"{overall_time:.2f}ms   \t {overall_gb:.2f} GB\t {overall_gb/(overall_time/1e3):.2f}GB/s"
    )
    print()


class DebugAutotuner(CachingAutotuner):
    def __init__(self, *args, regex_filter="", **kwargs):
        self.regex_filter = regex_filter
        super().__init__(*args, **kwargs)
        self.cached = None

    def run(self, *args, grid, stream):
        possible_names = _find_names(self)
        kernel_name = f"{max(possible_names, key=lambda x: len(x))}"
        if not re.match(self.regex_filter, kernel_name):
            return
        super().run(*args, grid=grid, stream=stream)
        (launcher,) = self.launchers

        if self.cached is None:
            ms = self.bench(launcher, *args, grid=grid)
            num_in_out_ptrs = len(
                [
                    arg_name
                    for arg_name in self.fn.arg_names
                    if arg_name.startswith("in_out_ptr")
                ]
            )
            num_gb = get_num_bytes(*args, num_in_out_args=num_in_out_ptrs) / 1e9
            gb_per_s = num_gb / (ms / 1e3)
            self.cached = (ms, num_gb, gb_per_s, kernel_name)
        else:
            ms, num_gb, gb_per_s, kernel_name = self.cached
        collected_calls.append((ms, num_gb, gb_per_s, kernel_name))
        print(
            create_bandwidth_info_str(ms, num_gb, gb_per_s, suffix=f" \t {kernel_name}")
        )


def hash_configs(configs: List[Config]):
    """
    Hash used to check for changes in configurations
    """
    hasher = hashlib.sha256()
    for cfg in configs:
        hasher.update(
            f"{sorted(cfg.kwargs.items())} {cfg.num_warps} {cfg.num_stages}\n".encode()
        )
    return hasher.hexdigest()


def load_cached_autotuning(
    cache_filename: str, configs_hash: str, configs: List[Config]
):
    """
    Read a cached autotuning result from disk
    """
    if not os.path.exists(cache_filename):
        return None

    with open(cache_filename) as fd:
        best_config = json.loads(fd.read())
    if best_config.pop("configs_hash", None) != configs_hash:
        return None

    if config.coordinate_descent_tuning and best_config.pop("found_by_coordesc", False):
        num_warps = best_config.pop("num_warps")
        num_stages = best_config.pop("num_stages")
        triton_config = Config(best_config, num_warps=num_warps, num_stages=num_stages)
        triton_config.found_by_coordesc = True
        return triton_config

    matching_configs = [
        cfg
        for cfg in configs
        if all(val == best_config.get(key) for key, val in cfg.kwargs.items())
        and cfg.num_warps == best_config.get("num_warps")
        and cfg.num_stages == best_config.get("num_stages")
    ]
    if len(matching_configs) != 1:
        return None

    return matching_configs[0]


def cached_autotune(
    size_hints: Optional[List[int]],
    configs: List[Config],
    meta,
    heuristic_type,
    filename=None,
):
    """
    A copy of triton.autotune that calls our subclass.  Our subclass
    has additional debugging, error handling, and on-disk caching.
    """
    configs = unique_configs(configs)
    assert len(configs) == 1 or filename
    save_cache_hook: Optional[Callable[[Any, Any], Any]]

    # on disk caching logic
    if filename is not None and (len(configs) > 1 or config.coordinate_descent_tuning):
        cache_filename = os.path.splitext(filename)[0] + ".best_config"
        configs_hash = hash_configs(configs)
        best_config = load_cached_autotuning(cache_filename, configs_hash, configs)
        if best_config:
            configs = [best_config]

        def save_cache_hook(cfg, found_by_coordesc=False):
            with open(cache_filename, "w") as fd:
                fd.write(
                    json.dumps(
                        {
                            **cfg.kwargs,
                            "num_warps": cfg.num_warps,
                            "num_stages": cfg.num_stages,
                            "configs_hash": configs_hash,
                            "found_by_coordesc": found_by_coordesc,
                        }
                    )
                )
            if log.isEnabledFor(logging.DEBUG):
                type_str = "coordesc" if found_by_coordesc else "heuristic"
                log.debug("Save %s tuning result to %s", type_str, cache_filename)

    else:
        save_cache_hook = None

    mutated_arg_names = meta.pop("mutated_arg_names", ())

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

        if config.profile_bandwidth:
            return DebugAutotuner(
                fn,
                meta=meta,
                regex_filter=config.profile_bandwidth_regex,
                configs=configs,
                save_cache_hook=save_cache_hook,
                mutated_arg_names=mutated_arg_names,
                heuristic_type=heuristic_type,
                size_hints=size_hints,
            )
        return CachingAutotuner(
            fn,
            meta=meta,
            configs=configs,
            save_cache_hook=save_cache_hook,
            mutated_arg_names=mutated_arg_names,
            heuristic_type=heuristic_type,
            size_hints=size_hints,
        )

    return decorator


def unique_configs(configs: List[Config]):
    """Remove duplicate configurations"""
    seen = set()
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
        max_block = config.triton.max_block[label]
        max_block_str = f'config.triton.max_block["{label}"]'
        assert max_block % block == 0, (
            f"TritonKernel.indexing assumes {label}BLOCK divides {max_block_str}"
            f" but {label}BLOCK={block} and {max_block_str}={max_block} (cfg={cfg})."
        )


def triton_config(
    size_hints, x, y=None, z=None, num_stages=1, num_elements_per_warp=256
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
    """
    # Ideally we want to read this from some device config

    # for a 2d size_hints [a, b], a should be mapped to YBLOCK rather than XBLOCK
    size_hints = list(reversed(size_hints))

    maxGridSize = [2147483647, 65535, 65535]

    target = conditional_product(x, y, z)
    if conditional_product(*size_hints) < target:
        target //= 8

    # shrink sizes to size hints
    x = min(x, size_hints[0])
    if y:
        y = min(y, size_hints[1])
    if z:
        z = min(z, size_hints[2])

    # if we are below original block size, scale up where we can;
    # or if the calculated grid size is larger than the limit, we bump up the corresponding dimension
    while x < size_hints[0] and (
        x * maxGridSize[0] < size_hints[0] or conditional_product(x, y, z) < target
    ):
        x *= 2
    while (
        y
        and y < size_hints[1]
        and (
            y * maxGridSize[1] < size_hints[1] or conditional_product(x, y, z) < target
        )
    ):
        y *= 2
    while (
        z
        and z < size_hints[2]
        and (
            z * maxGridSize[2] < size_hints[2] or conditional_product(x, y, z) < target
        )
    ):
        z *= 2

    cfg = {"XBLOCK": x}
    if y:
        cfg["YBLOCK"] = y
    if z:
        cfg["ZBLOCK"] = z
    num_warps = next_power_of_2(
        min(max(conditional_product(x, y, z) // num_elements_per_warp, 1), 8)
    )
    # we are going to arrive at 2 warps only if bs was too small due to
    # numel being too small. However to workaround some ptx bugs we still
    # want at least 4 warps if there's enough elements per thread
    # given that this is a rare situation, don't expect this to affect perf
    # in general
    # see https://github.com/pytorch/pytorch/pull/97950
    num_warps = max(num_warps, 4) if conditional_product(x, y, z) >= 128 else num_warps
    xnumel = size_hints[0]
    ynumel = size_hints[1] if y else None
    znumel = size_hints[2] if z else None
    check_config(cfg, xnumel=xnumel, ynumel=ynumel, znumel=znumel)
    return Config(cfg, num_warps=num_warps, num_stages=num_stages)


def triton_config_reduction(size_hints, x, r, num_stages=1, num_warps=None) -> Config:
    """
    Construct a reduction triton config with some adjustment heuristics
    based on size_hints. Size_hints is a tuple of numels in each tile
    dimension and will be rounded up to the nearest power of 2.
    """

    target = conditional_product(x, r)
    if conditional_product(*size_hints) < target:
        target //= 8

    # shrink sizes to size hints
    x = min(x, size_hints[0])
    r = min(r, size_hints[1])

    # if we are below original block size, scale up where we can
    while x < size_hints[0] and conditional_product(x, r) < target:
        x *= 2
    while r < size_hints[1] and conditional_product(x, r) < target:
        r *= 2

    cfg = {"XBLOCK": x, "RBLOCK": r}
    if num_warps is None:
        num_warps = conditional_product(x, r) // 128
    num_warps = next_power_of_2(min(max(num_warps, 2), 8))
    check_config(cfg, xnumel=size_hints[0])
    return Config(cfg, num_warps=num_warps, num_stages=num_stages)


def triton_config_tiled_reduction(size_hints, x, y, r, num_stages=1):
    """
    Construct a tile reduction triton config with some adjustment
    heuristics based on size_hints. Size_hints is a tuple of numels in
    each tile dimension and will be rounded up to the nearest power of 2.
    """

    target = conditional_product(x, y, r)
    if conditional_product(*size_hints) < target:
        target //= 8

    # shrink sizes to size hints
    x = min(x, size_hints[0])
    y = min(y, size_hints[1])
    r = min(r, size_hints[2])

    # if we are below original block size, scale up where we can
    while x < size_hints[0] and conditional_product(x, y, r) < target:
        x *= 2
    while r < size_hints[2] and conditional_product(x, y, r) < target:
        r *= 2
    while y < size_hints[1] and conditional_product(x, y, r) < target:
        y *= 2

    cfg = {"XBLOCK": x, "YBLOCK": y, "RBLOCK": r}
    num_warps = next_power_of_2(min(max(conditional_product(x, y, r) // 256, 1), 8))
    check_config(cfg, xnumel=size_hints[0], ynumel=size_hints[1])
    return Config(cfg, num_warps=num_warps, num_stages=num_stages)


def pointwise(size_hints, meta, tile_hint=None, filename=None):
    """
    Construct @triton.heuristics() based on size_hints.
    """
    numel = functools.reduce(operator.mul, size_hints)
    bs = max(256, min(numel // 128, 1024))

    hinted_configs = autotune_hints_to_configs(
        meta.get("autotune_hints", set()), size_hints, bs
    )

    if len(size_hints) == 1:
        if disable_pointwise_autotuning() and not (
            config.max_autotune or config.max_autotune_pointwise
        ):
            return cached_autotune(
                size_hints,
                [triton_config(size_hints, bs)],
                meta=meta,
                heuristic_type=HeuristicType.POINTWISE,
                filename=filename,
            )
        else:
            return cached_autotune(
                size_hints,
                [
                    triton_config(size_hints, bs, num_elements_per_warp=256),
                    triton_config(size_hints, bs // 2, num_elements_per_warp=64),
                    *hinted_configs,
                ],
                meta=meta,
                heuristic_type=HeuristicType.POINTWISE,
                filename=filename,
            )
    if len(size_hints) == 2:
        if (disable_pointwise_autotuning() or tile_hint == TileHint.SQUARE) and not (
            config.max_autotune or config.max_autotune_pointwise
        ):
            return cached_autotune(
                size_hints,
                [triton_config(size_hints, 32, 32)],
                meta=meta,
                heuristic_type=HeuristicType.POINTWISE,
                filename=filename,
            )
        return cached_autotune(
            size_hints,
            [
                triton_config(size_hints, 32, 32),
                triton_config(size_hints, 64, 64),  # ~8% better for fp16
                triton_config(size_hints, 256, 16),
                triton_config(size_hints, 16, 256),
                triton_config(size_hints, bs, 1),
                triton_config(size_hints, 1, bs),
                *hinted_configs,
            ],
            meta=meta,
            filename=filename,
            heuristic_type=HeuristicType.POINTWISE,
        )
    if len(size_hints) == 3:
        if disable_pointwise_autotuning():
            return cached_autotune(
                size_hints,
                [triton_config(size_hints, 16, 16, 16)],
                meta=meta,
                heuristic_type=HeuristicType.POINTWISE,
                filename=filename,
            )
        return cached_autotune(
            size_hints,
            [
                triton_config(size_hints, 16, 16, 16),
                triton_config(size_hints, 64, 8, 8),
                triton_config(size_hints, 8, 64, 8),
                triton_config(size_hints, 8, 8, 64),
                triton_config(size_hints, bs, 1, 1),
                triton_config(size_hints, 1, bs, 1),
                triton_config(size_hints, 1, 1, bs),
                *hinted_configs,
            ],
            meta=meta,
            filename=filename,
            heuristic_type=HeuristicType.POINTWISE,
        )
    raise NotImplementedError(f"size_hints: {size_hints}")


def reduction(size_hints, reduction_hint=False, meta=None, filename=None):
    """args to @triton.heuristics()"""
    assert meta is not None
    rnumel = size_hints[-1]
    if len(size_hints) == 2:
        contiguous_config = triton_config_reduction(
            size_hints, 1, (rnumel if 256 <= rnumel < 2048 else 2048)
        )
        outer_config = triton_config_reduction(size_hints, 128, 8)
        tiny_config = triton_config_reduction(
            size_hints, 2 * (256 // rnumel) if rnumel <= 256 else 1, min(rnumel, 2048)
        )
        if config.max_autotune or config.max_autotune_pointwise:
            pass  # skip all these cases
        elif reduction_hint == ReductionHint.INNER:
            return cached_autotune(
                size_hints,
                [contiguous_config],
                meta=meta,
                heuristic_type=HeuristicType.REDUCTION,
                filename=filename,
            )
        elif reduction_hint == ReductionHint.OUTER:
            return cached_autotune(
                size_hints,
                [outer_config],
                meta=meta,
                heuristic_type=HeuristicType.REDUCTION,
                filename=filename,
            )
        elif reduction_hint == ReductionHint.OUTER_TINY:
            return cached_autotune(
                size_hints,
                [tiny_config],
                meta=meta,
                heuristic_type=HeuristicType.REDUCTION,
                filename=filename,
            )
        if disable_pointwise_autotuning():
            return cached_autotune(
                size_hints,
                [triton_config_reduction(size_hints, 32, 128)],
                meta=meta,
                heuristic_type=HeuristicType.REDUCTION,
                filename=filename,
            )
        return cached_autotune(
            size_hints,
            [
                contiguous_config,
                outer_config,
                tiny_config,
                triton_config_reduction(size_hints, 64, 64),
                triton_config_reduction(size_hints, 8, 512),
                # halve the XBLOCK/RBLOCK compared to outer_config
                # TODO: this may only be beneficial when each iteration of the reduciton
                # is quite heavy. E.g. https://gist.github.com/shunting314/189a8ef69f90db9d614a823385147a72
                triton_config_reduction(size_hints, 64, 4, num_warps=8),
            ],
            meta=meta,
            filename=filename,
            heuristic_type=HeuristicType.REDUCTION,
        )
    raise NotImplementedError(f"size_hints: {size_hints}")


def persistent_reduction(size_hints, reduction_hint=False, meta=None, filename=None):
    xnumel, rnumel = size_hints

    configs = [
        triton_config_reduction(size_hints, xblock, rnumel)
        for xblock in (1, 8, 32, 128)
        if rnumel * xblock <= 4096 and xblock <= xnumel
    ]

    # TODO(jansel): we should be able to improve these heuristics
    if reduction_hint == ReductionHint.INNER and rnumel >= 256:
        configs = configs[:1]
    elif reduction_hint == ReductionHint.OUTER:
        configs = configs[-1:]
    elif reduction_hint == ReductionHint.OUTER_TINY:
        configs = [
            triton_config_reduction(
                size_hints, 2 * (256 // rnumel) if rnumel <= 256 else 1, rnumel
            )
        ]
    for c in configs:
        # we don't need RBLOCK for persistent reduction
        c.kwargs.pop("RBLOCK")

    if disable_pointwise_autotuning():
        configs = configs[:1]

    return cached_autotune(
        size_hints,
        configs,
        meta=meta,
        filename=filename,
        heuristic_type=HeuristicType.PERSISTENT_REDUCTION,
    )


def template(num_stages, num_warps, meta, filename=None):
    """
    Compile a triton template
    """
    return cached_autotune(
        None,
        [triton.Config({}, num_stages=num_stages, num_warps=num_warps)],
        meta=meta,
        heuristic_type=HeuristicType.TEMPLATE,
        filename=filename,
    )


def foreach(meta, num_warps, filename=None):
    """
    Compile a triton foreach kernel
    """
    return cached_autotune(
        None,
        [triton.Config({}, num_stages=1, num_warps=num_warps)],
        meta=meta,
        heuristic_type=HeuristicType.TEMPLATE,
        filename=filename,
    )


def grid(*numels):
    """Helper function to compute triton grids"""

    if len(numels) == 1:
        xnumel, ynumel, znumel = numels[0], None, None
    elif len(numels) == 2:
        xnumel, ynumel, znumel = numels[1], numels[0], None
    elif len(numels) == 3:
        xnumel, ynumel, znumel = numels[2], numels[1], numels[0]
    else:
        raise AssertionError(f"invalid size for numels {len(numels)}")

    def get_grid_dim(numel, block):
        if numel is None:
            return 1
        return ceildiv(numel, block)

    def grid_fn(meta):
        return (
            get_grid_dim(xnumel, meta.get("XBLOCK", 1)),
            get_grid_dim(ynumel, meta.get("YBLOCK", None)),
            get_grid_dim(znumel, meta.get("ZBLOCK", None)),
        )

    return grid_fn
