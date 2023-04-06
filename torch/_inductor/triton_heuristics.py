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
from typing import List

import torch
from torch._dynamo.utils import dynamo_timed

from . import config
from .codecache import cache_dir, cubin_cache_dir

from .ir import ReductionHint, TileHint
from .utils import (
    ceildiv,
    conditional_product,
    create_bandwidth_info_str,
    do_bench,
    get_num_bytes,
    has_triton,
    next_power_of_2,
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


class CachingAutotuner(KernelInterface):
    """
    Simplified version of Triton autotuner that has no invalidation
    key and caches the best config to disk to improve cold start times.
    Unlike the main triton Autotuner, this version can precompile all
    configs, and does not rely on the Triton JIT.
    """

    def __init__(self, fn, meta, configs, save_cache_hook, mutated_arg_names):
        super().__init__()
        self.fn = fn
        self.meta = meta
        self.save_cache_hook = save_cache_hook
        self.mutated_arg_names = mutated_arg_names
        self.configs = configs
        self.launchers = []
        self.lock = threading.Lock()
        if os.getenv("TRITON_CACHE_DIR") is None:
            os.environ["TRITON_CACHE_DIR"] = os.path.join(
                cache_dir(),
                "triton",
                str(self.meta.get("device", 0)),
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

    def _precompile_config(self, cfg: Config, warm_cache_only_with_cc: int):
        """Ahead of time compile a given autotuner config."""
        compile_meta = copy.deepcopy(self.meta)
        for k, v in cfg.kwargs.items():
            compile_meta["constants"][self.fn.arg_names.index(k)] = v
        compile_meta["num_warps"] = cfg.num_warps
        compile_meta["num_stages"] = cfg.num_stages
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
        if config.triton.store_cubin:
            launcher.kernel_name = self.fn.__name__
            launcher.bin = binary

        return launcher

    def bench(self, launcher, *args, grid):
        """Measure the performance of a given launcher"""
        stream = get_cuda_stream(torch.cuda.current_device())

        def kernel_call():
            if launcher.config.pre_hook is not None:
                launcher.config.pre_hook(
                    {**zip(self.arg_names, args), **launcher.config.kwargs}
                )
            launcher(
                *args,
                grid=grid,
                stream=stream,
            )

        return do_bench(kernel_call, rep=40, fast_flush=True)

    @dynamo_timed
    def benchmark_all_configs(self, *args, **kwargs):
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

        timings = {
            launcher: self.bench(launcher, *cloned_args, **kwargs)[0]
            for launcher in self.launchers
        }
        return timings

    def autotune_to_one_config(self, *args, **kwargs):
        """Do the actual autotuning"""
        timings = self.benchmark_all_configs(*args, **kwargs)
        self.launchers = [builtins.min(timings, key=timings.get)]
        if self.save_cache_hook:
            self.save_cache_hook(self.launchers[0].config)

    def save_cuda_kernel(self, grid, stream, launcher):
        from .codegen.wrapper import KernelParamCache

        # Make sure kernel_name is enough for distiguishing kernels
        assert config.triton.unique_kernel_names

        if callable(grid):
            grid_x, grid_y, grid_z = grid(launcher.config.kwargs)
        else:
            grid_x, grid_y, grid_z = grid

        kernel_name = launcher.kernel_name
        cubin_path = os.path.join(cubin_cache_dir(), f"{kernel_name}.cubin")
        with open(cubin_path, "wb") as f:
            f.write(launcher.bin.asm["cubin"])

        params = {
            "mangled_name": launcher.bin.metadata["name"],
            "grid_x": grid_x,
            "grid_y": grid_y,
            "grid_z": grid_z,
            "num_warps": launcher.bin.num_warps,
            "shared_mem": launcher.bin.shared,
            "stream": stream,
        }
        with self.lock:
            if KernelParamCache.cache.get(kernel_name, None):
                assert (
                    KernelParamCache.cache[kernel_name].get("mangled_name", None)
                    == launcher.bin.metadata["name"]
                )
            else:
                KernelParamCache.cache[kernel_name] = params

    def run(self, *args, grid, stream):
        if len(self.launchers) != 1:
            if len(self.launchers) == 0:
                self.precompile()
            if len(self.launchers) > 1:
                self.autotune_to_one_config(*args, grid=grid)

        if config.triton.store_cubin:
            self.save_cuda_kernel(grid, stream, self.launchers[0])

        (launcher,) = self.launchers
        if launcher.config.pre_hook is not None:
            launcher.config.pre_hook(
                {**zip(self.arg_names, args), **launcher.config.kwargs}
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
    for frame in iter(lambda: frame.f_back, None):
        frame.f_locals
    obj_names = []
    for referrer in gc.get_referrers(obj):
        if isinstance(referrer, dict):
            for k, v in referrer.items():
                if v is obj:
                    obj_names.append(k)
    return obj_names


collected_calls = []


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

    def run(self, *args, grid, stream):
        possible_names = _find_names(self)
        kernel_name = f"{max(possible_names, key=lambda x: len(x))}"
        if not re.match(self.regex_filter, kernel_name):
            return
        super().run(*args, grid=grid, stream=stream)
        (launcher,) = self.launchers

        ms = self.bench(launcher, *args, grid=grid)[0]
        num_in_out_ptrs = len(
            [
                arg_name
                for arg_name in self.fn.arg_names
                if arg_name.startswith("in_out_ptr")
            ]
        )
        num_gb = get_num_bytes(*args, num_in_out_args=num_in_out_ptrs) / 1e9
        gb_per_s = num_gb / (ms / 1e3)

        collected_calls.append((ms, num_gb, gb_per_s, kernel_name)),
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
            f"{sorted(cfg.kwargs.items())} {cfg.num_warps} {cfg.num_stages}\n".encode(
                "utf-8"
            )
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

    with open(cache_filename, "r") as fd:
        best_config = json.loads(fd.read())
    if best_config.get("configs_hash") != configs_hash:
        return None

    matching_configs = [
        cfg
        for cfg in configs
        if all(val == best_config.get(key) for key, val in cfg.kwargs.items()) and cfg.num_warps == best_config.get("num_warps") and cfg.num_stages == best_config.get("num_stages")
    ]
    if len(matching_configs) != 1:
        return None

    return matching_configs[0]


def cached_autotune(
    configs: List[Config],
    meta,
    filename=None,
):
    """
    A copy of triton.autotune that calls our subclass.  Our subclass
    has additional debugging, error handling, and on-disk caching.
    """
    configs = unique_configs(configs)
    assert len(configs) == 1 or filename

    # on disk caching logic
    if filename is not None and len(configs) > 1:
        cache_filename = os.path.splitext(filename)[0] + ".best_config"
        configs_hash = hash_configs(configs)
        best_config = load_cached_autotuning(cache_filename, configs_hash, configs)
        if best_config:
            configs = [best_config]

        def save_cache_hook(cfg):
            with open(cache_filename, "w") as fd:
                fd.write(json.dumps({**cfg.kwargs, "num_warps": cfg.num_warps, "num_stages": cfg.num_stages, "configs_hash": configs_hash}))

    else:
        save_cache_hook = None

    mutated_arg_names = meta.pop("mutated_arg_names", ())

    def decorator(fn):
        if config.profile_bandwidth:
            return DebugAutotuner(
                fn,
                meta=meta,
                regex_filter=config.profile_bandwidth_regex,
                configs=configs,
                save_cache_hook=save_cache_hook,
                mutated_arg_names=mutated_arg_names,
            )
        return CachingAutotuner(
            fn,
            meta=meta,
            configs=configs,
            save_cache_hook=save_cache_hook,
            mutated_arg_names=mutated_arg_names,
        )

    return decorator


def unique_configs(configs: List[Config]):
    """Remove duplicate configurations"""
    seen = set()
    pruned_configs = []

    def config_to_hashable(cfg):
        items = list(cfg.kwargs.items())
        items.append(("num_warps", cfg.num_warps))
        items.append(("num_stages", cfg.num_stages))
        return tuple(items)
        
    for cfg in configs:
        key = config_to_hashable(cfg)
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


def triton_config(size_hints, x, y=None, z=None, num_stages=1) -> Config:
    """
    Construct a pointwise triton config with some adjustment heuristics
    based on size_hints. Size_hints is a tuple of numels in each tile
    dimension and will be rounded up to the nearest power of 2.
    """
    # Ideally we want to read this from some device config
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
    num_warps = next_power_of_2(min(max(conditional_product(x, y, z) // 256, 1), 8))
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


def triton_config_reduction(size_hints, x, r, num_stages=2) -> Config:
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
    num_warps = next_power_of_2(min(max(conditional_product(x, r) // 128, 2), 8))
    check_config(cfg, xnumel=size_hints[0])
    return Config(cfg, num_warps=num_warps, num_stages=num_stages)


def triton_config_tiled_reduction(size_hints, x, y, r, num_stages=2):
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

    if len(size_hints) == 1:
        configs = [triton_config(size_hints, bs)]
        if config.max_autotune:
            configs.extend([
                # improve 1.832x for https://gist.github.com/shunting314/69b5055193148ade349ac7e58c85d2d9 
                Config({"XBLOCK": 256}, num_warps=8, num_stages=1),
                # improve 1.031x for https://gist.github.com/shunting314/339dd078cb9711536e4539dbb50b9032
                Config({"XBLOCK": 512}, num_warps=2, num_stages=1),
                # improve 1.016x for https://gist.github.com/shunting314/a5e6ee5cf8700ad3d43ec6853db4b236
                Config({"XBLOCK": 512}, num_warps=4, num_stages=1),
                # improve 1.02x for
                #   https://gist.github.com/shunting314/deecb85f7b15d5cd3331de589c230df2,
                #   https://gist.github.com/shunting314/07a541af896b7f4f099a1195419e407f,
                #   https://gist.github.com/shunting314/cfdd0c07479de3afdd415636e72b9e04
                Config({"XBLOCK": 512}, num_warps=1, num_stages=1),
                # improve 1.012x for https://gist.github.com/shunting314/3c2df5cb11062195203afab3f6b08c4d 
                Config({"XBLOCK": 256}, num_warps=2, num_stages=1),
                # improve 1.011x for https://gist.github.com/shunting314/600ecb80dc7339be4f08bea75c40cdbd
                Config({"XBLOCK": 512}, num_warps=8, num_stages=1),
                # improve 1.056x for https://gist.github.com/shunting314/83084eb255cb7df7776fbea20663048f
                Config({"XBLOCK": 512}, num_warps=4, num_stages=1),
                # improve 1.091x for https://gist.github.com/shunting314/bc5019bdfa78421a72822ce312df9639
                Config({"XBLOCK": 1024}, num_warps=2, num_stages=1),
                # improve 1.100x for https://gist.github.com/shunting314/5761558133f815b49984f094cc2a2e2c
                Config({"XBLOCK": 1024}, num_warps=4, num_stages=1),
                # improve 1.077x for https://gist.github.com/shunting314/f7f63c847d555c3551b1ba7c17973b10 
                Config({"XBLOCK": 1024}, num_warps=2, num_stages=1),
                # improve 1.071x for https://gist.github.com/shunting314/5a7c0d81152b45df7c953c06f57b368a
                Config({"XBLOCK": 2048}, num_warps=4, num_stages=1),
                # improve 1.011x for https://gist.github.com/shunting314/77082710f11d1701b2652d4359033890
                # improve 1.111x for https://gist.github.com/shunting314/facc4258391dd0a28ee16e1c1c7a579b
                Config({"XBLOCK": 1024}, num_warps=8, num_stages=1),
                # improve 1.091x for https://gist.github.com/shunting314/cc2345425c415c30719e36ccd0d31cf4
                Config({"XBLOCK": 256}, num_warps=1, num_stages=1),
                # improve 1.011x for https://gist.github.com/shunting314/00ebfd2615ccab2c1e3530404f4a1b4e
                Config({"XBLOCK": 2048}, num_warps=2, num_stages=1),
                # improve 1.111x for https://gist.github.com/shunting314/189a1d0e59f8eb759fa3db24a594218c
                Config({"XBLOCK": 512}, num_warps=4, num_stages=1),
            ])
        return cached_autotune(configs, meta=meta, filename=filename)
    if len(size_hints) == 2:
        if (
            not config.triton.autotune_pointwise or tile_hint == TileHint.SQUARE
        ) and not (config.max_autotune or config.max_autotune_pointwise):
            return cached_autotune([triton_config(size_hints, 32, 32)], meta=meta)
        return cached_autotune(
            [
                triton_config(size_hints, 32, 32),
                triton_config(size_hints, 64, 64),  # ~8% better for fp16
                triton_config(size_hints, 256, 16),
                triton_config(size_hints, 16, 256),
                triton_config(size_hints, bs, 1),
                triton_config(size_hints, 1, bs),
                # improve 1.04x for https://gist.github.com/shunting314/ec7db8a92ddec406ddd802813cfe0d50
                Config({"XBLOCK": 16, "YBLOCK": 256}, num_warps=16, num_stages=1),
                # improve 1.067x for https://gist.github.com/shunting314/75f801f9b353f0dd90911a4185bf3aa3
                Config({"XBLOCK": 32, "YBLOCK": 64}, num_warps=8, num_stages=1),
                # improve 1.054x for https://gist.github.com/shunting314/46ca21dba033156b1de9e87ac2d9b8a0
                Config({"XBLOCK": 8, "YBLOCK": 512}, num_warps=8, num_stages=1),
                # improve 1.015x for https://gist.github.com/shunting314/cae08cd11e635a9954e942692856138a
                Config({"XBLOCK": 16, "YBLOCK": 64}, num_warps=8, num_stages=1),
            ],
            meta=meta,
            filename=filename,
        )
    if len(size_hints) == 3:
        if not config.triton.autotune_pointwise:
            return cached_autotune([triton_config(size_hints, 16, 16, 16)], meta=meta)
        return cached_autotune(
            [
                triton_config(size_hints, 16, 16, 16),
                triton_config(size_hints, 64, 8, 8),
                triton_config(size_hints, 8, 64, 8),
                triton_config(size_hints, 8, 8, 64),
                triton_config(size_hints, bs, 1, 1),
                triton_config(size_hints, 1, bs, 1),
                triton_config(size_hints, 1, 1, bs),
            ],
            meta=meta,
            filename=filename,
        )
    raise NotImplementedError(f"size_hints: {size_hints}")


def reduction(size_hints, reduction_hint=False, meta=None, filename=None):
    """args to @triton.heuristics()"""
    assert meta is not None
    rnumel = size_hints[-1]
    if len(size_hints) == 2:
        contiguous_config = triton_config_reduction(
            size_hints, 1, (rnumel if 256 <= rnumel < 2048 else 2048), num_stages=1
        )
        outer_config = triton_config_reduction(size_hints, 128, 8)
        tiny_config = triton_config_reduction(
            size_hints, 2 * (256 // rnumel) if rnumel <= 256 else 1, min(rnumel, 2048)
        )
        if config.max_autotune or config.max_autotune_pointwise:
            pass  # skip all these cases
        elif reduction_hint == ReductionHint.INNER:
            return cached_autotune([contiguous_config], meta=meta)
        elif reduction_hint == ReductionHint.OUTER:
            return cached_autotune([outer_config], meta=meta)
        elif reduction_hint == ReductionHint.OUTER_TINY:
            return cached_autotune([tiny_config], meta=meta)
        if not config.triton.autotune_pointwise:
            return cached_autotune(
                [triton_config_reduction(size_hints, 32, 128)], meta=meta
            )
        return cached_autotune(
            [
                contiguous_config,
                outer_config,
                tiny_config,
                triton_config_reduction(size_hints, 64, 64),
                triton_config_reduction(size_hints, 8, 512),
                # improve 1.121x for https://gist.github.com/shunting314/6267da87c6524dab29a3e33f14ff91db
                Config({"XBLOCK": 1, "RBLOCK": 4096}, num_warps=8, num_stages=1),
                # improve 1.074x for https://gist.github.com/shunting314/bb36b37a3d049a29ff6bada9dbe18fb8
                Config({"XBLOCK": 2, "RBLOCK": 2048}, num_warps=8, num_stages=1),
                # improve 1.143x for https://gist.github.com/shunting314/ae6a0d3c63409bed7759ee7bdcde01a8
                Config({"XBLOCK": 1, "RBLOCK": 1024}, num_warps=16, num_stages=1),
                # improve 1.033x for https://gist.github.com/shunting314/d1036afd5210e19da3144c24f3b475d8
                Config({"XBLOCK": 1, "RBLOCK": 1024}, num_warps=8, num_stages=1),
                # improve 1.033x for https://gist.github.com/shunting314/b15e21af1fe9033cbffa9cfb0f575e12
                Config({"XBLOCK": 1, "RBLOCK": 2048}, num_warps=16, num_stages=1),
                # improve 1.185x for https://gist.github.com/shunting314/1afc463bf01cb75672ce3b418d4c66f3
                Config({"XBLOCK": 1, "RBLOCK": 512}, num_warps=8, num_stages=1),
                # improve 1.098x for https://gist.github.com/shunting314/8b7c83c4f9134110065e95d3daae2a56
                Config({"XBLOCK": 2, "RBLOCK": 1024}, num_warps=8, num_stages=1),
                # imporve 1.035x for https://gist.github.com/shunting314/5258e534ce028871a5b0d1be9033f67c
                Config({"XBLOCK": 2, "RBLOCK": 256}, num_warps=8, num_stages=1),
                # improve 1.014x for https://gist.github.com/shunting314/fd73b6ce32e8fd3ebd9d3c7a8a5e9995
                Config({"XBLOCK": 2, "RBLOCK": 512}, num_warps=8, num_stages=1),
                # improve 1.067x for https://gist.github.com/shunting314/0bafa894f1735f255a379c2107f5e979 
                #         1.080x for https://gist.github.com/shunting314/c7c56a558d2043f260cb00ba266aa910 
                Config({"XBLOCK": 64, "RBLOCK": 128}, num_warps=16, num_stages=2),
                # improve 1.143x for https://gist.github.com/shunting314/ac92b583c52c46e2f2e65540d0e75c2c
                Config({"XBLOCK": 64, "RBLOCK": 8}, num_warps=8, num_stages=2),
                # improve 1.152x for https://gist.github.com/shunting314/10c52fc9bf7ba5044b75a3d7b2ecb6a7 
                Config({"XBLOCK": 1, "RBLOCK": 256}, num_warps=8, num_stages=1),
                # improve 1.396x for https://gist.github.com/shunting314/9bf9ee7c0200966d5858d5efca0adc9c
                Config({"XBLOCK": 1, "RBLOCK": 1024}, num_warps=32, num_stages=1),
                # improve 1.031x for https://gist.github.com/shunting314/f3fac6407e4d379e0aa111484aae0ad1
                Config({"XBLOCK": 1, "RBLOCK": 1024}, num_warps=16, num_stages=2),
                # improve 1.043x for https://gist.github.com/shunting314/9bae27a5c41e9a6995b9bbe37dbb39a5
                Config({"XBLOCK": 1, "RBLOCK": 128}, num_warps=4, num_stages=2),
                # improve 1.031x for https://gist.github.com/shunting314/7fe0b75ef88c251f1611ba74f3c0b87a
                Config({"XBLOCK": 2, "RBLOCK": 128}, num_warps=4, num_stages=1),
                # improve 1.031x for https://gist.github.com/shunting314/d48e9470416b6ea507e48be119b7ed34
                Config({"XBLOCK": 4, "RBLOCK": 256}, num_warps=8, num_stages=1),
                # improve 1.198x for https://gist.github.com/shunting314/87858731b5dc210caf566a2a40da1326
                Config({"XBLOCK": 32, "RBLOCK": 64}, num_warps=16, num_stages=2),
                # improve 1.033x for https://gist.github.com/shunting314/71397bbd0e2514cb164dcc1147071cbd
                Config({"XBLOCK": 8, "RBLOCK": 128}, num_warps=8, num_stages=2),
                # improve 1.120x for https://gist.github.com/shunting314/e9506fce9d7273a9ffe4fbfce445e635
                Config({"XBLOCK": 4, "RBLOCK": 256}, num_warps=8, num_stages=1),
                # improve 1.01x for https://gist.github.com/shunting314/62a5af3f87296d0913be98d9b2c402da
                Config({"XBLOCK": 2, "RBLOCK": 256}, num_warps=4, num_stages=1),
                # improve 1.008x for https://gist.github.com/shunting314/c9f06f1c815afcb003a81c0fcb1d23ae
                Config({"XBLOCK": 2, "RBLOCK": 256}, num_warps=1, num_stages=1),
                # improve 1.215x for https://gist.github.com/shunting314/ca9a43df53085100a3bedd496edfa5f2
                Config({"XBLOCK": 16, "RBLOCK": 512}, num_warps=4, num_stages=2),
                # improve 1.009x for https://gist.github.com/shunting314/c6523b799a43058dd163f4a59f04cc25
                Config({"XBLOCK": 16, "RBLOCK": 128}, num_warps=4, num_stages=1),
                # improve 1.222x for https://gist.github.com/shunting314/b3fa8e0c48d4d9517811b1b42c339291
                Config({"XBLOCK": 2, "RBLOCK": 512}, num_warps=4, num_stages=1),
                # improve 1.178x for https://gist.github.com/shunting314/73b0e86287b994a313d889eb75d7e8b8
                Config({"XBLOCK": 64, "RBLOCK": 8}, num_warps=4, num_stages=2),
                # improve 1.051x for https://gist.github.com/shunting314/2720a4c58e2feb7abec58d461025c3ee
                Config({"XBLOCK": 32, "RBLOCK": 8}, num_warps=8, num_stages=2),
                # improve 1.038x for https://gist.github.com/shunting314/4eff23aa5f1b45a865cd9cc9fc732056
                Config({"XBLOCK": 256, "RBLOCK": 8}, num_warps=8, num_stages=2),
                # improve 1.111x for https://gist.github.com/shunting314/99b46948de675a4e367b91c0c2341583
                Config({"XBLOCK": 8, "RBLOCK": 128}, num_warps=8, num_stages=1),
                # improve 1.111x for https://gist.github.com/shunting314/dac96e99060c2ba463b2ed0ba396eaee
                Config({"XBLOCK": 8, "RBLOCK": 128}, num_warps=4, num_stages=2),
            ],
            meta=meta,
            filename=filename,
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

    if config.max_autotune:
        configs.extend([
            # improve by 1.043x for https://gist.github.com/shunting314/f5f5beb553f139be7ea3c122173a2288
            Config({"XBLOCK": 2, "RBLOCK": rnumel}, num_warps=2, num_stages=2),
            # improve by 1.147x for https://gist.github.com/shunting314/552ee9d73d0bec3426c5886d992b1654
            #            1.133x for https://gist.github.com/shunting314/014112d71f9f692249eaed9d6325884d
            Config({"XBLOCK": 4, "RBLOCK": rnumel}, num_warps=4, num_stages=2),
            # improve 1.017x for https://gist.github.com/shunting314/f9fa8a0fd58ba8031be7903cea57b535
            Config({"XBLOCK": 2, "RBLOCK": rnumel}, num_warps=4, num_stages=2),
        ])

    return cached_autotune(
        configs,
        meta=meta,
        filename=filename,
    )


def template(num_stages, num_warps, meta, filename=None):
    """
    Compile a triton template
    """
    return cached_autotune(
        [triton.Config({}, num_stages=num_stages, num_warps=num_warps)], meta=meta
    )


def grid(xnumel, ynumel=None, znumel=None):
    """Helper function to compute triton grids"""

    def get_grid_dim(numel, block):
        if numel is None:
            return 1
        return ceildiv(numel, block)

    def grid_fn(meta):
        return (
            get_grid_dim(xnumel, meta.get("XBLOCK", None)),
            get_grid_dim(ynumel, meta.get("YBLOCK", None)),
            get_grid_dim(znumel, meta.get("ZBLOCK", None)),
        )

    return grid_fn
