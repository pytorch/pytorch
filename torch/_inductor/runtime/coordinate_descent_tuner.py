# mypy: allow-untyped-defs
import copy
import itertools
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from torch.utils._ordered_set import OrderedSet

from ..utils import get_max_numwarps
from .hints import TRITON_MAX_BLOCK
from .runtime_utils import red_text, triton_config_to_hashable


if TYPE_CHECKING:
    from .triton_compat import triton


log = logging.getLogger(__name__)


def get_field(config, name):
    if name == "num_warps":
        return config.num_warps
    elif name == "num_stages":
        return config.num_stages
    elif name == "waves_per_eu":
        return config.kwargs.get(name, int(8 // config.num_warps))
    else:
        return config.kwargs.get(name, None)


def set_field(config, name, value):
    if name == "num_warps":
        config.num_warps = value
    elif name == "num_stages":
        config.num_stages = value
    else:
        config.kwargs[name] = value


class CoordescTuner:
    """
    The coordinate descent tuner. Tune one field/coordinate at a time.

    TODO will it be necessary to tune multiple fields simultaneously.


    TODO: what if both increasing and decreasing a field can improve perf.
          i.e., there are multiple local optima..
    """

    def __init__(
        self,
        is_mm=False,
        is_native_matmul=False,
        is_mix_order_reduction=False,
        name="unknown",
        size_hints=None,
        inductor_meta=None,
        frozen_fields=None,
    ):
        self.is_mm = is_mm  # we will tune num_stages for mm

        # Native matmul codegen assumes ZBLOCK=1 always.
        # This is because 3d tl.dot is slow and so we want to tile y and x only.
        # tl.dot also does not support size smaller than 16; we put this restriction.
        self.is_native_matmul = is_native_matmul
        assert not (self.is_mm and self.is_native_matmul)
        self.is_mix_order_reduction = is_mix_order_reduction
        self.cached_benchmark_results = {}
        self.name = name
        self.size_hints = size_hints
        self.inductor_meta = inductor_meta or {}
        self.frozen_fields: OrderedSet[str] = (
            OrderedSet(frozen_fields) if frozen_fields is not None else OrderedSet()
        )

    def get_config_max(self, prefix: str) -> int:
        max_block = TRITON_MAX_BLOCK[prefix.upper()]
        size_hint = self.size_hints.get(prefix) if self.size_hints is not None else None
        return min(max_block, size_hint) if size_hint is not None else max_block

    def get_warpsmax(self):
        # Avoid querying device directly if device properties are populated in inductor_meta
        warp_size = self.inductor_meta.get("warp_size")
        max_threads_per_block = self.inductor_meta.get("max_threads_per_block")
        if warp_size and max_threads_per_block:
            return max_threads_per_block // warp_size
        else:
            return get_max_numwarps()

    def cache_benchmark_result(self, config, timing):
        self.cached_benchmark_results[triton_config_to_hashable(config)] = timing

    def lookup_in_cache(self, config):
        return self.cached_benchmark_results.get(triton_config_to_hashable(config))

    def call_func(self, func, config):
        found = self.lookup_in_cache(config)
        if found is not None:
            log.debug("  CACHED")
            return found
        timing = func(config)
        self.cache_benchmark_result(config, timing)
        return timing

    @property
    def tunable_fields(self) -> list[str]:
        out: list[str] = [
            "XBLOCK",
            "YBLOCK",
            "ZBLOCK",
            # NOTE: we should not tune R0_BLOCK for persistent reduction.
            # We rely on the fact that persistent reduction's triton.Config
            # does not have the R0_BLOCK field to guarantee that.
            "R0_BLOCK",
            "R1_BLOCK",
            # the following 3 are for mm
            "BLOCK_M",
            "BLOCK_N",
            "BLOCK_K",
            "num_warps",
        ]
        if self.is_mm:
            out.append("num_stages")
        if self.inductor_meta.get("is_hip") is True:
            out.append("waves_per_eu")
        if self.is_native_matmul:
            out.append("num_stages")
            out.remove("ZBLOCK")  # ZBLOCK=1 always in native matmul

        if self.is_mix_order_reduction:
            # unlike TritonConfig.num_stages, this one is
            # put in TritonConfig.kwargs["NUM_STAGES"] and is used to
            # control the stage of pipelining of tl.range.
            out.append("NUM_STAGES")

        # Combo-kernel per-subkernel block fields (e.g. XBLOCK_0/1, YBLOCK_0/1)
        # come from the worker-side metadata in ``inductor_meta``. Prepend
        # them so coordesc iterates them alongside the base fields. Read
        # live each call so any post-construction mutation of
        # ``inductor_meta`` is observed.
        combo_fields: list[str] = self.inductor_meta.get(
            "combo_coordesc_field_order", []
        )
        out = combo_fields + out
        return [f for f in out if f not in self.frozen_fields]

    def value_too_large(self, name: str, val: int) -> bool:
        field_limits = self.inductor_meta.get("combo_coordesc_field_limits")
        if isinstance(field_limits, dict) and name in field_limits:
            return val > field_limits[name]

        block_suffix = "BLOCK"
        if name.endswith(block_suffix):
            prefix = name.strip(block_suffix).lower()
            return val > self.get_config_max(prefix)
        if name == "num_warps":
            return val > self.get_warpsmax()
        if name == "waves_per_eu":
            return val > 8

        return False

    def value_too_small(self, name: str, val: int) -> bool:
        # In native matmul, block size should be >= 16 for tl.dot
        if self.is_native_matmul:
            if name in ["YBLOCK", "XBLOCK", "R0_BLOCK"]:
                return val < 16

        # Break if value becomes 0/neg
        return val <= 0

    def get_neighbour_values(self, name, orig_val, radius=None, include_self=False):
        """
        Get neighbour values in 'radius' steps. The original value is not
        returned as its own neighbour.
        """
        if radius is None:
            radius = 1
        if name == "NUM_STAGES":
            # we see cases that
            # NUM_STAGES=1 is better than NUM_STAGES=2
            # while NUM_STAGES=1 is worse than NUM_STAGES=3
            radius = max(radius, 2)

        assert radius >= 1

        def update(cur_val, inc=True):
            if name in ["num_stages", "NUM_STAGES"]:
                if inc:
                    return cur_val + 1
                else:
                    return cur_val - 1
            else:
                if inc:
                    return cur_val * 2
                else:
                    return cur_val // 2

        out = []
        # increment loop
        cur_val = orig_val
        for _ in range(radius):
            cur_val = update(cur_val, True)
            if self.value_too_large(name, cur_val):
                break
            out.append(cur_val)

        # decrement loop
        cur_val = orig_val
        for _ in range(radius):
            cur_val = update(cur_val, False)
            if self.value_too_small(name, cur_val):
                break
            out.append(cur_val)

        if include_self:
            out.append(orig_val)
        return out

    @staticmethod
    def has_improvement(baseline, test):
        threshold = 0.001  # 0.1%
        return test is not None and test < baseline * (1 - threshold)

    def is_valid_config(self, config) -> bool:
        if self.is_mix_order_reduction:
            # Mix order reduction has an extra constraint that
            # we should not tune XBLOCK beyond RSPLIT_SIZE
            xblock = config.kwargs["XBLOCK"]
            split_size = config.kwargs["RSPLIT_SIZE"]
            return xblock <= split_size
        return True

    def get_all_tuning_directions(
        self,
        # pyrefly: ignore [missing-attribute]
        config: "triton.Config",
    ) -> list["triton.Config"]:  # pyrefly: ignore [missing-attribute]
        """Return the Cartesian product of neighbour values across every
        tunable field, as a list of valid candidate configs."""
        candidate_values_list = []
        effective_fields = []
        for field in self.tunable_fields:
            old_value = get_field(config, field)
            if old_value is None:
                continue
            radius = self.inductor_meta.get("coordinate_descent_search_radius", 1)
            candidate_values = self.get_neighbour_values(
                field,
                old_value,
                radius=radius,
                include_self=True,
            )
            candidate_values_list.append(candidate_values)
            effective_fields.append(field)

        configs = []
        for choice in itertools.product(*candidate_values_list):
            assert len(choice) == len(effective_fields)
            candidate = copy.deepcopy(config)
            for new_val, field in zip(choice, effective_fields):
                set_field(candidate, field, new_val)
            if self.is_valid_config(candidate):
                configs.append(candidate)
        return configs

    def check_all_tuning_directions(
        self,
        # pyrefly: ignore [missing-attribute]
        func: Callable[["triton.Config"], float],
        best_config,
        best_timing,
    ):
        """
        Check all directions. We only do this once the regular coordinate
        descent tuning find no better choices any more.
        We only have a few tunable fields, so this should be fine.
        """
        improved = False
        for candidate_config in self.get_all_tuning_directions(best_config):
            cmp_res, candidate_timing = self.compare_config(
                func, candidate_config, best_config, best_timing
            )
            if cmp_res:
                improved = True
                best_config = candidate_config
                best_timing = candidate_timing

        return improved, best_config, best_timing

    def compare_config(self, func, candidate_config, best_config, best_timing):
        """
        Check if candidate_config is better than best_config.

        Return a tuple of (compare_result, candidate_timing).
        compare_result is true iff candidate_config is better.
        """
        log.debug("Try config %s", candidate_config)
        try:
            candidate_timing = self.call_func(func, candidate_config)
        except Exception as e:
            log.debug("Got exception %s", e)
            return False, float("inf")

        if self.has_improvement(best_timing, candidate_timing):
            log.debug(
                "Tune from %s %f -> %s %f",
                best_config,
                best_timing,
                candidate_config,
                candidate_timing,
            )

            return True, candidate_timing
        return False, candidate_timing

    def get_neighbour_configs(
        self,
        # pyrefly: ignore [missing-attribute]
        config: "triton.Config",
        field: str,
    ) -> list["triton.Config"]:  # pyrefly: ignore [missing-attribute]
        """Return every valid neighbour of ``config`` along ``field``
        only — the per-axis counterpart of ``get_neighbour_values``.

        Precondition: ``field`` must be present on ``config``. Callers
        iterating ``tunable_fields`` against arbitrary configs must
        filter empty fields up front; ``autotune`` does."""
        cur_val = get_field(config, field)
        assert cur_val is not None, f"field {field!r} not present on config"
        neighbours = []
        for next_val in self.get_neighbour_values(field, cur_val):
            candidate = copy.deepcopy(config)
            set_field(candidate, field, next_val)
            if self.is_valid_config(candidate):
                neighbours.append(candidate)
        return neighbours

    def autotune(
        self,
        # pyrefly: ignore [missing-attribute]
        func: Callable[["triton.Config"], float],
        # pyrefly: ignore [missing-attribute]
        baseline_config: "triton.Config",
        baseline_timing: float | None = None,
    ) -> "triton.Config":  # pyrefly: ignore  # missing-attribute
        """
        Perform coordinate descent autotuning starting from a baseline configuration.
        """
        if baseline_timing is None:
            baseline_timing = self.call_func(func, baseline_config)

        log.debug("= Do coordinate descent tuning for %s =", self.name)
        log.debug(
            "%s: Baseline Config %s, baseline timing %f",
            self.name,
            baseline_config,
            baseline_timing,
        )
        improved = True
        best_config = baseline_config
        best_timing = baseline_timing

        while improved:
            improved = False

            # Gauss-Seidel: walk one field at a time, applying any
            # improvement immediately so subsequent fields see the
            # updated working point.
            for field in self.tunable_fields:
                # Some kernels don't have R0_BLOCK / YBLOCK / ZBLOCK,
                # so the field may be absent from ``best_config``.
                # Filter here so ``get_neighbour_configs`` can assume
                # the field is present.
                if get_field(best_config, field) is None:
                    continue
                for candidate_config in self.get_neighbour_configs(best_config, field):
                    cmp_res, candidate_timing = self.compare_config(
                        func, candidate_config, best_config, best_timing
                    )
                    if cmp_res:
                        improved = True
                        best_config, best_timing = candidate_config, candidate_timing

            if not improved and self.inductor_meta.get(
                "coordinate_descent_check_all_directions"
            ):
                old_best_timing = best_timing
                improved, best_config, best_timing = self.check_all_tuning_directions(
                    func, best_config, best_timing
                )

                if improved:
                    msg = red_text(
                        "%s: Coordinate descend tuning found improvement of %.3fx by looking in all directions."
                    )
                    log.debug(
                        msg,
                        self.name,
                        old_best_timing / best_timing,
                    )

        log.debug(
            "%s: Improve from %s %f -> %s %f, %.3fx",
            self.name,
            baseline_config,
            baseline_timing,
            best_config,
            best_timing,
            baseline_timing / best_timing,
        )

        return best_config

    @staticmethod
    def autotune_single_field(fn, init_val, min_val=None, max_val=None):
        """
        fn is a function that takes the field value and returns the benchmarking result
        init_val is the starting point of autotuning.

        Should work well for parabola like curve. Here is a real example
        for split-size of mix-order-reduction: https://github.com/pytorch/pytorch/pull/166461
        """
        cache = {}

        def _bench(val):
            if val not in cache:
                cache[val] = fn(val)
                # print(f"split size {val} -> {cache[val]:.3f} ms")
            return cache[val]

        if min_val is None:
            min_val = 1
        if max_val is None:
            max_val = 2**30  # some arbitrary large value

        best_val = init_val
        improved = True
        while improved:
            improved = False
            candlist = [best_val // 2, best_val * 2]
            for cand in candlist:
                cand = max(cand, min_val)
                cand = min(cand, max_val)

                if _bench(cand) < _bench(best_val):
                    best_val = cand
                    improved = True

        return best_val
