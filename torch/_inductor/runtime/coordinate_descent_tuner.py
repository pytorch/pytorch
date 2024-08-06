# mypy: allow-untyped-defs
import copy
import itertools
import logging
from typing import Callable, Optional

from .hints import TRITON_MAX_BLOCK
from .runtime_utils import red_text, triton_config_to_hashable


try:
    import triton
except ImportError:
    triton = None

log = logging.getLogger(__name__)


def get_field(config, name):
    if name == "num_warps":
        return config.num_warps
    elif name == "num_stages":
        return config.num_stages
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
        self, is_mm=False, name="unknown", size_hints=None, inductor_meta=None
    ):
        self.is_mm = is_mm  # we will tune num_stages for mm
        self.cached_benchmark_results = {}
        self.name = name
        self.size_hints = size_hints
        self.inductor_meta = inductor_meta or {}

    def prefix_to_size_hint(self, prefix: str) -> Optional[int]:
        size_hint_idx = {"X": 0, "Y": 1, "Z": 2, "R": -1}[prefix]

        have_size_hint = (
            self.size_hints is not None
            and len(self.size_hints) > 0
            and len(self.size_hints) > size_hint_idx
        )
        return self.size_hints[size_hint_idx] if have_size_hint else None

    def get_config_max(self, prefix: str) -> int:
        max_block = TRITON_MAX_BLOCK[prefix]
        size_hint = self.prefix_to_size_hint(prefix)
        return min(max_block, size_hint) if size_hint is not None else max_block

    def get_warpsmax(self):
        # Currently, CUDA has a maximum of 1024 threads, so 32 is the max
        # number of warps.
        return 1024 // 32

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
    def tunable_fields(self):
        out = [
            "XBLOCK",
            "YBLOCK",
            "ZBLOCK",
            # NOTE: we should not tune RBLOCK for persistent reduction.
            # We rely on the fact that persistent reduction's triton.Config
            # does not have the RBLOCK field to guarantee that.
            "RBLOCK",
            # the following 3 are for mm
            "BLOCK_M",
            "BLOCK_N",
            "BLOCK_K",
            "num_warps",
        ]
        if self.is_mm:
            out.append("num_stages")

        return out

    def value_too_large(self, name: str, val: int) -> bool:
        if name in {"XBLOCK", "YBLOCK", "ZBLOCK", "RBLOCK"}:
            return val > self.get_config_max(name[0])
        if name == "num_warps":
            return val > self.get_warpsmax()

        return False

    def get_neighbour_values(self, name, orig_val, radius=1, include_self=False):
        """
        Get neighbour values in 'radius' steps. The original value is not
        returned as it's own neighbour.
        """
        assert radius >= 1

        def update(cur_val, inc=True):
            if name == "num_stages":
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
            if cur_val <= 0:
                break
            out.append(cur_val)

        if include_self:
            out.append(orig_val)
        return out

    @staticmethod
    def has_improvement(baseline, test):
        threshold = 0.001  # 0.1%
        return test is not None and test < baseline * (1 - threshold)

    def check_all_tuning_directions(
        self,
        func: Callable[["triton.Config"], float],
        best_config,
        best_timing,
    ):
        """
        Check all directions. We only do this once the regular coordinate
        descent tuning find no better choices any more.
        We only have a few tunable fields, so this should be fine.
        """
        candidate_values_list = []
        effective_fields = []
        for field in self.tunable_fields:
            old_value = get_field(best_config, field)
            if old_value is None:
                continue
            candidate_values = self.get_neighbour_values(
                field,
                old_value,
                radius=self.inductor_meta.get("coordinate_descent_search_radius", 1),
                include_self=True,
            )
            candidate_values_list.append(candidate_values)
            effective_fields.append(field)

        choices = itertools.product(*candidate_values_list)
        improved = False
        for choice in choices:
            assert len(choice) == len(effective_fields)
            candidate_config = copy.deepcopy(best_config)
            for new_val, field in zip(choice, effective_fields):
                set_field(candidate_config, field, new_val)
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

        Return a touple of (compare_result, candidate_timing).
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

    def autotune(
        self,
        func: Callable[["triton.Config"], float],
        baseline_config: "triton.Config",
        baseline_timing: Optional[float] = None,
    ) -> "triton.Config":
        if baseline_timing is None:
            baseline_timing = self.call_func(func, baseline_config)

        log.debug("= Do coordinate descent tuning for %s =", self.name)
        log.debug(
            "Baseline Config %s, baseline timing %f", baseline_config, baseline_timing
        )
        improved = True
        best_config = baseline_config
        best_timing = baseline_timing
        tunable_fields = self.tunable_fields

        while improved:
            improved = False

            for name in tunable_fields:
                cur_val = get_field(best_config, name)
                # some kernel don't have RBLOCK/YBLOCK/ZBLOCK. So cur_val may be None
                if cur_val is None:
                    continue

                # It's possible that candidate_values is empty.
                # E.g., if XBLOCK is 1 initially and size_hint for x is also 1.
                # We would not try either larger or smaller XBLOCK in this case.
                candidate_values = self.get_neighbour_values(name, cur_val)

                for next_val in candidate_values:
                    candidate_config = copy.deepcopy(best_config)
                    set_field(candidate_config, name, next_val)

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
                        "Coordinate descend tuning found improvement of %.3fx by looking in all directions."
                    )
                    log.debug(
                        msg,
                        old_best_timing / best_timing,
                    )

        log.debug(
            "Improve from %s %f -> %s %f, %.3fx",
            baseline_config,
            baseline_timing,
            best_config,
            best_timing,
            baseline_timing / best_timing,
        )

        return best_config
