import copy
import logging
from typing import Callable, Optional

import triton

from . import config

from .utils import triton_config_to_hashable

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

    TODO will it be necessary to tune multiple fields simultanuously.


    TODO: what if both increasing and descreasing a field can improve perf.
          i.e., there are multiple local optima..
    """

    def __init__(self, is_mm=False, name="unknown"):
        self.is_mm = is_mm  # we will tune num_stages for mm
        self.cached_benchmark_results = {}
        self.name = name

        # for backtracking when checking all directions
        self.best_config = None
        self.best_timing = None
        self.cur_config = None

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

    @staticmethod
    def get_neighbour_values(name, orig_val, radius=1):
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
            out.append(cur_val)

        # decrement loop
        cur_val = orig_val
        for _ in range(radius):
            cur_val = update(cur_val, False)
            if cur_val <= 0:
                break
            out.append(cur_val)

        return out

    @staticmethod
    def has_improvement(baseline, test):
        threshold = 0.001  # 0.1%
        return test is not None and test < baseline * (1 - threshold)

    def check_all_dir(
        self, func: Callable[[triton.Config], float], next_field_idx: int = 0
    ):
        """
        Check all directions. We only do this once the regular coordinate
        descent tuning find no better choices any more.
        We only have a few tunable fields, so this should be fine.
        """
        if next_field_idx == len(self.tunable_fields):
            # check if self.cur_config is better
            # do deepcopy before running func since func may associate the
            # config to the launcher.  If we do deepcopy after calling func,
            # we may not able to find the launcher based on the copy of the config.
            # That's because triton.Config uses class object's hash function.
            config_copy = copy.deepcopy(self.cur_config)
            cmp_res, candidate_timing = self.compare_config(
                func, config_copy, self.best_config, self.best_timing
            )
            if cmp_res:
                self.best_config = config_copy
                self.best_timing = candidate_timing
            else:
                config_copy = None
            return cmp_res

        field = self.tunable_fields[next_field_idx]

        # go thru different tuning choices for field
        old_val = get_field(self.cur_config, field)

        out = self.check_all_dir(func, next_field_idx + 1)

        # we can tune this field only if the value in the config is not None
        if old_val is not None:
            candidate_values = self.get_neighbour_values(
                field, old_val, radius=config.coordinate_descent_search_radius
            )
            assert len(candidate_values) > 0

            for next_val in candidate_values:
                set_field(self.cur_config, field, next_val)
                # operands for 'or' matters due to short-circuit
                out = self.check_all_dir(func, next_field_idx + 1) or out

            # recover the old value for field
            set_field(self.cur_config, field, old_val)

        return out

    def compare_config(self, func, candidate_config, best_config, best_timing):
        """
        Check if candidate_config is better than best_config.

        Return a touple of (compare_result, candidate_timing).
        compare_result is true iff condidate_config is better.
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
        func: Callable[[triton.Config], float],
        baseline_config: triton.Config,
        baseline_timing: Optional[float] = None,
    ) -> triton.Config:
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

                candidate_values = self.get_neighbour_values(name, cur_val)
                assert len(candidate_values) > 0

                for next_val in candidate_values:
                    candidate_config = copy.deepcopy(best_config)
                    set_field(candidate_config, name, next_val)

                    cmp_res, candidate_timing = self.compare_config(
                        func, candidate_config, best_config, best_timing
                    )
                    if cmp_res:
                        best_config, best_timing = candidate_config, candidate_timing

            if not improved and config.coordinate_descent_check_all_dir:
                self.best_config = best_config
                self.best_timing = best_timing
                self.cur_config = copy.deepcopy(self.best_config)

                improved = self.check_all_dir(func)

                if improved:
                    import colorama

                    msg = (
                        colorama.Fore.RED
                        + "LOOKING IN ALL DIR FOUND BETTER CONFIGS, IMPROVE %.3fx"
                        + colorama.Fore.RESET
                    )
                    log.debug(
                        msg,
                        best_timing / self.best_timing,
                    )

                best_config = self.best_config
                best_timing = self.best_timing
                self.best_config = self.best_timing = self.cur_config = None

        log.debug(
            "Improve from %s %f -> %s %f, %.3fx",
            baseline_config,
            baseline_timing,
            best_config,
            best_timing,
            baseline_timing / best_timing,
        )

        return best_config
