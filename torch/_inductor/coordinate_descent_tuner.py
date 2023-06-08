import copy
import logging
from typing import Callable, Optional

from .utils import has_triton, triton_config_to_hashable

if has_triton():
    import triton
else:
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

    TODO will it be necessary to tune multiple fields simultanuously.


    TODO: what if both increasing and descreasing a field can improve perf.
          i.e., there are multiple local optima..
    """

    def __init__(self, is_mm=False, name="unknown"):
        self.is_mm = is_mm  # we will tune num_stages for mm
        self.cached_benchmark_results = {}
        self.name = name

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
    def get_neighbour_values(name, cur_val):
        lhs_val = None
        rhs_val = None
        if name == "num_stages":
            lhs_val = cur_val - 1
            rhs_val = cur_val + 1
        else:
            lhs_val = cur_val // 2
            rhs_val = cur_val * 2

        out = []
        if lhs_val > 0:
            out.append(lhs_val)
        out.append(rhs_val)
        return out

    @staticmethod
    def has_improvement(baseline, test):
        threshold = 0.001  # 0.1%
        return test is not None and test < baseline * (1 - threshold)

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

                candidate_values = self.get_neighbour_values(name, cur_val)
                assert len(candidate_values) > 0

                for next_val in candidate_values:
                    candidate_config = copy.deepcopy(best_config)
                    set_field(candidate_config, name, next_val)
                    log.debug("Try config %s", candidate_config)
                    try:
                        candidate_timing = self.call_func(func, candidate_config)
                    except Exception as e:
                        log.debug("Got exception %s", e)
                        continue

                    if self.has_improvement(best_timing, candidate_timing):
                        improved = True
                        log.debug(
                            "Tune from %s %f -> %s %f",
                            best_config,
                            best_timing,
                            candidate_config,
                            candidate_timing,
                        )
                        best_timing = candidate_timing
                        best_config = candidate_config

        log.debug(
            "Improve from %s %f -> %s %f, %.3fx",
            baseline_config,
            baseline_timing,
            best_config,
            best_timing,
            baseline_timing / best_timing,
        )

        return best_config
