import triton
from typing import Optional, Callable
import copy

def get_field(config, name):
    if name == "num_warps":
        return config.num_warps
    elif name == "num_stages":
        return config.num_stages
    else:
        return config.kwargs.get(name, None)

def set_field(config, name, value):
    """
    name should always be valid since we should have already called
    get_field with the name.
    """
    if name == "num_warps":
        config.num_warps = value
    elif name == "num_stages":
        config.num_stages = value
    else:
        config.kwargs[name] = value


class CoordescTuner:
    """
    The coordinate descent tuner.
    """

    def __init__(self, is_mm=False):
        self.is_mm = is_mm  # we will tune num_stages for mm

    @property
    def tunable_fields(self):
        out = [
            "XBLOCK",

            # for reduction
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
    def get_explore_values(name, cur_val):
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
        threshold = 0.00001
        return test is not None and test < baseline * (1 - threshold)

    def autotune(self, func: Callable[triton.Config, float], baseline_config: triton.Config, baseline_timing: Optional[float]=None):
        if baseline_timing is None:
            baseline_timing = func(baseline_config)
        print(f"Baselien Config {baseline_config}, baseline timing {baseline_timing}")
        improved = True
        best_config = baseline_config
        best_timing = baseline_timing
        tunable_fields = self.tunable_fields
        while improved:
            improved = False

            for name in tunable_fields:
                cur_val = get_field(best_config, name)
                # some kernel don't have RBLOCK. So cur_val may be None
                if cur_val is None:
                    continue
                
                candidate_values = self.get_explore_values(name, cur_val)
                assert len(candidate_values) > 0

                for next_val in candidate_values:
                    candidate_config = copy.deepcopy(best_config)
                    set_field(candidate_config, name, next_val)
                    candidate_timing = func(candidate_config)
                    
                    if self.has_improvement(best_timing, candidate_timing):
                        improved = True
                        print(f"Tune from {best_config} {best_timing} -> {candidate_config} {candidate_timing}")
                        best_timing = candidate_timing
                        best_config = candidate_config
                        break
        
        print(f"Improve from {baseline_config} {baseline_timing} -> {best_config} {best_timing}, {baseline_timing / best_timing:.3f}x")
