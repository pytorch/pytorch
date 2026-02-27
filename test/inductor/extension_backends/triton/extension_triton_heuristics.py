from typing import Any

from torch._inductor.runtime import triton_heuristics
from torch._inductor.runtime.triton_heuristics import user_autotune  # noqa: F401


EXTENSION_TRITON_META_FIELD = "extension_custom_field"


class ExtensionCachingAutotuner(triton_heuristics.CachingAutotuner):
    def _create_compile_meta(
        self,
        cfg: triton_heuristics.Config,
    ) -> dict[str, Any]:
        if EXTENSION_TRITON_META_FIELD not in self.triton_meta:
            raise AssertionError
        compile_meta = super()._create_compile_meta(cfg)
        if EXTENSION_TRITON_META_FIELD not in compile_meta:
            raise AssertionError
        return compile_meta


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
    configs = [triton_heuristics.Config({"XBLOCK": 32})]
    return triton_heuristics.cached_autotune(
        size_hints,
        configs,
        triton_meta=triton_meta,
        inductor_meta=inductor_meta,
        heuristic_type=triton_heuristics.HeuristicType.POINTWISE,
        filename=filename,
        caching_autotuner_cls=ExtensionCachingAutotuner,
    )
