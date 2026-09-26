# Owner(s): ["module: inductor"]
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

from torch.testing._internal.common_utils import run_tests, TestCase


if TYPE_CHECKING:
    from types import ModuleType


# test/inductor/ -> test/ -> caffe2/
_CAFFE2_ROOT = Path(__file__).resolve().parents[2]
_GEN_PATH = _CAFFE2_ROOT / "torchgen" / "gen_aoti_shim_symbol_prefix.py"
_HEADER_PATH = _CAFFE2_ROOT / "torch/csrc/inductor/aoti_torch/c/shim_symbol_prefix.h"

# A sample of symbols that were silently dropped by the old hand-maintained
# list; each exercises a different shim header (core, cpu fallback, cuda
# fallback, allocator, dtype table).
_CRITICAL_TOKENS = [
    "aoti_torch_index_put_out",
    "aoti_torch_cpu_view_dtype",
    "aoti_torch_is_defined",
    "aoti_torch_get_size",
    "aoti_torch_get_stride",
    "aoti_torch_cuda_caching_allocator_raw_alloc",
    "aoti_torch_cuda_caching_allocator_raw_delete",
    "aoti_torch__mm_plus_mm_out",
    "aoti_torch_call_dispatcher",
    "aoti_torch_dtype_float64",
    "aoti_torch_dtype_complex64",
]


def _load_generator() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "gen_aoti_shim_symbol_prefix", _GEN_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load generator from {_GEN_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AotiShimSymbolPrefixTest(TestCase):
    def test_header_is_up_to_date(self) -> None:
        gen = _load_generator()
        expected = gen.generate_header(gen.caffe2_root())
        actual = _HEADER_PATH.read_text()
        self.assertEqual(
            expected,
            actual,
            "shim_symbol_prefix.h is stale; regenerate with "
            "`python torchgen/gen_aoti_shim_symbol_prefix.py --output "
            f"{gen.HEADER_REL_PATH}`.",
        )

    def test_critical_tokens_present(self) -> None:
        content = _HEADER_PATH.read_text()
        for token in _CRITICAL_TOKENS:
            self.assertIn(f"#define {token} ", content)


if __name__ == "__main__":
    run_tests()
