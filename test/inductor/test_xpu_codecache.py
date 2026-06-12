# Owner(s): ["module: inductor"]

import ctypes
import unittest

import torch
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.codecache import XPUCodeCache
from torch._inductor.codegen.xpu.compile_utils import icpx_exist
from torch._inductor.exc import XPUCompileError
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import fresh_cache
from torch.testing._internal.triton_utils import requires_xpu_and_triton


_SOURCE_CODE = r"""

#include <sycl/sycl.hpp>

extern "C" {

__attribute__((__visibility__("default")))
int saxpy(int n, float a, float *x, float *y) {
    sycl::queue q{sycl::gpu_selector_v};
    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
        y[i] = a * x[i] + y[i];
    }).wait();
    return 0;
}

}
"""


@unittest.skipUnless(icpx_exist(), "requires icpx")
class TestXPUCodeCache(InductorTestCase):
    @requires_xpu_and_triton
    def test_xpu_load(self):
        with fresh_cache():
            # Test both .o and .so compilation.
            (
                object_file_path,
                object_hash_key,
                source_code_path0,
            ) = XPUCodeCache.compile(_SOURCE_CODE, "o")
            dll_wrapper, so_hash_key, source_code_path1 = XPUCodeCache.load(
                _SOURCE_CODE, "so"
            )
            self.assertEqual(source_code_path0, source_code_path1)
            self.assertEqual(object_hash_key, so_hash_key)

            # Test load and call functions in .so.
            x = torch.rand(10).float().xpu()
            y = torch.rand(10).float().xpu()
            a = 5.0
            expected_y = a * x + y
            dll_wrapper.saxpy(
                ctypes.c_int(10),
                ctypes.c_float(a),
                ctypes.c_void_p(x.data_ptr()),
                ctypes.c_void_p(y.data_ptr()),
            )
            torch.testing.assert_close(y, expected_y)

    @requires_xpu_and_triton
    def test_compilation_error(self):
        with fresh_cache():
            # Reference an undefined symbol to force a compile error.
            error_source_code = _SOURCE_CODE.replace(
                "parallel_for", "parallel_for_undefined", 1
            )
            with self.assertRaises(XPUCompileError):
                XPUCodeCache.compile(error_source_code, "o")

    @requires_xpu_and_triton
    def test_async_compile(self):
        with fresh_cache():
            async_compile = AsyncCompile()
            compiled_res = async_compile.xpu(_SOURCE_CODE, "so")
            async_compile.wait(globals())

            # Test load and call functions in .so.
            x = torch.rand(5).float().xpu()
            y = torch.rand(5).float().xpu()
            a = 2.0
            expected_y = a * x + y
            compiled_res.result().saxpy(
                ctypes.c_int(5),
                ctypes.c_float(a),
                ctypes.c_void_p(x.data_ptr()),
                ctypes.c_void_p(y.data_ptr()),
            )
            torch.testing.assert_close(y, expected_y)

    @requires_xpu_and_triton
    def test_cache_hit(self):
        with fresh_cache():
            # First compilation.
            _, hash_key1, source_path1 = XPUCodeCache.compile(_SOURCE_CODE, "o")

            # Second compilation of the same source should hit cache.
            _, hash_key2, source_path2 = XPUCodeCache.compile(_SOURCE_CODE, "o")

            # Verify hash consistency for cache hits.
            self.assertEqual(hash_key1, hash_key2)
            self.assertEqual(source_path1, source_path2)

            # Test with .so as well.
            dll_wrapper1, so_hash1, _ = XPUCodeCache.load(_SOURCE_CODE, "so")
            dll_wrapper2, so_hash2, _ = XPUCodeCache.load(_SOURCE_CODE, "so")

            self.assertEqual(so_hash1, so_hash2)
            # Both loads should return the same cached wrapper.
            self.assertIs(dll_wrapper1, dll_wrapper2)



class TestXPUCodeCacheClear(InductorTestCase):
    def test_cache_clear_clears_dll_cache(self):
        from unittest.mock import MagicMock

        XPUCodeCache.dll_cache["fake_path.so"] = MagicMock()
        self.assertGreater(len(XPUCodeCache.dll_cache), 0)

        XPUCodeCache.cache_clear()

        self.assertEqual(
            len(XPUCodeCache.dll_cache),
            0,
            "dll_cache should be empty after cache_clear()",
        )

    def test_cache_clear_clears_parent_cache(self):
        from unittest.mock import MagicMock

        XPUCodeCache.cache["fake_key"] = MagicMock()
        self.assertGreater(len(XPUCodeCache.cache), 0)

        XPUCodeCache.cache_clear()

        self.assertEqual(
            len(XPUCodeCache.cache),
            0,
            "cache should be empty after cache_clear()",
        )


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests("xpu")
