import time

import torch.testing._internal.common_utils as common

from torch.utils.cpp_extension import (
    load_inline,
    remove_extension_h_precompiler_headers,
)


def load_inline_without_pch():
    source_orig = """
    at::Tensor sin_add_orig(at::Tensor x, at::Tensor y) {
        return x.sin() + y.sin();
    }
    """
    remove_extension_h_precompiler_headers()
    start = time.time()
    module = load_inline(
        name="inline_extension_orig",
        cpp_sources=[source_orig],
        functions=["sin_add_orig"],
        use_pch=False,
    )
    end = time.time()
    return end - start


def load_inline_with_pch():
    source_pch = """
    at::Tensor sin_add_pch(at::Tensor x, at::Tensor y) {
        return x.sin() + y.sin();
    }
    """
    start = time.time()
    module = load_inline(
        name="inline_extension_pch",
        cpp_sources=[source_pch],
        functions=["sin_add_pch"],
        use_pch=True,
    )
    end = time.time()
    return end - start


def load_inline_gen_pch():
    source_gen = """
    at::Tensor sin_add_gen(at::Tensor x, at::Tensor y) {
        return x.sin() + y.sin();
    }
    """
    start = time.time()
    module = load_inline(
        name="inline_extension_gen",
        cpp_sources=[source_gen],
        functions=["sin_add_gen"],
        use_pch=True,
    )
    end = time.time()
    return end - start


class TestCppExtensionPCH(common.TestCase):
    def test_pch(self):
        # genarate PCH
        time_gen_pch = load_inline_gen_pch()
        print(f"compile time, gen pch: {time_gen_pch}")

        time_pch = load_inline_with_pch()
        time_orig = load_inline_without_pch()
        print(f"compile time, origin: {time_orig}, with pch: {time_pch}")


if __name__ == "__main__":
    common.run_tests()
