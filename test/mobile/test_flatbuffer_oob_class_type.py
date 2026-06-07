# Regression test for out-of-bounds read in the mobile flatbuffer loader.
#
# A malformed flatbuffer model can set Function.class_type to an index that is
# out of range for the module's object_types table. Before the fix,
# FlatbufferLoader::parseModule indexed all_types_ with that value via the
# unchecked std::vector::operator[] and dereferenced the result, causing a
# SIGSEGV while loading. The loader must reject such a module with a clean
# RuntimeError instead of crashing.
#
# The buffer below was produced with the flatbuffers runtime (see craft.py in
# the PR description): one Function ivalue with class_type=100 and an empty
# object_types table. It passes VerifyModuleBuffer but exercises the bug. It is
# embedded as base64 so the test has no third-party build/runtime dependency.
#
# Suggested location: test/mobile/test_lite_script_module.py (or
# test/cpp/jit/test_flatbuffer.cpp for a C++ equivalent).

import base64
import unittest

import torch
from torch.testing._internal.common_utils import TestCase, run_tests

# 176-byte malformed mobile flatbuffer module (class_type index out of range).
_MALFORMED_MODULE_B64 = (
    "JAAAAFBUTUYAABoAFAAQAAAAAAAAAAwAAAAAAAgAAAAAAAQAGgAAAP///38MAAAADAAAAAkA"
    "AAAAAAAAAQAAAAwAAAAIAAoACQAEAAgAAAAcAAAAABAWABwAGAAUABAADAAIAAAAAAAAAAQA"
    "FgAAAGQAAAAUAAAAFAAAABQAAAAUAAAAFAAAAAAAAAAAAAAAAAAAAAAAAAATAAAAX190b3Jj"
    "aF9fLk0uZm9yd2FyZAA="
)


class TestFlatbufferLoaderRobustness(TestCase):
    def test_out_of_range_class_type_is_rejected(self):
        buf = base64.b64decode(_MALFORMED_MODULE_B64)
        # Must raise (bounds check in getType), not segfault.
        with self.assertRaises(RuntimeError):
            torch._C._load_mobile_module_from_bytes(buf)


if __name__ == "__main__":
    run_tests()
