import os
import sys

import torch
from typing import List, Any, Dict, Tuple, Optional

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase

if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )

# Tests for torch.jit.isinstance2
class TestIsinstance2(JitTestCase):
    def test_int(self):
        def int_test(x: Any):
            assert torch.jit.isinstance2(x, int)
            assert not torch.jit.isinstance2(x, float)

        x = 1
        self.checkScript(int_test, (x,))

    def test_float(self):
        def float_test(x: Any):
            assert torch.jit.isinstance2(x, float)
            assert not torch.jit.isinstance2(x, int)

        x = 1.0
        self.checkScript(float_test, (x,))

    def test_bool(self):
        def bool_test(x: Any):
            assert torch.jit.isinstance2(x, bool)
            assert not torch.jit.isinstance2(x, float)

        x = False
        self.checkScript(bool_test, (x,))

    def test_list(self):
        def list_str_test(x: Any):
            assert torch.jit.isinstance2(x, List[str])
            assert not torch.jit.isinstance2(x, List[int])

        x = ["1", "2", "3"]
        self.checkScript(list_str_test, (x,))

    def test_dict(self):
        def dict_str_int_test(x: Any):
            assert torch.jit.isinstance2(x, Dict[str, int])
            assert not torch.jit.isinstance2(x, Dict[int, str])

        x = {"a": 1, "b": 2}
        self.checkScript(dict_str_int_test, (x,))

    def test_tuple(self):
        def tuple_test(x: Any):
            assert torch.jit.isinstance2(x, Tuple[str, int, str])
            assert not torch.jit.isinstance2(x, Tuple[int, str, str])
            assert not torch.jit.isinstance2(x, Tuple[str])

        x = ("a", 1, "b")
        self.checkScript(tuple_test, (x,))

    def test_optional(self):
        def optional_test(x: Any):
            assert torch.jit.isinstance2(x, Optional[torch.Tensor])
            assert not torch.jit.isinstance2(x, Optional[str])
            # TODO: successful torch.jit.isinstance2 makes sets type?

        x = torch.ones(3, 3)
        self.checkScript(optional_test, (x,))

    def test_optional_none(self):
        def optional_test_none(x: Any):
            assert torch.jit.isinstance2(x, Optional[torch.Tensor])
            # assert not torch.jit.isinstance2(x, Optional[str])
            # TODO: above line fails in TS interpreter need to investigate

        x = None
        self.checkScript(optional_test_none, (x,))

    def test_list_nested(self):
        def list_nested(x: Any):
            assert torch.jit.isinstance2(x, List[Dict[str, int]])
            assert not torch.jit.isinstance2(x, List[List[str]])

        x = [{"a": 1, "b": 2}, {"aa": 11, "bb": 22}]
        self.checkScript(list_nested, (x,))

    def test_dict_nested(self):
        def dict_nested(x: Any):
            assert torch.jit.isinstance2(x, Dict[str, Tuple[str, str, str]])
            assert not torch.jit.isinstance2(x, Dict[str, Tuple[int, int, int]])

        x = {"a": ("aa", "aa", "aa"), "b": ("bb", "bb", "bb")}
        self.checkScript(dict_nested, (x,))

    def test_tuple_nested(self):
        def tuple_nested(x: Any):
            assert torch.jit.isinstance2(
                x, Tuple[Dict[str, Tuple[str, str, str]], List[bool], Optional[str]]
            )
            assert not torch.jit.isinstance2(x, Dict[str, Tuple[int, int, int]])
            assert not torch.jit.isinstance2(x, Tuple[str])

        x = (
            {"a": ("aa", "aa", "aa"), "b": ("bb", "bb", "bb")},
            [True, False, True],
            None,
        )
        self.checkScript(tuple_nested, (x,))

    def test_optional_nested(self):
        def optional_nested(x: Any):
            assert torch.jit.isinstance2(x, Optional[List[str]])

        x = ["a", "b", "c"]
        self.checkScript(optional_nested, (x,))

    def test_list_tensor_type_true(self):
        def list_tensor_type_true(x: Any):
            assert torch.jit.isinstance2(x, List[torch.Tensor])

        x = [torch.rand(3, 3), torch.rand(4, 3)]
        self.checkScript(list_tensor_type_true, (x,))

    def test_tensor_type_false(self):
        def list_tensor_type_false(x: Any):
            assert not torch.jit.isinstance2(x, List[torch.Tensor])

        x = [1, 2, 3]
        self.checkScript(list_tensor_type_false, (x,))

    def test_in_if(self):
        def list_in_if(x: Any):
            if torch.jit.isinstance2(x, List[int]):
                assert True
            if torch.jit.isinstance2(x, List[str]):
                assert not True

        x = [1, 2, 3]
        self.checkScript(list_in_if, (x,))

    def test_if_else(self):
        def list_in_if_else(x: Any):
            if torch.jit.isinstance2(x, Tuple[str, str, str]):
                assert True
            else:
                assert not True

        x = ("a", "b", "c")
        self.checkScript(list_in_if_else, (x,))

    def test_in_while_loop(self):
        def list_in_while_loop(x: Any):
            count = 0
            while torch.jit.isinstance2(x, List[Dict[str, int]]) and count <= 0:
                count = count + 1
            assert count == 1

        x = [{"a": 1, "b": 2}, {"aa": 11, "bb": 22}]
        self.checkScript(list_in_while_loop, (x,))

    def test_switch_on_type(self):
        def list_switch_on_type(obj: Any):
            hit = False
            if torch.jit.isinstance2(obj, List[torch.Tensor]):
                hit = not hit
                for el in obj:
                    # perform some tensor operation
                    y = el.clamp(0, 0.5)
            if torch.jit.isinstance2(obj, Dict[str, str]):
                hit = not hit
                str_cat = ""
                for val in obj.values():
                    str_cat = str_cat + val
                assert "111222" == str_cat
            assert hit

        x = [torch.rand(3, 3), torch.rand(4, 3)]
        self.checkScript(list_switch_on_type, (x,))
        x = {"1": "111", "2": "222"}
        self.checkScript(list_switch_on_type, (x,))
