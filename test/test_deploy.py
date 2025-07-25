# Owner(s): ["oncall: package/deploy"]

import textwrap
import types

from torch.testing._internal.common_utils import run_tests, TestCase
from torch.utils._freeze import Freezer, PATH_MARKER


class TestFreezer(TestCase):
    """Tests the freeze.py script"""

    def test_compile_string(self):
        freezer = Freezer(True)
        code_str = textwrap.dedent(
            """
            class MyCls:
                def __init__(self) -> None:
                    pass
            """
        )
        co = freezer.compile_string(code_str)
        num_co = 0

        def verify_filename(co: types.CodeType):
            nonlocal num_co

            if not isinstance(co, types.CodeType):
                return

            self.assertEqual(PATH_MARKER, co.co_filename)
            num_co += 1

            for nested_co in co.co_consts:
                verify_filename(nested_co)

        verify_filename(co)
        # there is at least one nested code object besides the top level one
        self.assertTrue(num_co >= 2)


if __name__ == "__main__":
    run_tests()
