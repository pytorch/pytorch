# Owner(s): ["module: dynamo"]
# flake8: noqa: F401, F403, F405
# ruff: noqa: F401,F403,F405
try:
    from ._test_misc_common import *
    from ._test_misc_extra import (
        DynamoOpPromotionTests,
        MiscTestsDevice,
        MiscTestsPyTree,
        TestCustomFunction,
        TestTracer,
    )
    from ._test_misc_part1 import MiscTestsPart1
    from ._test_misc_part2 import MiscTestsPart2
    from ._test_misc_part3 import MiscTestsPart3
    from ._test_misc_part4 import MiscTestsPart4
    from ._test_misc_part5 import MiscTestsPart5
    from ._test_misc_part6 import MiscTestsPart6
    from ._test_misc_part7 import MiscTestsPart7
    from ._test_misc_part8 import MiscTestsPart8
except ImportError:
    from _test_misc_common import *
    from _test_misc_extra import (
        DynamoOpPromotionTests,
        MiscTestsDevice,
        MiscTestsPyTree,
        TestCustomFunction,
        TestTracer,
    )
    from _test_misc_part1 import MiscTestsPart1
    from _test_misc_part2 import MiscTestsPart2
    from _test_misc_part3 import MiscTestsPart3
    from _test_misc_part4 import MiscTestsPart4
    from _test_misc_part5 import MiscTestsPart5
    from _test_misc_part6 import MiscTestsPart6
    from _test_misc_part7 import MiscTestsPart7
    from _test_misc_part8 import MiscTestsPart8


class MiscTests(
    MiscTestsPart1,
    MiscTestsPart2,
    MiscTestsPart3,
    MiscTestsPart4,
    MiscTestsPart5,
    MiscTestsPart6,
    MiscTestsPart7,
    MiscTestsPart8,
    torch._inductor.test_case.TestCase,
):
    pass


instantiate_parametrized_tests(MiscTestsPyTree)

devices = ("cuda", "hpu", "xpu")
instantiate_device_type_tests(
    MiscTestsDevice, globals(), only_for=devices, allow_xpu=True
)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
