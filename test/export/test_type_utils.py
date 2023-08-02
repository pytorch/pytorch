from typing import List, Union, Dict
from torch._export.serde.type_utils import check, Succeed, Fail
from torch.testing._internal.common_utils import (
    run_tests,
    TestCase,
)


class JustATest:
    pass


class JustAnotherTest:
    pass


class TestUnionTypeCheck(TestCase):
    """The _Union"""

    def test_type_utils_check(self):
        self.assertEqual(
            check(["yada", "hurray"], List[int]),
            Succeed()
        )

        self.assertEqual(
            check({"python": "py", "racket": "rkt"}, Union[List[int], Dict[str, str], float]),
            Succeed()
        )

        self.assertEqual(
            check(JustATest(), JustATest),
            Succeed()
        )

        failed_check = check(JustATest(), JustAnotherTest)
        self.assertTrue(isinstance(failed_check, Fail))
        self.assertIn("Expected JustAnotherTest", failed_check.msg[0])


if __name__ == '__main__':
    run_tests()
