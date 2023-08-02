from typing import List, Union, Dict
from torch._export.serde.type_utils import check, Succeed, Fail, NotSure
from torch.testing._internal.common_utils import (
    run_tests,
    TestCase,
)


class JustATest:
    def __repr__(self) -> str:
        return "JustATest()"


class JustAnotherTest:
    def __repr__(self) -> str:
        return "JustAnotherTest()"


class TestUnionTypeCheck(TestCase):
    """The _Union"""

    def test_type_utils_check_lists(self):
        """
        Tests various list values
        Demonstrates the structure of error messages
        """

        self.assertEqual(check([[42, 120], [10, 5]], List[List[int]]), Succeed())

        self.assertEqual(
            check(["yada", "hurray"], List[int]),
            Fail(
                [
                    "['yada', 'hurray'] is not a List[int].",
                    "Expected int from yada, but got str."
                ]
            )
        )

        self.assertEqual(
            check([[42, 120], [10, "im_a_str"]], List[List[int]]),
            Fail(
                [
                    "[[42, 120], [10, 'im_a_str']] is not a List[List[int]].",
                    "[10, 'im_a_str'] is not a List[int].",
                    "Expected int from im_a_str, but got str."
                ]
            )
        )


    def test_type_utils_check(self) -> None:
        """
        Tests dictionaries, unions, and classes
        """

        self.assertEqual(
            check({"python": "py", "racket": "rkt"}, Union[List[int], Dict[str, str], float]),
            Succeed()
        )

        self.assertEqual(
            check({"five": 5}, Union[List[int], Dict[str, str], float]),
            Fail(
                ["{'five': 5} is not of type Union[List[int], Dict[str, str], float]."]
            )
        )

        self.assertEqual(
            check(JustATest(), JustATest),
            Succeed()
        )

        self.assertEqual(
            check(JustATest(), JustAnotherTest),
            Fail(["Expected JustAnotherTest from JustATest(), but got JustATest."])
        )

        self.assertEqual(
            check("fourty_two", 42), NotSure(v="fourty_two", t=42)
        )


if __name__ == '__main__':
    run_tests()
