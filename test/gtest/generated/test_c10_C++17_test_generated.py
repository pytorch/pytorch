import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/c10_C++17_test"


class Testif_constexpr(TestCase):
    cpp_name = "if_constexpr"

    def test_whenIsTrue_thenReturnsTrueCase(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "whenIsTrue_thenReturnsTrueCase")

    def test_whenIsFalse_thenReturnsFalseCase(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "whenIsFalse_thenReturnsFalseCase")

    def test_worksWithMovableOnlyTypes_withIdentityArg(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "worksWithMovableOnlyTypes_withIdentityArg"
        )

    def test_worksWithMovableOnlyTypes_withoutIdentityArg(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "worksWithMovableOnlyTypes_withoutIdentityArg"
        )

    def test_otherCaseCanHaveInvalidCode(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "otherCaseCanHaveInvalidCode")

    def test_worksWithoutElseCase_withIdentityArg(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "worksWithoutElseCase_withIdentityArg")

    def test_worksWithoutElseCase_withoutIdentityArg(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "worksWithoutElseCase_withoutIdentityArg"
        )

    def test_returnTypeCanDiffer_withIdentityArg(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "returnTypeCanDiffer_withIdentityArg")

    def test_returnTypeCanDiffer_withoutIdentityArg(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "returnTypeCanDiffer_withoutIdentityArg"
        )


if __name__ == "__main__":
    run_tests()
