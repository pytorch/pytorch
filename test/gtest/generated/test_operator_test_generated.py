import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/operator_test"


class TestOperatorDeathTest(TestCase):
    cpp_name = "OperatorDeathTest"

    def test_DISABLED_CannotAccessRepeatedParameterWithWrongType(self):
        run_cpp_test(
            TEST_BINARY,
            self.cpp_name,
            "DISABLED_CannotAccessRepeatedParameterWithWrongType",
        )


class TestOperatorTest(TestCase):
    cpp_name = "OperatorTest"

    def test_DeviceTypeRegistryWorks(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DeviceTypeRegistryWorks")

    def test_RegistryWorks(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RegistryWorks")

    def test_RegistryWrongDevice(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "RegistryWrongDevice")

    def test_ExceptionWorks(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ExceptionWorks")

    def test_FallbackIfEngineDoesNotBuild(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "FallbackIfEngineDoesNotBuild")

    def test_MultipleEngineChoices(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "MultipleEngineChoices")

    def test_CannotUseUninitializedBlob(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CannotUseUninitializedBlob")

    def test_TestParameterAccess(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestParameterAccess")

    def test_CannotAccessParameterWithWrongType(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "CannotAccessParameterWithWrongType")

    def test_TestDefaultValue(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestDefaultValue")

    def test_TestSetUp(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestSetUp")

    def test_TestSetUpInputOutputCount(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestSetUpInputOutputCount")

    def test_TestOutputValues(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestOutputValues")


class TestNetTest(TestCase):
    cpp_name = "NetTest"

    def test_TestScaffoldingSimpleNet(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestScaffoldingSimpleNet")

    def test_TestScaffoldingDAGNet(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "TestScaffoldingDAGNet")


class TestOperatorGradientRegistryTest(TestCase):
    cpp_name = "OperatorGradientRegistryTest"

    def test_GradientSimple(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "GradientSimple")


class TestEnginePrefTest(TestCase):
    cpp_name = "EnginePrefTest"

    def test_PerOpEnginePref(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PerOpEnginePref")

    def test_GlobalEnginePref(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "GlobalEnginePref")

    def test_GlobalEnginePrefAndPerOpEnginePref(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "GlobalEnginePrefAndPerOpEnginePref")

    def test_GlobalEnginePrefAndPerOpEnginePrefAndOpDef(self):
        run_cpp_test(
            TEST_BINARY, self.cpp_name, "GlobalEnginePrefAndPerOpEnginePrefAndOpDef"
        )

    def test_SetOpEnginePref(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SetOpEnginePref")

    def test_SetDefaultEngine(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "SetDefaultEngine")


class TestRequiredArg(TestCase):
    cpp_name = "RequiredArg"

    def test_Basic(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "Basic")


class TestIsTestArg(TestCase):
    cpp_name = "IsTestArg"

    def test_standard(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "standard")

    def test_non_standard(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "non_standard")


if __name__ == "__main__":
    run_tests()
