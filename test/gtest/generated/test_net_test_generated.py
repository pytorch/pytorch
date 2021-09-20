import torch
from torch.testing._internal.common_utils import TestCase, run_tests, run_cpp_test

TEST_BINARY = "build/bin/net_test"


class TestNetDeathTest(TestCase):
    cpp_name = "NetDeathTest"

    def test_DeclaredOutputNotMet(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DeclaredOutputNotMet")


class TestNetTest(TestCase):
    cpp_name = "NetTest"

    def test_ConstructionNoDeclaredInputOutput(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ConstructionNoDeclaredInputOutput")

    def test_ConstructionDeclaredInput(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ConstructionDeclaredInput")

    def test_ConstructionDeclaredOutput(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ConstructionDeclaredOutput")

    def test_DeclaredInputInsufficient(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DeclaredInputInsufficient")

    def test_DISABLED_ChainingForLinearModel(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DISABLED_ChainingForLinearModel")

    def test_DISABLED_ChainingForFork(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DISABLED_ChainingForFork")

    def test_DISABLED_ChainingForForkJoin(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DISABLED_ChainingForForkJoin")

    def test_DISABLED_ChainingForwardBackward(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DISABLED_ChainingForwardBackward")

    def test_DISABLED_ChainingForHogwildModel(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DISABLED_ChainingForHogwildModel")

    def test_DISABLED_FailingOperator(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DISABLED_FailingOperator")

    def test_OperatorWithExecutorHelper(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "OperatorWithExecutorHelper")

    def test_DISABLED_OperatorWithDisabledEvent(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DISABLED_OperatorWithDisabledEvent")

    def test_ExecutorOverride(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ExecutorOverride")

    def test_AsyncEmptyNet(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "AsyncEmptyNet")

    def test_DISABLED_RunAsyncFailure(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "DISABLED_RunAsyncFailure")

    def test_NoTypeNet(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "NoTypeNet")

    def test_PendingOpsAndNetFailure(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "PendingOpsAndNetFailure")

    def test_AsyncErrorOpTest(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "AsyncErrorOpTest")

    def test_AsyncErrorTimingsTest(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "AsyncErrorTimingsTest")

    def test_ChainErrorTest(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ChainErrorTest")

    def test_ProfDAGNetErrorTest(self):
        run_cpp_test(TEST_BINARY, self.cpp_name, "ProfDAGNetErrorTest")


if __name__ == "__main__":
    run_tests()
