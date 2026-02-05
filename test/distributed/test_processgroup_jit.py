# Owner(s): ["oncall: distributed"]

import unittest

import torch
import torch.distributed as dist
from torch.testing._internal.common_utils import (
    run_tests,
    scoped_load_inline,
    skipIfTorchDynamo,
    TestCase,
)


class DummyProcessGroup(dist.ProcessGroup):
    """Minimal ProcessGroup for testing."""

    def __init__(self, rank: int = 0, world_size: int = 1):
        super().__init__(rank, world_size)

    def getBackendName(self) -> str:
        return "dummy-jit-test"


@unittest.skipIf(not dist.is_available(), "requires distributed")
@skipIfTorchDynamo("JIT/C++ extension test")
class TestProcessGroupCapsule(TestCase):
    @scoped_load_inline
    def test_processgroup_capsule_discrimination(self, load_inline):
        """Test C++ can distinguish string vs ProcessGroup vs other capsule types."""
        cpp_source = """
#include <torch/extension.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

// A different custom class (not ProcessGroup) for testing discrimination
struct OtherCapsuleType : public torch::CustomClassHolder {
    int64_t value;
    explicit OtherCapsuleType(int64_t v) : value(v) {}
    ~OtherCapsuleType() override = default;
};

// Check what type of IValue we received
// Returns: 0=string, 1=ProcessGroup capsule, 2=other capsule, 3=int, 4=tensor, -1=other
int64_t check_ivalue_type(const c10::IValue& value) {
    if (value.isString()) {
        return 0;
    } else if (value.isCapsule()) {
        // Use dynamic_cast for runtime type checking
        auto capsule = value.toCapsule();
        auto* raw_ptr = capsule.get();
        if (dynamic_cast<c10d::ProcessGroup*>(raw_ptr) != nullptr) {
            return 1;  // ProcessGroup capsule
        }
        return 2;  // Other capsule type
    } else if (value.isInt()) {
        return 3;
    } else if (value.isTensor()) {
        return 4;
    }
    return -1;
}

// Extract info from ProcessGroup capsule - mimics get_process_group() pattern
torch::Tensor get_pg_info(const c10::IValue& group) {
    TORCH_CHECK(group.isCapsule(), "Expected capsule, got ", group.tagKind());
    auto capsule = group.toCapsule();
    auto* raw_ptr = capsule.get();
    auto* pg = dynamic_cast<c10d::ProcessGroup*>(raw_ptr);
    TORCH_CHECK(pg != nullptr, "Capsule is not a ProcessGroup");
    int64_t rank = pg->getRank();
    int64_t size = pg->getSize();
    return torch::tensor({rank, size});
}

// Create an OtherCapsuleType instance wrapped as capsule
c10::IValue create_other_capsule(int64_t value) {
    auto obj = c10::make_intrusive<OtherCapsuleType>(value);
    return c10::IValue::make_capsule(obj);
}

TORCH_LIBRARY(test_pg_capsule, m) {
    m.def("check_ivalue_type(Any value) -> int", check_ivalue_type);
    m.def("get_pg_info(Any group) -> Tensor", get_pg_info);
    m.def("create_other_capsule(int value) -> Any", create_other_capsule);
}
"""
        load_inline(
            name="test_pg_capsule",
            cpp_sources=cpp_source,
            extra_cflags=["-DUSE_DISTRIBUTED"],
            is_python_module=False,
            verbose=True,
        )

        pg = DummyProcessGroup(rank=2, world_size=8)

        # Test 1: String is recognized as string (type 0)
        self.assertEqual(torch.ops.test_pg_capsule.check_ivalue_type("my_group"), 0)

        # Test 2: ProcessGroup is recognized as ProcessGroup capsule (type 1)
        self.assertEqual(torch.ops.test_pg_capsule.check_ivalue_type(pg), 1)

        # Test 3: Other capsule type is recognized as other capsule (type 2)
        other_capsule = torch.ops.test_pg_capsule.create_other_capsule(42)
        self.assertEqual(torch.ops.test_pg_capsule.check_ivalue_type(other_capsule), 2)

        # Test 4: Integer is recognized as int (type 3), not capsule
        self.assertEqual(torch.ops.test_pg_capsule.check_ivalue_type(123), 3)

        # Test 5: Tensor is recognized as tensor (type 4), not capsule
        self.assertEqual(
            torch.ops.test_pg_capsule.check_ivalue_type(torch.tensor([1, 2, 3])), 4
        )

        # Test 6: Can extract ProcessGroup info via capsule
        info = torch.ops.test_pg_capsule.get_pg_info(pg)
        self.assertEqual(info[0].item(), 2)  # rank
        self.assertEqual(info[1].item(), 8)  # size

        # Test 7: Other capsule fails when treated as ProcessGroup
        with self.assertRaises(RuntimeError):
            torch.ops.test_pg_capsule.get_pg_info(other_capsule)


if __name__ == "__main__":
    run_tests()
