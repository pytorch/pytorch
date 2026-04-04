// Checks the behaviour of:
//  aoti_torch_new_stable_ivalue
//  aoti_torch_delete_stable_ivalue

#include <gtest/gtest.h>
#include "torch/csrc/inductor/aoti_torch/c/macros.h"

#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <limits>

TEST(AotiTorchStableIValue, TestStableIValueUse) {
  StableIValue* a = nullptr;
  ASSERT_EQ(aoti_torch_new_stable_ivalue(&a), AOTI_TORCH_SUCCESS);
  // Check if it is now a valid pointer.
  ASSERT_NE(a, nullptr);
  // Assign to verify that we can actually write to the full u64 size, this will
  // show up in valgrind if we were to allocate less than the required size.
  *a = std::numeric_limits<StableIValue>::max();

  // Free the value again.
  EXPECT_EQ(aoti_torch_delete_stable_ivalue(a), AOTI_TORCH_SUCCESS);

  // Freeing a nullptr should result in a failure.
  StableIValue* b = nullptr;
  ASSERT_EQ(aoti_torch_delete_stable_ivalue(b), AOTI_TORCH_FAILURE);

  // Trying to allocate with a pointer into which can't be assigned is a
  // failure.
  StableIValue** ret_value = nullptr;
  ASSERT_EQ(aoti_torch_new_stable_ivalue(ret_value), AOTI_TORCH_FAILURE);
}

TEST(AotiTorch, TestStableIValueLargeAmount) {
  // Next, do a bunch of allocations and frees, such that if the delete isn't
  // working but the allocation is we see a clear memory leak in valgrind.
  StableIValue* v = nullptr;
  const size_t total_allocation_bytes = 10'000'000;
  const size_t number_of_allocations =
      total_allocation_bytes / sizeof(StableIValue);
  for (size_t i = 0; i < number_of_allocations; i++) {
    ASSERT_EQ(aoti_torch_new_stable_ivalue(&v), AOTI_TORCH_SUCCESS);
    ASSERT_NE(v, nullptr);
    ASSERT_EQ(aoti_torch_delete_stable_ivalue(v), AOTI_TORCH_SUCCESS);
    v = nullptr;
  }
}
