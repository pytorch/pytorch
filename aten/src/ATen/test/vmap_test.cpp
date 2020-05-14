#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/BatchedTensorImpl.h>

using namespace at;

namespace {

TEST(VmapTest, TestBatchedTensor) {
  {
    Tensor x = addBatchDim(ones({2, 3, 4}), /*lvl=*/1, /*dim=*/1);
    std::vector<int64_t> expected_size = {2, 4};
    ASSERT_EQ(x.sizes(), expected_size);
    ASSERT_EQ(x.dim(), 2);
    ASSERT_EQ(x.numel(), 8);
    ASSERT_THROW(x.strides(), c10::Error);
    ASSERT_THROW(x.is_contiguous(), c10::Error);
    ASSERT_THROW(x.storage(), c10::Error);
    ASSERT_THROW(x.storage_offset(), c10::Error);
  }
  {
    // Test multiple batch dims
    Tensor x = addBatchDim(ones({2, 3, 4}), /*lvl=*/1, /*dim=*/1);
    x = addBatchDim(x, /*lvl=*/2, /*dim=*/1);
    std::vector<int64_t> expected_size = {2};
    ASSERT_EQ(x.sizes(), expected_size);
    ASSERT_EQ(x.dim(), 1);
    ASSERT_EQ(x.numel(), 2);
  }
  {
    // Test vmap tensor dimensionality limit

    // Should not throw
    std::vector<int64_t> sizes(kVmapMaxTensorDims, 1);
    Tensor x = addBatchDim(ones(sizes), /*lvl=*/1, /*dim=*/1);

    // Should throw
    std::vector<int64_t> too_many_sizes(kVmapMaxTensorDims + 1, 1);
    auto big_dim_tensor = ones(too_many_sizes);
    ASSERT_THROW(addBatchDim(big_dim_tensor, /*lvl=*/1, /*dim=*/1), c10::Error);
  }
}

}
