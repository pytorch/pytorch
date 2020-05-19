#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/BatchedTensorImpl.h>
#include <ATen/BatchingUtils.h>

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
  {
    // Create a "scalar" BatchedTensor. Should not crash.
    Tensor tensor = addBatchDim(ones({3}), /*lvl*/1, /*dim*/0);
  }
}
TEST(VmapTest, TestBatchedTensorActualDim) {
  {
    // No batch dims
    Tensor tensor = makeBatched(ones({2, 3, 5, 7}), {});
    auto* batched = maybeGetBatched(tensor);
    ASSERT_EQ(batched->actualDim(0), 0);
    ASSERT_EQ(batched->actualDim(1), 1);
    ASSERT_EQ(batched->actualDim(3), 3);

    // Test wrap around
    ASSERT_EQ(batched->actualDim(-1), 3);
    ASSERT_EQ(batched->actualDim(-4), 0);
    ASSERT_THROW(batched->actualDim(-5), c10::Error);
    ASSERT_THROW(batched->actualDim(4), c10::Error);

    // test wrap_dim = False
    ASSERT_THROW(batched->actualDim(-1, /*wrap_dim*/false), c10::Error);
    ASSERT_THROW(batched->actualDim(-4, /*wrap_dim*/false), c10::Error);
  }
  {
    // Single batch dim at front
    Tensor tensor = makeBatched(ones({2, 3, 5, 7}), {{/*lvl*/1, /*dim*/0}});
    auto* batched = maybeGetBatched(tensor);
    ASSERT_EQ(batched->actualDim(0), 1);
    ASSERT_EQ(batched->actualDim(2), 3);
    ASSERT_EQ(batched->actualDim(-1), 3);
    ASSERT_THROW(batched->actualDim(3), c10::Error);
  }
  {
    // Single batch dim in middle
    Tensor tensor = makeBatched(ones({2, 3, 5, 7}), {{/*lvl*/1, /*dim*/1}});
    auto* batched = maybeGetBatched(tensor);
    ASSERT_EQ(batched->actualDim(0), 0);
    ASSERT_EQ(batched->actualDim(1), 2);
    ASSERT_EQ(batched->actualDim(2), 3);
  }
  {
    // Single batch dim at end
    Tensor tensor = makeBatched(ones({2, 3, 5, 7}), {{/*lvl*/1, /*dim*/1}});
    auto* batched = maybeGetBatched(tensor);
    ASSERT_EQ(batched->actualDim(0), 0);
    ASSERT_EQ(batched->actualDim(2), 3);
    ASSERT_EQ(batched->actualDim(-1), 3);
  }
  {
    // Multiple (2) batch dims at front
    Tensor tensor = makeBatched(
        ones({2, 3, 5, 7}),
        {{/*lvl*/1, /*dim*/0}, {/*lvl*/2, /*dim*/1}});
    auto* batched = maybeGetBatched(tensor);
    ASSERT_EQ(batched->actualDim(0), 2);
    ASSERT_EQ(batched->actualDim(1), 3);
  }
  {
    // Multiple (2) batch dims, misc places
    Tensor tensor = makeBatched(
        ones({2, 3, 5, 7}),
        {{/*lvl*/1, /*dim*/1}, {/*lvl*/2, /*dim*/3}});
    auto* batched = maybeGetBatched(tensor);
    ASSERT_EQ(batched->actualDim(0), 0);
    ASSERT_EQ(batched->actualDim(1), 2);
    ASSERT_EQ(batched->actualDim(-1), 2);
    ASSERT_EQ(batched->actualDim(-2), 0);
  }
  {
    // ActualDim on kVmapMaxTensorDims sized underlying tensor
    auto tensor = ones({});
    for (int64_t i = 0; i < kVmapMaxTensorDims; i++) {
      tensor = tensor.unsqueeze(0);
    }
    ASSERT_EQ(tensor.dim(), kVmapMaxTensorDims);

    auto batched = addBatchDim(tensor, /*lvl*/1, /*dim*/0);
    auto* batched_impl = maybeGetBatched(batched);
    ASSERT_EQ(
        batched_impl->actualDim(kVmapMaxTensorDims - 2),
        kVmapMaxTensorDims - 1);
    ASSERT_EQ(
        batched_impl->actualDim(-1),
        kVmapMaxTensorDims - 1);
  }
}
TEST(VmapTest, TestUnpackBatched) {
  {
    // Call unpackBatched on a Tensor not backed by BatchedTensorImpl
    Tensor out;
    BatchDimsRef out_bdims;

    auto tensor = at::randn({2, 3, 5, 7});
    std::tie(out, out_bdims) = unpackBatched(tensor);
    ASSERT_TRUE(out.is_same(tensor));
    ASSERT_EQ(out_bdims.size(), 0);
  }
  {
    // Call unpackBatched on a Tensor backed by BatchedTensorImpl
    Tensor out;
    BatchDimsRef out_bdims;

    auto tensor = at::randn({2, 3, 5, 7});
    BatchDims bdims = {{/*lvl*/1, /*dim*/0}, {/*lvl*/2, /*dim*/2}};
    auto batched_tensor = makeBatched(tensor, bdims);
    std::tie(out, out_bdims) = unpackBatched(batched_tensor);

    ASSERT_TRUE(out.is_same(tensor));
    ASSERT_EQ(out_bdims.size(), bdims.size());
    ASSERT_EQ(out_bdims[0].level(), 1);
    ASSERT_EQ(out_bdims[0].dim(), 0);
    ASSERT_EQ(out_bdims[1].level(), 2);
    ASSERT_EQ(out_bdims[1].dim(), 2);
  }
}

TEST(VmapTest, TestMoveBatchDimsToFront) {
  {
    auto result = moveBatchDimsToFront({});
    ASSERT_EQ(result.size(), 0);
  }
  {
    BatchDims bdims = {{/*lvl*/1, /*dim*/2}, {/*lvl*/3, /*dim*/0}};
    auto result = moveBatchDimsToFront(bdims);
    ASSERT_EQ(result.size(), bdims.size());
    ASSERT_EQ(result[0].level(), 1);
    ASSERT_EQ(result[0].dim(), 0);
    ASSERT_EQ(result[1].level(), 3);
    ASSERT_EQ(result[1].dim(), 1);
  }
}

TEST(VmapTest, TestMoveTensorBatchDimsToFront) {
  {
    // Nullary case (bdims = {})
    auto tensor = ones({2, 3, 5});
    auto result = moveBatchDimsToFront(tensor, {});
    ASSERT_TRUE(result.is_same(tensor));
  }
  {
    // Batch dims are already at the front
    auto tensor = ones({2, 3, 5});
    BatchDims bdims = {{/*lvl*/1, /*dim*/0}, {/*lvl*/3, /*dim*/1}};
    auto result = moveBatchDimsToFront(tensor, bdims);
    ASSERT_TRUE(result.is_same(tensor));
  }
  {
    // Single batch dim, not at front
    auto tensor = ones({2, 3, 5});
    BatchDims bdims = {{/*lvl*/1, /*dim*/1}};
    auto result = moveBatchDimsToFront(tensor, bdims);
    ASSERT_EQ(result.data_ptr(), tensor.data_ptr());
    ASSERT_TRUE(at::allclose(result, tensor.permute({1, 0, 2})));
  }
  {
    // Multiple batch dims, not at front. 
    auto tensor = ones({2, 3, 5});
    BatchDims bdims = {{/*lvl*/1, /*dim*/1}, {/*lvl*/2,/*dim*/2}, {/*lvl*/3,/*dim*/0}};
    auto result = moveBatchDimsToFront(tensor, bdims);
    ASSERT_EQ(result.data_ptr(), tensor.data_ptr());
    ASSERT_TRUE(at::allclose(result, tensor.permute({1, 2, 0})));
  }
}

// Basic test for BatchedTensor::sum.
// NB: We don't need to write tests in C++ for batching rules if we can test them
// in Python via the vmap API. These are here to bootstrap that process.
TEST(VmapTest, TestBatchedTensorSum) {
  {
    // Simple: single batch dim, single reduce dim
    Tensor x = at::randn({2, 3, 5, 7});
    Tensor out;

    Tensor batched_x = makeBatched(x, {{/*lvl*/1, /*dim*/0}});
    Tensor batched_out = batched_x.sum(0);
    std::tie(out, std::ignore) = unpackBatched(batched_out);

    ASSERT_TRUE(at::allclose(out, x.sum(1)));
  }
  {
    // single batch dim, -1 reduce dim handling
    Tensor x = at::randn({2, 3});
    Tensor out;

    Tensor batched_x = makeBatched(x, {{/*lvl*/1, /*dim*/1}});
    Tensor batched_out = batched_x.sum(-1);
    std::tie(out, std::ignore) = unpackBatched(batched_out);

    ASSERT_TRUE(at::allclose(out, x.sum(0)));
  }
  {
    // single batch dim, multiple reduce dim
    Tensor x = at::randn({2, 3, 5, 7});
    Tensor out;

    Tensor batched_x = makeBatched(x, {{/*lvl*/1, /*dim*/1}});
    Tensor batched_out = batched_x.sum(std::vector<int64_t>{0, 1});
    std::tie(out, std::ignore) = unpackBatched(batched_out);

    ASSERT_TRUE(at::allclose(out, x.sum(std::vector<int64_t>{0, 2})));
  }
  {
    // multiple batch dim, multiple reduce dim
    Tensor x = at::randn({2, 3, 5, 7});
    Tensor out;

    Tensor batched_x = makeBatched(x, {{/*lvl*/1, /*dim*/0}, {/*lvl*/2, /*dim*/1}});
    Tensor batched_out = batched_x.sum(std::vector<int64_t>{0, 1});
    std::tie(out, std::ignore) = unpackBatched(batched_out);

    ASSERT_TRUE(at::allclose(out, x.sum(std::vector<int64_t>{2, 3})));
  }
}

}
