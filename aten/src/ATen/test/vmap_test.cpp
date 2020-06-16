#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/BatchedTensorImpl.h>
#include <ATen/VmapTransforms.h>

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

// returns {{lvl=0,dim=0}, {lvl=1,dim=1}, ..., {lvl=kVmapNumLevels-1,dim=kVmapNumLevels-1}};
static BatchDims maxBatchDimsAtFront() {
  BatchDims result;
  for (int64_t lvl = 0; lvl < kVmapNumLevels; lvl++) {
    result.emplace_back(lvl, /*dim=*/lvl);
  }
  return result;
}

TEST(VmapTest, TestBatchedTensorMaxLevel) {
  {
    // Should not throw
    auto tensor = ones({2, 3, 4});
    makeBatched(ones({2, 3, 4}), {{/*lvl*/kVmapNumLevels - 1, /*dim*/0}});
  }
  {
    auto tensor = ones({2, 3, 4});
    ASSERT_THROW(
        makeBatched(ones({2, 3, 4}), {{/*lvl*/kVmapNumLevels, /*dim*/0}}),
        c10::Error);
  }
  {
    auto tensor = ones({2, 3, 4});
    ASSERT_THROW(
        makeBatched(ones({2, 3, 4}), {{/*lvl*/kVmapNumLevels + 5, /*dim*/0}}),
        c10::Error);
  }
  {
    // create a BatchedTensor with kVmapNumLevels levels.
    // Should not throw
    auto tensor = ones(std::vector<int64_t>(kVmapNumLevels, 1));
    makeBatched(tensor, maxBatchDimsAtFront());
  }
  {
    // create a BatchedTensor with kVmapNumLevels+1 levels.
    auto tensor = ones(std::vector<int64_t>(kVmapNumLevels + 1, 1));
    auto batch_dims = maxBatchDimsAtFront();
    batch_dims.emplace_back(/*lvl*/kVmapNumLevels, /*dim*/kVmapNumLevels);
    ASSERT_THROW(makeBatched(tensor, batch_dims), c10::Error);
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
TEST(VmapTest, TestMultiBatchVmapTransform) {
  {
    // Input is regular Tensor
    auto tensor = ones({2, 3, 5});
    ASSERT_THROW(MultiBatchVmapTransform::logicalToPhysical(tensor), c10::Error);
  }
  {
    // Input is BatchedTensor, Batch dims are already at the front
    auto tensor = ones({2, 3, 5});
    BatchDims bdims = {{/*lvl*/1, /*dim*/0}, {/*lvl*/3, /*dim*/1}};
    auto batched = makeBatched(tensor, bdims);

    auto result = MultiBatchVmapTransform::logicalToPhysical(batched);
    ASSERT_TRUE(result.tensor().is_same(tensor));
  }
  {
    // Single batch dim, not at front
    auto tensor = ones({2, 3, 5});
    BatchDims bdims = {{/*lvl*/1, /*dim*/1}};
    auto batched = makeBatched(tensor, bdims);

    auto result = MultiBatchVmapTransform::logicalToPhysical(batched);
    ASSERT_EQ(result.tensor().data_ptr(), tensor.data_ptr());
    ASSERT_TRUE(at::allclose(result.tensor(), tensor.permute({1, 0, 2})));
  }
  {
    // Multiple batch dims, not at front.
    auto tensor = ones({2, 3, 5});
    BatchDims bdims = {{/*lvl*/1, /*dim*/1}, {/*lvl*/2,/*dim*/2}, {/*lvl*/3,/*dim*/0}};
    auto batched = makeBatched(tensor, bdims);

    auto result = MultiBatchVmapTransform::logicalToPhysical(batched);
    ASSERT_EQ(result.tensor().data_ptr(), tensor.data_ptr());
    ASSERT_TRUE(at::allclose(result.tensor(), tensor.permute({1, 2, 0})));
  }
  {
    // Edge case: kVmapNumLevels levels; batch dims are already at front.

    // sizes=[2, 1, 3, 1, 1, 7, 1, 1, 1, 1, ...]
    auto sizes = std::vector<int64_t>(kVmapNumLevels, 1);
    sizes[0] = 2;
    sizes[2] = 3;
    sizes[5] = 7;

    // bdims = {{lvl=0,dim=0,lvl=1,dim=1,...,{lvl=63,dim=63}}
    auto batch_dims = maxBatchDimsAtFront();
    auto tensor = ones(sizes);

    auto batched = makeBatched(tensor, batch_dims);
    auto result = MultiBatchVmapTransform::logicalToPhysical(batched);
    ASSERT_TRUE(result.tensor().is_same(tensor));
  }
  {
    // Edge case: kVmapNumLevels levels; batch dims are not at front

    // sizes=[1, 3, 2, 1, 1, 7, 1, 1, 1, 1, ..., 1, 1, 5]
    auto sizes = std::vector<int64_t>(kVmapNumLevels, 1);
    sizes[1] = 3;
    sizes[2] = 2;
    sizes[5] = 7;
    sizes[kVmapNumLevels - 1] = 5;

    // The goal is to permute sizes such that the final sizes are:
    // [2, 3, 5, 7, 1, 1, 1, 1, 1, ...]
    auto expected_result_sizes = std::vector<int64_t>(kVmapNumLevels, 1);
    expected_result_sizes[0] = 2;
    expected_result_sizes[1] = 3;
    expected_result_sizes[2] = 5;
    expected_result_sizes[3] = 7;

    // bdims = {{0, 2}, {1, 1}, {2, 63}, {3, 5}, {4, 0}, {5, 3}, {6, 4},
    //          {7, 6}, {8, 7}, {9, 8}, ..., {63, 62}}
    BatchDims batch_dims = {
      {0, 2}, {1, 1}, {2, kVmapNumLevels - 1}, {3, 5}, {4, 0}, {5, 3}, {6, 4}
    };
    for (int64_t level = 7; level < kVmapNumLevels; level++ ) {
      batch_dims.emplace_back(level, /*dim=*/level - 1);
    }
    auto tensor = ones(sizes);

    auto batched = makeBatched(tensor, batch_dims);
    auto result = MultiBatchVmapTransform::logicalToPhysical(batched);
    ASSERT_EQ(result.tensor().data_ptr(), tensor.data_ptr());
    ASSERT_EQ(result.tensor().sizes(), expected_result_sizes);
  }
}
TEST(VmapTest, TestVmapPhysicalViewGetPhysicalDim) {
  VmapPhysicalView physical_view(ones({2, 3, 4, 5, 6}), 1 | 4);

  // Positive dims
  ASSERT_EQ(physical_view.getPhysicalDim(0), 2);
  ASSERT_EQ(physical_view.getPhysicalDim(1), 3);
  ASSERT_EQ(physical_view.getPhysicalDim(2), 4);
  ASSERT_THROW(physical_view.getPhysicalDim(3), c10::Error);

  // Negative dims (testing wrap dim behavior)
  ASSERT_EQ(physical_view.getPhysicalDim(-1), 4);
  ASSERT_EQ(physical_view.getPhysicalDim(-2), 3);
  ASSERT_EQ(physical_view.getPhysicalDim(-3), 2);
  ASSERT_THROW(physical_view.getPhysicalDim(-4), c10::Error);
}
TEST(VmapTest, TestVmapPhysicalViewGetPhysicalDims) {
  VmapPhysicalView physical_view(ones({2, 3, 4, 5, 6}), 2 | 8 | 16);

  ASSERT_EQ(
      physical_view.getPhysicalDims({0, 1, -1, -2}),
      std::vector<int64_t>({3, 4, 4, 3}));

  ASSERT_THROW(physical_view.getPhysicalDims({2, 0}), c10::Error);
  ASSERT_THROW(physical_view.getPhysicalDims({0, -3}), c10::Error);
}

static void checkBatchDimsEqual(BatchDimsRef bdims, BatchDimsRef expected_bdims) {
  ASSERT_EQ(bdims.size(), expected_bdims.size());
  for (int64_t idx = 0; idx < bdims.size(); idx++) {
    ASSERT_EQ(bdims[idx].dim(), expected_bdims[idx].dim());
    ASSERT_EQ(bdims[idx].level(), expected_bdims[idx].level());
  }
}

TEST(VmapTest, TestVmapPhysicalViewNewLogicalFromPhysical) {
  {
    // Simple case: single level
    VmapPhysicalView physical_view(ones({2, 3, 4}), /*levels = {2}*/4);
    Tensor physical = ones({2, 6, 7});

    auto result = physical_view.newLogicalFromPhysical(physical);
    auto* batched = maybeGetBatched(result);
    ASSERT_TRUE(batched != nullptr);
    ASSERT_TRUE(batched->value().is_same(physical));
    checkBatchDimsEqual(batched->bdims(), {{2, 0}});
  }
  {
    // Multiple levels
    VmapPhysicalView physical_view(ones({2, 3, 4, 5, 6}), /*levels = {1, 3, 4}*/2 | 8 | 16);
    Tensor physical = ones({2, 3, 4, 7});

    auto result = physical_view.newLogicalFromPhysical(physical);
    auto* batched = maybeGetBatched(result);
    ASSERT_TRUE(batched != nullptr);
    ASSERT_TRUE(batched->value().is_same(physical));
    checkBatchDimsEqual(batched->bdims(), {{1, 0}, {3, 1}, {4, 2}});
  }
  {
    // Logical dimensions is [].
    VmapPhysicalView physical_view(ones({2}), /*levels = {2}*/4);
    Tensor physical = ones({2});

    auto result = physical_view.newLogicalFromPhysical(physical);
    auto* batched = maybeGetBatched(result);
    ASSERT_TRUE(batched != nullptr);
    ASSERT_TRUE(batched->value().is_same(physical));
    checkBatchDimsEqual(batched->bdims(), {{2, 0}});
  }
  {
    // Failure case: incompatible batch sizes
    VmapPhysicalView physical_view(ones({2, 3, 4, 5, 6}), 2 | 8 | 16);
    ASSERT_THROW(physical_view.newLogicalFromPhysical(ones({2, 2, 4, 7})), c10::Error);
    ASSERT_THROW(physical_view.newLogicalFromPhysical(ones({2, 3})), c10::Error);
    ASSERT_THROW(physical_view.newLogicalFromPhysical(ones({2, 1, 1})), c10::Error);
  }
}

// Basic test for BatchedTensor::sum.
// NB: We don't need to write tests in C++ for batching rules if we can test them
// in Python via the vmap API. These are here to bootstrap that process.
TEST(VmapTest, TestBatchedTensorSum) {
  {
    // Simple: single batch dim, single reduce dim
    Tensor x = at::randn({2, 3, 5, 7});

    Tensor batched_x = makeBatched(x, {{/*lvl*/1, /*dim*/0}});
    Tensor batched_out = batched_x.sum(0);
    const auto& out = maybeGetBatched(batched_out)->value();

    ASSERT_TRUE(at::allclose(out, x.sum(1)));
  }
  {
    // single batch dim, -1 reduce dim handling
    Tensor x = at::randn({2, 3});

    Tensor batched_x = makeBatched(x, {{/*lvl*/1, /*dim*/1}});
    Tensor batched_out = batched_x.sum(-1);
    const auto& out = maybeGetBatched(batched_out)->value();

    ASSERT_TRUE(at::allclose(out, x.sum(0)));
  }
  {
    // single batch dim, multiple reduce dim
    Tensor x = at::randn({2, 3, 5, 7});

    Tensor batched_x = makeBatched(x, {{/*lvl*/1, /*dim*/1}});
    Tensor batched_out = batched_x.sum(std::vector<int64_t>{0, 1});
    const auto& out = maybeGetBatched(batched_out)->value();

    ASSERT_TRUE(at::allclose(out, x.sum(std::vector<int64_t>{0, 2})));
  }
  {
    // multiple batch dim, multiple reduce dim
    Tensor x = at::randn({2, 3, 5, 7});

    Tensor batched_x = makeBatched(x, {{/*lvl*/1, /*dim*/0}, {/*lvl*/2, /*dim*/1}});
    Tensor batched_out = batched_x.sum(std::vector<int64_t>{0, 1});
    const auto& out = maybeGetBatched(batched_out)->value();

    ASSERT_TRUE(at::allclose(out, x.sum(std::vector<int64_t>{2, 3})));
  }
}


}
