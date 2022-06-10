#include <gtest/gtest.h>

#include <ATen/ATen.h>

using namespace at;

// TODO: failing sizes {4, 1, 4, 1}
std::vector<std::vector<int64_t>> sizes = {{4, 4, 4, 4}, {4, 4, 1, 1}, {4, 1, 4, 4}, {4, 1, 1, 4}, {1, 4, 1, 4}, {1, 4, 4, 1}};

inline bool CheckStrideIndices(const Tensor& t, at::MemoryFormat format) {
  size_t n_dim = t.dim();
  std::vector<size_t> stride_indices(n_dim);
  if (format == at::MemoryFormat::ChannelsLast) {
    // stride_indices_ should be {1, n-1, n-2, ..., 2, 0}
    std::iota(stride_indices.rbegin() + 1, stride_indices.rend() - 1, 2);
    stride_indices[0] = 1;
    stride_indices[n_dim - 1] = 0;
  } else if (format == at::MemoryFormat::Contiguous) {
    // stride_indices_ should be {n-1, n-2, n-3, ..., 0}
    std::iota(stride_indices.rbegin(), stride_indices.rend(), 0);
  } else {
    TORCH_INTERNAL_ASSERT(false, "not recognized memory format");
  }

  // testing computeStrideProps with `IValue ival(t)` somehow doesn't work on CI
  // with onnx; The function works fine within, but stride properties is somehow
  // altered in ival->type()->cast<TensorType>();
  auto tt = TensorType::create(c10::nullopt, c10::nullopt, t.sizes(), t.strides(), c10::nullopt);
  TORCH_INTERNAL_ASSERT(tt->stride_properties().isComplete(), "complete stride properties is needed for the test");

  auto index_iter = stride_indices.begin();
  for (const auto& opt_stride : *tt->stride_properties().sizes()) {
    if (*index_iter++ != opt_stride->stride_index_.value()) {
      return false;
    }
  }

  return true;
}

TEST(StridePropertiesTest, StrideIndicesTest) {
  // NOLINTNEXTLINE(performance-for-range-copy)
  for (const auto& size : sizes) {
    Tensor t = at::rand(size);
    for (auto memory_format : {at::MemoryFormat::ChannelsLast, at::MemoryFormat::Contiguous}) {
      t.resize_(size, memory_format);
      EXPECT_TRUE(CheckStrideIndices(t, memory_format));
    }
  }
}

TEST(StridePropertiesTest, ZeroStrideIndicesEagerConsistencyTest) {
  auto permuted_tensor = at::rand({6, 3, 1, 5, 2}).permute({0, 3, 2, 1, 4}); // permute dim-1 & dim-3
  auto tensor = permuted_tensor.expand({6, 5, 4, 3, 2}); // expand dim-2

  auto temp = TensorType::create(c10::nullopt, c10::nullopt, tensor.sizes(), tensor.strides(), c10::nullopt);

  // TensorIterator would preserve stride order, this is the eager reference
  auto eager_tensor = tensor.relu();
  auto ref_type = TensorType::create(c10::nullopt, c10::nullopt, eager_tensor.sizes(), eager_tensor.strides(), c10::nullopt);

  TORCH_INTERNAL_ASSERT(temp->stride_properties().isComplete() &&
      temp->stride_properties().isComplete(), "complete stride properties is needed for the test");
  auto ref_iter = (*(ref_type->stride_properties().sizes())).begin();
  for (const auto& opt_stride : *temp->stride_properties().sizes()) {
    EXPECT_TRUE(opt_stride->stride_index_.value() == (*ref_iter)->stride_index_.value());
    ref_iter++;
  }
}

TEST(StridePropertiesTest, ExpandedStrideIndicesTest) {
  Tensor t = at::rand({1});
  // note: expand with dimension of size 1 is tricky as stride is different
  // depending on the order of the unsqueezed dimension.
  t = t.expand({4, 4, 4});
  EXPECT_TRUE(CheckStrideIndices(t, at::MemoryFormat::Contiguous));
}

TEST(StridePropertiesTest, SlicedStrideIndicesTest) {
  // Sliced tensor shouldn't have changed stride order
  Tensor t = at::rand({16, 4}).slice(1, 0, 4, 4);

  auto temp = TensorType::create(c10::nullopt, c10::nullopt, t.sizes(), t.strides(), c10::nullopt);
  TORCH_INTERNAL_ASSERT(temp->stride_properties().isComplete() &&
      temp->stride_properties().isComplete(), "complete stride properties is needed for the test");
  std::vector<size_t> stride_indices(2);
  std::iota(stride_indices.rbegin(), stride_indices.rend(), 0);

  auto index_iter = stride_indices.begin();
  for (const auto& opt_stride : *temp->stride_properties().sizes()) {
    EXPECT_TRUE(*index_iter++ == opt_stride->stride_index_.value());
  }
}
