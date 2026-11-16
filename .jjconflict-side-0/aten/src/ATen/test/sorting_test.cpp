#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/native/Sorting.h>

#include <cstdint>
#include <vector>

using namespace at;

namespace {

// Each slice along `dim` holds 0, 1, ..., size(dim) - 1, stated independently
// of the implementation.
std::vector<int64_t> expectedIndices(const Tensor& indices, int64_t dim) {
  int64_t inner = 1;
  for (int64_t i = dim + 1; i < indices.dim(); ++i) {
    inner *= indices.size(i);
  }
  std::vector<int64_t> expected(indices.numel());
  for (int64_t k = 0; k < indices.numel(); ++k) {
    expected[k] = (k / inner) % indices.size(dim);
  }
  return expected;
}

void checkFillIndices(Tensor indices, int64_t dim) {
  indices.fill_(-1);
  native::_fill_indices(indices, dim);
  const auto contiguous = indices.contiguous();
  const auto* data = contiguous.const_data_ptr<int64_t>();
  EXPECT_EQ(
      std::vector<int64_t>(data, data + contiguous.numel()),
      expectedIndices(contiguous, dim))
      << indices.sizes() << " dim=" << dim;
}

} // namespace

TEST(FillIndicesTest, Contiguous) {
  const std::vector<std::pair<std::vector<int64_t>, int64_t>> cases = {
      {{0}, 0},
      {{1}, 0},
      {{5}, 0},
      {{1000}, 0},
      {{3, 5}, 1},
      {{3, 5}, 0},
      {{2, 3, 4}, 2},
      {{2, 3, 4}, 1},
      {{2, 3, 4}, 0},
      {{3, 5, 1}, 1},
      {{3, 0}, 1},
      {{0, 3}, 1},
  };
  for (const auto& [sizes, dim] : cases) {
    checkFillIndices(at::empty(sizes, kLong), dim);
  }
}

TEST(FillIndicesTest, NonContiguous) {
  checkFillIndices(at::empty({5, 3}, kLong).t(), 1);
  checkFillIndices(at::empty({5, 3}, kLong).t(), 0);
  checkFillIndices(at::empty({4, 6}, kLong).slice(1, 0, 6, 2), 1);
}
