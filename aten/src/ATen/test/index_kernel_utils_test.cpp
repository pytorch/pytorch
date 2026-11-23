#include <gtest/gtest.h>

#include <numeric>
#include <ostream>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/native/cpu/IndexKernelUtils.h>

using namespace at::native;

namespace {

constexpr int64_t kElem = static_cast<int64_t>(sizeof(int64_t));

struct IndexerTestCase {
  int64_t source_len; // flat source, filled with iota (source[k] == k)
  std::vector<int64_t> sizes; // size of each indexed dimension
  std::vector<int64_t> strides; // source stride per dimension, in elements
  std::vector<std::vector<int64_t>> indices; // contents of each index tensor
  std::vector<int64_t> expected; // element visited at each iteration position
};

template <typename T>
std::ostream& printToStream(std::ostream& out, const T& x) {
  return out << x;
}

template <typename T>
std::ostream& printToStream(std::ostream& out, const std::vector<T>& x) {
  out << '[';
  for (size_t i = 0; i < x.size(); ++i) {
    if (i != 0) {
      out << ", ";
    }
    printToStream(out, x[i]);
  }
  return out << ']';
}

std::ostream& operator<<(std::ostream& os, const IndexerTestCase& tc) {
  os << "IndexerTestCase{source_len=" << tc.source_len << ", sizes=";
  printToStream(os, tc.sizes);
  os << ", strides=";
  printToStream(os, tc.strides);
  os << ", indices=";
  printToStream(os, tc.indices);
  os << ", expected=";
  printToStream(os, tc.expected);
  return os << "}";
}

std::vector<int64_t> run(IndexerTestCase tc) {
  std::vector<int64_t> source(tc.source_len);
  std::iota(source.begin(), source.end(), 0);

  std::vector<char*> index_ptrs;
  std::vector<int64_t> indexer_strides;
  index_ptrs.reserve(tc.indices.size());
  indexer_strides.reserve(tc.indices.size());
  for (auto& idx : tc.indices) {
    index_ptrs.push_back(reinterpret_cast<char*>(idx.data()));
    indexer_strides.push_back(kElem);
  }
  std::vector<int64_t> byte_strides;
  byte_strides.reserve(tc.strides.size());
  for (const auto s : tc.strides) {
    byte_strides.push_back(s * kElem);
  }

  Indexer indexer(
      static_cast<int64_t>(tc.indices.size()),
      index_ptrs.data(),
      indexer_strides.data(),
      tc.sizes,
      byte_strides);

  const bool single = tc.indices.size() == 1;
  std::vector<int64_t> visited;
  visited.reserve(tc.indices.front().size());
  for (const auto i : c10::irange(tc.indices.front().size())) {
    const int64_t offset = indexer.get(i);
    if (single) {
      EXPECT_EQ(offset, indexer.get_1(i));
    }
    visited.push_back(source[offset / kElem]);
  }
  return visited;
}

} // namespace

TEST(IndexerTest, GathersExpectedElements) {
  const std::vector<IndexerTestCase> cases = {
      // Identity indices walk the source in order.
      {.source_len = 5,
       .sizes = {5},
       .strides = {1},
       .indices = {{0, 1, 2, 3, 4}},
       .expected = {0, 1, 2, 3, 4}},
      // The index value, not the position, decides what is read.
      {.source_len = 5,
       .sizes = {5},
       .strides = {1},
       .indices = {{2, 0, 4, 1, 3}},
       .expected = {2, 0, 4, 1, 3}},
      // The same element may be read more than once.
      {.source_len = 5,
       .sizes = {5},
       .strides = {1},
       .indices = {{2, 2, 0, 4, 4}},
       .expected = {2, 2, 0, 4, 4}},
      // Negative indices wrap by adding the dimension size.
      {.source_len = 5,
       .sizes = {5},
       .strides = {1},
       .indices = {{-1, -5, -3}},
       .expected = {4, 0, 2}},
      // A stride of 2 means index v lands on source element 2*v.
      {.source_len = 10,
       .sizes = {5},
       .strides = {2},
       .indices = {{0, 1, 2, 3, 4}},
       .expected = {0, 2, 4, 6, 8}},
      // Two index tensors: 4x3 iota matrix, element [r, c] equals r*3 + c.
      {.source_len = 12,
       .sizes = {4, 3},
       .strides = {3, 1},
       .indices = {{0, 3, 2}, {1, 0, 2}},
       .expected = {1, 9, 8}},
  };

  for (const auto& tc : cases) {
    SCOPED_TRACE(tc);
    EXPECT_EQ(run(tc), tc.expected);
  }
}
