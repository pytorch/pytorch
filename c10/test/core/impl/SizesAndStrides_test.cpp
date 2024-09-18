#include <gtest/gtest.h>

#include <c10/core/impl/SizesAndStrides.h>
#include <c10/util/irange.h>

#ifdef __clang__
#pragma clang diagnostic ignored "-Wself-assign-overloaded"
#endif

using namespace c10;
using namespace c10::impl;

// NOLINTBEGIN(*conversion*, *multiplication*)
static void checkData(
    const SizesAndStrides& sz,
    IntArrayRef sizes,
    IntArrayRef strides) {
  EXPECT_EQ(sizes.size(), strides.size())
      << "bad test case: size() of sizes and strides don't match";
  EXPECT_EQ(sz.size(), sizes.size());

  int idx = 0;
  for (auto x : sizes) {
    EXPECT_EQ(sz.size_at_unchecked(idx), x) << "index: " << idx;
    EXPECT_EQ(sz.size_at(idx), x) << "index: " << idx;
    EXPECT_EQ(sz.sizes_data()[idx], x) << "index: " << idx;
    EXPECT_EQ(*(sz.sizes_begin() + idx), x) << "index: " << idx;
    idx++;
  }
  EXPECT_EQ(sz.sizes_arrayref(), sizes);

  idx = 0;
  for (auto x : strides) {
    EXPECT_EQ(sz.stride_at_unchecked(idx), x) << "index: " << idx;
    EXPECT_EQ(sz.stride_at(idx), x) << "index: " << idx;
    EXPECT_EQ(sz.strides_data()[idx], x) << "index: " << idx;
    EXPECT_EQ(*(sz.strides_begin() + idx), x) << "index: " << idx;

    idx++;
  }
  EXPECT_EQ(sz.strides_arrayref(), strides);
}

TEST(SizesAndStridesTest, DefaultConstructor) {
  SizesAndStrides sz;
  checkData(sz, {0}, {1});
  // Can't test size_at() out of bounds because it just asserts for now.
}

TEST(SizesAndStridesTest, SetSizes) {
  SizesAndStrides sz;
  sz.set_sizes({5, 6, 7, 8});
  checkData(sz, {5, 6, 7, 8}, {1, 0, 0, 0});
}

TEST(SizesAndStridesTest, Resize) {
  SizesAndStrides sz;

  sz.resize(2);

  // Small to small growing.
  checkData(sz, {0, 0}, {1, 0});

  // Small to small growing, again.
  sz.resize(5);
  checkData(sz, {0, 0, 0, 0, 0}, {1, 0, 0, 0, 0});

  for (const auto ii : c10::irange(sz.size())) {
    sz.size_at_unchecked(ii) = ii + 1;
    sz.stride_at_unchecked(ii) = 2 * (ii + 1);
  }

  checkData(sz, {1, 2, 3, 4, 5}, {2, 4, 6, 8, 10});

  // Small to small, shrinking.
  sz.resize(4);
  checkData(sz, {1, 2, 3, 4}, {2, 4, 6, 8});

  // Small to small with no size change.
  sz.resize(4);
  checkData(sz, {1, 2, 3, 4}, {2, 4, 6, 8});

  // Small to small, growing back so that we can confirm that our "new"
  // data really does get zeroed.
  sz.resize(5);
  checkData(sz, {1, 2, 3, 4, 0}, {2, 4, 6, 8, 0});

  // Small to big.
  sz.resize(6);

  checkData(sz, {1, 2, 3, 4, 0, 0}, {2, 4, 6, 8, 0, 0});

  sz.size_at_unchecked(5) = 6;
  sz.stride_at_unchecked(5) = 12;

  checkData(sz, {1, 2, 3, 4, 0, 6}, {2, 4, 6, 8, 0, 12});

  // Big to big, growing.
  sz.resize(7);

  checkData(sz, {1, 2, 3, 4, 0, 6, 0}, {2, 4, 6, 8, 0, 12, 0});

  // Big to big with no size change.
  sz.resize(7);

  checkData(sz, {1, 2, 3, 4, 0, 6, 0}, {2, 4, 6, 8, 0, 12, 0});

  sz.size_at_unchecked(6) = 11;
  sz.stride_at_unchecked(6) = 22;

  checkData(sz, {1, 2, 3, 4, 0, 6, 11}, {2, 4, 6, 8, 0, 12, 22});

  // Big to big, shrinking.
  sz.resize(6);
  checkData(sz, {1, 2, 3, 4, 0, 6}, {2, 4, 6, 8, 0, 12});

  // Grow back to make sure "new" elements get zeroed in big mode too.
  sz.resize(7);
  checkData(sz, {1, 2, 3, 4, 0, 6, 0}, {2, 4, 6, 8, 0, 12, 0});

  // Finally, big to small.

  // Give it different data than it had when it was small to avoid
  // getting it right by accident (i.e., because of leftover inline
  // storage when going small to big).
  for (const auto ii : c10::irange(sz.size())) {
    sz.size_at_unchecked(ii) = ii - 1;
    sz.stride_at_unchecked(ii) = 2 * (ii - 1);
  }

  checkData(sz, {-1, 0, 1, 2, 3, 4, 5}, {-2, 0, 2, 4, 6, 8, 10});

  sz.resize(5);
  checkData(sz, {-1, 0, 1, 2, 3}, {-2, 0, 2, 4, 6});
}

TEST(SizesAndStridesTest, SetAtIndex) {
  SizesAndStrides sz;

  sz.resize(5);
  sz.size_at(4) = 42;
  sz.stride_at(4) = 23;

  checkData(sz, {0, 0, 0, 0, 42}, {1, 0, 0, 0, 23});

  sz.resize(6);
  sz.size_at(5) = 43;
  sz.stride_at(5) = 24;

  checkData(sz, {0, 0, 0, 0, 42, 43}, {1, 0, 0, 0, 23, 24});
}

TEST(SizesAndStridesTest, SetAtIterator) {
  SizesAndStrides sz;

  sz.resize(5);
  *(sz.sizes_begin() + 4) = 42;
  *(sz.strides_begin() + 4) = 23;

  checkData(sz, {0, 0, 0, 0, 42}, {1, 0, 0, 0, 23});

  sz.resize(6);
  *(sz.sizes_begin() + 5) = 43;
  *(sz.strides_begin() + 5) = 24;

  checkData(sz, {0, 0, 0, 0, 42, 43}, {1, 0, 0, 0, 23, 24});
}

TEST(SizesAndStridesTest, SetViaData) {
  SizesAndStrides sz;

  sz.resize(5);
  *(sz.sizes_data() + 4) = 42;
  *(sz.strides_data() + 4) = 23;

  checkData(sz, {0, 0, 0, 0, 42}, {1, 0, 0, 0, 23});

  sz.resize(6);
  *(sz.sizes_data() + 5) = 43;
  *(sz.strides_data() + 5) = 24;

  checkData(sz, {0, 0, 0, 0, 42, 43}, {1, 0, 0, 0, 23, 24});
}

static SizesAndStrides makeSmall(int offset = 0) {
  SizesAndStrides small;
  small.resize(3);
  for (const auto ii : c10::irange(small.size())) {
    small.size_at_unchecked(ii) = ii + 1 + offset;
    small.stride_at_unchecked(ii) = 2 * (ii + 1 + offset);
  }

  return small;
}

static SizesAndStrides makeBig(int offset = 0) {
  SizesAndStrides big;
  big.resize(8);
  for (const auto ii : c10::irange(big.size())) {
    big.size_at_unchecked(ii) = ii - 1 + offset;
    big.stride_at_unchecked(ii) = 2 * (ii - 1 + offset);
  }

  return big;
}

static void checkSmall(const SizesAndStrides& sm, int offset = 0) {
  std::vector<int64_t> sizes(3), strides(3);
  for (const auto ii : c10::irange(3)) {
    sizes[ii] = ii + 1 + offset;
    strides[ii] = 2 * (ii + 1 + offset);
  }
  checkData(sm, sizes, strides);
}

static void checkBig(const SizesAndStrides& big, int offset = 0) {
  std::vector<int64_t> sizes(8), strides(8);
  for (const auto ii : c10::irange(8)) {
    sizes[ii] = ii - 1 + offset;
    strides[ii] = 2 * (ii - 1 + offset);
  }
  checkData(big, sizes, strides);
}

TEST(SizesAndStridesTest, MoveConstructor) {
  SizesAndStrides empty;

  SizesAndStrides movedEmpty(std::move(empty));

  // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  EXPECT_EQ(empty.size(), 0);
  EXPECT_EQ(movedEmpty.size(), 1);
  checkData(movedEmpty, {0}, {1});

  SizesAndStrides small = makeSmall();
  checkSmall(small);

  SizesAndStrides movedSmall(std::move(small));
  checkSmall(movedSmall);
  // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  EXPECT_EQ(small.size(), 0);

  SizesAndStrides big = makeBig();
  checkBig(big);

  SizesAndStrides movedBig(std::move(big));
  checkBig(movedBig);
  // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  EXPECT_EQ(big.size(), 0);
}

TEST(SizesAndStridesTest, CopyConstructor) {
  SizesAndStrides empty;

  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  SizesAndStrides copiedEmpty(empty);

  EXPECT_EQ(empty.size(), 1);
  EXPECT_EQ(copiedEmpty.size(), 1);
  checkData(empty, {0}, {1});
  checkData(copiedEmpty, {0}, {1});

  SizesAndStrides small = makeSmall();
  checkSmall(small);

  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  SizesAndStrides copiedSmall(small);
  checkSmall(copiedSmall);
  checkSmall(small);

  SizesAndStrides big = makeBig();
  checkBig(big);

  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  SizesAndStrides copiedBig(big);
  checkBig(big);
  checkBig(copiedBig);
}

TEST(SizesAndStridesTest, CopyAssignmentSmallToSmall) {
  SizesAndStrides smallTarget = makeSmall();
  SizesAndStrides smallCopyFrom = makeSmall(1);

  checkSmall(smallTarget);
  checkSmall(smallCopyFrom, 1);

  smallTarget = smallCopyFrom;

  checkSmall(smallTarget, 1);
  checkSmall(smallCopyFrom, 1);
}

TEST(SizesAndStridesTest, MoveAssignmentSmallToSmall) {
  SizesAndStrides smallTarget = makeSmall();
  SizesAndStrides smallMoveFrom = makeSmall(1);

  checkSmall(smallTarget);
  checkSmall(smallMoveFrom, 1);

  smallTarget = std::move(smallMoveFrom);

  checkSmall(smallTarget, 1);
  // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  EXPECT_EQ(smallMoveFrom.size(), 0);
}

TEST(SizesAndStridesTest, CopyAssignmentSmallToBig) {
  SizesAndStrides bigTarget = makeBig();
  SizesAndStrides smallCopyFrom = makeSmall();

  checkBig(bigTarget);
  checkSmall(smallCopyFrom);

  bigTarget = smallCopyFrom;

  checkSmall(bigTarget);
  checkSmall(smallCopyFrom);
}

TEST(SizesAndStridesTest, MoveAssignmentSmallToBig) {
  SizesAndStrides bigTarget = makeBig();
  SizesAndStrides smallMoveFrom = makeSmall();

  checkBig(bigTarget);
  checkSmall(smallMoveFrom);

  bigTarget = std::move(smallMoveFrom);

  checkSmall(bigTarget);
  // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  EXPECT_EQ(smallMoveFrom.size(), 0);
}

TEST(SizesAndStridesTest, CopyAssignmentBigToBig) {
  SizesAndStrides bigTarget = makeBig();
  SizesAndStrides bigCopyFrom = makeBig(1);

  checkBig(bigTarget);
  checkBig(bigCopyFrom, 1);

  bigTarget = bigCopyFrom;

  checkBig(bigTarget, 1);
  checkBig(bigCopyFrom, 1);
}

TEST(SizesAndStridesTest, MoveAssignmentBigToBig) {
  SizesAndStrides bigTarget = makeBig();
  SizesAndStrides bigMoveFrom = makeBig(1);

  checkBig(bigTarget);
  checkBig(bigMoveFrom, 1);

  bigTarget = std::move(bigMoveFrom);

  checkBig(bigTarget, 1);
  // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  EXPECT_EQ(bigMoveFrom.size(), 0);
}

TEST(SizesAndStridesTest, CopyAssignmentBigToSmall) {
  SizesAndStrides smallTarget = makeSmall();
  SizesAndStrides bigCopyFrom = makeBig();

  checkSmall(smallTarget);
  checkBig(bigCopyFrom);

  smallTarget = bigCopyFrom;

  checkBig(smallTarget);
  checkBig(bigCopyFrom);
}

TEST(SizesAndStridesTest, MoveAssignmentBigToSmall) {
  SizesAndStrides smallTarget = makeSmall();
  SizesAndStrides bigMoveFrom = makeBig();

  checkSmall(smallTarget);
  checkBig(bigMoveFrom);

  smallTarget = std::move(bigMoveFrom);

  checkBig(smallTarget);
  // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  EXPECT_EQ(bigMoveFrom.size(), 0);
}

TEST(SizesAndStridesTest, CopyAssignmentSelf) {
  SizesAndStrides small = makeSmall();
  SizesAndStrides big = makeBig();

  checkSmall(small);
  checkBig(big);

  // NOLINTNEXTLINE(clang-diagnostic-self-assign-overloaded)
  small = small;
  checkSmall(small);

  // NOLINTNEXTLINE(clang-diagnostic-self-assign-overloaded)
  big = big;
  checkBig(big);
}

// Avoid failures due to -Wall -Wself-move.
static void selfMove(SizesAndStrides& x, SizesAndStrides& y) {
  x = std::move(y);
}

TEST(SizesAndStridesTest, MoveAssignmentSelf) {
  SizesAndStrides small = makeSmall();
  SizesAndStrides big = makeBig();

  checkSmall(small);
  checkBig(big);

  selfMove(small, small);
  checkSmall(small);

  selfMove(big, big);
  checkBig(big);
}
// NOLINTEND(*conversion*, *multiplication*)
