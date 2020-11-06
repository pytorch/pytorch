#include <gtest/gtest.h>

#include <c10/core/impl/SizesAndStrides.h>

using namespace c10;
using namespace c10::impl;

static void checkData(const SizesAndStrides& sz, IntArrayRef sizes, IntArrayRef strides) {
  EXPECT_EQ(sizes.size(), strides.size()) << "bad test case: size() of sizes and strides don't match";
  EXPECT_EQ(sz.size(), sizes.size());

  int idx = 0;
  for (auto x: sizes) {
    EXPECT_EQ(sz.size_at_unchecked(idx), x);
    EXPECT_EQ(sz.size_at(idx), x);
    EXPECT_EQ(sz.sizes_data()[idx], x);
    EXPECT_EQ(*(sz.sizes_begin() + idx), x);
    idx++;
  }

  idx = 0;
  for (auto x: strides) {
    EXPECT_EQ(sz.stride_at_unchecked(idx), x);
    EXPECT_EQ(sz.stride_at(idx), x);
    EXPECT_EQ(sz.strides_data()[idx], x);
    EXPECT_EQ(*(sz.strides_begin() + idx), x);

    idx++;
  }
}

TEST(SizesAndStridesTest, DefaultConstructor) {
  SizesAndStrides sz;
  checkData(sz, {0}, {1});
  // Can't test size_at() out of bounds because it just asserts for now.
}

TEST(SizesAndStridesTest, Resize) {
  SizesAndStrides sz;

  sz.resize(2);

  // Small to small.
  checkData(sz, {0, 0}, {1, 0});

  // Small to small, again.
  sz.resize(5);
  checkData(sz, {0, 0, 0, 0, 0}, {1, 0, 0, 0, 0});

  for (int ii = 0; ii < sz.size(); ++ii) {
    sz.size_at_unchecked(ii) = ii + 1;
    sz.stride_at_unchecked(ii) = 2 * (ii + 1);
  }

  checkData(sz, {1, 2, 3, 4, 5}, {2, 4, 6, 8, 10});

  // Small to big.
  sz.resize(6);

  checkData(sz, {1, 2, 3, 4, 5, 0}, {2, 4, 6, 8, 10, 0});

  sz.size_at_unchecked(5) = 6;
  sz.stride_at_unchecked(5) = 12;

  checkData(sz, {1, 2, 3, 4, 5, 6}, {2, 4, 6, 8, 10, 12});

  // Big to big.
  sz.resize(7);

  checkData(sz, {1, 2, 3, 4, 5, 6, 0}, {2, 4, 6, 8, 10, 12, 0});

  // Finally, big to small.

  // Give it different data than it had when it was small to avoid
  // getting it right by accident (i.e., because of leftover inline
  // storage when going small to big).
  for (int ii = 0; ii < sz.size(); ++ii) {
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
  for (int ii = 0; ii < small.size(); ++ii) {
    small.size_at_unchecked(ii) = ii + 1 + offset;
    small.stride_at_unchecked(ii) = 2 * (ii + 1 + offset);
  }

  return small;
}

static SizesAndStrides makeBig(int offset = 0) {
  SizesAndStrides big;
  big.resize(8);
  for (int ii = 0; ii < big.size(); ++ii) {
    big.size_at_unchecked(ii) = ii - 1 + offset;
    big.stride_at_unchecked(ii) = 2 * (ii - 1 + offset);
  }

  return big;
}

TEST(SizesAndStridesTest, MoveConstructor) {
  SizesAndStrides empty;

  SizesAndStrides movedEmpty(std::move(empty));

  EXPECT_EQ(empty.size(), 0);
  EXPECT_EQ(movedEmpty.size(), 1);
  checkData(movedEmpty, {0}, {1});

  SizesAndStrides small = makeSmall();
  checkData(small, {1, 2, 3}, {2, 4, 6});

  SizesAndStrides movedSmall(std::move(small));
  checkData(movedSmall, {1, 2, 3}, {2, 4, 6});
  EXPECT_EQ(small.size(), 0);

  SizesAndStrides big = makeBig();
  checkData(big, {-1, 0, 1, 2, 3, 4, 5, 6}, {-2, 0, 2, 4, 6, 8, 10, 12});

  SizesAndStrides movedBig(std::move(big));
  checkData(movedBig, {-1, 0, 1, 2, 3, 4, 5, 6}, {-2, 0, 2, 4, 6, 8, 10, 12});
  EXPECT_EQ(big.size(), 0);
}

TEST(SizesAndStridesTest, CopyConstructor) {
  SizesAndStrides empty;

  SizesAndStrides copiedEmpty(empty);

  EXPECT_EQ(empty.size(), 1);
  EXPECT_EQ(copiedEmpty.size(), 1);
  checkData(empty, {0}, {1});
  checkData(copiedEmpty, {0}, {1});

  SizesAndStrides small = makeSmall();
  checkData(small, {1, 2, 3}, {2, 4, 6});

  SizesAndStrides copiedSmall(small);
  checkData(copiedSmall, {1, 2, 3}, {2, 4, 6});
  checkData(small, {1, 2, 3}, {2, 4, 6});

  SizesAndStrides big = makeBig();
  checkData(big, {-1, 0, 1, 2, 3, 4, 5, 6}, {-2, 0, 2, 4, 6, 8, 10, 12});

  SizesAndStrides copiedBig(big);
  checkData(big, {-1, 0, 1, 2, 3, 4, 5, 6}, {-2, 0, 2, 4, 6, 8, 10, 12});
  checkData(copiedBig, {-1, 0, 1, 2, 3, 4, 5, 6}, {-2, 0, 2, 4, 6, 8, 10, 12});
}

TEST(SizesAndStridesTest, CopyAssignmentSmallToSmall) {
  SizesAndStrides smallTarget = makeSmall();
  SizesAndStrides smallCopyFrom = makeSmall(1);

  checkData(smallTarget, {1, 2, 3}, {2, 4, 6});
  checkData(smallCopyFrom, {2, 3, 4}, {4, 6, 8});

  smallTarget = smallCopyFrom;

  checkData(smallCopyFrom, {2, 3, 4}, {4, 6, 8});
  checkData(smallTarget, {2, 3, 4}, {4, 6, 8});
}

TEST(SizesAndStridesTest, MoveAssignmentSmallToSmall) {
  SizesAndStrides smallTarget = makeSmall();
  SizesAndStrides smallMoveFrom = makeSmall(1);

  checkData(smallTarget, {1, 2, 3}, {2, 4, 6});
  checkData(smallMoveFrom, {2, 3, 4}, {4, 6, 8});

  smallTarget = std::move(smallMoveFrom);

  checkData(smallTarget, {2, 3, 4}, {4, 6, 8});
  EXPECT_EQ(smallMoveFrom.size(), 0);
}

TEST(SizesAndStridesTest, CopyAssignmentSmallToBig) {
  SizesAndStrides bigTarget = makeBig();
  SizesAndStrides smallCopyFrom = makeSmall();

  checkData(bigTarget, {-1, 0, 1, 2, 3, 4, 5, 6}, {-2, 0, 2, 4, 6, 8, 10, 12});
  checkData(smallCopyFrom, {1, 2, 3}, {2, 4, 6});

  bigTarget = smallCopyFrom;

  checkData(bigTarget, {1, 2, 3}, {2, 4, 6});
  checkData(smallCopyFrom, {1, 2, 3}, {2, 4, 6});
}

TEST(SizesAndStridesTest, MoveAssignmentSmallToBig) {
  SizesAndStrides bigTarget = makeBig();
  SizesAndStrides smallMoveFrom = makeSmall();

  checkData(bigTarget, {-1, 0, 1, 2, 3, 4, 5, 6}, {-2, 0, 2, 4, 6, 8, 10, 12});
  checkData(smallMoveFrom, {1, 2, 3}, {2, 4, 6});

  bigTarget = std::move(smallMoveFrom);

  checkData(bigTarget, {1, 2, 3}, {2, 4, 6});
  EXPECT_EQ(smallMoveFrom.size(), 0);
}

TEST(SizesAndStridesTest, CopyAssignmentBigToBig) {
  SizesAndStrides bigTarget = makeBig();
  SizesAndStrides bigCopyFrom = makeBig(1);

  checkData(bigTarget, {-1, 0, 1, 2, 3, 4, 5, 6}, {-2, 0, 2, 4, 6, 8, 10, 12});
  checkData(bigCopyFrom, {0, 1, 2, 3, 4, 5, 6, 7}, {0, 2, 4, 6, 8, 10, 12, 14});

  bigTarget = bigCopyFrom;

  checkData(bigTarget, {0, 1, 2, 3, 4, 5, 6, 7}, {0, 2, 4, 6, 8, 10, 12, 14});
  checkData(bigCopyFrom, {0, 1, 2, 3, 4, 5, 6, 7}, {0, 2, 4, 6, 8, 10, 12, 14});
}

TEST(SizesAndStridesTest, MoveAssignmentBigToBig) {
  SizesAndStrides bigTarget = makeBig();
  SizesAndStrides bigMoveFrom = makeBig(1);

  checkData(bigTarget, {-1, 0, 1, 2, 3, 4, 5, 6}, {-2, 0, 2, 4, 6, 8, 10, 12});
  checkData(bigMoveFrom, {0, 1, 2, 3, 4, 5, 6, 7}, {0, 2, 4, 6, 8, 10, 12, 14});

  bigTarget = std::move(bigMoveFrom);

  checkData(bigTarget, {0, 1, 2, 3, 4, 5, 6, 7}, {0, 2, 4, 6, 8, 10, 12, 14});
  EXPECT_EQ(bigMoveFrom.size(), 0);
}

TEST(SizesAndStridesTest, CopyAssignmentBigToSmall) {
  SizesAndStrides smallTarget = makeSmall();
  SizesAndStrides bigCopyFrom = makeBig();

  checkData(smallTarget, {1, 2, 3}, {2, 4, 6});
  checkData(bigCopyFrom, {-1, 0, 1, 2, 3, 4, 5, 6}, {-2, 0, 2, 4, 6, 8, 10, 12});

  smallTarget = bigCopyFrom;

  checkData(smallTarget, {-1, 0, 1, 2, 3, 4, 5, 6}, {-2, 0, 2, 4, 6, 8, 10, 12});
  checkData(bigCopyFrom, {-1, 0, 1, 2, 3, 4, 5, 6}, {-2, 0, 2, 4, 6, 8, 10, 12});
}

TEST(SizesAndStridesTest, MoveAssignmentBigToSmall) {
  SizesAndStrides smallTarget = makeSmall();
  SizesAndStrides bigMoveFrom = makeBig();

  checkData(smallTarget, {1, 2, 3}, {2, 4, 6});
  checkData(bigMoveFrom, {-1, 0, 1, 2, 3, 4, 5, 6}, {-2, 0, 2, 4, 6, 8, 10, 12});

  smallTarget = std::move(bigMoveFrom);

  checkData(smallTarget, {-1, 0, 1, 2, 3, 4, 5, 6}, {-2, 0, 2, 4, 6, 8, 10, 12});
  EXPECT_EQ(bigMoveFrom.size(), 0);
}
