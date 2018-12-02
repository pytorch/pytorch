#include "gtest/gtest.h"

#include "ATen/ATen.h"
#include "ATen/native/ResultType.h"

using namespace at;

void testResultType(Type& weakType, Type& strongType, Type& promotedType) {
  auto strongWrapped = ones({}, strongType).mul(5);
  strongWrapped.unsafeGetTensorImpl()->set_wrapped_number(true);
  auto strongScalar = ones({}, strongType).mul(5);
  auto strongTensor = ones({3, 3}, strongType).mul(5);

  auto weakWrapped = ones({}, weakType).mul(5);
  weakWrapped.unsafeGetTensorImpl()->set_wrapped_number(true);
  auto weakScalar = ones({}, weakType).mul(5);
  auto weakTensor = ones({3, 3}, weakType).mul(5);

  // Single argument
  EXPECT_EQ(weakType, resultType(weakWrapped));
  EXPECT_EQ(weakType, resultType(weakScalar));
  EXPECT_EQ(weakType, resultType(weakTensor));

  // Wrapped scalars only
  EXPECT_EQ(weakType, resultType({weakWrapped, weakWrapped}));
  EXPECT_EQ(strongType, resultType({strongWrapped, strongWrapped}));
  EXPECT_EQ(strongType, resultType({strongWrapped, weakWrapped}));
  EXPECT_EQ(strongType, resultType({weakWrapped, strongWrapped}));

  // 0-dim only promotion
  EXPECT_EQ(weakType, resultType({weakScalar, weakScalar}));
  EXPECT_EQ(strongType, resultType({strongScalar, strongScalar}));
  EXPECT_EQ(strongType, resultType({strongScalar, weakScalar}));
  EXPECT_EQ(strongType, resultType({weakScalar, strongScalar}));

  // Non 0-dim tensor only promotion
  EXPECT_EQ(weakType, resultType({weakTensor, weakTensor}));
  EXPECT_EQ(strongType, resultType({strongTensor, strongTensor}));
  EXPECT_EQ(strongType, resultType({strongTensor, weakTensor}));
  EXPECT_EQ(strongType, resultType({weakTensor, strongTensor}));

  // 0-dim precedes wrapped scalars
  EXPECT_EQ(weakType, resultType({weakScalar, strongWrapped}));
  EXPECT_EQ(weakType, resultType({strongWrapped, weakScalar}));

  // Non 0-dim precedes 0-dim
  EXPECT_EQ(weakType, resultType({weakTensor, strongScalar}));
  EXPECT_EQ(weakType, resultType({strongScalar, weakTensor}));

  // Associativity
  EXPECT_EQ(weakType, resultType({weakTensor, strongScalar, weakWrapped}));
  EXPECT_EQ(weakType, resultType({weakWrapped, weakTensor, strongScalar}));
  EXPECT_EQ(strongType, resultType({weakTensor, strongScalar, strongTensor}));
  EXPECT_EQ(strongType, resultType({strongTensor, weakTensor, strongScalar}));
}

TEST(TestResultType, ResultTypeTestCPU) {
  testResultType(CPU(kInt), CPU(kFloat), CPU(kFloat));
}

TEST(TestResultType, ResultTypeTestGPU) {
  if (at::hasCUDA()) {
    testResultType(CUDA(kInt), CUDA(kFloat), CUDA(kFloat));
  }
}

TEST(TestResultType, ResultTypeTestFailBackendMismatch) {
  if (at::hasCUDA()) {
    auto cpuTensor = ones({3, 3}, CPU(kInt)).mul(5);
    auto cudaTensor = ones({3, 3}, CUDA(kInt)).mul(5);
    EXPECT_THROW(resultType({cpuTensor, cudaTensor}), c10::Error);
  }
}
