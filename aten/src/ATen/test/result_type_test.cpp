#include "gtest/gtest.h"

#include "ATen/ATen.h"
#include "ATen/native/ResultType.h"

using namespace at;

void testResultType(Type& weakType, Type& strongType) {
  Tensor strongWrapped = ones({}).mul(5).to(strongType);
  strongWrapped.unsafeGetTensorImpl()->set_wrapped_number(true);
  Tensor strongZeroDim = ones({}).mul(5).to(strongType);
  Tensor strongMultiDim = ones({3, 3}).mul(5).to(strongType);

  Tensor weakWrapped = ones({}).mul(5).to(weakType);
  weakWrapped.unsafeGetTensorImpl()->set_wrapped_number(true);
  Tensor weakZeroDim = ones({}).mul(5).to(weakType);
  Tensor weakMultiDim = ones({3, 3}).mul(5).to(weakType);

  // Single argument
  EXPECT_EQ(weakType, resultType(weakWrapped));
  EXPECT_EQ(weakType, resultType(weakZeroDim));
  EXPECT_EQ(weakType, resultType(weakMultiDim));

  // Wrapped scalars only
  EXPECT_EQ(weakType, resultType({weakWrapped, weakWrapped}));
  EXPECT_EQ(strongType, resultType({strongWrapped, strongWrapped}));
  EXPECT_EQ(strongType, resultType({strongWrapped, weakWrapped}));
  EXPECT_EQ(strongType, resultType({weakWrapped, strongWrapped}));

  // 0-dim only promotion
  EXPECT_EQ(weakType, resultType({weakZeroDim, weakZeroDim}));
  EXPECT_EQ(strongType, resultType({strongZeroDim, strongZeroDim}));
  EXPECT_EQ(strongType, resultType({strongZeroDim, weakZeroDim}));
  EXPECT_EQ(strongType, resultType({weakZeroDim, strongZeroDim}));

  // Non 0-dim tensor only promotion
  EXPECT_EQ(weakType, resultType({weakMultiDim, weakMultiDim}));
  EXPECT_EQ(strongType, resultType({strongMultiDim, strongMultiDim}));
  EXPECT_EQ(strongType, resultType({strongMultiDim, weakMultiDim}));
  EXPECT_EQ(strongType, resultType({weakMultiDim, strongMultiDim}));

  // 0-dim precedes wrapped scalars
  EXPECT_EQ(weakType, resultType({weakZeroDim, strongWrapped}));
  EXPECT_EQ(weakType, resultType({strongWrapped, weakZeroDim}));

  // Non 0-dim precedes 0-dim
  EXPECT_EQ(weakType, resultType({weakMultiDim, strongZeroDim}));
  EXPECT_EQ(weakType, resultType({strongZeroDim, weakMultiDim}));

  // Associativity
  EXPECT_EQ(weakType, resultType({weakMultiDim, strongZeroDim, weakWrapped}));
  EXPECT_EQ(weakType, resultType({weakWrapped, weakMultiDim, strongZeroDim}));
  EXPECT_EQ(strongType, resultType({weakMultiDim, strongZeroDim, strongMultiDim}));
  EXPECT_EQ(strongType, resultType({strongMultiDim, weakMultiDim, strongZeroDim}));
}

TEST(TestResultType, ResultTypeTestCPU) {
  testResultType(CPU(kChar), CPU(kInt));
  testResultType(CPU(kInt), CPU(kFloat));
  testResultType(CPU(kFloat), CPU(kDouble));
}

TEST(TestResultType, ResultTypeTestGPU) {
  if (at::hasCUDA()) {
    testResultType(CUDA(kChar), CUDA(kInt));
    testResultType(CUDA(kInt), CUDA(kFloat));
    testResultType(CUDA(kFloat), CUDA(kDouble));
  }
}

TEST(TestResultType, ResultTypeTestFailBackendMismatch) {
  if (at::hasCUDA()) {
    auto cpuTensor = ones({3, 3}, CPU(kInt)).mul(5);
    auto cudaTensor = ones({3, 3}, CUDA(kInt)).mul(5);
    EXPECT_THROW(resultType({cpuTensor, cudaTensor}), c10::Error);
  }
}
