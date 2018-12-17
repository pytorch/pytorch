#include "gtest/gtest.h"

#include "ATen/ATen.h"
#include "ATen/native/ResultType.h"

using namespace at;

template<typename T1, typename T2>
void expectEqualElement(const T1& a, const T2& b, const char* test) = delete;

template<>
void expectEqualElement(const Tensor& a, const Tensor& b, const char* test) {
  EXPECT_TRUE(a.scalar_type() == b.scalar_type()) << test;
  EXPECT_TRUE(a.equal(b)) << test;
}

template<>
void expectEqualElement(const Scalar& a, const Scalar& b, const char* test) {
  if (a.isIntegral()) {
    EXPECT_TRUE(b.isIntegral()) << test;
    EXPECT_EQ(a.toLong(), b.toLong()) << test;
  } else if (b.isFloatingPoint()) {
    EXPECT_TRUE(b.isFloatingPoint()) << test;
    EXPECT_EQ(a.toDouble(), b.toDouble()) << test;
  } else if (b.isComplex()) {
    EXPECT_TRUE(b.isComplex()) << test;
    EXPECT_EQ(a.toComplexDouble(), b.toComplexDouble()) << test;
  } else {
    AT_ERROR("Unknown scalar type in test");
  }
}

template<int i, typename T>
struct expectEqualTuples {
  void operator()(const T& a, const T& b, const char* test) {
    expectEqualElement(std::get<i>(a), std::get<i>(b), test);
    expectEqualTuples<i-1, T>()(a, b, test);
  }
};

template<typename T>
  struct expectEqualTuples<0, T> {
    void operator()(const T& a, const T& b, const char* test) {
    expectEqualElement(std::get<0>(a), std::get<0>(b), test);
  }
};

#define EXPECT_TUPLE(a, ...)                              \
  expectEqualTuples<std::tuple_size<decltype(a)>::value - 1, decltype(a)>()( \
      a, std::make_tuple(__VA_ARGS__), #a)

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

void testPromoteOperands(Type& weakType, Type& mediumType, Type& strongType) {
  Scalar strongScalar = ones({}).mul(5).to(strongType).item();
  Tensor strongZeroDim = ones({}).mul(5).to(strongType);
  Tensor strongMultiDim = ones({3, 3}).mul(5).to(strongType);

  Scalar mediumScalar = ones({}).mul(5).to(mediumType).item();
  Tensor mediumZeroDim = ones({}).mul(5).to(mediumType);
  Tensor mediumMultiDim = ones({3, 3}).mul(5).to(mediumType);

  Scalar weakScalar = ones({}).mul(5).to(weakType).item();
  Tensor weakZeroDim = ones({}).mul(5).to(weakType);
  Tensor weakMultiDim = ones({3, 3}).mul(5).to(weakType);

  // Unary no promotion.
  EXPECT_TUPLE(promoteOperands(mediumScalar), mediumScalar);
  EXPECT_TUPLE(promoteOperands(mediumZeroDim), mediumZeroDim);
  EXPECT_TUPLE(promoteOperands(mediumMultiDim), mediumMultiDim);

  // Binary no promotion.
  EXPECT_TUPLE(
      promoteOperands(mediumScalar, mediumScalar),
      mediumScalar, mediumScalar);
  EXPECT_TUPLE(
      promoteOperands(mediumZeroDim, mediumZeroDim),
      mediumZeroDim, mediumZeroDim);
  EXPECT_TUPLE(
      promoteOperands(mediumMultiDim, mediumMultiDim),
      mediumMultiDim, mediumMultiDim);

  // Binary same order promotion.
  EXPECT_TUPLE(
      promoteOperands(mediumScalar, weakScalar),
      mediumScalar, mediumScalar);
  EXPECT_TUPLE(
      promoteOperands(weakScalar, mediumScalar),
      mediumScalar, mediumScalar);
  EXPECT_TUPLE(
      promoteOperands(mediumScalar, strongScalar),
      strongScalar, strongScalar);
  EXPECT_TUPLE(
      promoteOperands(mediumZeroDim, weakZeroDim),
      mediumZeroDim, mediumZeroDim);
  EXPECT_TUPLE(
      promoteOperands(weakZeroDim, mediumZeroDim),
      mediumZeroDim, mediumZeroDim);
  EXPECT_TUPLE(
      promoteOperands(mediumZeroDim, strongZeroDim),
      strongZeroDim, strongZeroDim);
  EXPECT_TUPLE(
      promoteOperands(mediumMultiDim, weakMultiDim),
      mediumMultiDim, mediumMultiDim);
  EXPECT_TUPLE(
      promoteOperands(weakMultiDim, mediumMultiDim),
      mediumMultiDim, mediumMultiDim);
  EXPECT_TUPLE(
      promoteOperands(mediumMultiDim, strongMultiDim),
      strongMultiDim, strongMultiDim);

  // Binary higher order prevails.
  EXPECT_TUPLE(
      promoteOperands(mediumScalar, weakZeroDim),
      weakScalar, weakZeroDim);
  EXPECT_TUPLE(
      promoteOperands(mediumZeroDim, strongScalar),
      mediumZeroDim, mediumScalar);
  EXPECT_TUPLE(
      promoteOperands(mediumScalar, weakMultiDim),
      weakScalar, weakMultiDim);
  EXPECT_TUPLE(
      promoteOperands(mediumMultiDim, strongScalar),
      mediumMultiDim, mediumScalar);
  EXPECT_TUPLE(
      promoteOperands(mediumZeroDim, weakMultiDim),
      weakZeroDim, weakMultiDim);
  EXPECT_TUPLE(
      promoteOperands(mediumMultiDim, strongZeroDim),
      mediumMultiDim, mediumZeroDim);

  // Ternary no promotion.
  EXPECT_TUPLE(
      promoteOperands(mediumScalar, mediumScalar, mediumScalar),
      mediumScalar, mediumScalar, mediumScalar);
  EXPECT_TUPLE(
      promoteOperands(mediumZeroDim, mediumZeroDim, mediumZeroDim),
      mediumZeroDim, mediumZeroDim, mediumZeroDim);
  EXPECT_TUPLE(
      promoteOperands(mediumMultiDim, mediumMultiDim, mediumMultiDim),
      mediumMultiDim, mediumMultiDim, mediumMultiDim);

  // Ternary higher order prevails.
  EXPECT_TUPLE(
      promoteOperands(strongScalar, mediumZeroDim, weakMultiDim),
      weakScalar, weakZeroDim, weakMultiDim);

  // From r-values.
  EXPECT_TUPLE(
      promoteOperands(ones({}, weakType), Scalar(4.0)),
      ones({}, weakType), ones({}).mul(4.0).to(weakType).item());
}

TEST(TestResultType, PromoteOperandsTestCPU) {
  testPromoteOperands(CPU(kInt), CPU(kFloat), CPU(kDouble));
}

TEST(TestResultType, PromoteOperandsTestGPU) {
  if (at::hasCUDA()) {
    testPromoteOperands(CUDA(kInt), CUDA(kFloat), CUDA(kDouble));
  }
}

void testCastOperands(Type& intType, Type& longType, Type& floatType, Type& doubleType) {
  Scalar intScalar = ones({}).mul(5).to(intType).item();
  Tensor intZeroDim = ones({}).mul(5).to(intType);
  Tensor intMultiDim = ones({3, 3}).mul(5).to(intType);

  Scalar longScalar = ones({}).mul(5).to(longType).item();
  Tensor longZeroDim = ones({}).mul(5).to(longType);
  Tensor longMultiDim = ones({3, 3}).mul(5).to(longType);

  Scalar floatScalar = ones({}).mul(5).to(floatType).item();
  Tensor floatZeroDim = ones({}).mul(5).to(floatType);
  Tensor floatMultiDim = ones({3, 3}).mul(5).to(floatType);

  Scalar doubleScalar = ones({}).mul(5).to(doubleType).item();
  Tensor doubleZeroDim = ones({}).mul(5).to(doubleType);
  Tensor doubleMultiDim = ones({3, 3}).mul(5).to(doubleType);

  // No restriction.
  EXPECT_TUPLE(castOperands(c10::nullopt, intScalar, longScalar),
      longScalar, longScalar);
  EXPECT_TUPLE(castOperands(c10::nullopt, intMultiDim, doubleScalar),
      intMultiDim, intScalar);
  EXPECT_TUPLE(castOperands(c10::nullopt, doubleZeroDim, floatMultiDim),
      floatZeroDim, floatMultiDim);

  // Cast to result type.
  EXPECT_TUPLE(castOperands(kLong, intScalar, longScalar),
      longScalar, longScalar);
  EXPECT_TUPLE(castOperands(kInt, intMultiDim, floatScalar),
      intMultiDim, intScalar);
  EXPECT_TUPLE(castOperands(kFloat, doubleZeroDim, floatMultiDim),
      floatZeroDim, floatMultiDim);

  // Upcast into dtype.
  EXPECT_TUPLE(castOperands(kFloat, intScalar, longScalar),
      floatScalar, floatScalar);
  EXPECT_TUPLE(castOperands(kFloat, intMultiDim, floatScalar),
      floatMultiDim, floatScalar);
  EXPECT_TUPLE(castOperands(kDouble, doubleZeroDim, floatMultiDim),
      doubleZeroDim, doubleMultiDim);

  // Downcast into dtype (same-kind).
  EXPECT_TUPLE(castOperands(kInt, intScalar, longScalar),
      intScalar, intScalar);
  EXPECT_TUPLE(castOperands(kFloat, doubleZeroDim, doubleMultiDim),
      floatZeroDim, floatMultiDim);

  // Cannot downcast into lower kind dtype.
  EXPECT_THROW(castOperands(kInt, intScalar, floatScalar), c10::Error);
  EXPECT_THROW(castOperands(kLong, doubleZeroDim, floatMultiDim), c10::Error);

  // Invalid dtype.
  EXPECT_THROW(castOperands(Tensor().scalar_type()), c10::Error);
}

TEST(TestResultType, CastOperandsTestCPU) {
  testCastOperands(CPU(kInt), CPU(kLong), CPU(kFloat), CPU(kDouble));
}

TEST(TestResultType, CastOperandsTestGPU) {
  if (at::hasCUDA()) {
    testCastOperands(CUDA(kInt), CUDA(kLong), CUDA(kFloat), CUDA(kDouble));
  }
}
