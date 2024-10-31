#include <ATen/test/vec_test_all_types.h>
#include <c10/util/irange.h>
namespace {
#if GTEST_HAS_TYPED_TEST
    template <typename T>
    class Memory : public ::testing::Test {};
    template <typename T>
    class Arithmetics : public ::testing::Test {};
    template <typename T>
    class Comparison : public ::testing::Test {};
    template <typename T>
    class Bitwise : public ::testing::Test {};
    template <typename T>
    class MinMax : public ::testing::Test {};
    template <typename T>
    class Nan : public ::testing::Test {};
    template <typename T>
    class Interleave : public ::testing::Test {};
    template <typename T>
    class SignManipulation : public ::testing::Test {};
    template <typename T>
    class SignManipulationHalfPrecision : public ::testing::Test {};
    template <typename T>
    class Rounding : public ::testing::Test {};
    template <typename T>
    class SqrtAndReciprocal : public ::testing::Test {};
    template <typename T>
    class SqrtAndReciprocalReal : public ::testing::Test {};
    template <typename T>
    class FractionAndRemainderReal : public ::testing::Test {};
    template <typename T>
    class Trigonometric : public ::testing::Test {};
    template <typename T>
    class ErrorFunctions : public ::testing::Test {};
    template <typename T>
    class Exponents : public ::testing::Test {};
    template <typename T>
    class Hyperbolic : public ::testing::Test {};
    template <typename T>
    class InverseTrigonometric : public ::testing::Test {};
    template <typename T>
    class InverseTrigonometricReal : public ::testing::Test {};
    template <typename T>
    class LGamma : public ::testing::Test {};
    template <typename T>
    class Logarithm : public ::testing::Test {};
    template <typename T>
    class LogarithmReals : public ::testing::Test {};
    template <typename T>
    class Pow : public ::testing::Test {};
    template <typename T>
    class RangeFactories : public ::testing::Test {};
    template <typename T>
    class BitwiseFloatsAdditional : public ::testing::Test {};
    template <typename T>
    class BitwiseFloatsAdditional2 : public ::testing::Test {};
    template <typename T>
    class RealTests : public ::testing::Test {};
    template <typename T>
    class ComplexTests : public ::testing::Test {};
    template <typename T>
    class QuantizationTests : public ::testing::Test {};
    template <typename T>
    class Quantization8BitWithTailTests : public ::testing::Test {};
    template <typename T>
    class FunctionalTests : public ::testing::Test {};
    template <typename T>
    class FunctionalTestsReducedFloat : public ::testing::Test {};
    template <typename T>
    class InfiniteTests : public ::testing::Test {};
    template <typename T>
    class VecConvertTests : public ::testing::Test {};
    template <typename T>
    class VecConvertTestsReducedFloat : public ::testing::Test {};
    template <typename T>
    class VecMaskTests : public ::testing::Test {};
    using RealFloatTestedTypes = ::testing::Types<vfloat, vdouble>;
    using FloatTestedTypes = ::testing::Types<vfloat, vdouble, vcomplex, vcomplexDbl>;
    using ALLTestedTypes = ::testing::Types<vfloat, vdouble, vcomplex, vlong, vint, vshort, vqint8, vquint8, vqint>;
    using QuantTestedTypes = ::testing::Types<vqint8, vquint8, vqint>;
#if (defined(CPU_CAPABILITY_AVX2) ||  defined(CPU_CAPABILITY_AVX512))  && !defined(_MSC_VER)
    using Quantization8BitWithTailTestedTypes =
        ::testing::Types<vqint8, vquint8>;
#endif
    using RealFloatIntTestedTypes = ::testing::Types<vfloat, vdouble, vlong, vint, vshort>;
    using FloatIntTestedTypes = ::testing::Types<vfloat, vdouble, vcomplex, vcomplexDbl, vlong, vint, vshort>;
    using ComplexTypes = ::testing::Types<vcomplex, vcomplexDbl>;
    using ReducedFloatTestedTypes = ::testing::Types<vBFloat16, vHalf>;
    TYPED_TEST_SUITE(Memory, ALLTestedTypes);
    TYPED_TEST_SUITE(Arithmetics, FloatIntTestedTypes);
    TYPED_TEST_SUITE(Comparison, RealFloatIntTestedTypes);
    TYPED_TEST_SUITE(Bitwise, FloatIntTestedTypes);
    TYPED_TEST_SUITE(MinMax, RealFloatIntTestedTypes);
    TYPED_TEST_SUITE(Nan, RealFloatTestedTypes);
    TYPED_TEST_SUITE(Interleave, RealFloatIntTestedTypes);
    TYPED_TEST_SUITE(SignManipulation, FloatIntTestedTypes);
    TYPED_TEST_SUITE(SignManipulationHalfPrecision, ReducedFloatTestedTypes);
    TYPED_TEST_SUITE(Rounding, RealFloatTestedTypes);
    TYPED_TEST_SUITE(SqrtAndReciprocal, FloatTestedTypes);
    TYPED_TEST_SUITE(SqrtAndReciprocalReal, RealFloatTestedTypes);
    TYPED_TEST_SUITE(FractionAndRemainderReal, RealFloatTestedTypes);
    TYPED_TEST_SUITE(Trigonometric, RealFloatTestedTypes);
    TYPED_TEST_SUITE(ErrorFunctions, RealFloatTestedTypes);
    TYPED_TEST_SUITE(Exponents, RealFloatTestedTypes);
    TYPED_TEST_SUITE(Hyperbolic, RealFloatTestedTypes);
    TYPED_TEST_SUITE(InverseTrigonometricReal, RealFloatTestedTypes);
    TYPED_TEST_SUITE(InverseTrigonometric, FloatTestedTypes);
    TYPED_TEST_SUITE(LGamma, RealFloatTestedTypes);
    TYPED_TEST_SUITE(Logarithm, FloatTestedTypes);
    TYPED_TEST_SUITE(LogarithmReals, RealFloatTestedTypes);
    TYPED_TEST_SUITE(Pow, RealFloatTestedTypes);
    TYPED_TEST_SUITE(RealTests, RealFloatTestedTypes);
    TYPED_TEST_SUITE(RangeFactories, FloatIntTestedTypes);
    TYPED_TEST_SUITE(BitwiseFloatsAdditional, RealFloatTestedTypes);
    TYPED_TEST_SUITE(BitwiseFloatsAdditional2, FloatTestedTypes);
    TYPED_TEST_SUITE(QuantizationTests, QuantTestedTypes);
    TYPED_TEST_SUITE(InfiniteTests, RealFloatTestedTypes);
#if (defined(CPU_CAPABILITY_AVX2) ||  defined(CPU_CAPABILITY_AVX512))  && !defined(_MSC_VER)
    TYPED_TEST_SUITE(
        Quantization8BitWithTailTests,
        Quantization8BitWithTailTestedTypes);
#endif
    TYPED_TEST_SUITE(FunctionalTests, RealFloatIntTestedTypes);
    TYPED_TEST_SUITE(FunctionalTestsReducedFloat, ReducedFloatTestedTypes);
    TYPED_TEST_SUITE(VecConvertTests, RealFloatIntTestedTypes);
    TYPED_TEST_SUITE(VecConvertTestsReducedFloat, ReducedFloatTestedTypes);
    TYPED_TEST_SUITE(VecMaskTests, RealFloatIntTestedTypes);
    TYPED_TEST(Memory, UnAlignedLoadStore) {
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        constexpr size_t b_size = vec::size() * sizeof(VT);
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers,cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN unsigned char ref_storage[128 * b_size];
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers,cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN unsigned char storage[128 * b_size];
        auto seed = TestSeed();
        ValueGen<unsigned char> generator(seed);
        for (auto& x : ref_storage) {
            x = generator.get();
        }
        // test counted load stores
#if defined(CPU_CAPABILITY_VSX) || defined(CPU_CAPABILITY_ZVECTOR)
        for (int i = 1; i < 2 * vec::size(); i++) {
            vec v = vec::loadu(ref_storage, i);
            v.store(storage);
            size_t count = std::min(i * sizeof(VT), b_size);
            bool cmp = (std::memcmp(ref_storage, storage, count) == 0);
            ASSERT_TRUE(cmp) << "Failure Details:\nTest Seed to reproduce: " << seed
                << "\nCount: " << i;
            if (::testing::Test::HasFailure()) {
                break;
            }
            // clear storage
            std::memset(storage, 0, b_size);
        }
#endif
        // testing unaligned load store
        for (size_t offset = 0; offset < b_size; offset += 1) {
            unsigned char* p1 = ref_storage + offset;
            unsigned char* p2 = storage + offset;
            for (; p1 + b_size <= std::end(ref_storage); p1 += b_size, p2 += b_size) {
                vec v = vec::loadu(p1);
                v.store(p2);
            }
            size_t written = p1 - ref_storage - offset;
            bool cmp = (std::memcmp(ref_storage + offset, storage + offset, written) == 0);
            ASSERT_TRUE(cmp) << "Failure Details:\nTest Seed to reproduce: " << seed
                << "\nMismatch at unaligned offset: " << offset;
            if (::testing::Test::HasFailure()) {
                break;
            }
            // clear storage
            std::memset(storage, 0, sizeof storage);
        }
    }
    TYPED_TEST(SignManipulation, Absolute) {
        using vec = TypeParam;
        bool checkRelativeErr = is_complex<ValueType<TypeParam>>();
        test_unary<vec>(
            NAME_INFO(absolute), RESOLVE_OVERLOAD(local_abs),
            [](vec v) { return v.abs(); },
            createDefaultUnaryTestCase<vec>(TestSeed(), false, checkRelativeErr),
            RESOLVE_OVERLOAD(filter_int_minimum));
    }
    TYPED_TEST(SignManipulation, Negate) {
        using vec = TypeParam;
        // negate overflows for minimum on int and long
        test_unary<vec>(
            NAME_INFO(negate), std::negate<ValueType<vec>>(),
            [](vec v) { return v.neg(); },
            createDefaultUnaryTestCase<vec>(TestSeed()),
            RESOLVE_OVERLOAD(filter_int_minimum));
    }
    TYPED_TEST(SignManipulationHalfPrecision, AbsNegate) {
      typedef enum  {
        ABS,
        NEGATE
      } SignOpType;
      using vec = TypeParam;
      using VT = UholdType<TypeParam>;
      using RT = float; // reference
      float atol = 0.01f;
      float rtol = 0.01f;

      auto cmp = [&](RT ref, VT val) {
        return std::abs(ref - RT(val)) <= atol + rtol * std::abs(val);
      };

#define APPLY_FN_AND_STORE(VEC_TYPE)                            \
      [&](SignOpType op_type, VEC_TYPE& x_fp_vec, void *x_fp) { \
        if (op_type == SignOpType::NEGATE) {                    \
          x_fp_vec.neg().store(x_fp);                           \
        } else {                                                \
          x_fp_vec.abs().store(x_fp);                           \
        }                                                       \
      }

      auto apply_fn_and_store_ref = APPLY_FN_AND_STORE(vfloat);
      auto apply_fn_and_store_half = APPLY_FN_AND_STORE(vec);

      auto half_precision_ut = [&](SignOpType op_type) {
        constexpr auto N = vec::size();
        CACHE_ALIGN RT x_fp[N];
        CACHE_ALIGN VT x_hp[N];
        auto seed = TestSeed();
        ValueGen<RT> generator(RT(-1), RT(1), seed);
        for (const auto i : c10::irange(N)) {
            x_fp[i] = generator.get();
            x_hp[i] = VT(x_fp[i]);
        }
        auto x_fp_vec = vfloat::loadu(x_fp);
        apply_fn_and_store_ref(op_type, x_fp_vec, x_fp);
        x_fp_vec = vfloat::loadu(x_fp + vfloat::size());
        apply_fn_and_store_ref(op_type, x_fp_vec, x_fp + vfloat::size());

        auto x_hp_vec = vec::loadu(x_hp);
        apply_fn_and_store_half(op_type, x_hp_vec, x_hp);

        for (int64_t len = 0; len < N; len++) {
            ASSERT_TRUE(cmp(x_fp[len], x_hp[len])) << "Failure Details:\nTest Seed to reproduce: " << seed
                << "\nabs/negate, Length: " << len << "; fp32: " << x_fp[len] << "; bf16/fp16: " << RT(x_hp[len]);
        }
      };

      half_precision_ut(SignOpType::ABS);
      half_precision_ut(SignOpType::NEGATE);
    }
    TYPED_TEST(Rounding, Round) {
        using vec = TypeParam;
        using UVT = UvalueType<TypeParam>;
        UVT case1 = -658.5f;
        UVT exp1 = -658.f;
        UVT case2 = -657.5f;
        UVT exp2 = -658.f;
        auto test_case = TestingCase<vec>::getBuilder()
            .addDomain(CheckWithinDomains<UVT>{ { {-1000, 1000}} })
            .addCustom({ {case1},exp1 })
            .addCustom({ {case2},exp2 })
            .setTrialCount(64000)
            .setTestSeed(TestSeed());
        test_unary<vec>(
            NAME_INFO(round),
            RESOLVE_OVERLOAD(at::native::round_impl),
            [](vec v) { return v.round(); },
            test_case);
    }
    TYPED_TEST(Rounding, Ceil) {
        using vec = TypeParam;
        test_unary<vec>(
            NAME_INFO(ceil),
            RESOLVE_OVERLOAD(std::ceil),
            [](vec v) { return v.ceil(); },
            createDefaultUnaryTestCase<vec>(TestSeed()));
    }
    TYPED_TEST(Rounding, Floor) {
        using vec = TypeParam;
        test_unary<vec>(
            NAME_INFO(floor),
            RESOLVE_OVERLOAD(std::floor),
            [](vec v) { return v.floor(); },
            createDefaultUnaryTestCase<vec>(TestSeed()));
    }
    TYPED_TEST(Rounding, Trunc) {
        using vec = TypeParam;
        test_unary<vec>(
            NAME_INFO(trunc),
            RESOLVE_OVERLOAD(std::trunc),
            [](vec v) { return v.trunc(); },
            createDefaultUnaryTestCase<vec>(TestSeed()));
    }
    TYPED_TEST(SqrtAndReciprocal, Sqrt) {
        using vec = TypeParam;
        test_unary<vec>(
            NAME_INFO(sqrt),
            RESOLVE_OVERLOAD(local_sqrt),
            [](vec v) { return v.sqrt(); },
            createDefaultUnaryTestCase<vec>(TestSeed(), false, true));
    }
    TYPED_TEST(SqrtAndReciprocalReal, RSqrt) {
        using vec = TypeParam;
        test_unary<vec>(
            NAME_INFO(rsqrt),
            rsqrt<ValueType<vec>>,
            [](vec v) { return v.rsqrt(); },
            createDefaultUnaryTestCase<vec>(TestSeed()),
            RESOLVE_OVERLOAD(filter_zero));
    }
    TYPED_TEST(SqrtAndReciprocalReal, Reciprocal) {
        using vec = TypeParam;
        test_unary<vec>(
            NAME_INFO(reciprocal),
            reciprocal<ValueType<vec>>,
            [](vec v) { return v.reciprocal(); },
            createDefaultUnaryTestCase<vec>(TestSeed()),
            RESOLVE_OVERLOAD(filter_zero));
    }
    TYPED_TEST(FractionAndRemainderReal, Frac) {
      using vec = TypeParam;
      test_unary<vec>(
          NAME_INFO(frac),
          RESOLVE_OVERLOAD(frac),
          [](vec v) { return v.frac(); },
          createDefaultUnaryTestCase<vec>(TestSeed(), false, true));
    }
    TYPED_TEST(FractionAndRemainderReal, Fmod) {
      using vec = TypeParam;
      test_binary<vec>(
          NAME_INFO(fmod),
          RESOLVE_OVERLOAD(std::fmod),
          [](vec v0, vec v1) { return v0.fmod(v1); },
          createDefaultBinaryTestCase<vec>(TestSeed()),
          RESOLVE_OVERLOAD(filter_fmod));
    }
    TYPED_TEST(Trigonometric, Sin) {
        using vec = TypeParam;
        using UVT = UvalueType<TypeParam>;
        auto test_case = TestingCase<vec>::getBuilder()
            .addDomain(CheckWithinDomains<UVT>{ { {-4096, 4096}}, true, 1.2e-7f})
            .addDomain(CheckWithinDomains<UVT>{ { {-8192, 8192}}, true, 3.0e-7f})
            .setTrialCount(8000)
            .setTestSeed(TestSeed());
        test_unary<vec>(
            NAME_INFO(sin),
            RESOLVE_OVERLOAD(std::sin),
            [](vec v) { return v.sin(); },
            test_case);
    }
    TYPED_TEST(Trigonometric, Cos) {
        using vec = TypeParam;
        using UVT = UvalueType<TypeParam>;
        auto test_case = TestingCase<vec>::getBuilder()
            .addDomain(CheckWithinDomains<UVT>{ { {-4096, 4096}}, true, 1.2e-7f})
            .addDomain(CheckWithinDomains<UVT>{ { {-8192, 8192}}, true, 3.0e-7f})
            .setTrialCount(8000)
            .setTestSeed(TestSeed());
        test_unary<vec>(
            NAME_INFO(cos),
            RESOLVE_OVERLOAD(std::cos),
            [](vec v) { return v.cos(); },
            test_case);
    }
    TYPED_TEST(Trigonometric, Tan) {
        using vec = TypeParam;
        test_unary<vec>(
            NAME_INFO(tan),
            RESOLVE_OVERLOAD(std::tan),
            [](vec v) { return v.tan(); },
            createDefaultUnaryTestCase<vec>(TestSeed()));
    }
    TYPED_TEST(Hyperbolic, Tanh) {
        using vec = TypeParam;
        test_unary<vec>(
            NAME_INFO(tanH),
            RESOLVE_OVERLOAD(std::tanh),
            [](vec v) { return v.tanh(); },
            createDefaultUnaryTestCase<vec>(TestSeed()));
    }
    TYPED_TEST(Hyperbolic, Sinh) {
        using vec = TypeParam;
        using UVT = UvalueType<TypeParam>;
        auto test_case =
            TestingCase<vec>::getBuilder()
            .addDomain(CheckWithinDomains<UVT>{ { {-88, 88}}, true, getDefaultTolerance<UVT>()})
            .setTrialCount(65536)
            .setTestSeed(TestSeed());
        test_unary<vec>(
            NAME_INFO(sinh),
            RESOLVE_OVERLOAD(std::sinh),
            [](vec v) { return v.sinh(); },
            test_case);
    }
    TYPED_TEST(Hyperbolic, Cosh) {
        using vec = TypeParam;
        using UVT = UvalueType<TypeParam>;
        auto test_case =
            TestingCase<vec>::getBuilder()
            .addDomain(CheckWithinDomains<UVT>{ { {-88, 88}}, true, getDefaultTolerance<UVT>()})
            .setTrialCount(65536)
            .setTestSeed(TestSeed());
        test_unary<vec>(
            NAME_INFO(cosh),
            RESOLVE_OVERLOAD(std::cosh),
            [](vec v) { return v.cosh(); },
            test_case);
    }
    TYPED_TEST(InverseTrigonometric, Asin) {
        using vec = TypeParam;
        using UVT = UvalueType<TypeParam>;
        bool checkRelativeErr = is_complex<ValueType<TypeParam>>();
        auto test_case =
            TestingCase<vec>::getBuilder()
            .addDomain(CheckWithinDomains<UVT>{ { {-10, 10}}, checkRelativeErr, getDefaultTolerance<UVT>() })
            .setTrialCount(125536)
            .setTestSeed(TestSeed());
        test_unary<vec>(
            NAME_INFO(asin),
            RESOLVE_OVERLOAD(local_asin),
            [](vec v) { return v.asin(); },
            test_case);
    }
    TYPED_TEST(InverseTrigonometric, ACos) {
        using vec = TypeParam;
        using UVT = UvalueType<TypeParam>;
        bool checkRelativeErr = is_complex<ValueType<TypeParam>>();
        auto test_case =
            TestingCase<vec>::getBuilder()
            .addDomain(CheckWithinDomains<UVT>{ { {-10, 10}}, checkRelativeErr, getDefaultTolerance<UVT>() })
            .setTrialCount(125536)
            .setTestSeed(TestSeed());
        test_unary<vec>(
            NAME_INFO(acos),
            RESOLVE_OVERLOAD(local_acos),
            [](vec v) { return v.acos(); },
            test_case);
    }
    TYPED_TEST(InverseTrigonometric, ATan) {
        bool checkRelativeErr = is_complex<ValueType<TypeParam>>();
        using vec = TypeParam;
        using UVT = UvalueType<TypeParam>;
        auto test_case =
            TestingCase<vec>::getBuilder()
            .addDomain(CheckWithinDomains<UVT>{ { {-100, 100}}, checkRelativeErr, getDefaultTolerance<UVT>()})
            .setTrialCount(65536)
            .setTestSeed(TestSeed());
        test_unary<vec>(
            NAME_INFO(atan),
            RESOLVE_OVERLOAD(std::atan),
            [](vec v) { return v.atan(); },
            test_case,
            RESOLVE_OVERLOAD(filter_zero));
    }
    TYPED_TEST(Logarithm, Log) {
        using vec = TypeParam;
        test_unary<vec>(
            NAME_INFO(log),
            RESOLVE_OVERLOAD(std::log),
            [](const vec& v) { return v.log(); },
            createDefaultUnaryTestCase<vec>(TestSeed()));
    }
    TYPED_TEST(LogarithmReals, Log2) {
        using vec = TypeParam;
        test_unary<vec>(
            NAME_INFO(log2),
            RESOLVE_OVERLOAD(local_log2),
            [](const vec& v) { return v.log2(); },
            createDefaultUnaryTestCase<vec>(TestSeed()));
    }
    TYPED_TEST(Logarithm, Log10) {
        using vec = TypeParam;
        test_unary<vec>(
            NAME_INFO(log10),
            RESOLVE_OVERLOAD(std::log10),
            [](const vec& v) { return v.log10(); },
            createDefaultUnaryTestCase<vec>(TestSeed()));
    }
    TYPED_TEST(LogarithmReals, Log1p) {
        using vec = TypeParam;
        using UVT = UvalueType<TypeParam>;
        auto test_case =
            TestingCase<vec>::getBuilder()
            .addDomain(CheckWithinDomains<UVT>{ { {-1, 1000}}, true, getDefaultTolerance<UVT>()})
            .addDomain(CheckWithinDomains<UVT>{ { {1000, 1.e+30}}, true, getDefaultTolerance<UVT>()})
            .setTrialCount(65536)
            .setTestSeed(TestSeed());
        test_unary<vec>(
            NAME_INFO(log1p),
            RESOLVE_OVERLOAD(std::log1p),
            [](const vec& v) { return v.log1p(); },
            test_case);
    }
    TYPED_TEST(Exponents, Exp) {
        using vec = TypeParam;
        test_unary<vec>(
            NAME_INFO(exp),
            RESOLVE_OVERLOAD(std::exp),
            [](const vec& v) { return v.exp(); },
            createDefaultUnaryTestCase<vec>(TestSeed()));
    }
    TYPED_TEST(Exponents, Expm1) {
        using vec = TypeParam;
        test_unary<vec>(
            NAME_INFO(expm1),
            RESOLVE_OVERLOAD(std::expm1),
            [](const vec& v) { return v.expm1(); },
            createDefaultUnaryTestCase<vec>(TestSeed(), false, true));
    }
    TYPED_TEST(ErrorFunctions, Erf) {
        using vec = TypeParam;
        test_unary<vec>(
            NAME_INFO(erf),
            RESOLVE_OVERLOAD(std::erf),
            [](const vec& v) { return v.erf(); },
            createDefaultUnaryTestCase<vec>(TestSeed(), false, true));
    }
    TYPED_TEST(ErrorFunctions, Erfc) {
        using vec = TypeParam;
        test_unary<vec>(
            NAME_INFO(erfc),
            RESOLVE_OVERLOAD(std::erfc),
            [](const vec& v) { return v.erfc(); },
            createDefaultUnaryTestCase<vec>(TestSeed(), false, true));
    }
    TYPED_TEST(ErrorFunctions, Erfinv) {
        using vec = TypeParam;
        test_unary<vec>(
            NAME_INFO(erfinv),
            RESOLVE_OVERLOAD(calc_erfinv),
            [](const vec& v) { return v.erfinv(); },
            createDefaultUnaryTestCase<vec>(TestSeed(), false, true));
    }
    TYPED_TEST(Nan, IsNan) {
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN VT test_vals[vec::size()];
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN VT expected_vals[vec::size()];
        auto vals = 1 << (vec::size());
        for (const auto val : c10::irange(vals)) {
          for (int i = 0; i < vec::size(); ++i) {
            if (val & (1 << i)) {
              test_vals[i] = std::numeric_limits<VT>::quiet_NaN();
              // All bits are set to 1 if true, otherwise 0.
              // same rule as at::Vectorized<T>::binary_pred.
              std::memset(static_cast<void*>(&expected_vals[i]), 0xFF, sizeof(VT));
            } else {
              test_vals[i] = (VT)0.123;
              std::memset(static_cast<void*>(&expected_vals[i]), 0, sizeof(VT));
            }
          }
          vec actual = vec::loadu(test_vals).isnan();
          vec expected = vec::loadu(expected_vals);
          AssertVectorized<vec>(NAME_INFO(isnan), expected, actual).check();
        }
    }
    TYPED_TEST(LGamma, LGamma) {
        using vec = TypeParam;
        using UVT = UvalueType<vec>;
        UVT tolerance = getDefaultTolerance<UVT>();
        // double: 2e+305  float: 4e+36 (https://sleef.org/purec.xhtml#eg)
        UVT maxCorrect = std::is_same_v<UVT, float> ? (UVT)4e+36 : (UVT)2e+305;
        TestingCase<vec> testCase = TestingCase<vec>::getBuilder()
            .addDomain(CheckWithinDomains<UVT>{ { {(UVT)-100, (UVT)0}}, true, tolerance})
            .addDomain(CheckWithinDomains<UVT>{ { {(UVT)0, (UVT)1000 }}, true, tolerance})
            .addDomain(CheckWithinDomains<UVT>{ { {(UVT)1000, maxCorrect }}, true, tolerance})
            .setTestSeed(TestSeed());
        test_unary<vec>(
            NAME_INFO(lgamma),
            RESOLVE_OVERLOAD(std::lgamma),
            [](vec v) { return v.lgamma(); },
            testCase);
    }
    TYPED_TEST(InverseTrigonometricReal, ATan2) {
        using vec = TypeParam;
        test_binary<vec>(
            NAME_INFO(atan2),
            RESOLVE_OVERLOAD(std::atan2),
            [](vec v0, vec v1) {
                return v0.atan2(v1);
            },
            createDefaultBinaryTestCase<vec>(TestSeed()));
    }
    TYPED_TEST(Pow, Pow) {
        using vec = TypeParam;
        test_binary<vec>(
            NAME_INFO(pow),
            RESOLVE_OVERLOAD(std::pow),
            [](vec v0, vec v1) { return v0.pow(v1); },
            createDefaultBinaryTestCase<vec>(TestSeed(), false, true));
    }
    TYPED_TEST(RealTests, Hypot) {
        using vec = TypeParam;
        test_binary<vec>(
            NAME_INFO(hypot),
            RESOLVE_OVERLOAD(std::hypot),
            [](vec v0, vec v1) { return v0.hypot(v1); },
            createDefaultBinaryTestCase<vec>(TestSeed(), false, true));
    }
    TYPED_TEST(RealTests, NextAfter) {
        using vec = TypeParam;
        test_binary<vec>(
            NAME_INFO(nextafter),
            RESOLVE_OVERLOAD(std::nextafter),
            [](vec v0, vec v1) { return v0.nextafter(v1); },
            createDefaultBinaryTestCase<vec>(TestSeed(), false, true));
    }
    TYPED_TEST(Interleave, Interleave) {
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        constexpr auto N = vec::size() * 2LL;
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN VT vals[N];
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN VT interleaved[N];
        auto seed = TestSeed();
        ValueGen<VT> generator(seed);
        for (VT& v : vals) {
            v = generator.get();
        }
        copy_interleave(vals, interleaved);
        auto a = vec::loadu(vals);
        auto b = vec::loadu(vals + vec::size());
        auto cc = interleave2(a, b);
        AssertVectorized<vec>(NAME_INFO(Interleave FirstHalf), std::get<0>(cc), vec::loadu(interleaved)).check(true);
        AssertVectorized<vec>(NAME_INFO(Interleave SecondHalf), std::get<1>(cc), vec::loadu(interleaved + vec::size())).check(true);
    }
    TYPED_TEST(Interleave, DeInterleave) {
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        constexpr auto N = vec::size() * 2LL;
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN VT vals[N];
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN VT interleaved[N];
        auto seed = TestSeed();
        ValueGen<VT> generator(seed);
        for (VT& v : vals) {
            v = generator.get();
        }
        copy_interleave(vals, interleaved);
        // test interleaved with vals this time
        auto a = vec::loadu(interleaved);
        auto b = vec::loadu(interleaved + vec::size());
        auto cc = deinterleave2(a, b);
        AssertVectorized<vec>(NAME_INFO(DeInterleave FirstHalf), std::get<0>(cc), vec::loadu(vals)).check(true);
        AssertVectorized<vec>(NAME_INFO(DeInterleave SecondHalf), std::get<1>(cc), vec::loadu(vals + vec::size())).check(true);
    }
    TYPED_TEST(Arithmetics, Plus) {
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        test_binary<vec>(
            NAME_INFO(plus),
            std::plus<VT>(),
            [](const vec& v0, const vec& v1) -> vec {
                return v0 + v1;
            },
            createDefaultBinaryTestCase<vec>(TestSeed()),
                RESOLVE_OVERLOAD(filter_add_overflow));
    }
    TYPED_TEST(Arithmetics, Minus) {
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        test_binary<vec>(
            NAME_INFO(minus),
            std::minus<VT>(),
            [](const vec& v0, const vec& v1) -> vec {
                return v0 - v1;
            },
            createDefaultBinaryTestCase<vec>(TestSeed()),
                RESOLVE_OVERLOAD(filter_sub_overflow));
    }
    TYPED_TEST(Arithmetics, Multiplication) {
        using vec = TypeParam;
        test_binary<vec>(
            NAME_INFO(mult),
            RESOLVE_OVERLOAD(local_multiply),
            [](const vec& v0, const vec& v1) { return v0 * v1; },
            createDefaultBinaryTestCase<vec>(TestSeed(), false, true),
            RESOLVE_OVERLOAD(filter_mult_overflow));
    }
    TYPED_TEST(Arithmetics, Division) {
        using vec = TypeParam;
        TestSeed seed;
        test_binary<vec>(
            NAME_INFO(division),
            RESOLVE_OVERLOAD(local_division),
            [](const vec& v0, const vec& v1) { return v0 / v1; },
            createDefaultBinaryTestCase<vec>(seed),
            RESOLVE_OVERLOAD(filter_div_ub));
    }
    TYPED_TEST(Bitwise, BitAnd) {
        using vec = TypeParam;
        test_binary<vec>(
            NAME_INFO(bit_and),
            RESOLVE_OVERLOAD(local_and),
            [](const vec& v0, const vec& v1) { return v0 & v1; },
            createDefaultBinaryTestCase<vec>(TestSeed(), true));
    }
    TYPED_TEST(Bitwise, BitOr) {
        using vec = TypeParam;
        test_binary<vec>(
            NAME_INFO(bit_or),
            RESOLVE_OVERLOAD(local_or),
            [](const vec& v0, const vec& v1) { return v0 | v1; },
            createDefaultBinaryTestCase<vec>(TestSeed(), true));
    }
    TYPED_TEST(Bitwise, BitXor) {
        using vec = TypeParam;
        test_binary<vec>(
            NAME_INFO(bit_xor),
            RESOLVE_OVERLOAD(local_xor),
            [](const vec& v0, const vec& v1) { return v0 ^ v1; },
            createDefaultBinaryTestCase<vec>(TestSeed(), true));
    }
    TYPED_TEST(Comparison, Equal) {
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        test_binary<vec>(
            NAME_INFO(== ),
            [](const VT& v1, const VT& v2) {return func_cmp(std::equal_to<VT>(), v1, v2); },
            [](const vec& v0, const vec& v1) { return v0 == v1; },
            createDefaultBinaryTestCase<vec>(TestSeed(), true));
    }
    TYPED_TEST(Comparison, NotEqual) {
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        test_binary<vec>(
            NAME_INFO(!= ),
            [](const VT& v1, const VT& v2) {return func_cmp(std::not_equal_to<VT>(), v1, v2); },
            [](const vec& v0, const vec& v1) { return v0 != v1; },
            createDefaultBinaryTestCase<vec>(TestSeed(), true));
    }
    TYPED_TEST(Comparison, Greater) {
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        test_binary<vec>(
            NAME_INFO(> ),
            [](const VT& v1, const VT& v2) {return func_cmp(std::greater<VT>(), v1, v2); },
            [](const vec& v0, const vec& v1) { return v0 > v1; },
            createDefaultBinaryTestCase<vec>(TestSeed(), true));
    }
    TYPED_TEST(Comparison, Less) {
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        test_binary<vec>(
            NAME_INFO(< ),
            [](const VT& v1, const VT& v2) {return func_cmp(std::less<VT>(), v1, v2); },
            [](const vec& v0, const vec& v1) { return v0 < v1; },
            createDefaultBinaryTestCase<vec>(TestSeed(), true));
    }
    TYPED_TEST(Comparison, GreaterEqual) {
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        test_binary<vec>(
            NAME_INFO(>= ),
            [](const VT& v1, const VT& v2) {return func_cmp(std::greater_equal<VT>(), v1, v2); },
            [](const vec& v0, const vec& v1) { return v0 >= v1; },
            createDefaultBinaryTestCase<vec>(TestSeed(), true));
    }
    TYPED_TEST(Comparison, LessEqual) {
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        test_binary<vec>(
            NAME_INFO(<= ),
            [](const VT& v1, const VT& v2) {return func_cmp(std::less_equal<VT>(), v1, v2); },
            [](const vec& v0, const vec& v1) { return v0 <= v1; },
            createDefaultBinaryTestCase<vec>(TestSeed(), true));
    }
    TYPED_TEST(MinMax, Minimum) {
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        test_binary<vec>(
            NAME_INFO(minimum),
            minimum<VT>,
            [](const vec& v0, const vec& v1) {
                return minimum(v0, v1);
            },
            createDefaultBinaryTestCase<vec>(TestSeed()));
    }
    TYPED_TEST(MinMax, Maximum) {
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        test_binary<vec>(
            NAME_INFO(maximum),
            maximum<VT>,
            [](const vec& v0, const vec& v1) {
                return maximum(v0, v1);
            },
            createDefaultBinaryTestCase<vec>(TestSeed()));
    }
    TYPED_TEST(MinMax, ClampMin) {
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        test_binary<vec>(
            NAME_INFO(clamp min),
            clamp_min<VT>,
            [](const vec& v0, const vec& v1) {
                return clamp_min(v0, v1);
            },
            createDefaultBinaryTestCase<vec>(TestSeed()));
    }
    TYPED_TEST(MinMax, ClampMax) {
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        test_binary<vec>(
            NAME_INFO(clamp max),
            clamp_max<VT>,
            [](const vec& v0, const vec& v1) {
                return clamp_max(v0, v1);
            },
            createDefaultBinaryTestCase<vec>(TestSeed()));
    }
    TYPED_TEST(MinMax, Clamp) {
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        test_ternary<vec>(
            NAME_INFO(clamp), clamp<VT>,
            [](const vec& v0, const vec& v1, const vec& v2) {
                return clamp(v0, v1, v2);
            },
            createDefaultTernaryTestCase<vec>(TestSeed()),
                RESOLVE_OVERLOAD(filter_clamp));
    }
    TYPED_TEST(MinMax, ClampVecN) {
        using VT = ValueType<TypeParam>;
        using vec = at::vec::VectorizedN<VT, 1>;
        test_ternary<vec>(
            NAME_INFO(clamp), clamp<VT>,
            [](const vec& v0, const vec& v1, const vec& v2) {
                return clamp(v0, v1, v2);
            },
            createDefaultTernaryTestCase<vec>(TestSeed()),
                RESOLVE_OVERLOAD(filter_clamp));
    }
    TYPED_TEST(BitwiseFloatsAdditional, ZeroMask) {
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN VT test_vals[vec::size()];
        //all sets will be within 0  2^(n-1)
        auto power_sets = 1 << (vec::size());
        for (const auto expected : c10::irange(power_sets)) {
            // generate test_val based on expected
            for (int i = 0; i < vec::size(); ++i)
            {
                if (expected & (1 << i)) {
                    test_vals[i] = (VT)0;
                }
                else {
                    test_vals[i] = (VT)0.897;
                }
            }
            int actual = vec::loadu(test_vals).zero_mask();
            ASSERT_EQ(expected, actual) << "Failure Details:\n"
                << std::hex << "Expected:\n#\t" << expected
                << "\nActual:\n#\t" << actual;
        }
    }
    TYPED_TEST(BitwiseFloatsAdditional, Convert) {
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        using IntVT = at::vec::int_same_size_t<VT>;

        // verify float to int
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN VT input1[vec::size()];
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN IntVT expected_vals1[vec::size()];
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN IntVT actual_vals1[vec::size()];
        for (int64_t i = 0; i < vec::size(); i++) {
            input1[i] = (VT)i * (VT)2.1 + (VT)0.5;
            expected_vals1[i] = static_cast<IntVT>(input1[i]);
        }
        at::vec::convert(input1, actual_vals1, vec::size());
        auto expected1 = VecType<IntVT>::loadu(expected_vals1);
        auto actual1 = VecType<IntVT>::loadu(actual_vals1);
        if (AssertVectorized<VecType<IntVT>>(NAME_INFO(test_convert_to_int), expected1, actual1).check()) {
          return;
        }

        // verify int to float
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN IntVT input2[vec::size()];
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN VT expected_vals2[vec::size()];
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN VT actual_vals2[vec::size()];
        for (int64_t i = 0; i < vec::size(); i++) {
            input2[i] = (IntVT)i * (IntVT)2 + (IntVT)1;
            expected_vals2[i] = (VT)input2[i];
        }
        at::vec::convert(input2, actual_vals2, vec::size());
        auto expected2 = vec::loadu(expected_vals2);
        auto actual2 = vec::loadu(actual_vals2);
        AssertVectorized<vec>(NAME_INFO(test_convert_to_float), expected2, actual2).check();
    }
    TYPED_TEST(BitwiseFloatsAdditional, Fmadd) {
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;

        auto test_case = TestingCase<vec>::getBuilder()
          .addDomain(CheckWithinDomains<VT>{
              {{(VT)-1000, (VT)1000}, {(VT)-1000, (VT)1000}, {(VT)-1000, (VT)1000}},
              true, getDefaultTolerance<VT>()})
          .setTestSeed(TestSeed());

        test_ternary<vec>(
            NAME_INFO(fmadd), RESOLVE_OVERLOAD(local_fmadd),
            [](const vec& v0, const vec& v1, const vec& v2) {
                return at::vec::fmadd(v0, v1, v2);
            },
            test_case,
            RESOLVE_OVERLOAD(filter_fmadd));
    }
    TYPED_TEST(BitwiseFloatsAdditional, FmaddVecN) {
        using VT = ValueType<TypeParam>;
        using vec = at::vec::VectorizedN<VT, 1>;

        auto test_case = TestingCase<vec>::getBuilder()
          .addDomain(CheckWithinDomains<VT>{
              {{(VT)-1000, (VT)1000}, {(VT)-1000, (VT)1000}, {(VT)-1000, (VT)1000}},
              true, getDefaultTolerance<VT>()})
          .setTestSeed(TestSeed());

        test_ternary<vec>(
            NAME_INFO(fmadd), RESOLVE_OVERLOAD(local_fmadd),
            [](const vec& v0, const vec& v1, const vec& v2) {
                return at::vec::fmadd(v0, v1, v2);
            },
            test_case,
            RESOLVE_OVERLOAD(filter_fmadd));
    }
#if defined(CPU_CAPABILITY_NEON)
    TEST(BitwiseFloatsAdditional, HalfToFloatFmadd) {
        using vec = vhalf;
        using VT = ValueType<vec>;

        auto test_case = TestingCase<vec>::getBuilder()
          .addDomain(CheckWithinDomains<VT>{
              {{(VT)-1000, (VT)1000}, {(VT)-1000, (VT)1000}, {(VT)-1000, (VT)1000}},
              true, getDefaultTolerance<VT>()})
          .setTestSeed(TestSeed());

        test_ternary<vec>(
            NAME_INFO(half_to_float_fmadd), RESOLVE_OVERLOAD(local_fmadd),
            [](const vec& v0, const vec& v1, const vec& v2) {
              const auto [v2_float0, v2_float1] = convert_half_float(v2);
              const auto [result_float0, result_float1] = at::vec::fmadd(v0, v1, v2_float0, v2_float1);
              return convert_float_half(result_float0, result_float1);
            },
            test_case,
            RESOLVE_OVERLOAD(filter_fmadd));
    }
#endif
    template<typename vec, typename VT, int64_t mask>
    typename std::enable_if_t<(mask < 0 || mask> 255), void>
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    test_blend(VT expected_val[vec::size()], VT a[vec::size()], VT b[vec::size()])
    {
    }
    template<typename vec, typename VT, int64_t mask>
    typename std::enable_if_t<(mask >= 0 && mask <= 255), void>
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    test_blend(VT expected_val[vec::size()], VT a[vec::size()], VT b[vec::size()]) {
        // generate expected_val
        int64_t m = mask;
        for (int64_t i = 0; i < vec::size(); i++) {
            expected_val[i] = (m & 0x01) ? b[i] : a[i];
            m = m >> 1;
        }
        // test with blend
        auto vec_a = vec::loadu(a);
        auto vec_b = vec::loadu(b);
        auto expected = vec::loadu(expected_val);
        auto actual = vec::template blend<mask>(vec_a, vec_b);
        auto mask_str = std::string("\nblend mask: ") + std::to_string(mask);
        if (AssertVectorized<vec>(std::string(NAME_INFO(test_blend)) + mask_str, expected, actual).check()) return;
        test_blend<vec, VT, mask - 1>(expected_val, a, b);
    }
    template<typename vec, typename VT, int64_t idx, int64_t N>
    std::enable_if_t<(!is_complex<VT>::value && idx == N), bool>
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    test_blendv(VT expected_val[vec::size()], VT a[vec::size()], VT b[vec::size()], VT mask[vec::size()]) {
        using bit_rep = BitType<VT>;
        // generate expected_val
        for (int64_t i = 0; i < vec::size(); i++) {
            bit_rep hex_mask = 0;
            hex_mask=c10::bit_cast<bit_rep>(mask[i]);
            expected_val[i] = (hex_mask & 0x01) ? b[i] : a[i];
        }
        // test with blendv
        auto vec_a = vec::loadu(a);
        auto vec_b = vec::loadu(b);
        auto vec_m = vec::loadu(mask);
        auto expected = vec::loadu(expected_val);
        auto actual = vec::blendv(vec_a, vec_b, vec_m);
        auto mask_str = std::string("\nblendv mask: ");
        for (int64_t i = 0; i < vec::size(); i++) {
            mask_str += std::to_string(mask[i]) + " ";
        }
        if (AssertVectorized<vec>(std::string(NAME_INFO(test_blendv)) + mask_str, expected, actual).check()) {
            return false;
        }
        return true;
    }
    template<typename vec, typename VT, int64_t idx, int64_t N>
    std::enable_if_t<(!is_complex<VT>::value && idx != N), bool>
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    test_blendv(VT expected_val[vec::size()], VT a[vec::size()], VT b[vec::size()], VT mask[vec::size()]) {
        // shuffle mask and do blendv test
        VT m = mask[idx];
        if (!test_blendv<vec, VT, idx+1, N>(expected_val, a, b, mask)) return false;
        if (m != (VT)0) {
          mask[idx] = (VT)0;
        }
        else {
          int64_t hex_mask = 0xFFFFFFFFFFFFFFFF;
          std::memcpy(&mask[idx], &hex_mask, sizeof(VT));
        }
        if (!test_blendv<vec, VT, idx+1, N>(expected_val, a, b, mask)) return false;
        mask[idx] = m;
        return true;
    }
    template<typename T, int N>
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    void blend_init(T(&a)[N], T(&b)[N]) {
        a[0] = (T)1.0;
        b[0] = a[0] + (T)N;
        for (const auto i : c10::irange(1, N)) {
            a[i] = a[i - 1] + (T)(1.0);
            b[i] = b[i - 1] + (T)(1.0);
        }
    }
    TYPED_TEST(BitwiseFloatsAdditional, Blendv) {
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN VT a[vec::size()];
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN VT b[vec::size()];
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN VT mask[vec::size()] = {0};
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN VT expected_val[vec::size()];
        blend_init(a, b);
        test_blendv<vec, VT, 0, vec::size()>(expected_val, a, b, mask);
    }
// NOTE: In this test, blend<mask> is not required to implement SVE Vectorized::set.
// so, this test is disabled for SVE.
#if !defined(CPU_CAPABILITY_SVE)
    TYPED_TEST(BitwiseFloatsAdditional2, Blend) {
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN VT a[vec::size()];
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN VT b[vec::size()];
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN VT expected_val[vec::size()];
        blend_init(a, b);
        constexpr int64_t power_sets = 1LL << (vec::size());
        test_blend<vec, VT, power_sets - 1>(expected_val, a, b);
    }
#endif
    template<typename vec, typename VT>
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    void test_set(VT expected_val[vec::size()], VT a[vec::size()], VT b[vec::size()], int64_t count){
        if (count < 0) return;
        //generate expected_val
        for (int64_t i = 0; i < vec::size(); i++) {
            expected_val[i] = (i < count) ? b[i] : a[i];
        }
        // test with set
        auto vec_a = vec::loadu(a);
        auto vec_b = vec::loadu(b);
        auto expected = vec::loadu(expected_val);
        auto actual = vec::set(vec_a, vec_b, count);

        auto count_str = std::string("\ncount: ") + std::to_string(count);
        if (AssertVectorized<vec>(std::string(NAME_INFO(test_set)) + count_str, expected, actual).check()) {
          return;
        }
        test_set<vec, VT>(expected_val, a, b, (count == 0 ? -1 : count / 2));
    }
    TYPED_TEST(BitwiseFloatsAdditional2, Set) {
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN VT a[vec::size()];
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN VT b[vec::size()];
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN VT expected_val[vec::size()];
        blend_init(a, b);
        test_set<vec, VT>(expected_val, a, b, vec::size());
    }
    template<typename T>
    std::enable_if_t<!is_complex<T>::value, void>
    arange_init(T& base, T& step) {
        base = (T)5.0;
        step = (T)2.0;
    }
    template<typename T>
    std::enable_if_t<is_complex<T>::value, void>
    arange_init(T& base, T& step) {
       base = T(5.0, 5.0);
       step = T(2.0, 3.0);
    }
    TYPED_TEST(RangeFactories, Arange) {
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        using UVT = UvalueType<TypeParam>;
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN VT expected_val[vec::size()];
        VT base, step;
        arange_init(base, step);
        for (int64_t i = 0; i < vec::size(); i++) {
            expected_val[i] = base + VT((UVT)i) * step;
        }
        auto expected = vec::loadu(expected_val);
        auto actual = vec::arange(base, step);
        AssertVectorized<vec>(NAME_INFO(test_arange), expected, actual).check();
    }
    TEST(ComplexTests, TestComplexFloatImagRealConj) {
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers,cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        float aa[] = { 1.5488e-28,2.5488e-28,3.5488e-28,4.5488e-28,5.5488e-28,6.5488e-28,7.5488e-28,8.5488e-28,
                       9.5488e-28,10.5488e-28,11.5488e-28,12.5488e-28,13.5488e-28,14.5488e-28,15.5488e-28,16.5488e-28};
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers,cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        float exp[] = { aa[0],0,aa[2],0,aa[4],0,aa[6],0,aa[8],0,aa[10],0,aa[12],0,aa[14],0 };
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers,cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        float exp3[] = { aa[1],0,aa[3],0,aa[5],0,aa[7],0,aa[9],0,aa[11],0,aa[13],0,aa[15],0 };
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers,cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        float exp4[] = { 1.5488e-28, -2.5488e-28,3.5488e-28,-4.5488e-28,
                         5.5488e-28,-6.5488e-28,7.5488e-28,-8.5488e-28,
                         9.5488e-28,-10.5488e-28,11.5488e-28,-12.5488e-28,
                         13.5488e-28,-14.5488e-28,15.5488e-28,-16.5488e-28 };
        auto a = vcomplex::loadu(aa);
        auto actual1 = a.real();
        auto actual3 = a.imag();
        auto actual4 = a.conj();
        auto expected1 = vcomplex::loadu(exp);
        auto expected3 = vcomplex::loadu(exp3);
        auto expected4 = vcomplex::loadu(exp4);
        AssertVectorized<vcomplex>(NAME_INFO(complex real), expected1, actual1).check();
        AssertVectorized<vcomplex>(NAME_INFO(complex imag), expected3, actual3).check();
        AssertVectorized<vcomplex>(NAME_INFO(complex conj), expected4, actual4).check();
    }
    TEST(ComplexTests, TestComplexConstructor) {
        auto actual1 = vcomplex(1.0);
        auto expected1 = vcomplex(Complex<float>(1.0));
        AssertVectorized<vcomplex>(NAME_INFO(complex constructor), expected1, actual1).check();
    }
    TYPED_TEST(QuantizationTests, Quantize) {
        using vec = TypeParam;
        using underlying = ValueType<vec>;
        constexpr int trials = 4000;
        // NOLINTNEXTLINE(bugprone-signed-char-misuse)
        constexpr int min_val = std::numeric_limits<underlying>::min();
        constexpr int max_val = std::numeric_limits<underlying>::max();
        constexpr int el_count = vfloat::size();
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN float unit_float_vec[el_count];
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN underlying expected_qint_vals[vec::size()];
        typename vec::float_vec_return_type  float_ret;
        auto seed = TestSeed();
        //zero point
        ValueGen<int> generator_zp(min_val, max_val, seed);
        //scale
        ValueGen<float> generator_sc(1.f, 15.f, seed.add(1));
        //value
        float minv = static_cast<float>(static_cast<double>(min_val) * 2.0);
        float maxv = static_cast<float>(static_cast<double>(max_val) * 2.0);
        ValueGen<float> gen(minv, maxv, seed.add(2));
        for ([[maybe_unused]] const auto i : c10::irange(trials)) {
          float scale = generator_sc.get();
          float inv_scale = 1.0f / static_cast<float>(scale);
          auto zero_point_val = generator_zp.get();
          int index = 0;
          for (int j = 0; j < vec::float_num_vecs(); j++) {
            // generate vals
            for (auto& v : unit_float_vec) {
              v = gen.get();
              expected_qint_vals[index] =
                  quantize_val<underlying>(scale, zero_point_val, v);
              index++;
            }
            float_ret[j] = vfloat::loadu(unit_float_vec);
          }
          auto expected = vec::loadu(expected_qint_vals);
          auto actual =
              vec::quantize(float_ret, scale, zero_point_val, inv_scale);
          if (AssertVectorized<vec>(NAME_INFO(Quantize), expected, actual)
                  .check())
            return;
        } // trials;
    }
#if (defined(CPU_CAPABILITY_AVX2) ||  defined(CPU_CAPABILITY_AVX512))  && !defined(_MSC_VER)
    // This test case aims to test at::vec::QuantizeAvx512 and
    // at::vec::QuantizeAVX2 which do not support CPU_CAPABILITY_DEFAULT case
    TYPED_TEST(Quantization8BitWithTailTests, QuantizeTile) {
      using vec = TypeParam;
      using underlying = ValueType<vec>;
      constexpr int trials = 4000;
      // NOLINTNEXTLINE(bugprone-signed-char-misuse)
      constexpr int min_val = std::numeric_limits<underlying>::min();
      constexpr int max_val = std::numeric_limits<underlying>::max();
      constexpr int el_count = vfloat::size();
      // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
      CACHE_ALIGN float unit_float_vec[el_count];
      // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
      CACHE_ALIGN underlying expected_qint_vals[vec::size()];
      CACHE_ALIGN underlying actual_qint_vals[vec::size()];
      constexpr int tile_size = vec::size() - 1;
      typename vec::float_vec_return_type float_ret;
      auto seed = TestSeed();
      // zero point
      ValueGen<int> generator_zp(min_val, max_val, seed);
      // scale
      ValueGen<float> generator_sc(1.f, 15.f, seed.add(1));
      // value
      float minv = static_cast<float>(static_cast<double>(min_val) * 2.0);
      float maxv = static_cast<float>(static_cast<double>(max_val) * 2.0);
      ValueGen<float> gen(minv, maxv, seed.add(2));
      for ([[maybe_unused]] const auto i : c10::irange(trials)) {
        float scale = generator_sc.get();
        float inv_scale = 1.0f / static_cast<float>(scale);
        auto zero_point_val = generator_zp.get();
        int index = 0;
        for (int j = 0; j < vec::float_num_vecs(); j++) {
          // generate vals
          for (auto& v : unit_float_vec) {
            v = gen.get();
            expected_qint_vals[index] =
                quantize_val<underlying>(scale, zero_point_val, v);
            index++;
          }
          float_ret[j] = vfloat::loadu(unit_float_vec);
        }
#if defined(CPU_CAPABILITY_AVX512)
        at::vec::QuantizeAvx512(
            (float*)float_ret.data(),
            actual_qint_vals,
            tile_size,
            inv_scale,
            zero_point_val);
#endif
#if defined(CPU_CAPABILITY_AVX2)
        at::vec::QuantizeAvx2(
            (float*)float_ret.data(),
            actual_qint_vals,
            tile_size,
            inv_scale,
            zero_point_val);
#endif
        expected_qint_vals[tile_size] = 0;
        actual_qint_vals[tile_size] = 0;
        auto expected = vec::loadu(expected_qint_vals);
        auto actual = vec::loadu(actual_qint_vals);
        if (AssertVectorized<vec>(NAME_INFO(QuantizeTile), expected, actual)
                .check())
          return;
      } // trials;
    }
#endif
    TYPED_TEST(QuantizationTests, DeQuantize) {
        using vec = TypeParam;
        using underlying = ValueType<vec>;
        constexpr bool is_large = sizeof(underlying) > 1;
        constexpr int trials = is_large ? 4000 : std::numeric_limits<underlying>::max() / 2;
        constexpr int min_val = is_large ? -2190 : std::numeric_limits<underlying>::min();
        constexpr int max_val = is_large ? 2199 : std::numeric_limits<underlying>::max();
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN float unit_exp_vals[vfloat::size()];
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN underlying qint_vals[vec::size()];
#if  defined(CHECK_DEQUANT_WITH_LOW_PRECISION)
        std::cout << "Dequant will be tested with relative error " << 1.e-3f << std::endl;
#endif
        auto seed = TestSeed();
        ValueGen<int> generator(min_val, max_val, seed.add(1));
        //scale
        ValueGen<float> generator_sc(1.f, 15.f, seed.add(2));
        for ([[maybe_unused]] const auto i : c10::irange(trials)) {
          float scale = generator_sc.get();
          int32_t zero_point_val = generator.get();
          float scale_zp_premul = -(scale * zero_point_val);
          vfloat vf_scale = vfloat{scale};
          vfloat vf_zp = vfloat{static_cast<float>(zero_point_val)};
          vfloat vf_scale_zp = vfloat{scale_zp_premul};
          // generate vals
          for (auto& x : qint_vals) {
            x = generator.get();
          }
          // get expected
          int index = 0;
          auto qint_vec = vec::loadu(qint_vals);
          auto actual_float_ret =
              qint_vec.dequantize(vf_scale, vf_zp, vf_scale_zp);
          for (int j = 0; j < vec::float_num_vecs(); j++) {
            for (auto& v : unit_exp_vals) {
              v = dequantize_val(scale, zero_point_val, qint_vals[index]);
              index++;
            }
            vfloat expected = vfloat::loadu(unit_exp_vals);
            const auto& actual = actual_float_ret[j];
#if  defined(CHECK_DEQUANT_WITH_LOW_PRECISION)
                if (AssertVectorized<vfloat>(NAME_INFO(DeQuantize), seed, expected, actual).check(false, true, 1.e-3f)) return;
#else
                if (AssertVectorized<vfloat>(NAME_INFO(DeQuantize), seed, expected, actual).check()) return;
#endif
            }
        } // trials;
    }
    TYPED_TEST(QuantizationTests, ReQuantizeFromInt) {
        using vec = TypeParam;
        using underlying = ValueType<vec>;
        constexpr int trials = 4000;
        constexpr int min_val = -65535;
        constexpr int max_val = 65535;
        constexpr int el_count = vint::size();
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN c10::qint32 unit_int_vec[el_count];
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN underlying expected_qint_vals[vec::size()];
        typename vec::int_vec_return_type  int_ret;
        auto seed = TestSeed();
        //zero point and value
        ValueGen<int32_t> generator(min_val, max_val, seed);
        //scale
        ValueGen<float> generator_sc(1.f, 15.f, seed.add(1));
        for ([[maybe_unused]] const auto i : c10::irange(trials)) {
          float multiplier = 1.f / (generator_sc.get());
          auto zero_point_val = generator.get();
          int index = 0;
          for (int j = 0; j < vec::float_num_vecs(); j++) {
            // generate vals
            for (auto& v : unit_int_vec) {
              v = c10::qint32(generator.get());
              expected_qint_vals[index] = requantize_from_int<underlying>(
                  multiplier, zero_point_val, v.val_);
              index++;
            }
            int_ret[j] = vqint::loadu(unit_int_vec);
          }
          auto expected = vec::loadu(expected_qint_vals);
          auto actual =
              vec::requantize_from_int(int_ret, multiplier, zero_point_val);
          if (AssertVectorized<vec>(
                  NAME_INFO(ReQuantizeFromInt), seed, expected, actual)
                  .check()) {
            return;
          }
        } // trials;
    }
    TYPED_TEST(QuantizationTests, WideningSubtract) {
        using vec = TypeParam;
        using underlying = ValueType<vec>;
        constexpr bool is_large = sizeof(underlying) > 1;
        constexpr int trials = is_large ? 4000 : std::numeric_limits<underlying>::max() / 2;
        // NOLINTNEXTLINE(bugprone-signed-char-misuse)
        constexpr int min_val = std::numeric_limits<underlying>::min();
        constexpr int max_val = std::numeric_limits<underlying>::max();
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN int32_t unit_exp_vals[vfloat::size()];
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN underlying qint_vals[vec::size()];
        // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
        CACHE_ALIGN underlying qint_b[vec::size()];
        typename vec::int_vec_return_type  expected_int_ret;
        auto seed = TestSeed();
        ValueGen<underlying> generator(min_val, max_val, seed);
        for ([[maybe_unused]] const auto i : c10::irange(trials)) {
          // generate vals
          for (int j = 0; j < vec::size(); j++) {
            qint_vals[j] = generator.get();
            qint_b[j] = generator.get();
            if constexpr (std::is_same_v<underlying, int>) {
              // filter overflow cases
              filter_sub_overflow(qint_vals[j], qint_b[j]);
            }
          }
          int index = 0;
          auto qint_vec = vec::loadu(qint_vals);
          auto qint_vec_b = vec::loadu(qint_b);
          auto actual_int_ret = qint_vec.widening_subtract(qint_vec_b);
          for (int j = 0; j < vec::float_num_vecs(); j++) {
            for (auto& v : unit_exp_vals) {
              v = widening_subtract(qint_vals[index], qint_b[index]);
              index++;
            }
            auto expected = vqint::loadu(unit_exp_vals);
            const auto& actual = actual_int_ret[j];
            if (AssertVectorized<vqint>(
                    NAME_INFO(WideningSubtract), seed, expected, actual)
                    .check())
              return;
          }
        } // trials;
    }
    TYPED_TEST(QuantizationTests, Relu) {
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        constexpr VT min_val = std::numeric_limits<VT>::min();
        constexpr VT max_val = std::numeric_limits<VT>::max();
        constexpr VT fake_zp = sizeof(VT) > 1 ? static_cast<VT>(65535) : static_cast<VT>(47);
        auto test_case = TestingCase<vec>::getBuilder()
            .addDomain(CheckWithinDomains<VT>{ { DomainRange<VT>{min_val, max_val}, DomainRange<VT>{(VT)0, (VT)fake_zp}} })
            .setTestSeed(TestSeed());
        test_binary<vec>(
            NAME_INFO(relu),
            RESOLVE_OVERLOAD(relu),
            [](const vec& v0, const vec& v1) {
                return v0.relu(v1);
            },
            test_case);
    }
    TYPED_TEST(QuantizationTests, Relu6) {
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        constexpr VT min_val = std::numeric_limits<VT>::min();
        constexpr VT max_val = std::numeric_limits<VT>::max();
        constexpr VT fake_zp = sizeof(VT) > 1 ? static_cast<VT>(65535) : static_cast<VT>(47);
        constexpr VT temp = sizeof(VT) > 1 ? static_cast<VT>(12345) : static_cast<VT>(32);
        constexpr VT fake_qsix = fake_zp + temp;
        auto test_case = TestingCase<vec>::getBuilder()
            .addDomain(CheckWithinDomains<VT>{
                {
                    DomainRange<VT>{min_val, max_val},
                        DomainRange<VT>{(VT)0, (VT)fake_zp},
                        DomainRange<VT>{(VT)fake_zp, (VT)fake_qsix}
                }})
            .setTestSeed(TestSeed());
        test_ternary<vec>(
            NAME_INFO(relu6),
            RESOLVE_OVERLOAD(relu6),
            [](/*const*/ vec& v0, const vec& v1, const vec& v2) {
                return  v0.relu6(v1, v2);
            },
            test_case);
    }
    TYPED_TEST(FunctionalTests, Map) {
        using vec = TypeParam;
        using VT = ValueType<TypeParam>;
        constexpr auto R = 2LL; // residual
        constexpr auto N = vec::size() + R;
        CACHE_ALIGN VT x1[N];
        CACHE_ALIGN VT x2[N];
        CACHE_ALIGN VT x3[N];
        CACHE_ALIGN VT x4[N];
        CACHE_ALIGN VT y[N];
        CACHE_ALIGN VT ref_y[N];
        auto seed = TestSeed();
        ValueGen<VT> generator(VT(-100), VT(100), seed);
        for (const auto i : c10::irange(N)) {
          x1[i] = generator.get();
          x2[i] = generator.get();
          x3[i] = generator.get();
          x4[i] = generator.get();
        }
        auto cmp = [&](VT* y, VT* ref_y) {
          AssertVectorized<vec>(NAME_INFO(Map), vec::loadu(y), vec::loadu(ref_y)).check(true);
          AssertVectorized<vec>(NAME_INFO(Map), vec::loadu(y + vec::size(), R), vec::loadu(ref_y + vec::size(), R)).check(true);
        };
        // test map: y = x1
        at::vec::map<VT>([](vec x) { return x; }, y, x1, N);
        for (const auto i : c10::irange(N)) { ref_y[i] = x1[i]; }
        cmp(y, ref_y);
        // test map2: y = x1 + x2
        at::vec::map2<VT>([](vec x1, vec x2) { return x1 + x2; }, y, x1, x2, N);
        for (const auto i : c10::irange(N)) { ref_y[i] = x1[i] + x2[i]; }
        cmp(y, ref_y);
        // test map3: y = x1 + x2 + x3
        at::vec::map3<VT>([](vec x1, vec x2, vec x3) { return x1 + x2 + x3; }, y, x1, x2, x3, N);
        for (const auto i : c10::irange(N)) { ref_y[i] = x1[i] + x2[i] + x3[i]; }
        cmp(y, ref_y);
        // test map4: y = x1 + x2 + x3 + x4
        at::vec::map4<VT>([](vec x1, vec x2, vec x3, vec x4) { return x1 + x2 + x3 + x4; }, y, x1, x2, x3, x4, N);
        for (const auto i : c10::irange(N)) { ref_y[i] = x1[i] + x2[i] + x3[i] + x4[i]; }
        cmp(y, ref_y);
    }
      TYPED_TEST(FunctionalTestsReducedFloat, Reduce) {
      using vec = TypeParam;
      // Can't use ValueType<TypeParam> here:
      // Vectorized<BFloat16>::value_type returns uint16_t on AVX2/AVX512
      using VT = UholdType<TypeParam>;
      using RT = float; // reference
      constexpr auto R = 2LL; // residual
      constexpr auto N = vec::size() * 2 + R;
      CACHE_ALIGN RT x_f1[N];
      CACHE_ALIGN RT x_f2[N];
      CACHE_ALIGN RT x_f3[N];
      CACHE_ALIGN VT x_b1[N];
      CACHE_ALIGN VT x_b2[N];
      CACHE_ALIGN VT x_b3[N];
      auto seed = TestSeed();
      ValueGen<RT> generator(RT(-1), RT(1), seed);
      for (const auto i : c10::irange(N)) {
        x_f1[i] = generator.get();
        x_f2[i] = generator.get();
        x_f3[i] = generator.get();
        x_b1[i] = VT(x_f1[i]);
        x_b2[i] = VT(x_f2[i]);
        x_b3[i] = VT(x_f3[i]);
      }
      float atol = 0.01f;
      float rtol = 0.01f;
      auto cmp = [=](RT ref, VT val) { return std::abs(ref - val) <= atol + rtol * std::abs(val); };
      auto sum = [](auto& x, auto& y) { return x + y; };
      auto max = [](auto& x, auto& y) { return at::vec::maximum(x, y); };
      // ReduceAll
      for (int64_t len = 1; len <= N; len++) {
        auto y1 = at::vec::reduce_all<RT>(sum, x_f1, len);
        auto y2 = at::vec::reduce_all<VT>(sum, x_b1, len);
        ASSERT_TRUE(cmp(y1, y2)) << "Failure Details:\nTest Seed to reproduce: " << seed
            << "\nreduce_all, Length: " << len << "; fp32: " << y1 << "; bf16: " << RT(y2);
      }
      // Reduce2All
      for (int64_t len = 1; len <= N; len++) {
        auto y1 = at::vec::reduce2_all<RT>(sum, max, x_f1, len);
        auto y2 = at::vec::reduce2_all<VT>(sum, max, x_b1, len);
        ASSERT_TRUE(cmp(y1.first, y2.first) && cmp(y1.second, y2.second)) << "Failure Details:\nTest Seed to reproduce: " << seed
            << "\nreduce2_all, Length: " << len << "; fp32(fun1): " << y1.first << "; bf16(fun1): " << RT(y2.first)
            << "; fp32(fun2): " << y1.second << "; bf16(fun2): " << y2.second;
      }
      // MapReduceAll
      for (int64_t len = 1; len <= N; len++) {
        auto y1 = at::vec::map_reduce_all<RT>([](auto x) { return x - x.exp(); }, sum, x_f1, len);
        auto y2 = at::vec::map_reduce_all<VT>([](auto x) { return x - x.exp(); }, sum, x_b1, len);
        ASSERT_TRUE(cmp(y1, y2)) << "Failure Details:\nTest Seed to reproduce: " << seed
            << "\nmap_reduce_all, Length: " << len << "; fp32: " << y1 << "; bf16: " << RT(y2);
      }
      // Map2ReduceAll
      for (int64_t len = 1; len <= N; len++) {
        auto y1 = at::vec::map2_reduce_all<RT>([](auto x, auto y) { return x * y; }, sum, x_f1, x_f2, len);
        auto y2 = at::vec::map2_reduce_all<VT>([](auto x, auto y) { return x * y; }, sum, x_b1, x_b2, len);
        ASSERT_TRUE(cmp(y1, y2)) << "Failure Details:\nTest Seed to reproduce: " << seed
            << "\nmap2_reduce_all, Length: " << len << "; fp32: " << y1 << "; bf16: " << RT(y2);
      }
      // Map3ReduceAll
      for (int64_t len = 1; len <= N; len++) {
        auto y1 = at::vec::map3_reduce_all<RT>([](auto x, auto y, auto z) { return x * y + z; }, sum, x_f1, x_f2, x_f3, len);
        auto y2 = at::vec::map3_reduce_all<VT>([](auto x, auto y, auto z) { return x * y + z; }, sum, x_b1, x_b2, x_b3, len);
        ASSERT_TRUE(cmp(y1, y2)) << "Failure Details:\nTest Seed to reproduce: " << seed
            << "\nmap3_reduce_all, Length: " << len << "; fp32: " << y1 << "; bf16: " << RT(y2);
      }
    }
    TYPED_TEST(FunctionalTestsReducedFloat, Map) {
      using vec = TypeParam;
      using VT = UholdType<TypeParam>;
      using RT = float; // reference
      constexpr auto R = 2LL; // residual
      constexpr auto N = vec::size() * 2 + R;
      CACHE_ALIGN RT x_f1[N];
      CACHE_ALIGN RT x_f2[N];
      CACHE_ALIGN RT x_f3[N];
      CACHE_ALIGN RT x_f4[N];
      CACHE_ALIGN VT x_b1[N];
      CACHE_ALIGN VT x_b2[N];
      CACHE_ALIGN VT x_b3[N];
      CACHE_ALIGN VT x_b4[N];
      CACHE_ALIGN RT y_f[N];
      CACHE_ALIGN VT y_b[N];
      auto seed = TestSeed();
      ValueGen<RT> generator(RT(-1), RT(1), seed);
      for (const auto i : c10::irange(N)) {
        x_f1[i] = generator.get();
        x_f2[i] = generator.get();
        x_f3[i] = generator.get();
        x_f4[i] = generator.get();
        x_b1[i] = VT(x_f1[i]);
        x_b2[i] = VT(x_f2[i]);
        x_b3[i] = VT(x_f3[i]);
        x_b4[i] = VT(x_f4[i]);
      }
      float atol = 0.01f;
      float rtol = 0.01f;
      auto cmp = [=](RT ref, VT val) { return std::abs(ref - val) <= atol + rtol * std::abs(val); };
      // Map
      for (int64_t len = 1; len <= N; len++) {
        at::vec::map<RT>([](auto x) { return x; }, y_f, x_f1, len);
        at::vec::map<VT>([](auto x) { return x; }, y_b, x_b1, len);
        for (const auto i : c10::irange(len)) {
          ASSERT_TRUE(cmp(y_f[i], y_b[i])) << "Failure Details:\nTest Seed to reproduce: " << seed
              << "\nmap, Length: " << len << "; index: " << i << "; fp32 reference: " << y_f[i] << "; bf16 value: " << RT(y_b[i]);
        }
      }
      // Map - For float32 in, reduced floating points out
      for (int64_t len = 1; len <= N; len++) {
        at::vec::map<RT>([](auto x) { return x; }, y_f, x_f1, len);
        at::vec::map<VT>([](auto x) { return x; }, y_b, x_f1, len);
        for (const auto i : c10::irange(len)) {
          ASSERT_TRUE(cmp(y_f[i], y_b[i])) << "Failure Details:\nTest Seed to reproduce: " << seed
              << "\nmap, Length: " << len << "; index: " << i << "; fp32 reference: " << y_f[i] << "; bf16 value: " << RT(y_b[i]);
        }
      }
      // Map2
      for (int64_t len = 1; len <= N; len++) {
        at::vec::map2<RT>([](auto x, auto y) { return x + y; }, y_f, x_f1, x_f2, len);
        at::vec::map2<VT>([](auto x, auto y) { return x + y; }, y_b, x_b1, x_b2, len);
        for (const auto i : c10::irange(len)) {
          ASSERT_TRUE(cmp(y_f[i], y_b[i])) << "Failure Details:\nTest Seed to reproduce: " << seed
              << "\nmap2, Length: " << len << "; index: " << i << "; fp32 reference: " << y_f[i] << "; bf16 value: " << RT(y_b[i]);
        }
      }
      // Map3
      for (int64_t len = 1; len <= N; len++) {
        at::vec::map3<RT>([](auto x, auto y, auto z) { return x + y * z; }, y_f, x_f1, x_f2, x_f3, len);
        at::vec::map3<VT>([](auto x, auto y, auto z) { return x + y * z; }, y_b, x_b1, x_b2, x_b3, len);
        for (const auto i : c10::irange(len)) {
          ASSERT_TRUE(cmp(y_f[i], y_b[i])) << "Failure Details:\nTest Seed to reproduce: " << seed
              << "\nmap3, Length: " << len << "; index: " << i << "; fp32 reference: " << y_f[i] << "; bf16 value: " << RT(y_b[i]);
        }
      }
      // Map4
      for (int64_t len = 1; len <= N; len++) {
         at::vec::map4<RT>([](auto x, auto y, auto z, auto w) { return x + y * z - w; }, y_f, x_f1, x_f2, x_f3, x_f4, len);
         at::vec::map4<VT>([](auto x, auto y, auto z, auto w) { return x + y * z - w; }, y_b, x_b1, x_b2, x_b3, x_b4, len);
         for (const auto i : c10::irange(len)) {
           ASSERT_TRUE(cmp(y_f[i], y_b[i])) << "Failure Details:\nTest Seed to reproduce: " << seed
               << "\nmap4, Length: " << len << "; index: " << i << "; fp32 reference: " << y_f[i] << "; bf16 value: " << RT(y_b[i]);
         }
      }
    }
    TEST(HalfConversionTest, HalfFloat) {
      float f32s[100];
      for (const auto i : c10::irange(100)) {
        f32s[i] = i + 0.3;
      }
      uint16_t u16;
      float x;
      for (const auto i : c10::irange(100)) {
      #if (defined(CPU_CAPABILITY_AVX2) || defined(CPU_CAPABILITY_AVX512)) && \
          !defined(__APPLE__)
        u16 = at::vec::float2half_scalar(f32s[i]);
        x = at::vec::half2float_scalar(u16);
      #else
        u16 = c10::detail::fp16_ieee_from_fp32_value(f32s[i]);
        x = c10::detail::fp16_ieee_to_fp32_value(u16);
      #endif

        EXPECT_EQ(u16, c10::detail::fp16_ieee_from_fp32_value(f32s[i]))
            << "Test failed for float to uint16 " << f32s[i] << "\n";
        EXPECT_EQ(x, c10::detail::fp16_ieee_to_fp32_value(u16))
            << "Test failed for uint16 to float " << u16 << "\n";
      }
    }
    TYPED_TEST(InfiniteTests, HasInfNan) {
      using vec = TypeParam;
      using VT = UholdType<TypeParam>;
      auto vec_size = vec::size();
      VT values[20];
      for (const auto i : c10::irange(20)) {
        values[i] = i + 0.3;
      }
      auto vec_val = vec::loadu(values);
      auto seed = TestSeed();
      ValueGen<int> generator(int(0), int(vec_size - 1), seed);
      int index = generator.get();
      int nanBits = 0x7FC00000;
      VT v_nan = static_cast<VT>(*(float *)&nanBits);
      values[index] = v_nan;
      auto vec_nan = vec::loadu(values);
      int infBits = 0x7F800000;
      VT v_pinf = static_cast<VT>(*(float *)&infBits);
      values[index] = v_pinf;
      auto vec_pinf = vec::loadu(values);
      int negInfBits = 0xFF800000;
      VT v_ninf  = static_cast<VT>(*(float *)&negInfBits);
      values[index] = v_ninf;
      auto vec_ninf = vec::loadu(values);

      ASSERT_TRUE(!(vec_val.has_inf_nan())) << "Test failed for normal value\n";
      ASSERT_TRUE(vec_nan.has_inf_nan()) << "Test failed for NAN\n";
      ASSERT_TRUE(vec_pinf.has_inf_nan()) << "Test failed for positive Infinity\n";
      ASSERT_TRUE(vec_ninf.has_inf_nan()) << "Test failed for negative Infinity\n";
    }
#if !defined(CPU_CAPABILITY_SVE)
    TYPED_TEST(VecConvertTests, Convert) {
      using vec = TypeParam;
      using src_t = ValueType<TypeParam>;
      constexpr auto N = vec::size();
    #define TEST_CONVERT_TO(dst_t)                                     \
      do {                                                             \
        CACHE_ALIGN src_t x[N];                                        \
        CACHE_ALIGN dst_t y[N];                                        \
        CACHE_ALIGN dst_t ref[N];                                      \
        auto seed = TestSeed();                                        \
        auto low = std::is_signed_v<dst_t> ? src_t(-100) : 0;          \
        ValueGen<src_t> generator(low, src_t(100), seed);              \
        for (const auto i : c10::irange(N)) {                          \
          x[i] = generator.get();                                      \
        }                                                              \
        for (const auto i : c10::irange(N)) {                          \
          ref[i] = static_cast<dst_t>(x[i]);                           \
        }                                                              \
        auto x_vec = vec::loadu(x);                                    \
        auto y_vec = at::vec::convert<dst_t>(x_vec);                   \
        constexpr int num_dst_elements =                               \
            std::min(N, at::vec::Vectorized<dst_t>::size());           \
        y_vec.store(y, num_dst_elements);                              \
        for (const auto i : c10::irange(num_dst_elements)) {           \
          ASSERT_EQ(y[i], ref[i])                                      \
              << "Failure Details:\nTest Seed to reproduce: " << seed  \
              << " x[" << i << "]=" << x[i] << " dst_t=" #dst_t;       \
        }                                                              \
        constexpr int dst_n = N / num_dst_elements;                    \
        auto y_vec_n = at::vec::convert<dst_t, dst_n, src_t, 1>(       \
            at::vec::VectorizedN<src_t, 1>(x_vec));                    \
        y_vec_n.store(y, N);                                           \
        for (const auto i : c10::irange(N)) {                          \
          ASSERT_EQ(y[i], ref[i])                                      \
              << "Failure Details:\nTest Seed to reproduce: " << seed  \
              << " x[" << i << "]=" << x[i] << " dst_t=" #dst_t;       \
        }                                                              \
      } while (0)
      TEST_CONVERT_TO(int8_t);
      TEST_CONVERT_TO(uint8_t);
      TEST_CONVERT_TO(int16_t);
      TEST_CONVERT_TO(uint16_t);
      TEST_CONVERT_TO(int32_t);
      TEST_CONVERT_TO(uint32_t);
      TEST_CONVERT_TO(int64_t);
      TEST_CONVERT_TO(uint64_t);
      TEST_CONVERT_TO(c10::BFloat16);
      TEST_CONVERT_TO(c10::Half);
      TEST_CONVERT_TO(float);
      TEST_CONVERT_TO(double);
    #undef TEST_CONVERT_TO
    }
    TYPED_TEST(VecConvertTestsReducedFloat, ConvertReduced) {
      using vec = TypeParam;
      using src_t = UholdType<TypeParam>;
      constexpr auto N = vec::size();
    #define TEST_CONVERT_TO(dst_t)                                     \
      do {                                                             \
        CACHE_ALIGN src_t x[N];                                        \
        CACHE_ALIGN dst_t y[N];                                        \
        CACHE_ALIGN dst_t ref[N];                                      \
        auto seed = TestSeed();                                        \
        auto low = std::is_signed_v<dst_t> ? src_t(-100.0) : src_t(0); \
        ValueGen<src_t> generator(low, src_t(100), seed);              \
        for (const auto i : c10::irange(N)) {                          \
          x[i] =  generator.get();                                     \
        }                                                              \
        for (const auto i : c10::irange(N)) {                          \
          ref[i] = static_cast<dst_t>(x[i]);                           \
        }                                                              \
        auto x_vec = vec::loadu(x);                                    \
        auto y_vec = at::vec::convert<dst_t>(x_vec);                   \
        constexpr int num_dst_elements =                               \
            std::min(N, at::vec::Vectorized<dst_t>::size());           \
        y_vec.store(y, num_dst_elements);                              \
        for (const auto i : c10::irange(num_dst_elements)) {           \
          ASSERT_EQ(y[i], ref[i])                                      \
              << "Failure Details:\nTest Seed to reproduce: " << seed  \
              << " x[" << i << "]=" << x[i] << " dst_t=" #dst_t;       \
        }                                                              \
        constexpr int dst_n = N / num_dst_elements;                    \
        auto y_vec_n = at::vec::convert<dst_t, dst_n, src_t, 1>(       \
            at::vec::VectorizedN<src_t, 1>(x_vec));                    \
        y_vec_n.store(y, N);                                           \
        for (const auto i : c10::irange(N)) {                          \
          ASSERT_EQ(y[i], ref[i])                                      \
              << "Failure Details:\nTest Seed to reproduce: " << seed  \
              << " x[" << i << "]=" << x[i] << " dst_t=" #dst_t;       \
        }                                                              \
      } while (0)
      TEST_CONVERT_TO(int8_t);
      TEST_CONVERT_TO(uint8_t);
      TEST_CONVERT_TO(float);
    #undef TEST_CONVERT_TO
    }
#endif
    TYPED_TEST(VecMaskTests, MaskedLoad) {
      using vec = TypeParam;
      using src_t = ValueType<TypeParam>;
      constexpr auto size = vec::size();

    #define TEST_MASK_LOAD(dst_t, mask_t, mask_n)                           \
      do {                                                                  \
        CACHE_ALIGN dst_t x[mask_n * size];                                 \
        CACHE_ALIGN dst_t y[mask_n * size];                                 \
        CACHE_ALIGN dst_t ref[mask_n * size];                               \
        auto seed = TestSeed();                                             \
        ValueGen<dst_t> generator(dst_t(-100), dst_t(100), seed);           \
        for (const auto i : c10::irange(mask_n * size)) {                   \
          x[i] = generator.get();                                           \
        }                                                                   \
        auto vec_mask = generate_vec_mask<mask_t, mask_n>(seed);            \
        constexpr int dst_size = at::vec::Vectorized<dst_t>::size();        \
        constexpr int dst_n = mask_n * size / dst_size;                     \
        constexpr int rnd_n = (mask_n * size + dst_size - 1) / dst_size;    \
        if constexpr(dst_n * dst_size >= mask_n * size) {                   \
            auto x_vec = vec_mask.template loadu<dst_t, rnd_n>(x);          \
            x_vec.store(y);                                                 \
            for (const auto i : c10::irange(mask_n * size)) {               \
                if (vec_mask.is_masked(i)) {                                \
                    ref[i] = x[i];                                          \
                } else {                                                    \
                    ref[i] = 0;                                             \
                }                                                           \
            }                                                               \
            for (const auto i : c10::irange(mask_n * size)) {               \
            ASSERT_EQ(y[i], ref[i])                                         \
                << "Failure Details:\nTest Seed to reproduce: " << seed;    \
            }                                                               \
        }                                                                   \
      } while (0)


    #define TEST_MASK_LOAD_N(N)                                      \
      TEST_MASK_LOAD(int8_t, src_t, N);                              \
      TEST_MASK_LOAD(uint8_t, src_t, N);                             \
      TEST_MASK_LOAD(int16_t, src_t, N);                             \
      TEST_MASK_LOAD(uint16_t, src_t, N);                            \
      TEST_MASK_LOAD(int32_t, src_t, N);                             \
      TEST_MASK_LOAD(uint32_t, src_t, N);                            \
      TEST_MASK_LOAD(int64_t, src_t, N);                             \
      TEST_MASK_LOAD(uint64_t, src_t, N);                            \
      TEST_MASK_LOAD(c10::BFloat16, src_t, N);                       \
      TEST_MASK_LOAD(c10::Half, src_t, N);                           \
      TEST_MASK_LOAD(float, src_t, N);                               \
      TEST_MASK_LOAD(double, src_t, N);

      TEST_MASK_LOAD_N(1)
      TEST_MASK_LOAD_N(2)
      TEST_MASK_LOAD_N(4)

    #undef TEST_MASK_LOAD
    #undef TEST_MASK_LOAD_N
    }
#if !defined(CPU_CAPABILITY_SVE)
    TYPED_TEST(VecMaskTests, MaskedCheck) {
      using VT = ValueType<TypeParam>;
      using vec = TypeParam;
      constexpr auto size = vec::size();
    #define TEST_MASK_CHECK_N(N)                                                           \
      do {                                                                                 \
        auto vec_mask = create_vec_mask<VT, N>(0);                                         \
        ASSERT_TRUE(vec_mask.all_zero()) << "all_zero check failed";                       \
        vec_mask = create_vec_mask<VT, N>(-1);                                             \
        ASSERT_TRUE(vec_mask.all_masked()) << "all_masked check failed";                   \
        vec_mask = create_vec_mask<VT, N>(2);                                              \
        for (int i = 0; i < N; i ++) {                                                     \
          ASSERT_TRUE(vec_mask.is_masked(1 + i * size)) << "is_masked(1) check failed";    \
          ASSERT_TRUE(!vec_mask.is_masked(0 + i * size)) << "!is_masked(0) check failed";  \
        }                                                                                  \
      } while (0)

      TEST_MASK_CHECK_N(1);
      TEST_MASK_CHECK_N(2);
      TEST_MASK_CHECK_N(4);

    #undef TEST_MASK_CHECK_N
    }
#endif
#if !defined(CPU_CAPABILITY_SVE)
    TYPED_TEST(VecMaskTests, ToFrom) {
      using vec = TypeParam;
      using VT = ValueType<TypeParam>;
      constexpr auto N = vec::size();
      auto vec_mask = at::vec::VecMask<VT, 1>::from(1);
      ASSERT_TRUE(vec_mask.all_masked()) << "expect all_masked with from(1)";
      vec_mask = at::vec::VecMask<VT, 1>::from(0);
      ASSERT_TRUE(vec_mask.all_zero()) << "expect all_zero with from(0)";

      CACHE_ALIGN VT x[N];
      CACHE_ALIGN VT y[N];
      auto seed = TestSeed();
      ValueGen<VT> generator(VT(0), VT(2), seed);
      for (const auto i : c10::irange(N)) {
        x[i] = generator.get();
      }
      auto x_vec = vec::loadu(x);
      vec_mask = at::vec::VecMask<VT, 1>::template from<VT, 1>(x_vec);
      auto y_vec = vec_mask.template to<VT, 1>();
      y_vec.store(y);
      for (const auto i : c10::irange(N)) {
        ASSERT_EQ(y[i] != 0, x[i] != 0)
            << "Failure Details:\nTest Seed to reproduce: " << seed;
      }
    }
#endif
#if !defined(CPU_CAPABILITY_SVE)
    TYPED_TEST(VecMaskTests, Cast) {
      using vec = TypeParam;
      using src_t = ValueType<TypeParam>;
      constexpr auto size = vec::size();

    #define TEST_MASK_CAST(dst_t, mask_t, mask_n)                      \
      do {                                                             \
        constexpr int num_dst_elements =                               \
            std::min(size, at::vec::Vectorized<dst_t>::size());        \
        constexpr int dst_n = mask_n * size / num_dst_elements;        \
        CACHE_ALIGN mask_t x[mask_n * size];                           \
        CACHE_ALIGN dst_t y[at::vec::VectorizedN<dst_t, dst_n>::size()]; \
        auto seed = TestSeed();                                        \
        auto vec_mask = generate_vec_mask<mask_t, mask_n>(seed);       \
        auto vec_mask_new = vec_mask.template cast<dst_t, dst_n>();    \
        vec_mask.template to<mask_t, mask_n>().store(x);               \
        vec_mask_new.template to<dst_t, dst_n>().store(y);             \
        for (const auto i : c10::irange(mask_n * size)) {              \
          ASSERT_EQ(y[i], x[i])                                        \
              << "Failure Details:\nTest Seed to reproduce: " << seed; \
        }                                                              \
      } while (0)

    #define TEST_MASK_CAST_N(N)                                      \
      TEST_MASK_CAST(int8_t, src_t, N);                              \
      TEST_MASK_CAST(uint8_t, src_t, N);                             \
      TEST_MASK_CAST(int16_t, src_t, N);                             \
      TEST_MASK_CAST(uint16_t, src_t, N);                            \
      TEST_MASK_CAST(int32_t, src_t, N);                             \
      TEST_MASK_CAST(uint32_t, src_t, N);                            \
      TEST_MASK_CAST(int64_t, src_t, N);                             \
      TEST_MASK_CAST(uint64_t, src_t, N);                            \
      TEST_MASK_CAST(c10::BFloat16, src_t, N);                       \
      TEST_MASK_CAST(c10::Half, src_t, N);                           \
      TEST_MASK_CAST(float, src_t, N);                               \
      TEST_MASK_CAST(double, src_t, N);

      TEST_MASK_CAST_N(1)
      TEST_MASK_CAST_N(2)
      TEST_MASK_CAST_N(4)

    #undef TEST_MASK_CAST
    #undef TEST_MASK_CAST_N
    }
#endif
#else
#error GTEST does not have TYPED_TEST
#endif
}  // namespace
