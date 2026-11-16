#include <c10/macros/Macros.h>
#include <c10/util/complex.h>
#include <c10/util/hash.h>
#include <gtest/gtest.h>
#include <sstream>
#include <tuple>
#include <type_traits>
#include <unordered_map>

#if (defined(__CUDACC__) || defined(__HIPCC__))
#define MAYBE_GLOBAL __global__
#else
#define MAYBE_GLOBAL
#endif

#define PI 3.141592653589793238463

namespace memory {

MAYBE_GLOBAL void test_size() {
  static_assert(sizeof(c10::complex<float>) == 2 * sizeof(float), "");
  static_assert(sizeof(c10::complex<double>) == 2 * sizeof(double), "");
}

MAYBE_GLOBAL void test_align() {
  static_assert(alignof(c10::complex<float>) == 2 * sizeof(float), "");
  static_assert(alignof(c10::complex<double>) == 2 * sizeof(double), "");
}

MAYBE_GLOBAL void test_pod() {
  static_assert(std::is_standard_layout<c10::complex<float>>::value, "");
  static_assert(std::is_standard_layout<c10::complex<double>>::value, "");
}

TEST(TestMemory, ReinterpretCast) {
  {
    std::complex<float> z(1, 2);
    c10::complex<float> zz = *reinterpret_cast<c10::complex<float>*>(&z);
    ASSERT_EQ(zz.real(), float(1));
    ASSERT_EQ(zz.imag(), float(2));
  }

  {
    c10::complex<float> z(3, 4);
    std::complex<float> zz = *reinterpret_cast<std::complex<float>*>(&z);
    ASSERT_EQ(zz.real(), float(3));
    ASSERT_EQ(zz.imag(), float(4));
  }

  {
    std::complex<double> z(1, 2);
    c10::complex<double> zz = *reinterpret_cast<c10::complex<double>*>(&z);
    ASSERT_EQ(zz.real(), double(1));
    ASSERT_EQ(zz.imag(), double(2));
  }

  {
    c10::complex<double> z(3, 4);
    std::complex<double> zz = *reinterpret_cast<std::complex<double>*>(&z);
    ASSERT_EQ(zz.real(), double(3));
    ASSERT_EQ(zz.imag(), double(4));
  }
}

#if defined(__CUDACC__) || defined(__HIPCC__)
TEST(TestMemory, ThrustReinterpretCast) {
  {
    thrust::complex<float> z(1, 2);
    c10::complex<float> zz = *reinterpret_cast<c10::complex<float>*>(&z);
    ASSERT_EQ(zz.real(), float(1));
    ASSERT_EQ(zz.imag(), float(2));
  }

  {
    c10::complex<float> z(3, 4);
    thrust::complex<float> zz = *reinterpret_cast<thrust::complex<float>*>(&z);
    ASSERT_EQ(zz.real(), float(3));
    ASSERT_EQ(zz.imag(), float(4));
  }

  {
    thrust::complex<double> z(1, 2);
    c10::complex<double> zz = *reinterpret_cast<c10::complex<double>*>(&z);
    ASSERT_EQ(zz.real(), double(1));
    ASSERT_EQ(zz.imag(), double(2));
  }

  {
    c10::complex<double> z(3, 4);
    thrust::complex<double> zz =
        *reinterpret_cast<thrust::complex<double>*>(&z);
    ASSERT_EQ(zz.real(), double(3));
    ASSERT_EQ(zz.imag(), double(4));
  }
}
#endif

} // namespace memory

namespace constructors {

template <typename scalar_t>
C10_HOST_DEVICE void test_construct_from_scalar() {
  constexpr scalar_t num1 = scalar_t(1.23);
  constexpr scalar_t num2 = scalar_t(4.56);
  constexpr scalar_t zero = scalar_t();
  static_assert(c10::complex<scalar_t>(num1, num2).real() == num1, "");
  static_assert(c10::complex<scalar_t>(num1, num2).imag() == num2, "");
  static_assert(c10::complex<scalar_t>(num1).real() == num1, "");
  static_assert(c10::complex<scalar_t>(num1).imag() == zero, "");
  static_assert(c10::complex<scalar_t>().real() == zero, "");
  static_assert(c10::complex<scalar_t>().imag() == zero, "");
}

template <typename scalar_t, typename other_t>
C10_HOST_DEVICE void test_construct_from_other() {
  constexpr other_t num1 = other_t(1.23);
  constexpr other_t num2 = other_t(4.56);
  constexpr scalar_t num3 = scalar_t(num1);
  constexpr scalar_t num4 = scalar_t(num2);
  static_assert(
      c10::complex<scalar_t>(c10::complex<other_t>(num1, num2)).real() == num3,
      "");
  static_assert(
      c10::complex<scalar_t>(c10::complex<other_t>(num1, num2)).imag() == num4,
      "");
}

MAYBE_GLOBAL void test_convert_constructors() {
  test_construct_from_scalar<float>();
  test_construct_from_scalar<double>();

  static_assert(
      std::is_convertible<c10::complex<float>, c10::complex<float>>::value, "");
  static_assert(
      !std::is_convertible<c10::complex<double>, c10::complex<float>>::value,
      "");
  static_assert(
      std::is_convertible<c10::complex<float>, c10::complex<double>>::value,
      "");
  static_assert(
      std::is_convertible<c10::complex<double>, c10::complex<double>>::value,
      "");

  static_assert(
      std::is_constructible<c10::complex<float>, c10::complex<float>>::value,
      "");
  static_assert(
      std::is_constructible<c10::complex<double>, c10::complex<float>>::value,
      "");
  static_assert(
      std::is_constructible<c10::complex<float>, c10::complex<double>>::value,
      "");
  static_assert(
      std::is_constructible<c10::complex<double>, c10::complex<double>>::value,
      "");

  test_construct_from_other<float, float>();
  test_construct_from_other<float, double>();
  test_construct_from_other<double, float>();
  test_construct_from_other<double, double>();
}

template <typename scalar_t>
C10_HOST_DEVICE void test_construct_from_std() {
  constexpr scalar_t num1 = scalar_t(1.23);
  constexpr scalar_t num2 = scalar_t(4.56);
  static_assert(
      c10::complex<scalar_t>(std::complex<scalar_t>(num1, num2)).real() == num1,
      "");
  static_assert(
      c10::complex<scalar_t>(std::complex<scalar_t>(num1, num2)).imag() == num2,
      "");
}

MAYBE_GLOBAL void test_std_conversion() {
  test_construct_from_std<float>();
  test_construct_from_std<double>();
}

#if defined(__CUDACC__) || defined(__HIPCC__)
template <typename scalar_t>
void test_construct_from_thrust() {
  constexpr scalar_t num1 = scalar_t(1.23);
  constexpr scalar_t num2 = scalar_t(4.56);
  ASSERT_EQ(
      c10::complex<scalar_t>(thrust::complex<scalar_t>(num1, num2)).real(),
      num1);
  ASSERT_EQ(
      c10::complex<scalar_t>(thrust::complex<scalar_t>(num1, num2)).imag(),
      num2);
}

TEST(TestConstructors, FromThrust) {
  test_construct_from_thrust<float>();
  test_construct_from_thrust<double>();
}
#endif

TEST(TestConstructors, UnorderedMap) {
  std::unordered_map<
      c10::complex<double>,
      c10::complex<double>,
      c10::hash<c10::complex<double>>>
      m;
  auto key1 = c10::complex<double>(2.5, 3);
  auto key2 = c10::complex<double>(2, 0);
  auto val1 = c10::complex<double>(2, -3.2);
  auto val2 = c10::complex<double>(0, -3);
  m[key1] = val1;
  m[key2] = val2;
  ASSERT_EQ(m[key1], val1);
  ASSERT_EQ(m[key2], val2);
}

} // namespace constructors

namespace assignment {

template <typename scalar_t>
constexpr c10::complex<scalar_t> one() {
  c10::complex<scalar_t> result(3, 4);
  result = scalar_t(1);
  return result;
}

MAYBE_GLOBAL void test_assign_real() {
  static_assert(one<float>().real() == float(1), "");
  static_assert(one<float>().imag() == float(), "");
  static_assert(one<double>().real() == double(1), "");
  static_assert(one<double>().imag() == double(), "");
}

constexpr std::tuple<c10::complex<double>, c10::complex<float>> one_two() {
  constexpr c10::complex<float> src(1, 2);
  c10::complex<double> ret0;
  c10::complex<float> ret1;
  ret0 = ret1 = src;
  return std::make_tuple(ret0, ret1);
}

MAYBE_GLOBAL void test_assign_other() {
  constexpr auto tup = one_two();
  static_assert(std::get<c10::complex<double>>(tup).real() == double(1), "");
  static_assert(std::get<c10::complex<double>>(tup).imag() == double(2), "");
  static_assert(std::get<c10::complex<float>>(tup).real() == float(1), "");
  static_assert(std::get<c10::complex<float>>(tup).imag() == float(2), "");
}

constexpr std::tuple<c10::complex<double>, c10::complex<float>> one_two_std() {
  constexpr std::complex<float> src(1, 1);
  c10::complex<double> ret0;
  c10::complex<float> ret1;
  ret0 = ret1 = src;
  return std::make_tuple(ret0, ret1);
}

MAYBE_GLOBAL void test_assign_std() {
  constexpr auto tup = one_two();
  static_assert(std::get<c10::complex<double>>(tup).real() == double(1), "");
  static_assert(std::get<c10::complex<double>>(tup).imag() == double(2), "");
  static_assert(std::get<c10::complex<float>>(tup).real() == float(1), "");
  static_assert(std::get<c10::complex<float>>(tup).imag() == float(2), "");
}

#if defined(__CUDACC__) || defined(__HIPCC__)
C10_HOST_DEVICE std::tuple<c10::complex<double>, c10::complex<float>>
one_two_thrust() {
  thrust::complex<float> src(1, 2);
  c10::complex<double> ret0;
  c10::complex<float> ret1;
  ret0 = ret1 = src;
  return std::make_tuple(ret0, ret1);
}

TEST(TestAssignment, FromThrust) {
  auto tup = one_two_thrust();
  ASSERT_EQ(std::get<c10::complex<double>>(tup).real(), double(1));
  ASSERT_EQ(std::get<c10::complex<double>>(tup).imag(), double(2));
  ASSERT_EQ(std::get<c10::complex<float>>(tup).real(), float(1));
  ASSERT_EQ(std::get<c10::complex<float>>(tup).imag(), float(2));
}
#endif

} // namespace assignment

namespace literals {

MAYBE_GLOBAL void test_complex_literals() {
  using namespace c10::complex_literals;
  static_assert(std::is_same<decltype(0.5_if), c10::complex<float>>::value, "");
  static_assert((0.5_if).real() == float(), "");
  static_assert((0.5_if).imag() == float(0.5), "");
  static_assert(
      std::is_same<decltype(0.5_id), c10::complex<double>>::value, "");
  static_assert((0.5_id).real() == float(), "");
  static_assert((0.5_id).imag() == float(0.5), "");

  static_assert(std::is_same<decltype(1_if), c10::complex<float>>::value, "");
  static_assert((1_if).real() == float(), "");
  static_assert((1_if).imag() == float(1), "");
  static_assert(std::is_same<decltype(1_id), c10::complex<double>>::value, "");
  static_assert((1_id).real() == double(), "");
  static_assert((1_id).imag() == double(1), "");
}

} // namespace literals

namespace real_imag {

template <typename scalar_t>
constexpr c10::complex<scalar_t> zero_one() {
  c10::complex<scalar_t> result;
  result.imag(scalar_t(1));
  return result;
}

template <typename scalar_t>
constexpr c10::complex<scalar_t> one_zero() {
  c10::complex<scalar_t> result;
  result.real(scalar_t(1));
  return result;
}

MAYBE_GLOBAL void test_real_imag_modify() {
  static_assert(zero_one<float>().real() == float(0), "");
  static_assert(zero_one<float>().imag() == float(1), "");
  static_assert(zero_one<double>().real() == double(0), "");
  static_assert(zero_one<double>().imag() == double(1), "");

  static_assert(one_zero<float>().real() == float(1), "");
  static_assert(one_zero<float>().imag() == float(0), "");
  static_assert(one_zero<double>().real() == double(1), "");
  static_assert(one_zero<double>().imag() == double(0), "");
}

} // namespace real_imag

namespace arithmetic_assign {

template <typename scalar_t>
constexpr c10::complex<scalar_t> p(scalar_t value) {
  c10::complex<scalar_t> result(scalar_t(2), scalar_t(2));
  result += value;
  return result;
}

template <typename scalar_t>
constexpr c10::complex<scalar_t> m(scalar_t value) {
  c10::complex<scalar_t> result(scalar_t(2), scalar_t(2));
  result -= value;
  return result;
}

template <typename scalar_t>
constexpr c10::complex<scalar_t> t(scalar_t value) {
  c10::complex<scalar_t> result(scalar_t(2), scalar_t(2));
  result *= value;
  return result;
}

template <typename scalar_t>
constexpr c10::complex<scalar_t> d(scalar_t value) {
  c10::complex<scalar_t> result(scalar_t(2), scalar_t(2));
  result /= value;
  return result;
}

template <typename scalar_t>
C10_HOST_DEVICE void test_arithmetic_assign_scalar() {
  constexpr c10::complex<scalar_t> x = p(scalar_t(1));
  static_assert(x.real() == scalar_t(3), "");
  static_assert(x.imag() == scalar_t(2), "");
  constexpr c10::complex<scalar_t> y = m(scalar_t(1));
  static_assert(y.real() == scalar_t(1), "");
  static_assert(y.imag() == scalar_t(2), "");
  constexpr c10::complex<scalar_t> z = t(scalar_t(2));
  static_assert(z.real() == scalar_t(4), "");
  static_assert(z.imag() == scalar_t(4), "");
  constexpr c10::complex<scalar_t> t = d(scalar_t(2));
  static_assert(t.real() == scalar_t(1), "");
  static_assert(t.imag() == scalar_t(1), "");
}

template <typename scalar_t, typename rhs_t>
constexpr c10::complex<scalar_t> p(
    scalar_t real,
    scalar_t imag,
    c10::complex<rhs_t> rhs) {
  c10::complex<scalar_t> result(real, imag);
  result += rhs;
  return result;
}

template <typename scalar_t, typename rhs_t>
constexpr c10::complex<scalar_t> m(
    scalar_t real,
    scalar_t imag,
    c10::complex<rhs_t> rhs) {
  c10::complex<scalar_t> result(real, imag);
  result -= rhs;
  return result;
}

template <typename scalar_t, typename rhs_t>
constexpr c10::complex<scalar_t> t(
    scalar_t real,
    scalar_t imag,
    c10::complex<rhs_t> rhs) {
  c10::complex<scalar_t> result(real, imag);
  result *= rhs;
  return result;
}

template <typename scalar_t, typename rhs_t>
constexpr c10::complex<scalar_t> d(
    scalar_t real,
    scalar_t imag,
    c10::complex<rhs_t> rhs) {
  c10::complex<scalar_t> result(real, imag);
  result /= rhs;
  return result;
}

template <typename scalar_t>
C10_HOST_DEVICE void test_arithmetic_assign_complex() {
  using namespace c10::complex_literals;
  constexpr c10::complex<scalar_t> x2 = p(scalar_t(2), scalar_t(2), 1.0_if);
  static_assert(x2.real() == scalar_t(2), "");
  static_assert(x2.imag() == scalar_t(3), "");
  constexpr c10::complex<scalar_t> x3 = p(scalar_t(2), scalar_t(2), 1.0_id);
  static_assert(x3.real() == scalar_t(2), "");

  static_assert(x3.imag() == scalar_t(3), "");

  constexpr c10::complex<scalar_t> y2 = m(scalar_t(2), scalar_t(2), 1.0_if);
  static_assert(y2.real() == scalar_t(2), "");
  static_assert(y2.imag() == scalar_t(1), "");
  constexpr c10::complex<scalar_t> y3 = m(scalar_t(2), scalar_t(2), 1.0_id);
  static_assert(y3.real() == scalar_t(2), "");

  static_assert(y3.imag() == scalar_t(1), "");

  constexpr c10::complex<scalar_t> z2 = t(scalar_t(1), scalar_t(-2), 1.0_if);
  static_assert(z2.real() == scalar_t(2), "");
  static_assert(z2.imag() == scalar_t(1), "");
  constexpr c10::complex<scalar_t> z3 = t(scalar_t(1), scalar_t(-2), 1.0_id);
  static_assert(z3.real() == scalar_t(2), "");
  static_assert(z3.imag() == scalar_t(1), "");

  constexpr c10::complex<scalar_t> t2 = d(scalar_t(-1), scalar_t(2), 1.0_if);
  static_assert(t2.real() == scalar_t(2), "");
  static_assert(t2.imag() == scalar_t(1), "");
  constexpr c10::complex<scalar_t> t3 = d(scalar_t(-1), scalar_t(2), 1.0_id);
  static_assert(t3.real() == scalar_t(2), "");
  static_assert(t3.imag() == scalar_t(1), "");
}

MAYBE_GLOBAL void test_arithmetic_assign() {
  test_arithmetic_assign_scalar<float>();
  test_arithmetic_assign_scalar<double>();
  test_arithmetic_assign_complex<float>();
  test_arithmetic_assign_complex<double>();
}

} // namespace arithmetic_assign

namespace arithmetic {

template <typename scalar_t>
C10_HOST_DEVICE void test_arithmetic_() {
  static_assert(
      c10::complex<scalar_t>(1, 2) == +c10::complex<scalar_t>(1, 2), "");
  static_assert(
      c10::complex<scalar_t>(-1, -2) == -c10::complex<scalar_t>(1, 2), "");

  static_assert(
      c10::complex<scalar_t>(1, 2) + c10::complex<scalar_t>(3, 4) ==
          c10::complex<scalar_t>(4, 6),
      "");
  static_assert(
      c10::complex<scalar_t>(1, 2) + scalar_t(3) ==
          c10::complex<scalar_t>(4, 2),
      "");
  static_assert(
      scalar_t(3) + c10::complex<scalar_t>(1, 2) ==
          c10::complex<scalar_t>(4, 2),
      "");

  static_assert(
      c10::complex<scalar_t>(1, 2) - c10::complex<scalar_t>(3, 4) ==
          c10::complex<scalar_t>(-2, -2),
      "");
  static_assert(
      c10::complex<scalar_t>(1, 2) - scalar_t(3) ==
          c10::complex<scalar_t>(-2, 2),
      "");
  static_assert(
      scalar_t(3) - c10::complex<scalar_t>(1, 2) ==
          c10::complex<scalar_t>(2, -2),
      "");

  static_assert(
      c10::complex<scalar_t>(1, 2) * c10::complex<scalar_t>(3, 4) ==
          c10::complex<scalar_t>(-5, 10),
      "");
  static_assert(
      c10::complex<scalar_t>(1, 2) * scalar_t(3) ==
          c10::complex<scalar_t>(3, 6),
      "");
  static_assert(
      scalar_t(3) * c10::complex<scalar_t>(1, 2) ==
          c10::complex<scalar_t>(3, 6),
      "");

  static_assert(
      c10::complex<scalar_t>(-5, 10) / c10::complex<scalar_t>(3, 4) ==
          c10::complex<scalar_t>(1, 2),
      "");
  static_assert(
      c10::complex<scalar_t>(5, 10) / scalar_t(5) ==
          c10::complex<scalar_t>(1, 2),
      "");
  static_assert(
      scalar_t(25) / c10::complex<scalar_t>(3, 4) ==
          c10::complex<scalar_t>(3, -4),
      "");
}

MAYBE_GLOBAL void test_arithmetic() {
  test_arithmetic_<float>();
  test_arithmetic_<double>();
}

template <typename T, typename int_t>
void test_binary_ops_for_int_type_(T real, T img, int_t num) {
  c10::complex<T> c(real, img);
  ASSERT_EQ(c + num, c10::complex<T>(real + num, img));
  ASSERT_EQ(num + c, c10::complex<T>(num + real, img));
  ASSERT_EQ(c - num, c10::complex<T>(real - num, img));
  ASSERT_EQ(num - c, c10::complex<T>(num - real, -img));
  ASSERT_EQ(c * num, c10::complex<T>(real * num, img * num));
  ASSERT_EQ(num * c, c10::complex<T>(num * real, num * img));
  ASSERT_EQ(c / num, c10::complex<T>(real / num, img / num));
  ASSERT_EQ(
      num / c,
      c10::complex<T>(num * real / std::norm(c), -num * img / std::norm(c)));
}

template <typename T>
void test_binary_ops_for_all_int_types_(T real, T img, int8_t i) {
  test_binary_ops_for_int_type_<T, int8_t>(real, img, i);
  test_binary_ops_for_int_type_<T, int16_t>(real, img, i);
  test_binary_ops_for_int_type_<T, int32_t>(real, img, i);
  test_binary_ops_for_int_type_<T, int64_t>(real, img, i);
}

TEST(TestArithmeticIntScalar, All) {
  test_binary_ops_for_all_int_types_<float>(1.0, 0.1, 1);
  test_binary_ops_for_all_int_types_<double>(-1.3, -0.2, -2);
}

} // namespace arithmetic

namespace equality {

template <typename scalar_t>
C10_HOST_DEVICE void test_equality_() {
  static_assert(
      c10::complex<scalar_t>(1, 2) == c10::complex<scalar_t>(1, 2), "");
  static_assert(c10::complex<scalar_t>(1, 0) == scalar_t(1), "");
  static_assert(scalar_t(1) == c10::complex<scalar_t>(1, 0), "");
  static_assert(
      c10::complex<scalar_t>(1, 2) != c10::complex<scalar_t>(3, 4), "");
  static_assert(c10::complex<scalar_t>(1, 2) != scalar_t(1), "");
  static_assert(scalar_t(1) != c10::complex<scalar_t>(1, 2), "");
}

MAYBE_GLOBAL void test_equality() {
  test_equality_<float>();
  test_equality_<double>();
}

} // namespace equality

namespace io {

template <typename scalar_t>
void test_io_() {
  std::stringstream ss;
  c10::complex<scalar_t> a(1, 2);
  ss << a;
  ASSERT_EQ(ss.str(), "(1,2)");
  ss.str("(3,4)");
  ss >> a;
  ASSERT_TRUE(a == c10::complex<scalar_t>(3, 4));
}

TEST(TestIO, All) {
  test_io_<float>();
  test_io_<double>();
}

} // namespace io

namespace test_std {

template <typename scalar_t>
C10_HOST_DEVICE void test_callable_() {
  static_assert(std::real(c10::complex<scalar_t>(1, 2)) == scalar_t(1), "");
  static_assert(std::imag(c10::complex<scalar_t>(1, 2)) == scalar_t(2), "");
  std::abs(c10::complex<scalar_t>(1, 2));
  std::arg(c10::complex<scalar_t>(1, 2));
  static_assert(std::norm(c10::complex<scalar_t>(3, 4)) == scalar_t(25), "");
  static_assert(
      std::conj(c10::complex<scalar_t>(3, 4)) == c10::complex<scalar_t>(3, -4),
      "");
  c10::polar(float(1), float(PI / 2));
  c10::polar(double(1), double(PI / 2));
}

MAYBE_GLOBAL void test_callable() {
  test_callable_<float>();
  test_callable_<double>();
}

template <typename scalar_t>
void test_values_() {
  ASSERT_EQ(std::abs(c10::complex<scalar_t>(3, 4)), scalar_t(5));
  ASSERT_LT(std::abs(std::arg(c10::complex<scalar_t>(0, 1)) - PI / 2), 1e-6);
  ASSERT_LT(
      std::abs(
          c10::polar(scalar_t(1), scalar_t(PI / 2)) -
          c10::complex<scalar_t>(0, 1)),
      1e-6);
}

TEST(TestStd, BasicFunctions) {
  test_values_<float>();
  test_values_<double>();
  // CSQRT edge cases: checks for overflows which are likely to occur
  // if square root is computed using polar form
  ASSERT_LT(
      std::abs(std::sqrt(c10::complex<float>(-1e20, -4988429.2)).real()), 3e-4);
  ASSERT_LT(
      std::abs(std::sqrt(c10::complex<double>(-1e60, -4988429.2)).real()),
      3e-4);
}

} // namespace test_std
