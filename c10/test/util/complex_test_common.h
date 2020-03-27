#include <type_traits>
#include <tuple>
#include <sstream>
#include <c10/util/complex.h>
#include <gtest/gtest.h>

#if (defined(__CUDACC__) || defined(__HIPCC__)) && !defined(C10_HOST_DEVICE)
#define MAYBE_GLOBAL __global__
#else
#define MAYBE_GLOBAL
#endif

const double PI = 3.141592653589793238463;

namespace memory {

MAYBE_GLOBAL void test_size() {
  static_assert(sizeof(c10::complex<c10::Half>) == 4, "");
  static_assert(sizeof(c10::complex<float>) == 8, "");
  static_assert(sizeof(c10::complex<double>) == 16, "");
}

MAYBE_GLOBAL void test_align() {
  static_assert(alignof(c10::complex<c10::Half>) == 4, "");
  static_assert(alignof(c10::complex<float>) == 8, "");
  static_assert(alignof(c10::complex<double>) == 16, "");
}

}  // memory

namespace constructors {

template<typename scalar_t>
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

template<typename scalar_t, typename other_t>
C10_HOST_DEVICE void test_construct_from_other() {
  constexpr other_t num1 = other_t(1.23);
  constexpr other_t num2 = other_t(4.56);
  constexpr scalar_t num3 = scalar_t(num1);
  constexpr scalar_t num4 = scalar_t(num2);
  static_assert(c10::complex<scalar_t>(c10::complex<other_t>(num1, num2)).real() == num3, "");
  static_assert(c10::complex<scalar_t>(c10::complex<other_t>(num1, num2)).imag() == num4, "");
}

MAYBE_GLOBAL void test_convert_constructors() {
  test_construct_from_scalar<c10::Half>();
  test_construct_from_scalar<float>();
  test_construct_from_scalar<double>();

  static_assert(std::is_convertible<c10::complex<c10::Half>, c10::complex<c10::Half>>::value, "");
  static_assert(!std::is_convertible<c10::complex<float>, c10::complex<c10::Half>>::value, "");
  static_assert(!std::is_convertible<c10::complex<double>, c10::complex<c10::Half>>::value, "");
  static_assert(std::is_convertible<c10::complex<c10::Half>, c10::complex<float>>::value, "");
  static_assert(std::is_convertible<c10::complex<float>, c10::complex<float>>::value, "");
  static_assert(!std::is_convertible<c10::complex<double>, c10::complex<float>>::value, "");
  static_assert(std::is_convertible<c10::complex<c10::Half>, c10::complex<double>>::value, "");
  static_assert(std::is_convertible<c10::complex<float>, c10::complex<double>>::value, "");
  static_assert(std::is_convertible<c10::complex<double>, c10::complex<double>>::value, "");

  static_assert(std::is_constructible<c10::complex<c10::Half>, c10::complex<c10::Half>>::value, "");
  static_assert(std::is_constructible<c10::complex<float>, c10::complex<c10::Half>>::value, "");
  static_assert(std::is_constructible<c10::complex<double>, c10::complex<c10::Half>>::value, "");
  static_assert(std::is_constructible<c10::complex<c10::Half>, c10::complex<float>>::value, "");
  static_assert(std::is_constructible<c10::complex<float>, c10::complex<float>>::value, "");
  static_assert(std::is_constructible<c10::complex<double>, c10::complex<float>>::value, "");
  static_assert(std::is_constructible<c10::complex<c10::Half>, c10::complex<double>>::value, "");
  static_assert(std::is_constructible<c10::complex<float>, c10::complex<double>>::value, "");
  static_assert(std::is_constructible<c10::complex<double>, c10::complex<double>>::value, "");

  test_construct_from_other<c10::Half, c10::Half>();
  test_construct_from_other<c10::Half, float>();
  test_construct_from_other<c10::Half, double>();
  test_construct_from_other<float, c10::Half>();
  test_construct_from_other<float, float>();
  test_construct_from_other<float, double>();
  test_construct_from_other<double, c10::Half>();
  test_construct_from_other<double, float>();
  test_construct_from_other<double, double>();
}

template<typename scalar_t>
C10_HOST_DEVICE void test_construct_from_std() {
  constexpr scalar_t num1 = scalar_t(1.23);
  constexpr scalar_t num2 = scalar_t(4.56);
  static_assert(c10::complex<scalar_t>(std::complex<scalar_t>(num1, num2)).real() == num1, "");
  static_assert(c10::complex<scalar_t>(std::complex<scalar_t>(num1, num2)).imag() == num2, "");
}

MAYBE_GLOBAL void test_std_conversion() {
  test_construct_from_std<float>();
  test_construct_from_std<double>();
}

#if defined(__CUDACC__) || defined(__HIPCC__)
template<typename scalar_t>
void test_construct_from_thrust() {
  constexpr scalar_t num1 = scalar_t(1.23);
  constexpr scalar_t num2 = scalar_t(4.56);
  ASSERT_EQ(c10::complex<scalar_t>(thrust::complex<scalar_t>(num1, num2)).real(), num1);
  ASSERT_EQ(c10::complex<scalar_t>(thrust::complex<scalar_t>(num1, num2)).imag(), num2);
}
#endif

TEST(TestConstructors, FromThrust) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  // TODO(@zasdfgbnm): thrust::complex only support float and double, how do we handle c10::Half?
  test_construct_from_thrust<float>();
  test_construct_from_thrust<double>();
#endif
}


}  // constructors

namespace assignment {

template<typename scalar_t>
constexpr c10::complex<scalar_t> one() {
  c10::complex<scalar_t> result;
  result = scalar_t(1);
  return result;
}

MAYBE_GLOBAL void test_assign_real() {
  static_assert(one<c10::Half>().real() == c10::Half(1), "");
  static_assert(one<c10::Half>().imag() == c10::Half(), "");
  static_assert(one<float>().real() == float(1), "");
  static_assert(one<float>().imag() == float(), "");
  static_assert(one<double>().real() == double(1), "");
  static_assert(one<double>().imag() == double(), "");
}

constexpr std::tuple<c10::complex<double>, c10::complex<float>, c10::complex<c10::Half>> ones() {
  constexpr c10::complex<c10::Half> src(1, 1);
  c10::complex<double> ret0;
  c10::complex<float> ret1;
  c10::complex<c10::Half> ret2;
  ret0 = ret1 = ret2 = src;
  return std::make_tuple(ret0, ret1, ret2);
}

MAYBE_GLOBAL void test_assign_other() {
  constexpr auto tup = ones();
  static_assert(std::get<c10::complex<double>>(tup).real() == double(1), "");
  static_assert(std::get<c10::complex<double>>(tup).imag() == double(1), "");
  static_assert(std::get<c10::complex<float>>(tup).real() == float(1), "");
  static_assert(std::get<c10::complex<float>>(tup).imag() == float(1), "");
  static_assert(std::get<c10::complex<c10::Half>>(tup).real() == c10::Half(1), "");
  static_assert(std::get<c10::complex<c10::Half>>(tup).imag() == c10::Half(1), "");
}

constexpr std::tuple<c10::complex<double>, c10::complex<float>, c10::complex<c10::Half>> ones_std() {
  constexpr std::complex<c10::Half> src(1, 1);
  c10::complex<double> ret0;
  c10::complex<float> ret1;
  c10::complex<c10::Half> ret2;
  ret0 = ret1 = ret2 = src;
  return std::make_tuple(ret0, ret1, ret2);
}

MAYBE_GLOBAL void test_assign_std() {
  constexpr auto tup = ones_std();
  static_assert(std::get<c10::complex<double>>(tup).real() == double(1), "");
  static_assert(std::get<c10::complex<double>>(tup).imag() == double(1), "");
  static_assert(std::get<c10::complex<float>>(tup).real() == float(1), "");
  static_assert(std::get<c10::complex<float>>(tup).imag() == float(1), "");
  static_assert(std::get<c10::complex<c10::Half>>(tup).real() == c10::Half(1), "");
  static_assert(std::get<c10::complex<c10::Half>>(tup).imag() == c10::Half(1), "");
}

#if defined(__CUDACC__) || defined(__HIPCC__)
std::tuple<c10::complex<double>, c10::complex<float>> one_two_thrust() {
  thrust::complex<float> src(1, 2);
  c10::complex<double> ret0;
  c10::complex<float> ret1;
  ret0 = ret1 = src;
  return std::make_tuple(ret0, ret1);
}
#endif

TEST(TestAssignment, FromThrust) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  auto tup = one_two_thrust();
  ASSERT_EQ(std::get<c10::complex<double>>(tup).real(), double(1));
  ASSERT_EQ(std::get<c10::complex<double>>(tup).imag(), double(2));
  ASSERT_EQ(std::get<c10::complex<float>>(tup).real(), float(1));
  ASSERT_EQ(std::get<c10::complex<float>>(tup).imag(), float(2));
#endif
}

} // namespace assignment

namespace literals {

MAYBE_GLOBAL void test_complex_literals() {
  using namespace c10::complex_literals;
  static_assert(std::is_same<decltype(0.5_ih), c10::complex<c10::Half>>::value, "");
  static_assert((0.5_ih).real() == c10::Half(), "");
  static_assert((0.5_ih).imag() == c10::Half(0.5), "");
  static_assert(std::is_same<decltype(0.5_if), c10::complex<float>>::value, "");
  static_assert((0.5_if).real() == float(), "");
  static_assert((0.5_if).imag() == float(0.5), "");
  static_assert(std::is_same<decltype(0.5_id), c10::complex<double>>::value, "");
  static_assert((0.5_id).real() == float(), "");
  static_assert((0.5_id).imag() == float(0.5), "");
}

} // namespace literals

namespace real_imag {

template<typename scalar_t>
constexpr c10::complex<scalar_t> zero_one() {
  c10::complex<scalar_t> result;
  result.imag(scalar_t(1));
  return result;
}

template<typename scalar_t>
constexpr c10::complex<scalar_t> one_zero() {
  c10::complex<scalar_t> result;
  result.real(scalar_t(1));
  return result;
}

MAYBE_GLOBAL void test_real_imag_modify() {
  static_assert(zero_one<c10::Half>().real() == c10::Half(0), "");
  static_assert(zero_one<c10::Half>().imag() == c10::Half(1), "");
  static_assert(zero_one<float>().real() == float(0), "");
  static_assert(zero_one<float>().imag() == float(1), "");
  static_assert(zero_one<double>().real() == double(0), "");
  static_assert(zero_one<double>().imag() == double(1), "");

  static_assert(one_zero<c10::Half>().real() == c10::Half(1), "");
  static_assert(one_zero<c10::Half>().imag() == c10::Half(0), "");
  static_assert(one_zero<float>().real() == float(1), "");
  static_assert(one_zero<float>().imag() == float(0), "");
  static_assert(one_zero<double>().real() == double(1), "");
  static_assert(one_zero<double>().imag() == double(0), "");
}

} // namespace real_imag

namespace arithmetic_assign {

template<typename scalar_t>
constexpr c10::complex<scalar_t> p(scalar_t value) {
  c10::complex<scalar_t> result(scalar_t(2), scalar_t(2));
  result += value;
  return result;
}

template<typename scalar_t>
constexpr c10::complex<scalar_t> m(scalar_t value) {
  c10::complex<scalar_t> result(scalar_t(2), scalar_t(2));
  result -= value;
  return result;
}

template<typename scalar_t>
constexpr c10::complex<scalar_t> t(scalar_t value) {
  c10::complex<scalar_t> result(scalar_t(2), scalar_t(2));
  result *= value;
  return result;
}

template<typename scalar_t>
constexpr c10::complex<scalar_t> d(scalar_t value) {
  c10::complex<scalar_t> result(scalar_t(2), scalar_t(2));
  result /= value;
  return result;
}

template<typename scalar_t>
MAYBE_GLOBAL void test_arithmetic_assign_scalar() {
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

template<typename scalar_t, typename rhs_t>
constexpr c10::complex<scalar_t> p(scalar_t real, scalar_t imag, c10::complex<rhs_t> rhs) {
  c10::complex<scalar_t> result(real, imag);
  result += rhs;
  return result;
}

template<typename scalar_t, typename rhs_t>
constexpr c10::complex<scalar_t> m(scalar_t real, scalar_t imag, c10::complex<rhs_t> rhs) {
  c10::complex<scalar_t> result(real, imag);
  result -= rhs;
  return result;
}

template<typename scalar_t, typename rhs_t>
constexpr c10::complex<scalar_t> t(scalar_t real, scalar_t imag, c10::complex<rhs_t> rhs) {
  c10::complex<scalar_t> result(real, imag);
  result *= rhs;
  return result;
}

template<typename scalar_t, typename rhs_t>
constexpr c10::complex<scalar_t> d(scalar_t real, scalar_t imag, c10::complex<rhs_t> rhs) {
  c10::complex<scalar_t> result(real, imag);
  result /= rhs;
  return result;
}

template<typename scalar_t>
void test_arithmetic_assign_complex() {
  using namespace c10::complex_literals;
  constexpr c10::complex<scalar_t> x1 = p(scalar_t(2), scalar_t(2), 1.0_ih);
  static_assert(x1.real() == scalar_t(2), "");
  static_assert(x1.imag() == scalar_t(3), "");
  constexpr c10::complex<scalar_t> x2 = p(scalar_t(2), scalar_t(2), 1.0_if);
  static_assert(x2.real() == scalar_t(2), "");
  static_assert(x2.imag() == scalar_t(3), "");
  constexpr c10::complex<scalar_t> x3 = p(scalar_t(2), scalar_t(2), 1.0_id);
  static_assert(x3.real() == scalar_t(2), "");
#if !defined(__CUDACC__)
  // The following is flaky on nvcc
  static_assert(x3.imag() == scalar_t(3), "");
#endif

  constexpr c10::complex<scalar_t> y1 = m(scalar_t(2), scalar_t(2), 1.0_ih);
  static_assert(y1.real() == scalar_t(2), "");
  static_assert(y1.imag() == scalar_t(1), "");
  constexpr c10::complex<scalar_t> y2 = m(scalar_t(2), scalar_t(2), 1.0_if);
  static_assert(y2.real() == scalar_t(2), "");
  static_assert(y2.imag() == scalar_t(1), "");
  constexpr c10::complex<scalar_t> y3 = m(scalar_t(2), scalar_t(2), 1.0_id);
  static_assert(y3.real() == scalar_t(2), "");
#if !defined(__CUDACC__)
  // The following is flaky on nvcc
  static_assert(y3.imag() == scalar_t(1), "");
#endif

  constexpr c10::complex<scalar_t> z1 = t(scalar_t(1), scalar_t(-2), 1.0_ih);
  static_assert(z1.real() == scalar_t(2), "");
  static_assert(z1.imag() == scalar_t(1), "");
  constexpr c10::complex<scalar_t> z2 = t(scalar_t(1), scalar_t(-2), 1.0_if);
  static_assert(z2.real() == scalar_t(2), "");
  static_assert(z2.imag() == scalar_t(1), "");
  constexpr c10::complex<scalar_t> z3 = t(scalar_t(1), scalar_t(-2), 1.0_id);
  static_assert(z3.real() == scalar_t(2), "");
  static_assert(z3.imag() == scalar_t(1), "");

  constexpr c10::complex<scalar_t> t1 = d(scalar_t(-1), scalar_t(2), 1.0_ih);
  static_assert(t1.real() == scalar_t(2), "");
  static_assert(t1.imag() == scalar_t(1), "");
  constexpr c10::complex<scalar_t> t2 = d(scalar_t(-1), scalar_t(2), 1.0_if);
  static_assert(t2.real() == scalar_t(2), "");
  static_assert(t2.imag() == scalar_t(1), "");
  constexpr c10::complex<scalar_t> t3 = d(scalar_t(-1), scalar_t(2), 1.0_id);
  static_assert(t3.real() == scalar_t(2), "");
  static_assert(t3.imag() == scalar_t(1), "");
}

MAYBE_GLOBAL void test_arithmetic_assign() {
  test_arithmetic_assign_scalar<c10::Half>();
  test_arithmetic_assign_scalar<float>();
  test_arithmetic_assign_scalar<double>();
  test_arithmetic_assign_complex<c10::Half>();
  test_arithmetic_assign_complex<float>();
  test_arithmetic_assign_complex<double>();
}

} // namespace arithmetic_assign

namespace arithmetic {

template<typename scalar_t>
C10_HOST_DEVICE void test_arithmetic_() {
  static_assert(c10::complex<scalar_t>(1, 2) == +c10::complex<scalar_t>(1, 2), "");
  static_assert(c10::complex<scalar_t>(-1, -2) == -c10::complex<scalar_t>(1, 2), "");

  static_assert(c10::complex<scalar_t>(1, 2) + c10::complex<scalar_t>(3, 4) == c10::complex<scalar_t>(4, 6), "");
  static_assert(c10::complex<scalar_t>(1, 2) + scalar_t(3) == c10::complex<scalar_t>(4, 2), "");
  static_assert(scalar_t(3) + c10::complex<scalar_t>(1, 2)  == c10::complex<scalar_t>(4, 2), "");

  static_assert(c10::complex<scalar_t>(1, 2) - c10::complex<scalar_t>(3, 4) == c10::complex<scalar_t>(-2, -2), "");
  static_assert(c10::complex<scalar_t>(1, 2) - scalar_t(3) == c10::complex<scalar_t>(-2, 2), "");
  static_assert(scalar_t(3) - c10::complex<scalar_t>(1, 2)  == c10::complex<scalar_t>(2, -2), "");

  static_assert(c10::complex<scalar_t>(1, 2) * c10::complex<scalar_t>(3, 4) == c10::complex<scalar_t>(-5, 10), "");
  static_assert(c10::complex<scalar_t>(1, 2) * scalar_t(3) == c10::complex<scalar_t>(3, 6), "");
  static_assert(scalar_t(3) * c10::complex<scalar_t>(1, 2)  == c10::complex<scalar_t>(3, 6), "");

  static_assert(c10::complex<scalar_t>(-5, 10) / c10::complex<scalar_t>(3, 4) == c10::complex<scalar_t>(1, 2), "");
  static_assert(c10::complex<scalar_t>(5, 10) / scalar_t(5) == c10::complex<scalar_t>(1, 2), "");
  static_assert(scalar_t(25) / c10::complex<scalar_t>(3, 4)  == c10::complex<scalar_t>(3, -4), "");
}

MAYBE_GLOBAL void test_arithmetic() {
  test_arithmetic_<c10::Half>();
  test_arithmetic_<float>();
  test_arithmetic_<double>();
}

} // namespace arithmetic

namespace equality {

template<typename scalar_t>
C10_HOST_DEVICE void test_equality_() {
  static_assert(c10::complex<scalar_t>(1, 2) == c10::complex<scalar_t>(1, 2), "");
  static_assert(c10::complex<scalar_t>(1, 2) != c10::complex<scalar_t>(3, 4), "");
}

MAYBE_GLOBAL void test_equality() {
  test_equality_<c10::Half>();
  test_equality_<float>();
  test_equality_<double>();
}
  
} // namespace equality

namespace io {

template<typename scalar_t>
void test_io_() {
  std::stringstream ss;
  c10::complex<scalar_t> a(1, 2);
  ss << a;
  ASSERT_EQ(ss.str(), "(1,2)");
  ss.str("(3,4)");
  ss >> a;
  ASSERT_EQ(a, c10::complex<scalar_t>(3, 4));
}

TEST(TestIO, All) {
  test_io_<c10::Half>();
  test_io_<float>();
  test_io_<double>();
}

} // namespace io

namespace test_std {

template<typename scalar_t>
C10_HOST_DEVICE void test_callable_() {
  static_assert(std::real(c10::complex<scalar_t>(1, 2)) == scalar_t(1), "");
  static_assert(std::imag(c10::complex<scalar_t>(1, 2)) == scalar_t(2), "");
  std::abs(c10::complex<scalar_t>(1, 2));
  std::arg(c10::complex<scalar_t>(1, 2));
  static_assert(std::norm(c10::complex<scalar_t>(3, 4)) == scalar_t(25), "");
  static_assert(std::conj(c10::complex<scalar_t>(3, 4)) == c10::complex<scalar_t>(3, -4), "");
  // TODO(@zasdfgbnm): thrust::complex only support float and double, how do we handle c10::Half?
  c10::polar(float(1), float(PI / 2));
  c10::polar(double(1), double(PI / 2));
}

MAYBE_GLOBAL void test_callable() {
  test_callable_<c10::Half>();
  test_callable_<float>();
  test_callable_<double>();
}

template<typename scalar_t>
void test_values_() {
  ASSERT_EQ(std::abs(c10::complex<scalar_t>(3, 4)), scalar_t(5));
  ASSERT_LT(std::abs(std::arg(c10::complex<scalar_t>(0, 1)) - PI / 2), 1e-6);
  ASSERT_LT(std::abs(c10::polar(scalar_t(1), scalar_t(PI / 2)) - c10::complex<scalar_t>(0, 1)), 1e-6);
}

TEST(TestStd, BasicFunctions) {
  ASSERT_EQ(std::abs(c10::complex<c10::Half>(3, 4)), c10::Half(5));
  // FIXME: ASSERT_LT(std::abs(std::arg(c10::complex<c10::Half>(0, 1)) - PI / 2), 1e-6);
  // TODO(@zasdfgbnm): thrust::complex only support float and double, how do we handle c10::Half?
  test_values_<float>();
  test_values_<double>();
}

} // namespace test_std
