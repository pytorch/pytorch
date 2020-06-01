#pragma once

#include <complex>
#include <type_traits>
#include <c10/core/ScalarType.h>
#include <c10/util/C++17.h>
#include <ATen/detail/FunctionTraits.h>
#include <ATen/native/TensorIterator.h>

#if defined(__CUDACC__) || defined(__HIPCC__)
#include <thrust/complex.h>
#endif


// This file includes utilties for dynamic_casting done by TensorIterator, see CUDALoops.cuh and Loops.h.

// dynamic_casting handles when the types expected by the iterator do not match the types of the arguments
// to the function that is being called.
// On CUDA, the cast is currently pushed down into the kernel (for performance reasons).
// On CPU, there is currently an internal assert that a dynamic_cast is not needed.

namespace at { namespace native {

// this extra namespace (cppmap) is to avoid conflicting with at::detail when at:: is
// left off in native functions.
namespace cppmap { namespace detail {

// See [NOTE: Complex Operator Unification]
// CPPTypeAndStdComplexToScalarType is equivalent to CPPTypeToScalarType, but also includes mappings
// from all the complex types.
template <typename>
struct CPPTypeAndStdComplexToScalarType {
};

#define SPECIALIZE_CPPTypeAndStdComplexToScalarType(cpp_type, scalar_type)                  \
  template <>                                                                               \
  struct CPPTypeAndStdComplexToScalarType<cpp_type> {                                       \
    constexpr static c10::ScalarType value() { return c10::ScalarType::scalar_type; }       \
  };

AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(SPECIALIZE_CPPTypeAndStdComplexToScalarType)

#undef SPECIALIZE_CPPTypeeAndStdComplexToScalarType

template<>
struct CPPTypeAndStdComplexToScalarType<std::complex<float>> {
    constexpr static c10::ScalarType value() { return c10::ScalarType::ComplexFloat; }
};

template<>
struct CPPTypeAndStdComplexToScalarType<std::complex<double>> {
  constexpr static c10::ScalarType value() { return c10::ScalarType::ComplexDouble; }
};

#if defined(__CUDACC__) || defined(__HIPCC__)
template<>
struct CPPTypeAndStdComplexToScalarType<thrust::complex<float>> {
  constexpr static c10::ScalarType value() { return c10::ScalarType::ComplexFloat; }
};

template<>
struct CPPTypeAndStdComplexToScalarType<thrust::complex<double>> {
  constexpr static c10::ScalarType value() { return c10::ScalarType::ComplexDouble; }
};
#endif

// this shouldn't strictly be necessary, but needs some finagling to get to work
// with "fake" C++17 if_constexpr.
template<>
struct CPPTypeAndStdComplexToScalarType<void> {
  constexpr static c10::ScalarType value() { return c10::ScalarType::Undefined; }
};

}} //namespace cppmap::detail

// `needs_dynamic_casting` compares the types expected by iterator
// (i.e. dtypes of the operands) with the actual type of the arguments
// (and returns) of func_t
template<typename func_t, int nargs=function_traits<func_t>::arity>
struct needs_dynamic_casting {
  static bool check(TensorIterator& iter) {
    using traits = function_traits<func_t>;
    using cpp_type = typename traits::template arg<nargs - 1>::type;
    using cpp_map = cppmap::detail::CPPTypeAndStdComplexToScalarType<cpp_type>;

    if (iter.input_dtype(nargs-1) != cpp_map::value()) {
      return true;
    }
    return needs_dynamic_casting<func_t, nargs - 1>::check(iter);
  }
};

template<typename func_t>
struct needs_dynamic_casting<func_t, 0> {
  static bool check(TensorIterator& iter) {
    using traits = function_traits<func_t>;
    using cpp_type = typename traits::result_type;
    using cpp_map = cppmap::detail::CPPTypeAndStdComplexToScalarType<cpp_type>;

    // we could assert output numbers are correct here, but checks
    // (including arity) are currently pushed outside of this struct.
    return c10::guts::if_constexpr<std::is_void<cpp_type>::value>([]() {
      return false;
    }, /* else */ [&]() {
      return iter.dtype(0) != cpp_map::value();
    });
  }
};

}} //namespace at::native
