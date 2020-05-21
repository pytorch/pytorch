#pragma once

#include <type_traits>
#include <c10/core/ScalarType.h>
#include <c10/util/C++17.h>
#include <ATen/detail/FunctionTraits.h>
#include <ATen/native/TensorIterator.h>

// This file includes utilties for dynamic_casting done by TensorIterator, see CUDALoops.cuh and Loops.h.

// dynamic_casting handles when the types expected by the iterator do not match the types of the arguments
// to the function that is being called.
// On CUDA, the cast is currently pushed down into the kernel (for performance reasons).
// On CPU, there is currently an internal assert that a dynamic_cast is not needed.

namespace at { namespace native {

// `needs_dynamic_casting` compares the types expected by iterator
// (i.e. dtypes of the operands) with the actual type of the arguments
// (and returns) of func_t
template<typename func_t, int nargs=function_traits<func_t>::arity>
struct needs_dynamic_casting {
  static bool check(TensorIterator& iter) {
    using traits = function_traits<func_t>;
    if (iter.input_dtype(nargs-1) != c10::impl::CPPTypeToScalarType<typename traits::template arg<nargs - 1>::type>::value) {
      return true;
    }
    return needs_dynamic_casting<func_t, nargs - 1>::check(iter);
  }
};

template<typename func_t>
struct needs_dynamic_casting<func_t, 0> {
  static bool check(TensorIterator& iter) {
    using traits = function_traits<func_t>;

    // we could assert output numbers are correct here, but checks
    // (including arity) are currently pushed outside of this struct.
    return c10::guts::if_constexpr<std::is_void<typename traits::result_type>::value>(
      [&]() { return false; },
      [&]() { return iter.dtype(0) != c10::impl::CPPTypeToScalarType<typename traits::result_type>::value;}
    );
  }
};

}} //namespace at::native
