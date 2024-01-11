#pragma once

// Element-wise invoke. Invoking data load (`data`) and custom function (`f`)
// for an element specified by a logical index (`i`). Return the result of
// custom function. Custom function is a user-defined functor for
// scalar calculation.

#include <type_traits>

#include <ATen/detail/FunctionTraits.h>
#include <c10/macros/Macros.h>
#include <c10/core/DynamicCast.h>
#include <c10/core/ScalarType.h>

template <typename traits, typename func_t, typename index_t, size_t... INDEX>
C10_HOST_DEVICE typename traits::result_type invoke_impl(
    const func_t& f,
    char* const C10_RESTRICT data[],
    const index_t strides[],
    int i,
    std::index_sequence<INDEX...>) {
  (void)strides;
  (void)i;
  return f(c10::load<typename traits::template arg<INDEX>::type>(
      data[INDEX] + i * strides[INDEX])...);
}

template <
    typename func_t,
    typename index_t,
    typename traits = function_traits<func_t>>
C10_HOST_DEVICE typename traits::result_type invoke(
    const func_t& f,
    char* const C10_RESTRICT data[],
    const index_t strides[],
    int i) {
  using Indices = std::make_index_sequence<traits::arity>;
  return invoke_impl<traits>(f, data, strides, i, Indices{});
}

template <typename traits, typename func_t, typename index_t, size_t... I>
C10_HOST_DEVICE typename traits::result_type invoke_impl(
    const func_t& f,
    char* const C10_RESTRICT data[],
    const index_t strides[],
    const c10::ScalarType dtypes[],
    int i,
    std::index_sequence<I...>) {
  (void)strides;
  (void)i;
  return f(c10::fetch_and_cast<typename traits::template arg<I>::type>(
      dtypes[I], data[I] + i * strides[I])...);
}

template <
    typename func_t,
    typename index_t,
    typename traits = function_traits<func_t>>
C10_HOST_DEVICE typename traits::result_type invoke(
    const func_t& f,
    char* const C10_RESTRICT data[],
    const index_t strides[],
    const c10::ScalarType dtypes[],
    int i) {
  using Indices = std::make_index_sequence<traits::arity>;
  return invoke_impl<traits>(f, data, strides, dtypes, i, Indices{});
}
