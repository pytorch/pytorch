#pragma once

#include <torch/headeronly/core/Dispatch_v2.h>

// Get AT_DISPATCH_SWITCH and AT_DISPATCH_CASE:
#include <ATen/Dispatch.h>

// This is a new implementation of the AT_DISPATCH macro family from
// ATen/Dispatch.h
//
// The intended usage is:
//
//  ScalarType scalar_type;
//
//  AT_DISPATCH_V2(
//    scalar_type,
//    "debug string",
//    AT_WRAP([&] {
//      ... code to specialize with scalar_t ...
//    }),
//    kHalf,
//    AT_EXPAND(AT_ALL_TYPES),
//    ... as many types arguments as needed ...
//  )
//
// For example, given an old style:
//
//  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
//    kComplexHalf,
//    kHalf,
//    self.scalar_type(),
//    "_local_scalar_dense_cpu",
//    [&] {
//      scalar_t value = *self.data_ptr<scalar_t>();
//      r = Scalar(value);
//    }
//  )
//
// You now write:
//
//  AT_DISPATCH_V2(
//    self.scalar_type(),
//    "_local_scalar_dense_cpu",
//    AT_WRAP([&] {
//      scalar_t value = *self.data_ptr<scalar_t>();
//      r = Scalar(value);
//    }),
//    AT_EXPAND(AT_ALL_TYPES),
//    AT_EXPAND(AT_COMPLEX_TYPES),
//    kComplexHalf,
//    kHalf,
//  )
//
// Notably, it sports the following improvements:
//
//  - It is not necessary to specify the arity (e.g.,
//    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND{2,3,4,...})
//    when using the macro
//
//  - It is not necessary to specify each dtype individually; if
//    there is a set of related dtypes and you want to dispatch
//    over all of them, you can simply say, e.g., AT_EXPAND(AT_INTEGRAL_TYPES)
//    in your argument list.
//
// However, you must remember to wrap the payload body in AT_WRAP, or commas
// inside your lambda will be improperly handled.  Furthermore, if you more
// entries to ScalarType than can be supported by this macro, it will fail
// with an obscure error (due to attempting to concatenate AT_AP with
// something that is not a number).
//
// The implementation strategy is to use the count arguments trick
// (e.g., as described in https://stackoverflow.com/a/2124385/23845)
// to discover how many dtypes have been passed, and then dispatch to a
// hand-written macro for each arity that applies as many DISPATCH_CASE as
// necessary.  The hand-written macros can be regenerated for other arities
// with the script below.
//
// There is some delicacy in the implementation in controlling when
// macro expansion occurs, mediated with AT_EXPAND and AT_GUARD.  I mostly
// relied on GPT4 to help me get it right.

// Helper macros, kept for BC:
#define AT_AP_VAR(N, T, ...) \
  AT_EXPAND(AT_CONCAT(AT_AP, AT_NUM_ARGS(__VA_ARGS__))(AT_WRAP(N), __VA_ARGS__))

// See documentation above
#define AT_DISPATCH_V2(TYPE, NAME, BODY, ...) \
  AT_DISPATCH_V2_TMPL(                        \
      AT_DISPATCH_SWITCH,                     \
      AT_DISPATCH_CASE,                       \
      TYPE,                                   \
      NAME,                                   \
      AT_WRAP(BODY),                          \
      __VA_ARGS__)
