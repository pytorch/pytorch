#pragma once

#include <torch/csrc/Export.h>
#include <cstdint>
#include <string>
#include <unordered_map>

namespace torch {
namespace jit {
namespace tensorexpr {

// The external functions that could be called from NNC must have the same
// signature defined by `NNCExternalFunction`.
//
// Why this signature?
// It was picked for two reasons: 1) it should be generic enough to represent
// most of the ops we might want to call, 2) it should be possible to generate a
// code for this call in LLVM codegen.
// The first 5 parameters allow to pass any number of contiguous CPU tensors in
// case we need to run aten ops (TODO: support different devices). The first
// buffer in the array is assumed to be the output buffer. We couldn't use
// `at::Tensor` (or `c10::IValue`) type there directly as it would mean that
// we'd need to declare it in LLVM codegen in LLVM IR form, which would be very
// cumbersome and hard to maintain. Note that the dimensions of all tensors are
// concatenated into a single array buf_dims. We do not need to pass its length,
// since it can be deduced from total number of buffers and their ranks.
//
// The last 2 arguments allow to pass any non-tensor arguments encoded as an
// array of int64_t values. The way they are encoded is not specified and could
// be arbitrary - whatever the most convenient for the specific bridge function
// is.
//
// The bridge functions must not throw exceptions - properly propagating them
// from the generated code is too cumbersome, and thus all calls to functions
// that could throw must be wrapped with try-catch blocks.
using NNCExternalFunction = void (*)(
    int64_t bufs_num,
    void** buf_data,
    int64_t* buf_ranks,
    int64_t* buf_dims,
    int64_t* buf_strides,
    int8_t* buf_dtypes,
    int64_t args_num,
    int64_t* extra_args);

// Return a global map "function-name" -> "function-pointer" for all registered
// in NNC external functions
TORCH_API std::unordered_map<std::string, NNCExternalFunction>&
getNNCFunctionRegistry();

// To register a new external function in NNC one needs to create an instance of
// this struct
struct RegisterNNCExternalFunction {
  RegisterNNCExternalFunction(const std::string& name, NNCExternalFunction fn) {
    getNNCFunctionRegistry()[name] = fn;
  }
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
