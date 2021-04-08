#pragma once

#include <ATen/DimVector.h>
#include <c10/core/TensorOptions.h>
#include <ATen/core/Dimname.h>

namespace at {

class Tensor;

namespace impl {

// Use this to define the prototype for a meta function.  There are two
// versions; one that takes one argument (just the operator name), or FUNC2
// variant that takes two arguments (operator name and overload name).
//
// Example usage:
//
//    TORCH_META_FUNC2(add, Tensor) (
//      const Tensor& self, const Tensor& other
//    ) {
//      ... compute sizes and options ...
//      set_output(sizes, options);
//    }
//
#define TORCH_META_FUNC(name) void name::meta
#define TORCH_META_FUNC2(name, overload) void name##_##overload::meta

// Use this to define the prototype for an implementation.  This takes only
// one argument, which is the name of the dispatch key entry you're
// implementing.
//
// Example usage:
//
//    TORCH_IMPL_FUNC(add_cpu) (
//      Tensor& result, const Tensor& self, const Tensor& other
//    ) {
//      ... do the actual implementation ...
//    }
//
#define TORCH_IMPL_FUNC(name) void structured_##name::impl

// Base class for all structured kernel classes.  The set_output virtual
// method is varied depending whether or not the operator is
// functional/out/inplace, and could also be specialized for CPU/CUDA/etc
// (although presently it isn't).
//
// A notable subclass of this interface is TensorIteratorBase.
struct TORCH_API MetaBase {
  virtual void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides, TensorOptions options, DimnameList names) = 0;
  virtual const Tensor& maybe_get_output(int64_t output_idx) = 0;
  void set_output(IntArrayRef sizes, TensorOptions options) {
    set_output(0, sizes, {}, options, {});
  }
  void set_output(int64_t output_idx, IntArrayRef sizes, TensorOptions options) {
    set_output(output_idx, sizes, {}, options, {});
  }
  // Returns a reference to an undefined tensor if there is no presupplied
  // output
  const Tensor& maybe_get_output() { return maybe_get_output(0); }
  virtual ~MetaBase() {}
};

} // namespace impl

} // namespace at
