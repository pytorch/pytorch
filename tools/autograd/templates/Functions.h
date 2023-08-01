#pragma once

// ${generated_comment}

#include <ATen/ATen.h>
#include <ATen/core/functional.h>
#include <ATen/TensorGeometry.h>

#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/autograd/saved_variable.h"
#include <torch/csrc/Export.h>

#include <c10/core/SymIntArrayRef.h>

namespace torch { namespace autograd { namespace generated {

using at::Scalar;
using at::Tensor;
using at::IntArrayRef;
using at::ArrayRef;
using at::Type;
using at::TensorGeometry;
using at::ScalarType;
using c10::optional;
using c10::fmap;

inline std::vector<Tensor> unpack_list(at::ArrayRef<SavedVariable> xs) {
  // NB: we must explicitly do the conversion in the lambda, otherwise template
  // deduction will give a Tensor of Variable which is not convertible
  return fmap(xs, [](const SavedVariable& x) {
    return static_cast<Tensor>(x.unpack());
  });
}

inline c10::List<c10::optional<Tensor>> unpack_opt_list(at::ArrayRef<SavedVariable> xs) {
  torch::List<c10::optional<Tensor>> result;
  result.reserve(xs.size());
  for (const SavedVariable& v : xs) {
    auto var = v.unpack();
    result.push_back(var.defined() ? c10::optional<Tensor>(var) : c10::nullopt);
  }
  return result;
}

struct TypeAndSize {
  TypeAndSize() : options(at::TensorOptions()) {}
  /* implicit */
  TypeAndSize(const Tensor & t)
    : sym_sizes(t.sym_sizes().vec())
    , options(t.options()) {}

  Tensor zeros() { return at::zeros_symint(sym_sizes, options); }

  std::vector<c10::SymInt> sym_sizes;
  at::TensorOptions options;
};

${autograd_function_declarations}

}}} // namespace torch::autograd::generated
