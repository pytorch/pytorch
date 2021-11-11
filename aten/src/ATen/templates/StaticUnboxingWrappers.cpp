#include "torch/csrc/jit/runtime/operator.h"
#include "torch/csrc/jit/runtime/custom_operator.h"
#include "torch/csrc/jit/runtime/register_ops_utils.h"

#include "torch/csrc/autograd/profiler.h"
#include "torch/csrc/autograd/generated/variable_factories.h"

#include <ATen/ATen.h>
#include <ATen/core/functional.h>
#include <ATen/core/interned_strings.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// ${generated_comment}

// NOTE [Sharded File]: This file is generated in a sharded fashion to speed up
// incremental rebuilds. See the comment at the top of
// templates/VariableType.cpp for an analogous, in-depth discussion.
//
// Note that unlike VariableType.cpp, when sharding this file we take
// care to generate all overloads of a particular name in a single
// file and in a particular order. See gen_jit_dispatch.py for
// details.

namespace torch { namespace jit {

using autograd::Variable;
using autograd::variable_list;
using at::Scalar;
using at::ScalarType;
using at::Tensor;
using at::TensorOptions;
using at::DeviceGuard;

using ::c10::fmap;
using ::c10::filter;

namespace {

// TODO: remove the toOptionalTensor and toListOfOptionalTensor
// when we remove the undefined tensor semantic from TH

// XXX: This function is to specialize IValue for tensor type in
// interpreter, it should only be used in this file
at::Tensor toOptionalTensor(const IValue& v) {
  if (v.isNone()) {
    return at::Tensor();
  }
  return v.toTensor();
}

// XXX: This function is to specialize IValue for list of optional
// tensor type in interpreter, it should only be used in this file
std::vector<Tensor> toListOfOptionalTensor(const IValue& v) {
  // v is a list of optional tensor, loop over as generic list
  auto vlist = v.toList();
  std::vector<Tensor> res;

  for (const IValue &v: vlist) {
    res.emplace_back(toOptionalTensor(v));
  }
  return res;
}

template<typename T, size_t N>
std::array<T, N> as_array(const c10::List<c10::IValue>& list) {
  std::array<T, N> res;
  AT_ASSERT(list.size() == N);
  std::vector<T> vec;
  for (c10::IValue elem : list) {
    vec.push_back(elem.to<T>());
  }
  std::copy(vec.begin(), vec.end(), res.begin());
  return res;
}

RegisterOperators reg({
    OperatorGenerator(
        TORCH_SELECTIVE_SCHEMA("aten::get_device(Tensor self) -> int"),
        [](Stack & stack) {
          RECORD_FUNCTION("get_device", std::vector<c10::IValue>());
          auto result = at::get_device(
              (std::move(peek(stack, 0, 1))).toTensor()
          );
          drop(stack, 1);
          pack(stack, std::move(result));
        },
        aliasAnalysisFromSchema()
    ),
    OperatorGenerator(
        TORCH_SELECTIVE_SCHEMA("aten::storage_offset(Tensor self) -> int"),
        [](Stack & stack) {
          RECORD_FUNCTION("storage_offset", std::vector<c10::IValue>());
          auto result = ((std::move(peek(stack, 0, 1))).toTensor()).storage_offset();
          drop(stack, 1);
          pack(stack, std::move(result));
        },
        aliasAnalysisFromSchema()
    ),

    // Generated operators
    ${unboxed_ops}
});

} // anon namespace


}} // namespace torch::jit
