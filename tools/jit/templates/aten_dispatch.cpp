#include "aten_dispatch.h"
#include "torch/csrc/jit/interned_strings.h"
#include "torch/csrc/utils/functional.h"

#include <unordered_map>
#include <cstring>

// ${generated_comment}

namespace torch { namespace jit {

using autograd::Variable;
using autograd::variable_list;
using at::Scalar;
using at::Tensor;
using at::IntList;
using at::TensorList;
using operator_constructor = std::function<TensorOp(jit::Node*)>;

namespace {

variable_list pack_list(Tensor v) { return { std::move(v) }; }
variable_list pack_list(Scalar v) { return { v.toTensor() }; }
variable_list pack_list(std::vector<Tensor> t) { return fmap<Variable>(t); }
variable_list pack_list(std::tuple<Tensor, Tensor> v) {
  return { std::move(std::get<0>(v)), std::move(std::get<1>(v)) };
}
variable_list pack_list(std::tuple<Tensor, Tensor, Tensor> v) {
  return { std::get<0>(v), std::get<1>(v), std::get<2>(v) };
}

std::vector<Tensor> as_tensor_list(const variable_list& vars) {
  return fmap(vars, [](Variable v) { return static_cast<Tensor>(v); });
}

template<size_t N>
std::array<bool, N> as_bool_array(const std::vector<int64_t>& vec) {
  std::array<bool, N> res;
  JIT_ASSERT(vec.size() == N);
  std::copy(vec.begin(), vec.end(), res.begin());
  return res;
}

std::unordered_map<std::string, operator_constructor> constructors = {
  ${constructors}
};

std::string getDescriptor(jit::Node* n) {
  std::stringstream s;
  s << symbolToString(n->kind()) << "-" << n->inputs().size();
  std::vector<const char*> attr_names = fmap(n->attributeNames(), &symbolToString);
  std::sort(attr_names.begin(), attr_names.end(), [](const char *a, const char *b) {
    return std::strcmp(a, b) < 0;
  });
  for (const auto & name : attr_names)
    s << "-" << name;
  return s.str();
}

} // anonymous namespace

TensorOp getTensorOp(jit::Node* n) {
  auto signature = getDescriptor(n);
  try {
    return constructors.at(signature)(n);
  } catch (std::out_of_range &e) {
    throw std::runtime_error("Unsupported op descriptor: " + signature + ". "
                             "File a bug report.");
  }
};

}} // namespace torch::jit
