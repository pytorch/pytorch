#include "aten_dispatch.h"
#include "torch/csrc/autograd/profiler.h"
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

// a temporary Tensor that does not alter the refcount of impl on
// acquisition or release, avoids any refcounting in dispatch functions
struct TensorTemporary : public at::Tensor {
  explicit TensorTemporary(at::RefCounted * impl)
  : at::Tensor(static_cast<at::TensorImpl*>(impl), false /* do not retain*/) {}
  ~TensorTemporary() {
    detach(); // reset
  }
};

// same thing but creates a tensor list, only used rarely (e.g. for cat)
struct TensorTemporaryList {
  explicit TensorTemporaryList(const refcounted_list & ts) {
    tensors.reserve(ts.size());
    for(auto & t : ts) {
      tensors.push_back(at::Tensor(static_cast<at::TensorImpl*>(t), false /*do not retain*/));
    }
  }
  operator TensorList() const {
    return tensors;
  }
  ~TensorTemporaryList() {
    for(auto & t : tensors) {
      t.detach();
    }
  }
private:
  std::vector<at::Tensor> tensors;
};

using refcounted_list = std::vector<at::RefCounted*>;
void pack_list(refcounted_list & outputs, Tensor v) { outputs.push_back(v.detach()); }
void pack_list(refcounted_list & outputs, Scalar v) { outputs.push_back(v.toTensor().detach()); }
void pack_list(refcounted_list & outputs, std::vector<Tensor> && ts) {
  outputs.reserve(ts.size());
  for(auto & t : ts) {
    outputs.push_back(t.detach());
  }
}
void pack_list(refcounted_list & outputs, std::tuple<Tensor, Tensor> v) {
  outputs.push_back(std::get<0>(v).detach());
  outputs.push_back(std::get<1>(v).detach());
}
void pack_list(refcounted_list & outputs, std::tuple<Tensor, Tensor, Tensor> v) {
  outputs.push_back(std::get<0>(v).detach());
  outputs.push_back(std::get<1>(v).detach());
  outputs.push_back(std::get<2>(v).detach());
}

// A list of functions taking TensorList arguments (where we can't use
// the number of inputs to choose an overload).
std::unordered_set<Symbol> tensor_vararg_fns = {
  kcat,
};

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
  s << symbolToString(n->kind());
  if (tensor_vararg_fns.count(n->kind()) == 0)
    s << "-" << n->inputs().size();
  else
    s << "-*";
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
