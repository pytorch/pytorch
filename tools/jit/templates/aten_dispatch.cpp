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
struct TensorTemporary {
  explicit TensorTemporary(at::Retainable * impl)
  : temp(static_cast<at::TensorImpl*>(impl), false /* do not retain*/) {}
  const at::Tensor & value() {
    return temp;
  }
  ~TensorTemporary() {
    // don't reduce the refcount on deletion
    temp.detach();
  }
private:
  at::Tensor temp;
};

// same list of Tensors that does not alter the refcount on acquisition or
// release of the refcount temporaries, only used rarely (e.g. for cat)
struct TensorTemporaryList {
  explicit TensorTemporaryList(const list_of_retainable & ts) {
    tensors.reserve(ts.size());
    for(auto & t : ts) {
      tensors.push_back(at::Tensor(static_cast<at::TensorImpl*>(t), false /*do not retain*/));
    }
  }
  // TensorTemporaryList only exposes a TensorList,
  // not its underlying std::vector<at::Tensor>.
  // This ArrayRef has the desired semantics: if you get out an at::Tensor from it,
  // the refcount is bumped;
  // if you take a reference, it is only guaranteed to stay live as long as the ArrayRef is live,
  operator TensorList() const {
    return tensors;
  }
  ~TensorTemporaryList() {
    // we didnt retain the tensors when we created the list
    // so make sure we don't release them when we free it
    for(auto & t : tensors) {
      t.detach();
    }
  }
private:
  std::vector<at::Tensor> tensors;
};

using list_of_retainable = std::vector<at::Retainable*>;

// pack list takes the return values of aten functions and puts them into a
// refcounted list. Each pack_list variant takes a Tensor by value, ensuring
// it has a owning reference and then that reference is stolen ad added to the
// list_of_retainable output list.
// pack_list never operates on tensor temporaries.
void pack_list(list_of_retainable & outputs, Tensor v) { outputs.push_back(toRetainableSteal(std::move(v))); }
void pack_list(list_of_retainable & outputs, std::vector<Tensor> && ts) {
  outputs.reserve(ts.size());
  for(auto & t : ts) {
    outputs.push_back(toRetainableSteal(std::move(t)));
  }
}
void pack_list(list_of_retainable & outputs, std::tuple<Tensor, Tensor> v) {
  outputs.push_back(toRetainableSteal(std::move(std::get<0>(v))));
  outputs.push_back(toRetainableSteal(std::move(std::get<1>(v))));
}
void pack_list(list_of_retainable & outputs, std::tuple<Tensor, Tensor, Tensor> v) {
  outputs.push_back(toRetainableSteal(std::move(std::get<0>(v))));
  outputs.push_back(toRetainableSteal(std::move(std::get<1>(v))));
  outputs.push_back(toRetainableSteal(std::move(std::get<2>(v))));
}
void pack_list(list_of_retainable & outputs, std::tuple<Tensor, Tensor, Tensor, Tensor> v) {
  outputs.push_back(toRetainableSteal(std::move(std::get<0>(v))));
  outputs.push_back(toRetainableSteal(std::move(std::get<1>(v))));
  outputs.push_back(toRetainableSteal(std::move(std::get<2>(v))));
  outputs.push_back(toRetainableSteal(std::move(std::get<3>(v))));
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
