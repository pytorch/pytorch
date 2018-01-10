#include "aten_dispatch.h"
#include "torch/csrc/autograd/profiler.h"
#include "torch/csrc/jit/interned_strings.h"
#include "torch/csrc/utils/functional.h"

#include <unordered_map>
#include <cstring>
#include <tuple>

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

// The packer here is carefully written not to make any unnecessary
// copies.

// pack takes the return values of aten functions and puts them into a
// refcounted list. Takes an owning reference to a tensor and steals
// that reference, adding it to the list_of_retainable output list.
// pack never operates on tensor temporaries.
void pack(list_of_retainable & outputs, Tensor&& v) {
  outputs.push_back(toRetainableSteal(std::move(v)));
}
void pack(list_of_retainable & outputs, std::vector<Tensor>&& ts) {
  for(auto& t : ts) {
    outputs.push_back(toRetainableSteal(std::move(t)));
  }
}

template<std::size_t remaining, typename... Args>
struct TuplePacker
{
  // NB: *Not* a universal reference.
  static void execute(list_of_retainable & outputs, std::tuple<Args...> && t)
  {
    // NB: The move here does not "destroy" the entire tuple, that is
    // not what std::move does; only the particular tuple index
    // processed here gets stolen.
    pack(outputs, std::get<sizeof...(Args) - remaining>(std::move(t)));
    TuplePacker<remaining - 1, Args...>::execute(outputs, std::move(t));
  }
};

template<typename... Args>
struct TuplePacker<0, Args...>
{
  static void execute(list_of_retainable & outputs, std::tuple<Args...> && t) {};
};

template<typename... Args>
void pack_list(list_of_retainable & outputs, std::tuple<Args...> && t)
{
  TuplePacker<sizeof...(Args), Args...>::execute(outputs, std::move(t));
}

template<typename... Args>
void pack_list(list_of_retainable & outputs, Tensor && t)
{
  pack(outputs, std::move(t));
}

template<typename... Args>
void pack_list(list_of_retainable & outputs, std::vector<Tensor> && t)
{
  pack(outputs, std::move(t));
}

int deviceForInputs(const list_of_retainable & inputs) {
  if(inputs.size() == 0)
    return -1;
  auto t = TensorTemporary(inputs[0]);
  return t.value().type().is_cuda() ? (int) t.value().get_device() : -1;
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
  s << n->kind().toString();
  if (tensor_vararg_fns.count(n->kind()) == 0)
    s << "-" << n->inputs().size();
  else
    s << "-*";
  std::vector<const char*> attr_names = fmap(n->attributeNames(), [](Symbol x) { return x.toString(); });
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
