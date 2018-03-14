#include "aten_dispatch.h"
#include "torch/csrc/autograd/profiler.h"
#include "torch/csrc/jit/interned_strings.h"
#include "torch/csrc/jit/tensor_conversions.h"
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

namespace {

// The packer here is carefully written not to make any unnecessary
// copies.

// pack takes the return values of aten functions pushes them onto the stack
template<typename T>
void pack(Stack & stack, T&& v) {
  stack.push_back(as_tensor(std::move(v)));
}
template<>
void pack(Stack & stack, Tensor&& v) {
  stack.push_back(std::move(v));
}
template<>
void pack(Stack & stack, std::vector<Tensor>&& ts) {
  for(auto& t : ts) {
    stack.push_back(std::move(t));
  }
}

template<std::size_t remaining, typename... Args>
struct TuplePacker
{
  // NB: *Not* a universal reference.
  static void execute(Stack & stack, std::tuple<Args...> && t)
  {
    // NB: The move here does not "destroy" the entire tuple, that is
    // not what std::move does; only the particular tuple index
    // processed here gets stolen.
    pack(stack, std::get<sizeof...(Args) - remaining>(std::move(t)));
    TuplePacker<remaining - 1, Args...>::execute(stack, std::move(t));
  }
};

template<typename... Args>
struct TuplePacker<0, Args...>
{
  static void execute(Stack & stack, std::tuple<Args...> && t) {};
};

template<typename... Args>
void pack(Stack & stack, std::tuple<Args...> && t) {
  TuplePacker<sizeof...(Args), Args...>::execute(stack, std::move(t));
}

int deviceForInputs(Stack & stack, size_t N) {
  if(N == 0)
    return -1;
  auto & t = *(stack.end() - N);
  return t.type().is_cuda() ? (int) t.get_device() : -1;
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

ConstructorsMap constructors = {
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

ConstructorsMap::iterator findTensorOp(jit::Node* n) {
  auto signature = getDescriptor(n);
  return constructors.find(signature);
}
bool hasTensorOp(jit::Node* n) {
  return constructors.end() != findTensorOp(n);
}
TensorOp getTensorOp(jit::Node* n) {
  auto itr = findTensorOp(n);
  if (itr == constructors.end()) {
    throw std::runtime_error(
        "Unsupported op descriptor: " + getDescriptor(n) +
        ". "
        "File a bug report.");
  }
  return itr->second(n);
}

}} // namespace torch::jit
