#include "torch/csrc/jit/operator.h"
#include "torch/csrc/jit/custom_operator.h"

#include "torch/csrc/autograd/profiler.h"
#include "torch/csrc/jit/interned_strings.h"

#include "torch/csrc/utils/functional.h"
#include "torch/csrc/variable_tensor_functions.h"
#include "torch/csrc/autograd/generated/variable_factories.h"

#include <ATen/ATen.h>

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

namespace torch { namespace jit {

using autograd::Variable;
using autograd::variable_list;
using at::Scalar;
using at::Tensor;
using at::TensorOptions;
using at::DeviceGuard;

namespace {

int deviceForInputs(Stack & stack, size_t N) {
  if(N == 0)
    return -1;
  auto t = (stack.end() - N)->toTensor();
  return t.type().is_cuda() ? (int) t.get_device() : -1;
}

template<size_t N>
std::array<bool, N> as_bool_array(at::ArrayRef<int64_t> vec) {
  std::array<bool, N> res;
  JIT_ASSERT(vec.size() == N);
  std::copy(vec.begin(), vec.end(), res.begin());
  return res;
}

at::Device as_device(ArrayRef<int64_t> elements) {
  return at::Device(static_cast<at::Device::Type>(elements[0]), elements[1]);
}

RegisterOperators reg({
  ${constructors}
});

} // anon namespace


}} // namespace torch::jit
