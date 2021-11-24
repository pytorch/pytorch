#include "CodegenFunctions.h"
#include "Functions.h"

#include <ATen/Tensor.h>
#include <ATen/core/functional.h>
#include <ATen/core/interned_strings.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/stack.h>

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
namespace at {
namespace unboxing {
using at::Scalar;
using at::ScalarType;
using at::Tensor;
using at::TensorOptions;
using at::DeviceGuard;

using ::c10::fmap;
using ::c10::filter;
using torch::jit::peek;
using torch::jit::drop;
using torch::jit::pack;
using torch::jit::pop;

namespace {

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
}  // namespace
// Generated function declaration
${definitions}

} // namespace unboxing
} // namespace at
