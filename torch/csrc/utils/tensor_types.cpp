#include "tensor_types.h"

#include <sstream>
#include <unordered_map>

#include "torch/csrc/autograd/generated/VariableType.h"
#include "torch/csrc/Exceptions.h"

using namespace at;

namespace torch { namespace utils {

static const char* backend_to_string(const at::Type& type) {
  switch (type.backend()) {
    case at::kCPU: return "torch";
    case at::kCUDA: return "torch.cuda";
    case at::kSparseCPU: return "torch.sparse";
    case at::kSparseCUDA: return "torch.cuda.sparse";
    default: throw std::runtime_error("Unimplemented backend");
  }
}

std::string type_to_string(const at::Type& type) {
  std::ostringstream ss;
  ss << backend_to_string(type) << "." << toString(type.scalarType()) << "Tensor";
  return ss.str();
}

at::Type& type_from_string(const std::string& str) {
  static std::once_flag once;
  static std::unordered_map<std::string, Type*> map;
  std::call_once(once, []() {
    for (auto type : autograd::VariableType::allTypes()) {
      map.emplace(type_to_string(*type), type);
    }
  });

  auto it = map.find(str);
  if (it == map.end()) {
    throw ValueError("invalid type: '%s'", str.c_str());
  }
  return *it->second;
}

}} // namespace torch::utils
