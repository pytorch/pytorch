#include <Python.h>

#include "tensor_types.h"

#include "torch/csrc/autograd/generated/VariableType.h"
#include "torch/csrc/Exceptions.h"
#include "torch/csrc/tensor/python_tensor.h"

#include <sstream>
#include <unordered_map>

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

  if (str == "torch.Tensor") {
    return torch::tensor::get_default_tensor_type();
  }

  auto it = map.find(str);
  if (it == map.end()) {
    throw ValueError("invalid type: '%s'", str.c_str());
  }
  return *it->second;
}

std::vector<std::pair<Backend, ScalarType>> all_declared_types() {
  std::vector<std::pair<Backend, ScalarType>> ret;
  // can't easily iterate over enum classes
  std::vector<Backend> backends = { Backend::CPU, Backend::CUDA, Backend::SparseCPU, Backend::SparseCUDA };
  std::vector<ScalarType> scalar_types = { ScalarType::Byte, ScalarType::Char, ScalarType::Double, ScalarType::Float,
                                           ScalarType::Int, ScalarType::Long, ScalarType::Short, ScalarType::Half};
  for (auto& backend : backends) {
    for (auto& scalar_type : scalar_types) {
      // there is no sparse half types.
      if (scalar_type == ScalarType::Half && (backend == Backend::SparseCUDA || backend == Backend::SparseCPU)) {
        continue;
      }
      ret.emplace_back(std::make_pair(backend, scalar_type));
    }
  }

  return ret;
}

}} // namespace torch::utils
