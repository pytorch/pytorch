#include <Python.h>

#include <torch/csrc/utils/tensor_types.h>

#include <torch/csrc/autograd/generated/VariableType.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/tensor/python_tensor.h>

#include <sstream>
#include <unordered_map>
#include <algorithm>

using namespace at;

namespace torch { namespace utils {

static const char* backend_to_string(const at::Type& type) {
  switch (type.backend()) {
    case at::Backend::CPU: return "torch";
    case at::Backend::CUDA: return "torch.cuda";
    case at::Backend::SparseCPU: return "torch.sparse";
    case at::Backend::SparseCUDA: return "torch.cuda.sparse";
    // We split complex into its own backend, but keeping it the same here for now
    case at::Backend::ComplexCPU: return "torch";
    case at::Backend::ComplexCUDA: return "torch.cuda";
    default: AT_ERROR("Unimplemented backend ", type.backend());
  }
}

std::string type_to_string(const at::Type& type, const ScalarType scalar_type) {
  std::ostringstream ss;
  ss << backend_to_string(type) << "." << toString(scalar_type) << "Tensor";
  return ss.str();
}

std::pair<at::Type*, at::ScalarType> type_from_string(const std::string& str) {
  static std::string cuda_prefix("torch.cuda.");
  static std::once_flag cpu_once;
  static std::once_flag cuda_once;
  static std::unordered_map<std::string, std::pair<at::Type*, at::ScalarType>> cpu_map;
  static std::unordered_map<std::string, std::pair<at::Type*, at::ScalarType>> cuda_map;

  const std::unordered_map<std::string, std::pair<at::Type*, at::ScalarType>>* map = nullptr;

  if (str == "torch.Tensor") {
    return std::make_pair(&torch::tensors::get_default_tensor_type(),
                          torch::tensors::get_default_scalar_type());
  }

  if (std::mismatch(cuda_prefix.begin(), cuda_prefix.end(), str.begin()).first == cuda_prefix.end()) {
    // torch.cuda. is prefix of str
    std::call_once(cuda_once, []() {
      for (auto type : autograd::VariableType::allCUDATypes()) {
        for (int s = 0; s < static_cast<int>(ScalarType::NumOptions); s++) {
          cuda_map.emplace(type_to_string(*type, static_cast<ScalarType>(s)),
                           std::make_pair(type, static_cast<ScalarType>(s)));
        }
      }
    });
    map = &cuda_map;
  } else {
    std::call_once(cpu_once, []() {
      for (auto type : autograd::VariableType::allCPUTypes()) {
        for (int s = 0; s < static_cast<int>(ScalarType::NumOptions); s++) {
          cpu_map.emplace(type_to_string(*type, static_cast<ScalarType>(s)),
                          std::make_pair(type, static_cast<ScalarType>(s)));
        }
      }
    });
    map = &cpu_map;
  }

  auto it = map->find(str);
  if (it == map->end()) {
    throw ValueError("invalid type: '%s'", str.c_str());
  }
  return it->second;
}

std::vector<std::pair<Backend, ScalarType>> all_declared_types() {
  std::vector<std::pair<Backend, ScalarType>> ret;
  // can't easily iterate over enum classes
  std::vector<Backend> backends = { Backend::CPU, Backend::CUDA, Backend::SparseCPU, Backend::SparseCUDA };
  std::vector<ScalarType> scalar_types = { ScalarType::Byte, ScalarType::Char, ScalarType::Double, ScalarType::Float,
                                           ScalarType::Int, ScalarType::Long, ScalarType::Short, ScalarType::Half,
                                           ScalarType::Bool, ScalarType::BFloat16};
  for (auto& backend : backends) {
    for (auto& scalar_type : scalar_types) {
      // there is no sparse bool type.
      if (scalar_type == ScalarType::Bool && (backend == Backend::SparseCUDA || backend == Backend::SparseCPU)) {
        continue;
      }
      ret.emplace_back(std::make_pair(backend, scalar_type));
    }
  }

  return ret;
}

}} // namespace torch::utils
