#include <Python.h>

#include <torch/csrc/utils/tensor_types.h>

#include <ATen/Context.h>
#include <ATen/Formatting.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/autograd/generated/VariableType.h>
#include <torch/csrc/tensor/python_tensor.h>

#include <c10/util/CallOnce.h>

#include <algorithm>
#include <sstream>
#include <unordered_map>

using namespace at;

namespace torch::utils {

static const char* parse_privateuseone_backend(bool is_sparse = false) {
  static std::string backend_name = "torch." + get_privateuse1_backend();
  static std::string sparse_backend_name = backend_name + ".sparse";
  return is_sparse == false ? backend_name.c_str()
                            : sparse_backend_name.c_str();
}

const char* backend_to_string(const at::Backend& backend) {
  switch (backend) {
    case at::Backend::CPU:
      return "torch";
    case at::Backend::CUDA:
      return "torch.cuda";
    case at::Backend::XPU:
      return "torch.xpu";
    case at::Backend::IPU:
      return "torch.ipu";
    case at::Backend::SparseCPU:
      return "torch.sparse";
    case at::Backend::SparseCUDA:
      return "torch.cuda.sparse";
    case at::Backend::SparseXPU:
      return "torch.xpu.sparse";
    case at::Backend::QuantizedCPU:
      return "torch.quantized";
    case at::Backend::HPU:
      return "torch.hpu";
    case at::Backend::MPS:
      return "torch.mps";
    case at::Backend::MTIA:
      return "torch.mtia";
    case at::Backend::PrivateUse1:
      return parse_privateuseone_backend();
    case at::Backend::SparsePrivateUse1:
      return parse_privateuseone_backend(true);
    case at::Backend::Lazy:
      return "torch.lazy";
    case at::Backend::XLA:
      return "torch.xla";
    case at::Backend::Meta:
      return "torch.meta";
    default:
      TORCH_CHECK(false, "Unimplemented backend ", backend);
  }
}

std::string options_to_string(const at::TensorOptions& options) {
  std::ostringstream ss;
  ss << backend_to_string(options.backend()) << "."
     << toString(at::typeMetaToScalarType(options.dtype())) << "Tensor";
  return ss.str();
}

std::string type_to_string(const at::DeprecatedTypeProperties& type) {
  std::ostringstream ss;
  ss << backend_to_string(type.backend()) << "." << toString(type.scalarType())
     << "Tensor";
  return ss.str();
}

at::TensorOptions options_from_string(const std::string& str) {
  static std::string cuda_prefix("torch.cuda.");
  static std::string xpu_prefix("torch.xpu.");
  static std::string privateUser_prefix(
      std::string(parse_privateuseone_backend()) + ".");
  static c10::once_flag cpu_once;
  static c10::once_flag cuda_once;
  static c10::once_flag xpu_once;
  static c10::once_flag privateUser1_once;
  static std::unordered_map<std::string, at::DeprecatedTypeProperties*> cpu_map;
  static std::unordered_map<std::string, at::DeprecatedTypeProperties*> xpu_map;
  static std::unordered_map<std::string, at::DeprecatedTypeProperties*>
      cuda_map;
  static std::unordered_map<std::string, at::DeprecatedTypeProperties*>
      privateUser1_map;

  const std::unordered_map<std::string, at::DeprecatedTypeProperties*>* map =
      nullptr;

  if (str == "torch.Tensor") {
    auto backend =
        dispatchKeyToBackend(torch::tensors::get_default_dispatch_key());
    auto scalar_type = torch::tensors::get_default_scalar_type();
    return getDeprecatedTypeProperties(backend, scalar_type).options();
  }

  if (std::mismatch(cuda_prefix.begin(), cuda_prefix.end(), str.begin())
          .first == cuda_prefix.end()) {
    // torch.cuda. is prefix of str
    c10::call_once(cuda_once, []() {
      for (auto type : autograd::VariableType::allCUDATypes()) {
        cuda_map.emplace(type_to_string(*type), type);
      }
    });
    map = &cuda_map;
  } else if (
      std::mismatch(xpu_prefix.begin(), xpu_prefix.end(), str.begin()).first ==
      xpu_prefix.end()) {
    // torch.xpu. is prefix of str
    c10::call_once(xpu_once, []() {
      for (auto type : autograd::VariableType::allXPUTypes()) {
        xpu_map.emplace(type_to_string(*type), type);
      }
    });
    map = &xpu_map;
  } else if (
      std::mismatch(
          privateUser_prefix.begin(), privateUser_prefix.end(), str.begin())
          .first == privateUser_prefix.end()) {
    // torch.foo. foo is privateUser1 name
    c10::call_once(privateUser1_once, []() {
      for (auto type : autograd::VariableType::allPrivateUser1Types()) {
        privateUser1_map.emplace(type_to_string(*type), type);
      }
    });
    map = &privateUser1_map;
  } else {
    c10::call_once(cpu_once, []() {
      for (auto type : autograd::VariableType::allCPUTypes()) {
        cpu_map.emplace(type_to_string(*type), type);
      }
    });
    map = &cpu_map;
  }

  auto it = map->find(str);
  TORCH_CHECK_VALUE(it != map->end(), "invalid type: '", str, "'");
  return it->second->options();
}

std::vector<std::pair<Backend, ScalarType>> all_declared_types() {
  std::vector<std::pair<Backend, ScalarType>> ret;

  // NOTE: Do not add more types here. This list controls the creation
  // of legacy tensor types e.g. torch.cuda.FloatTensor which are
  // maintained for backwards-compatibility only.
  auto backends = {
      Backend::CPU, Backend::CUDA, Backend::SparseCPU, Backend::SparseCUDA};
  auto scalar_types = {
      ScalarType::Byte,
      ScalarType::Char,
      ScalarType::Double,
      ScalarType::Float,
      ScalarType::Int,
      ScalarType::Long,
      ScalarType::Short,
      ScalarType::Half,
      ScalarType::Bool,
      ScalarType::BFloat16};

  for (auto& backend : backends) {
    for (auto& scalar_type : scalar_types) {
      // there is no sparse bool type.
      if (scalar_type == ScalarType::Bool &&
          (backend == Backend::SparseCUDA || backend == Backend::SparseCPU)) {
        continue;
      }
      ret.emplace_back(backend, scalar_type);
    }
  }

  return ret;
}

} // namespace torch::utils
