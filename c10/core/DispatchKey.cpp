#include <c10/core/DispatchKey.h>

namespace c10 {

const char* toString(DispatchKey t) {
  switch (t) {
    case DispatchKey::Undefined:
      return "Undefined";
    case DispatchKey::CPUTensorId:
      return "CPUTensorId";
    case DispatchKey::CUDATensorId:
      return "CUDATensorId";
    case DispatchKey::SparseCPUTensorId:
      return "SparseCPUTensorId";
    case DispatchKey::SparseCUDATensorId:
      return "SparseCUDATensorId";
    case DispatchKey::MKLDNNTensorId:
      return "MKLDNNTensorId";
    case DispatchKey::OpenGLTensorId:
      return "OpenGLTensorId";
    case DispatchKey::OpenCLTensorId:
      return "OpenCLTensorId";
    case DispatchKey::IDEEPTensorId:
      return "IDEEPTensorId";
    case DispatchKey::HIPTensorId:
      return "HIPTensorId";
    case DispatchKey::SparseHIPTensorId:
      return "SparseHIPTensorId";
    case DispatchKey::MSNPUTensorId:
      return "MSNPUTensorId";
    case DispatchKey::XLATensorId:
      return "XLATensorId";
    case DispatchKey::MkldnnCPUTensorId:
      return "MkldnnCPUTensorId";
    case DispatchKey::VulkanTensorId:
      return "VulkanTensorId";
    case DispatchKey::QuantizedCPUTensorId:
      return "QuantizedCPUTensorId";
    case DispatchKey::VariableTensorId:
      return "VariableTensorId";
    case DispatchKey::BackendSelect:
      return "BackendSelect";
    case DispatchKey::TESTING_ONLY_GenericModeTensorId:
      return "TESTING_ONLY_GenericModeTensorId";
    case DispatchKey::AutocastTensorId:
      return "AutocastTensorId";
    case DispatchKey::TESTING_ONLY_GenericWrapperTensorId:
      return "TESTING_ONLY_GenericWrapperTensorId";
    default:
      return "UNKNOWN_TENSOR_TYPE_ID";
  }
}

std::ostream& operator<<(std::ostream& str, DispatchKey rhs) {
  return str << toString(rhs);
}

} // namespace c10
