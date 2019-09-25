#include "c10/core/TensorTypeId.h"

namespace c10 {

const char* toString(TensorTypeId t) {
  switch (t) {
    case TensorTypeId::UndefinedTensorId:
      return "UndefinedTensorId";
    case TensorTypeId::CPUTensorId:
      return "CPUTensorId";
    case TensorTypeId::CUDATensorId:
      return "CUDATensorId";
    case TensorTypeId::SparseCPUTensorId:
      return "SparseCPUTensorId";
    case TensorTypeId::SparseCUDATensorId:
      return "SparseCUDATensorId";
    case TensorTypeId::MKLDNNTensorId:
      return "MKLDNNTensorId";
    case TensorTypeId::OpenGLTensorId:
      return "OpenGLTensorId";
    case TensorTypeId::OpenCLTensorId:
      return "OpenCLTensorId";
    case TensorTypeId::IDEEPTensorId:
      return "IDEEPTensorId";
    case TensorTypeId::HIPTensorId:
      return "HIPTensorId";
    case TensorTypeId::SparseHIPTensorId:
      return "SparseHIPTensorId";
    case TensorTypeId::MSNPUTensorId:
      return "MSNPUTensorId";
    case TensorTypeId::XLATensorId:
      return "XLATensorId";
    case TensorTypeId::MkldnnCPUTensorId:
      return "MkldnnCPUTensorId";
    case TensorTypeId::QuantizedCPUTensorId:
      return "QuantizedCPUTensorId";
    case TensorTypeId::ComplexCPUTensorId:
      return "ComplexCPUTensorId";
    case TensorTypeId::ComplexCUDATensorId:
      return "ComplexCUDATensorId";
    case TensorTypeId::VariableTensorId:
      return "VariableTensorId";
    case TensorTypeId::TestingOnly_WrapperTensorId:
      return "TestingOnly_WrapperTensorId";
    default:
      return "UNKNOWN_TENSOR_TYPE_ID";
  }
}

std::ostream& operator<<(std::ostream& str, TensorTypeId rhs) {
  return str << toString(rhs);
}

} // namespace c10
