//  Copyright © 2022 Apple Inc.
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/TensorFactory.h>
#include <c10/core/ScalarType.h>
#include <unordered_map>

using namespace at::mps;

namespace at {
namespace native {
namespace mps {

std::string getBitSizeString(ScalarType scalar_type) {
  size_t scalarBitSize = c10::elementSize(scalar_type) * 8;
  TORCH_CHECK(scalarBitSize <= 64, "Unsupported data type: ", getMPSTypeString(scalar_type));
  return std::to_string(scalarBitSize) + "bit";

}

std::string getIndexFunctionName(ScalarType scalar_type, bool index_select, bool accumulate, bool serial=false) {
  std::string indexFunction = index_select ? "index_select_" :
                      (accumulate && (scalar_type != kBool)) ? "index_put_accumulate_" : (serial ? "index_put_serial_" : "index_put_");

  indexFunction += getBitSizeString(scalar_type);
  if (accumulate) {
    TORCH_CHECK(scalar_type == ScalarType::Float || scalar_type == ScalarType::Int, "Unsupported data type for accumulate case: ", getMPSTypeString(scalar_type));
    string dtypeString = (scalar_type == ScalarType::Float) ? "_float" : "_int";
    indexFunction += dtypeString;
  }
  return indexFunction;
}
}
}
}
