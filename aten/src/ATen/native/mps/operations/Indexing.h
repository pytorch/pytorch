//  Copyright © 2022 Apple Inc.

#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <ATen/Utils.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/TensorFactory.h>
#include <c10/core/ScalarType.h>
#include <torch/library.h>
#include <unordered_map>

using namespace at::mps;

namespace at {
namespace native {
namespace mps {

std::string getMetalScalarType(ScalarType scalar_type) {
  std::string res = "";
  switch (scalar_type) {
    case ScalarType::Float:
      res = "float"; break;
    case ScalarType::Half:
      res = "half";  break;
    case ScalarType::Long:
      res = "long";  break;
    case ScalarType::Int:
      res = "int";   break;
    case ScalarType::Short:
      res = "short"; break;
    case ScalarType::Char:
      res = "char"; break;
    case ScalarType::Byte:
      res = "uchar"; break;
    case ScalarType::Bool:
      res = "bool";  break;
    default:
      break;
  }
  return res;
}

std::string getIndexFunctionName(ScalarType scalar_type, bool index_select, bool accumulate) {
    std::string indexFunction = index_select ? "index_select_" :
                        (accumulate && (scalar_type != kBool)) ? "index_put_accumulate_" : "index_put_";

  return indexFunction + getMetalScalarType(scalar_type);
}
}
}
}
