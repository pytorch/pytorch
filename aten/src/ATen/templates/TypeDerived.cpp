// required for old g++ to compile PRId64 macros, see
// https://github.com/pytorch/pytorch/issues/3571
// for context
#define __STDC_FORMAT_MACROS

#include <ATen/${Type}.h>

// ${generated_comment}

$th_headers
$storage_tensor_headers
#include <ATen/${Generator}.h>
#include <c10/core/Allocator.h>
#include <ATen/DeviceGuard.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Utils.h>
#include <ATen/WrapDimUtils.h>
#include <c10/util/Half.h>
#include <c10/core/TensorImpl.h>
#include <c10/core/UndefinedTensorImpl.h>
#include <c10/util/Optional.h>

#include <cstddef>
#include <functional>
#include <memory>
#include <utility>

#include <ATen/Config.h>
$extra_cuda_headers

namespace at {

${Type}::${Type}()
  : ${DenseBackend}TypeDefault(${Backend}TensorId(), /*is_variable=*/false, /*is_undefined=*/false) {}

ScalarType ${Type}::scalarType() const {
  return ScalarType::${ScalarName};
}

caffe2::TypeMeta ${Type}::typeMeta() const {
    return caffe2::TypeMeta::Make<${ScalarType}>();
}

Backend ${Type}::backend() const {
  return Backend::${Backend};
}

const char * ${Type}::toString() const {
  return "${Type}";
}

TypeID ${Type}::ID() const {
  return ${TypeID};
}

/* example
Tensor * ${Type}::add(Tensor & a, Tensor & b) {
  std::cout << "add Tensor with backend ${Backend}\n";
  return &a;
}
*/

${type_derived_method_definitions}

}
