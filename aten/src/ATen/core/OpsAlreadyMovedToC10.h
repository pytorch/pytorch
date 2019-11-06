#pragma once

#include <c10/macros/Export.h>

namespace c10 {
struct OperatorName;
}

namespace at {

// list of ATen ops that come from native_functions.yaml
CAFFE2_API bool is_aten_op(const c10::OperatorName& opName);

}
