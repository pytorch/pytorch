#pragma once

#include <c10/macros/Export.h>

namespace c10 {
struct OperatorName;
}

namespace at {

// list of ATen ops that come from native_functions.yaml
CAFFE2_API bool is_aten_op_and_unboxing_is_already_handled_by_c10(const c10::OperatorName& opName);
CAFFE2_API bool is_aten_op_and_unboxing_is_not_handled_by_c10_yet(const c10::OperatorName& opName);

}
