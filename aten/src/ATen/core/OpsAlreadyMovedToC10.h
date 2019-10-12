#pragma once

#include <c10/macros/Export.h>

namespace c10 {
struct OperatorName;
}

namespace at {

/*
There's semantically three sets of operators:

- aten_ops_already_moved_to_c10
- aten_ops_not_moved_to_c10_yet
- non_aten_ops (e.g. custom ops)

register_c10_ops.cpp needs to decide between aten_ops_already_moved_to_c10
and union(aten_ops_not_moved_to_c10_yet, non_aten_ops).
The c10 operator registry needs to decide between aten_ops_not_moved_to_c10_yet
and union(aten_ops_already_moved_to_c10, non_aten_ops), which is different to what
register_c10_ops.cpp needs. We need to store two sets to be able to make both decisions.
*/

// list of ATen ops that got already moved to the c10 dispatcher
CAFFE2_API bool aten_op_is_already_moved_to_c10(const c10::OperatorName& opName);

// list of ATen ops that are still on the globalATenDispatch dispatcher.
CAFFE2_API bool aten_op_is_not_moved_to_c10_yet(const c10::OperatorName& opName);

}
