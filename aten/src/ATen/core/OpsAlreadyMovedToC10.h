#pragma once

#include <ATen/core/operator_name.h>
#include <unordered_set>

namespace at {

// list of ATen ops that got already moved to the c10 dispatcher
const std::unordered_set<c10::OperatorName>& aten_ops_already_moved_to_c10();

}
