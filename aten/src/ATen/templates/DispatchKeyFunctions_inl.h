#pragma once
// ${generated_comment}

// NB: The implementing C++ file is RegisterDispatchKey.cpp

// The only #includes we need are for custom classes that have defaults in the C++ API
#include <c10/core/MemoryFormat.h>
#include <c10/core/Scalar.h>
#include <ATen/core/Reduction.h>

// TODO: If necessary, consider adding <ATen/ops/{function}_key.h> headers
#ifdef TORCH_ASSERT_ONLY_METHOD_OPERATORS
#error This change adds a dependency on all pytorch operators, meaning the     \
  file will need to be re-compiled every time an operator is changed or added. \
  Consider including a specific operator from \
  <ATen/ops/{my_operator}_${dispatch_namespace}_dispatch.h>
#endif

$DispatchKeyFunctions_inl_includes
