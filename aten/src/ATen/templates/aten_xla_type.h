#pragma once
// ${generated_comment}

#include <ATen/Tensor.h>

namespace ${cpp_namespace} {

// Base ATEN Type class where the XLA specific overrides should be defined.
class AtenXlaType {
 public:
  static void InitializeAtenBindings();

  //////////////////////////////////////////////////////////////////////////////
  // ATEN API ovverrides in alphabetical order.
  // Note: The C++ signatures must match the ones listed within the following
  // pytorch folder file:
  //   torch/csrc/autograd/generated/RegistrationDeclarations.h
  /////////////////////////////////////////////////////////////////////////////
${dispatch_xla_declarations}
};

}  // namespace torch_xla
