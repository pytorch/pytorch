#pragma once

// an external backend might generate file within its code tree
// and check all the source files within the tree with clang-format.
// so, disable it since the backend might have a different config.
// clang-format off

// ${generated_comment}

#include <ATen/Tensor.h>

${namespace_prologue}

struct ${class_name} {

${dispatch_declarations}

};

#define EAGER_REGISTRATION ${eager_registration}

#if !EAGER_REGISTRATION
extern TORCH_API std::function<void(void)> Register${BackendName}${DispatchKey}Modules;
#endif

${namespace_epilogue}
