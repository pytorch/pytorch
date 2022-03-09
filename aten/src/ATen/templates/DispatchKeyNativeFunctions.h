#pragma once
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
