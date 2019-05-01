#pragma once
#include <ATen/Backend.h>
${extension_backend_headers}

namespace at {

template <typename FnPtr>
inline void register_extension_backend_op(
    Backend backend,
    const char * schema,
    FnPtr fn) {
      switch (backend) {
        ${extension_backend_register_switches}
        default:
          AT_ERROR("Invalid extension backend: ", toString(backend));
    }
}

} // namespace at
