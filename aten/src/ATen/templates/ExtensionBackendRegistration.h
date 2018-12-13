#include <ATen/Backend.h>
{extension_backend_headers}

template <typename FnPtr>
CAFFE2_API void register_extension_backend_op(
    Backend backend,
    std::string schema,
    FnPtr* fn) {
      switch (backend) {
        {extension_backend_register_switches}
        default:
          AT_ERROR("Invalid extension backend: ", toString(backend));
    }
