#pragma once

#include <c10/core/Device.h>
#include <c10/macros/Export.h>

namespace c10 {
struct TensorImpl;
}

namespace at {

TORCH_API void set_and_normalize_fake_device(
    c10::TensorImpl* impl,
    c10::Device device);

} // namespace at
