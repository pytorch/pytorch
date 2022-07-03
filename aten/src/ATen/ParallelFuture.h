#pragma once

#include <ATen/core/ivalue.h>
#include <c10/macros/Macros.h>
#include <functional>

namespace at {

// Launches intra-op parallel task, returns a future
TORCH_API c10::intrusive_ptr<c10::ivalue::Future> intraop_launch_future(
    std::function<void()> func);

} // namespace at
