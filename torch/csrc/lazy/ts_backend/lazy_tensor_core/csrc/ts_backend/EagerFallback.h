#pragma once

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/stack.h>

namespace torch_lazy_tensors {

void eager_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack,
                    c10::DeviceType device_type);

}
