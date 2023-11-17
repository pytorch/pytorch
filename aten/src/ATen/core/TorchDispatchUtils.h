#pragma once

#include <torch/library.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/Optional.h>
#include <c10/core/impl/TorchDispatchModeTLS.h>

namespace at {
namespace impl {

TORCH_API bool tensor_has_dispatch(const at::Tensor& t);
TORCH_API bool tensorlist_has_dispatch(at::ITensorListRef li);
TORCH_API bool tensorlist_has_dispatch(const c10::List<c10::optional<at::Tensor>>& li);
using c10::impl::dispatch_mode_enabled;

}}
