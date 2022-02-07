// Copyright 2004-present Facebook. All Rights Reserved.
#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>

namespace at { namespace native {

using stack_serial_fn = void(*)(Tensor &, ITensorList, int64_t);
DECLARE_DISPATCH(stack_serial_fn, stack_serial_stub);

}}  // namespace at::native
