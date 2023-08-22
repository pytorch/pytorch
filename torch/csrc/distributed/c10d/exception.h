// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stdexcept>

#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>

#define C10D_THROW_ERROR(err_type, msg) \
  throw ::c10d::err_type(               \
      {__func__, __FILE__, static_cast<uint32_t>(__LINE__)}, msg)

namespace c10d {

using c10::DistBackendError;

class TORCH_API SocketError : public DistBackendError {
  using DistBackendError::DistBackendError;
};

class TORCH_API TimeoutError : public DistBackendError {
  using DistBackendError::DistBackendError;
};

} // namespace c10d
