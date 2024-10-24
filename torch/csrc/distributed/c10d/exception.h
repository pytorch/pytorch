// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stdexcept>

#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>

// Utility macro similar to C10_THROW_ERROR, the major difference is that this
// macro handles exception types defined in the c10d namespace, whereas
// C10_THROW_ERROR requires an exception to be defined in the c10 namespace.
#define C10D_THROW_ERROR(err_type, msg) \
  throw ::c10d::err_type(               \
      {__func__, __FILE__, static_cast<uint32_t>(__LINE__)}, msg)

namespace c10d {

using c10::DistNetworkError;

class TORCH_API SocketError : public DistNetworkError {
  using DistNetworkError::DistNetworkError;
};

class TORCH_API TimeoutError : public DistNetworkError {
  using DistNetworkError::DistNetworkError;
};

} // namespace c10d
