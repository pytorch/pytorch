/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdexcept>

#ifndef _MSC_VER
extern "C" {
// This function is needed to avoid superfluous dependency on GNU OpenMP library
// when cuPTI is linked statically For more details see
// https://github.com/pytorch/pytorch/issues/51026
__attribute__((weak)) int acc_get_device_type() {
  throw std::runtime_error(
      "Dummy implementation of acc_get_device_type is not supposed to be called!");
}

} // extern "C"
#endif
