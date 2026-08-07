/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <source_location>
#include <string_view>

#include <pti/pti.h>

namespace KINETO_NAMESPACE {

using namespace libkineto;

// Used to enable future features in PTI before release.
#define PTI_VERSION_AT_LEAST(MAJOR, MINOR) \
  (PTI_VERSION_MAJOR > MAJOR ||            \
   (PTI_VERSION_MAJOR == MAJOR && PTI_VERSION_MINOR >= MINOR))

void XPUPTI_CALL(
    pti_result errCode,
    std::string_view message = "",
    std::source_location source_location = std::source_location::current());

using DeviceIndex_t = int8_t;

} // namespace KINETO_NAMESPACE
