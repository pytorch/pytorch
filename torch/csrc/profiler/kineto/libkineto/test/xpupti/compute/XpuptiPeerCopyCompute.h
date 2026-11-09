/*
 * Copyright (C) Intel Corporation
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "XpuptiScopeProfilerCompute.h"

#include <cstddef>

// Copies `bytes` from the first GPU to the second over a peer-to-peer link.
// Returns false, without copying, when there is no such pair of devices.
XPUPTI_COMPUTE_API bool CopyPeerToPeerOnXpu(std::size_t bytes);
