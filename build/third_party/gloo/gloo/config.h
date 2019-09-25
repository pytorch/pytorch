/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#define GLOO_VERSION_MAJOR 0
#define GLOO_VERSION_MINOR 5
#define GLOO_VERSION_PATCH 0

static_assert(
    GLOO_VERSION_MINOR < 100,
    "Programming error: you set a minor version that is too big.");
static_assert(
    GLOO_VERSION_PATCH < 100,
    "Programming error: you set a patch version that is too big.");

#define GLOO_VERSION                                         \
  (GLOO_VERSION_MAJOR * 10000 + GLOO_VERSION_MINOR * 100 +   \
   GLOO_VERSION_PATCH)

#define GLOO_USE_CUDA 0
#define GLOO_USE_NCCL 0
#define GLOO_USE_ROCM 0
#define GLOO_USE_RCCL 0
#define GLOO_USE_REDIS 0
#define GLOO_USE_IBVERBS 0
#define GLOO_USE_MPI 0
#define GLOO_USE_AVX 0
#define GLOO_USE_LIBUV 0

#define GLOO_HAVE_TRANSPORT_TCP 1
#define GLOO_HAVE_TRANSPORT_IBVERBS 0
#define GLOO_HAVE_TRANSPORT_UV 0
