// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <fmt/format.h>

#include <c10/util/Logging.h>

#define C10D_LOG_(n, lvl, ...)\
    LOG_IF(n, FLAGS_caffe2_log_level <= lvl) << fmt::format(__VA_ARGS__)

#define C10D_ERROR(...)   C10D_LOG_(ERROR,   2, __VA_ARGS__)
#define C10D_WARNING(...) C10D_LOG_(WARNING, 1, __VA_ARGS__)
#define C10D_INFO(...)    C10D_LOG_(INFO,    0, __VA_ARGS__)
