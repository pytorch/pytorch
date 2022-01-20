// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <fmt/format.h>

#include <c10/util/Logging.h>

#define C10D_ERROR(...)\
    LOG_IF(ERROR,   FLAGS_caffe2_log_level <= 2) << fmt::format(__VA_ARGS__)

#define C10D_WARNING(...)\
    LOG_IF(WARNING, FLAGS_caffe2_log_level <= 1) << fmt::format(__VA_ARGS__)

#define C10D_INFO(...)\
    LOG_IF(INFO,    FLAGS_caffe2_log_level <= 0) << fmt::format(__VA_ARGS__)
