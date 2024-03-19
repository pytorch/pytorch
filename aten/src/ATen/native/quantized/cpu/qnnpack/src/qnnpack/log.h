/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <inttypes.h>

#include <clog.h>

#ifndef PYTORCH_QNNP_LOG_LEVEL
#define PYTORCH_QNNP_LOG_LEVEL CLOG_WARNING
#endif

CLOG_DEFINE_LOG_DEBUG(
    pytorch_qnnp_log_debug,
    "QNNPACK",
    PYTORCH_QNNP_LOG_LEVEL);
CLOG_DEFINE_LOG_INFO(pytorch_qnnp_log_info, "QNNPACK", PYTORCH_QNNP_LOG_LEVEL);
CLOG_DEFINE_LOG_WARNING(
    pytorch_qnnp_log_warning,
    "QNNPACK",
    PYTORCH_QNNP_LOG_LEVEL);
CLOG_DEFINE_LOG_ERROR(
    pytorch_qnnp_log_error,
    "QNNPACK",
    PYTORCH_QNNP_LOG_LEVEL);
CLOG_DEFINE_LOG_FATAL(
    pytorch_qnnp_log_fatal,
    "QNNPACK",
    PYTORCH_QNNP_LOG_LEVEL);
