/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <clog.h>

CLOG_DEFINE_LOG_DEBUG(named_log_debug, "Unit Test", CLOG_DEBUG);
CLOG_DEFINE_LOG_INFO(named_log_info, "Unit Test", CLOG_INFO);
CLOG_DEFINE_LOG_WARNING(named_log_warning, "Unit Test", CLOG_WARNING);
CLOG_DEFINE_LOG_ERROR(named_log_error, "Unit Test", CLOG_ERROR);
CLOG_DEFINE_LOG_FATAL(named_log_fatal, "Unit Test", CLOG_FATAL);

CLOG_DEFINE_LOG_DEBUG(nameless_log_debug, NULL, CLOG_DEBUG);
CLOG_DEFINE_LOG_INFO(nameless_log_info, NULL, CLOG_INFO);
CLOG_DEFINE_LOG_WARNING(nameless_log_warning, NULL, CLOG_WARNING);
CLOG_DEFINE_LOG_ERROR(nameless_log_error, NULL, CLOG_ERROR);
CLOG_DEFINE_LOG_FATAL(nameless_log_fatal, NULL, CLOG_FATAL);

CLOG_DEFINE_LOG_DEBUG(suppressed_log_debug, NULL, CLOG_INFO);
CLOG_DEFINE_LOG_INFO(suppressed_log_info, NULL, CLOG_WARNING);
CLOG_DEFINE_LOG_WARNING(suppressed_log_warning, NULL, CLOG_ERROR);
CLOG_DEFINE_LOG_ERROR(suppressed_log_error, NULL, CLOG_FATAL);
CLOG_DEFINE_LOG_FATAL(suppressed_log_fatal, NULL, CLOG_NONE);

TEST(CLOG, debug) {
  named_log_debug("test debug message with a module name");
  nameless_log_debug("test debug message without a module name");
  suppressed_log_debug("test suppressed debug message");
}

TEST(CLOG, info) {
  named_log_info("test info message with a module name");
  nameless_log_info("test info message without a module name");
  suppressed_log_info("test suppressed info message");
}

TEST(CLOG, warning) {
  named_log_warning("test warning message with a module name");
  nameless_log_warning("test warning message without a module name");
  suppressed_log_warning("test suppressed warning message");
}

TEST(CLOG, error) {
  named_log_error("test error message with a module name");
  nameless_log_error("test error message without a module name");
  suppressed_log_error("test suppressed error message");
}
