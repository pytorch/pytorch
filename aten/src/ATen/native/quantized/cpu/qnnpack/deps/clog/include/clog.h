/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <inttypes.h>
#include <stdarg.h>
#include <stdlib.h>

#define CLOG_NONE 0
#define CLOG_FATAL 1
#define CLOG_ERROR 2
#define CLOG_WARNING 3
#define CLOG_INFO 4
#define CLOG_DEBUG 5

#ifndef CLOG_VISIBILITY
#if defined(__ELF__)
#define CLOG_VISIBILITY __attribute__((__visibility__("internal")))
#elif defined(__MACH__)
#define CLOG_VISIBILITY __attribute__((__visibility__("hidden")))
#else
#define CLOG_VISIBILITY
#endif
#endif

#ifndef CLOG_ARGUMENTS_FORMAT
#if defined(__GNUC__)
#define CLOG_ARGUMENTS_FORMAT __attribute__((__format__(__printf__, 1, 2)))
#else
#define CLOG_ARGUMENTS_FORMAT
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

CLOG_VISIBILITY void clog_vlog_debug(
    const char* module,
    const char* format,
    va_list args);
CLOG_VISIBILITY void clog_vlog_info(
    const char* module,
    const char* format,
    va_list args);
CLOG_VISIBILITY void clog_vlog_warning(
    const char* module,
    const char* format,
    va_list args);
CLOG_VISIBILITY void clog_vlog_error(
    const char* module,
    const char* format,
    va_list args);
CLOG_VISIBILITY void clog_vlog_fatal(
    const char* module,
    const char* format,
    va_list args);

#define CLOG_DEFINE_LOG_DEBUG(log_debug_function_name, module, level)   \
  CLOG_ARGUMENTS_FORMAT                                                 \
  inline static void log_debug_function_name(const char* format, ...) { \
    if (level >= CLOG_DEBUG) {                                          \
      va_list args;                                                     \
      va_start(args, format);                                           \
      clog_vlog_debug(module, format, args);                            \
      va_end(args);                                                     \
    }                                                                   \
  }

#define CLOG_DEFINE_LOG_INFO(log_info_function_name, module, level)    \
  CLOG_ARGUMENTS_FORMAT                                                \
  inline static void log_info_function_name(const char* format, ...) { \
    if (level >= CLOG_INFO) {                                          \
      va_list args;                                                    \
      va_start(args, format);                                          \
      clog_vlog_info(module, format, args);                            \
      va_end(args);                                                    \
    }                                                                  \
  }

#define CLOG_DEFINE_LOG_WARNING(log_warning_function_name, module, level) \
  CLOG_ARGUMENTS_FORMAT                                                   \
  inline static void log_warning_function_name(const char* format, ...) { \
    if (level >= CLOG_WARNING) {                                          \
      va_list args;                                                       \
      va_start(args, format);                                             \
      clog_vlog_warning(module, format, args);                            \
      va_end(args);                                                       \
    }                                                                     \
  }

#define CLOG_DEFINE_LOG_ERROR(log_error_function_name, module, level)   \
  CLOG_ARGUMENTS_FORMAT                                                 \
  inline static void log_error_function_name(const char* format, ...) { \
    if (level >= CLOG_ERROR) {                                          \
      va_list args;                                                     \
      va_start(args, format);                                           \
      clog_vlog_error(module, format, args);                            \
      va_end(args);                                                     \
    }                                                                   \
  }

#define CLOG_DEFINE_LOG_FATAL(log_fatal_function_name, module, level)   \
  CLOG_ARGUMENTS_FORMAT                                                 \
  inline static void log_fatal_function_name(const char* format, ...) { \
    if (level >= CLOG_FATAL) {                                          \
      va_list args;                                                     \
      va_start(args, format);                                           \
      clog_vlog_fatal(module, format, args);                            \
      va_end(args);                                                     \
    }                                                                   \
    abort();                                                            \
  }

#ifdef __cplusplus
} /* extern "C" */
#endif
