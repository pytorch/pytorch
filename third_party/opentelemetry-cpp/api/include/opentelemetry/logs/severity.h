// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "opentelemetry/nostd/string_view.h"
#include "opentelemetry/version.h"

OPENTELEMETRY_BEGIN_NAMESPACE
namespace logs
{

/**
 * Severity Levels assigned to log events, based on Log Data Model,
 * with the addition of kInvalid (mapped to a severity number of 0).
 *
 * See
 * https://github.com/open-telemetry/opentelemetry-specification/blob/main/specification/logs/data-model.md#field-severitynumber
 */
enum class Severity : uint8_t
{
  kInvalid = 0,
  kTrace   = 1,
  kTrace2  = 2,
  kTrace3  = 3,
  kTrace4  = 4,
  kDebug   = 5,
  kDebug2  = 6,
  kDebug3  = 7,
  kDebug4  = 8,
  kInfo    = 9,
  kInfo2   = 10,
  kInfo3   = 11,
  kInfo4   = 12,
  kWarn    = 13,
  kWarn2   = 14,
  kWarn3   = 15,
  kWarn4   = 16,
  kError   = 17,
  kError2  = 18,
  kError3  = 19,
  kError4  = 20,
  kFatal   = 21,
  kFatal2  = 22,
  kFatal3  = 23,
  kFatal4  = 24
};

const uint8_t kMaxSeverity = 255;

/**
 * Mapping of the severity enum above, to a severity text string (in all caps).
 * This severity text can be printed out by exporters. Capital letters follow the
 * spec naming convention.
 *
 * Included to follow the specification's recommendation to print both
 * severity number and text in each log record.
 */
const nostd::string_view SeverityNumToText[25] = {
    "INVALID", "TRACE",  "TRACE2", "TRACE3", "TRACE4", "DEBUG",  "DEBUG2", "DEBUG3", "DEBUG4",
    "INFO",    "INFO2",  "INFO3",  "INFO4",  "WARN",   "WARN2",  "WARN3",  "WARN4",  "ERROR",
    "ERROR2",  "ERROR3", "ERROR4", "FATAL",  "FATAL2", "FATAL3", "FATAL4"};

}  // namespace logs
OPENTELEMETRY_END_NAMESPACE
