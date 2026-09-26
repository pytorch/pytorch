/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "cupti_strings.h"

#include <string>

namespace libkineto {

const char* memcpyKindString(CUpti_ActivityMemcpyKind kind) {
  switch (kind) {
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD:
      return "HtoD";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH:
      return "DtoH";
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOA:
      return "HtoA";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOH:
      return "AtoH";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOA:
      return "AtoA";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOD:
      return "AtoD";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOA:
      return "DtoA";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD:
      return "DtoD";
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOH:
      return "HtoH";
    case CUPTI_ACTIVITY_MEMCPY_KIND_PTOP:
      return "PtoP";
    default:
      break;
  }
  return "<unknown>";
}

const char* memoryKindString(CUpti_ActivityMemoryKind kind) {
  switch (kind) {
    case CUPTI_ACTIVITY_MEMORY_KIND_UNKNOWN:
      return "Unknown";
    case CUPTI_ACTIVITY_MEMORY_KIND_PAGEABLE:
      return "Pageable";
    case CUPTI_ACTIVITY_MEMORY_KIND_PINNED:
      return "Pinned";
    case CUPTI_ACTIVITY_MEMORY_KIND_DEVICE:
      return "Device";
    case CUPTI_ACTIVITY_MEMORY_KIND_ARRAY:
      return "Array";
    case CUPTI_ACTIVITY_MEMORY_KIND_MANAGED:
      return "Managed";
    case CUPTI_ACTIVITY_MEMORY_KIND_DEVICE_STATIC:
      return "Device Static";
    case CUPTI_ACTIVITY_MEMORY_KIND_MANAGED_STATIC:
      return "Managed Static";
    case CUPTI_ACTIVITY_MEMORY_KIND_FORCE_INT:
      return "Force Int";
    default:
      return "Unrecognized";
  }
}

const char* overheadKindString(CUpti_ActivityOverheadKind kind) {
  switch (kind) {
    case CUPTI_ACTIVITY_OVERHEAD_UNKNOWN:
      return "Unknown";
    case CUPTI_ACTIVITY_OVERHEAD_DRIVER_COMPILER:
      return "Driver Compiler";
    case CUPTI_ACTIVITY_OVERHEAD_CUPTI_BUFFER_FLUSH:
      return "Buffer Flush";
    case CUPTI_ACTIVITY_OVERHEAD_CUPTI_INSTRUMENTATION:
      return "Instrumentation";
    case CUPTI_ACTIVITY_OVERHEAD_CUPTI_RESOURCE:
      return "Resource";
#if CUDART_VERSION >= 12040
    case CUPTI_ACTIVITY_OVERHEAD_RUNTIME_TRIGGERED_MODULE_LOADING:
      return "Runtime Triggered Module Loading";
    case CUPTI_ACTIVITY_OVERHEAD_LAZY_FUNCTION_LOADING:
      return "Lazy Function Loading";
    case CUPTI_ACTIVITY_OVERHEAD_COMMAND_BUFFER_FULL:
      return "Command Buffer Full";
#endif
#if CUDART_VERSION >= 12070
    case CUPTI_ACTIVITY_OVERHEAD_ACTIVITY_BUFFER_REQUEST:
      return "Activity Buffer Request";
#endif
    case CUPTI_ACTIVITY_OVERHEAD_FORCE_INT:
      return "Force Int";
    default:
      return "Unrecognized";
  }
}

namespace {

// CUPTI returns identifiers like `cudaLaunchKernel_v7000` whose trailing
// `_vNNNN` is the CUDA-version-introduced suffix. Strip it to match the
// existing trace-label convention, while preserving single-digit
// API-generation suffixes like `_v2` / `_v3` (lowest CUDA-version suffix is
// `_v3020` for CUDA 3.2, so 4+ trailing digits is unambiguous).
std::string lookupCbidName(CUpti_CallbackDomain domain, CUpti_CallbackId cbid) {
  const char* raw = nullptr;
  if (cuptiGetCallbackName(domain, cbid, &raw) != CUPTI_SUCCESS ||
      raw == nullptr) {
    return "INVALID";
  }
  std::string name = raw;
  if (name.empty()) {
    return "INVALID";
  }
  const auto vPos = name.find_last_not_of("0123456789");
  if (vPos != std::string::npos && vPos >= 1 && name[vPos] == 'v' &&
      name[vPos - 1] == '_' && name.size() - vPos - 1 >= 4) {
    name.resize(vPos - 1);
  }
  return name;
}

} // namespace

std::string runtimeCbidName(CUpti_CallbackId cbid) {
  return lookupCbidName(CUPTI_CB_DOMAIN_RUNTIME_API, cbid);
}

std::string driverCbidName(CUpti_CallbackId cbid) {
  return lookupCbidName(CUPTI_CB_DOMAIN_DRIVER_API, cbid);
}

// From
// https://docs.nvidia.com/cupti/modules.html#group__CUPTI__ACTIVITY__API_1g80e1eb47615e31021f574df8ebbe5d9a
//   enum CUpti_ActivitySynchronizationType
const char* syncTypeString(CUpti_ActivitySynchronizationType kind) {
  switch (kind) {
    case CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_EVENT_SYNCHRONIZE:
      return "Event Sync";
    case CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_STREAM_WAIT_EVENT:
      return "Stream Wait Event";
    case CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_STREAM_SYNCHRONIZE:
      return "Stream Sync";
    case CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_CONTEXT_SYNCHRONIZE:
      return "Context Sync";
    case CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_UNKNOWN:
    default:
      return "Unknown Sync";
  }
}
} // namespace libkineto
