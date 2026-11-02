/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <exception>
#include <functional>
#include <optional>
#include <span>
#include <vector>

#include <pti/pti.h>
#include <pti/pti_metrics_scope.h>

namespace KINETO_NAMESPACE {

class Config;

class XpuptiScopeProfilerApi {
 public:
  XpuptiScopeProfilerApi() = default;
  XpuptiScopeProfilerApi(const XpuptiScopeProfilerApi&) = delete;
  XpuptiScopeProfilerApi& operator=(const XpuptiScopeProfilerApi&) = delete;

  ~XpuptiScopeProfilerApi() = default;

  void enableScopeProfiler(const Config&);
  void disableScopeProfiler();
  void startScopeActivity();
  void stopScopeActivity();

  void processScopeTrace(
      const std::function<void(
          const pti_metrics_scope_record_t*,
          const pti_metrics_scope_record_metadata_t& metadata)>& handler);

 private:
  struct safe_pti_scope_collection_handle_t {
    safe_pti_scope_collection_handle_t(
        std::exception_ptr& exceptFromDestructor);
    ~safe_pti_scope_collection_handle_t() noexcept;

    operator pti_scope_collection_handle_t() {
      return handle_;
    }

    pti_scope_collection_handle_t handle_{};
    std::exception_ptr& exceptFromDestructor_;
  };

  std::optional<safe_pti_scope_collection_handle_t> scopeHandleOpt_;
  std::exception_ptr exceptFromScopeHandleDestructor_;
};

// Map requested device indices to PTI device handles, preserving order.
// Throws std::runtime_error if any index is out of [0, deviceCount).
std::vector<pti_device_handle_t> selectDeviceHandles(
    std::span<const pti_device_handle_t> handles,
    std::span<const int> indices);

} // namespace KINETO_NAMESPACE
