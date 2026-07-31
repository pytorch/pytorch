/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <optional>

#include <pti/pti_metrics_scope.h>

#include "XpuptiActivityApi.h"

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
      std::function<void(
          const pti_metrics_scope_record_t*,
          const pti_metrics_scope_record_metadata_t& metadata)> handler);

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

} // namespace KINETO_NAMESPACE
