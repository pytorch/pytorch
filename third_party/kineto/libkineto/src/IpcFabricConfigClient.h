/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#ifdef __linux__

// TODO(T90238193)
// @lint-ignore-every CLANGTIDY facebook-hte-RelativeInclude
// include logger before to enable ipc fabric to access LOG() macros
#ifdef ENABLE_IPC_FABRIC

#include "Logger.h"

// The following is required for LOG() macros to work below
// Include the IPC Fabric
#include "FabricManager.h"

#else

// Adds an empty implementation so compilation works.
namespace dynolog::ipcfabric {

class FabricManager {
 public:
  FabricManager(const FabricManager&) = delete;
  FabricManager& operator=(const FabricManager&) = delete;

  static std::unique_ptr<FabricManager> factory(
      std::string endpoint_name = "") {
    return nullptr;
  }
};

} // namespace dynolog::ipcfabric

#endif // ENABLE_IPC_FABRIC

namespace KINETO_NAMESPACE {

enum LibkinetoConfigType {
  NONE = 0,
  EVENTS = 0x1,
  ACTIVITIES = 0x2,
};

// IpcFabricConfigClient : connects to a daemon using the IPC Fabric
//   this can be used as a base class for other Daemon Config clients as well.
class IpcFabricConfigClient {
 public:
  IpcFabricConfigClient();
  virtual ~IpcFabricConfigClient() {}

  // Registers this application with the daemon
  virtual int32_t registerInstance(int32_t gpu);

  // Get the base config for libkineto
  virtual std::string getLibkinetoBaseConfig();

  // Get on demand configurations for tracing/counter collection
  // type is a bit mask, please see LibkinetoConfigType encoding above.
  virtual std::string getLibkinetoOndemandConfig(int32_t type);

  void setIpcFabricEnabled(bool enabled) {
    ipcFabricEnabled_ = enabled;
  }

 protected:
  // Temporarily keep both int and string job id until IPC related code is
  // updated to handle string job id.
  int64_t jobId_;
  std::string jobIdStr_;
  std::vector<int32_t> pids_;
  bool ipcFabricEnabled_;

  std::unique_ptr<dynolog::ipcfabric::FabricManager> fabricManager_;
};

#endif // __linux__
} // namespace KINETO_NAMESPACE
