/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <string>

#if !USE_GOOGLE_LOG
#include <memory>
#endif // !USE_GOOGLE_LOG
#ifdef __linux__
#include "IpcFabricConfigClient.h"
#endif // __linux__

namespace KINETO_NAMESPACE {

class IDaemonConfigLoader {
 public:
  virtual ~IDaemonConfigLoader() = default;

  // Return the base config from the daemon
  virtual std::string readBaseConfig() = 0;

  // Return a configuration string from the daemon, if one has been posted.
  virtual std::string readOnDemandConfig(bool activities) = 0;

  virtual void setCommunicationFabric(bool enabled) = 0;
};

// Basic Daemon Config Loader that uses IPCFabric for communication
// Only works on Linux based platforms
#ifdef __linux__
class DaemonConfigLoader : public IDaemonConfigLoader {
 public:
  DaemonConfigLoader() = default;

  // Return the base config from the daemon
  std::string readBaseConfig() override;

  // Return a configuration string from the daemon, if one has been posted.
  std::string readOnDemandConfig(bool activities) override;

  void setCommunicationFabric(bool enabled) override;

  IpcFabricConfigClient* getConfigClient();

  static void registerFactory();

 private:
  std::unique_ptr<IpcFabricConfigClient> configClient;
};
#endif // __linux__

} // namespace KINETO_NAMESPACE
