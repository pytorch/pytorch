/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef __linux__

// TODO(T90238193)
// @lint-ignore-every CLANGTIDY facebook-hte-RelativeInclude
#include "DaemonConfigLoader.h"
#include "ConfigLoader.h"
#include "Logger.h"

namespace KINETO_NAMESPACE {

// TODO : implications of this singleton being thread safe on forks?
IpcFabricConfigClient* DaemonConfigLoader::getConfigClient() {
  if (!configClient) {
    configClient = std::make_unique<IpcFabricConfigClient>();
  }
  return configClient.get();
}

std::string DaemonConfigLoader::readBaseConfig() {
  LOG(INFO) << "Reading base config";
  auto configClient_2 = getConfigClient();
  if (!configClient_2) {
    LOG_EVERY_N(WARNING, 10) << "Failed to read config: No dyno config client";
    return "";
  }
  return configClient_2->getLibkinetoBaseConfig();
}

std::string DaemonConfigLoader::readOnDemandConfig(bool activities) {
  auto configClient = getConfigClient();
  if (!configClient) {
    LOG_EVERY_N(WARNING, 10) << "Failed to read config: No dyno config client";
    return "";
  }
  int config_type = static_cast<int>(LibkinetoConfigType::NONE);
  if (activities) {
    config_type |= static_cast<int>(LibkinetoConfigType::ACTIVITIES);
  }
  return configClient->getLibkinetoOndemandConfig(config_type);
}

void DaemonConfigLoader::setCommunicationFabric(bool enabled) {
  LOG(INFO) << "Setting communication fabric enabled = " << enabled;
  auto configClient = getConfigClient();

  if (!configClient) {
    LOG(WARNING) << "Failed to read config: No dyno config client";
    // This is probably a temporary problem - return -1 to indicate error.
    return;
  }
  configClient->setIpcFabricEnabled(enabled);
}

void DaemonConfigLoader::registerFactory() {
  ConfigLoader::setDaemonConfigLoaderFactory([]() {
    auto loader = std::make_unique<DaemonConfigLoader>();
    loader->setCommunicationFabric(true);
    return loader;
  });
}

} // namespace KINETO_NAMESPACE
#endif // __linux__
