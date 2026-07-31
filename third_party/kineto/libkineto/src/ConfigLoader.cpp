/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "ConfigLoader.h"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <memory>
#include <utility>

#include "DaemonConfigLoader.h"

#include "Logger.h"

using namespace std::chrono;

namespace KINETO_NAMESPACE {

constexpr char kConfigFileEnvVar[] = "KINETO_CONFIG";
#ifdef __linux__
constexpr char kConfigFile[] = "/etc/libkineto.conf";
#else
constexpr char kConfigFile[] = "libkineto.conf";
#endif

constexpr std::chrono::seconds kConfigUpdateIntervalSecs(300);

// return an empty string if reading gets any errors. Otherwise a config string.
static std::string readConfigFromConfigFile(
    const char* filename,
    bool verbose = true) {
  // Read whole file into a string.
  std::ifstream file(filename);
  std::string conf;
  try {
    conf.assign(
        std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
  } catch (std::exception& e) {
    if (verbose) {
      VLOG(0) << "Error reading " << filename << ": " << e.what();
    }

    conf = "";
  }
  return conf;
}

static std::function<std::unique_ptr<IDaemonConfigLoader>()>&
daemonConfigLoaderFactory() {
  static std::function<std::unique_ptr<IDaemonConfigLoader>()> factory =
      nullptr;
  return factory;
}

void ConfigLoader::setDaemonConfigLoaderFactory(
    std::function<std::unique_ptr<IDaemonConfigLoader>()> factory) {
  daemonConfigLoaderFactory() = std::move(factory);
}

ConfigLoader& ConfigLoader::instance() {
  static ConfigLoader config_loader;
  return config_loader;
}

// return an empty string if polling gets any errors. Otherwise a config string.
std::string ConfigLoader::readOnDemandConfigFromDaemon(
    [[maybe_unused]] time_point<system_clock> now) {
  if (!daemonConfigLoader_) {
    return "";
  }
  bool activities = canHandlerAcceptConfig(ConfigKind::ActivityProfiler);
  return daemonConfigLoader_->readOnDemandConfig(activities);
}

ConfigLoader::ConfigLoader()
    : configUpdateIntervalSecs_(kConfigUpdateIntervalSecs),
      // on-demand config will be overwritten by the value read from the regular
      // config so the initial value is not important
      onDemandConfigUpdateIntervalSecs_(kConfigUpdateIntervalSecs),
      stopFlag_(false) {}

void ConfigLoader::startThread() {
  if (!updateThread_) {
    // Reset the stop flag so a thread started after a prior stopThread() runs
    // instead of exiting immediately on its first wakeup.
    stopFlag_ = false;
    // Create default base config here - at this point static initializers
    // of extensions should have run and registered all config feature factories
    std::scoped_lock lock(configLock_);
    if (!config_) {
      config_ = std::make_unique<Config>();
    }
    updateThread_ =
        std::make_unique<std::thread>(&ConfigLoader::updateConfigThread, this);
  }
}

void ConfigLoader::stopThread() {
  if (updateThread_) {
    stopFlag_ = true;
    {
      std::scoped_lock lock(updateThreadMutex_);
      updateThreadCondVar_.notify_one();
    }
    if (updateThread_->joinable()) {
      updateThread_->join();
    }
    updateThread_ = nullptr;
  }
}

void ConfigLoader::stopUpdateThread() {
  stopThread();
}

void ConfigLoader::resetDaemonConfigLoaderForTesting() {
  daemonConfigLoader_.reset();
}

bool ConfigLoader::waitForUpdateThreadLoopCountForTesting(
    uint64_t target,
    std::chrono::milliseconds timeout) {
  std::unique_lock<std::mutex> lock(loopCountMutex_);
  return loopCountCondVar_.wait_for(lock, timeout, [this, target] {
    return updateThreadLoopCount_.load(std::memory_order_acquire) >= target;
  });
}

std::chrono::seconds ConfigLoader::onDemandConfigUpdateIntervalForTesting() {
  std::scoped_lock lock(configLock_);
  // config_ is the authoritative source; updateConfigThread caches its value
  // into onDemandConfigUpdateIntervalSecs_ on each base-config refresh.
  return config_ ? config_->onDemandConfigUpdateIntervalSecs()
                 : onDemandConfigUpdateIntervalSecs_;
}

ConfigLoader::~ConfigLoader() {
  stopThread();
#if !USE_GOOGLE_LOG
  Logger::clearLoggerObservers();
#endif // !USE_GOOGLE_LOG
}

namespace {

const char* configFileName() {
  static const char* configFileName__ = []() {
    const char* configFileName_ = getenv(kConfigFileEnvVar);
    if (configFileName_ == nullptr) {
      configFileName_ = kConfigFile;
    }
    return configFileName_;
  }();
  return configFileName__;
}

} // namespace

IDaemonConfigLoader* ConfigLoader::daemonConfigLoader() {
  if (!daemonConfigLoader_ && daemonConfigLoaderFactory()) {
    daemonConfigLoader_ = daemonConfigLoaderFactory()();
    daemonConfigLoader_->setCommunicationFabric(config_->ipcFabricEnabled());
  }
  return daemonConfigLoader_.get();
}

const char* ConfigLoader::customConfigFileName() {
  return getenv(kConfigFileEnvVar);
}

std::string ConfigLoader::getConfString() {
  return readConfigFromConfigFile(configFileName(), false);
}

void ConfigLoader::updateBaseConfig() {
  // First try reading local config file
  // If that fails, read from daemon
  // TODO: Invert these once daemon path fully rolled out
  std::string config_str = readConfigFromConfigFile(configFileName());
  if (config_str.empty() && daemonConfigLoader()) {
    // If local config file was not successfully loaded (e.g. not found)
    // then try the daemon
    config_str = daemonConfigLoader()->readBaseConfig();
  }
  if (config_str != config_->source()) {
    std::scoped_lock lock(configLock_);
    config_ = std::make_unique<Config>();
    config_->parse(config_str);
    if (daemonConfigLoader()) {
      daemonConfigLoader()->setCommunicationFabric(config_->ipcFabricEnabled());
    }
    SET_LOG_VERBOSITY_LEVEL(
        config_->verboseLogLevel(), config_->verboseLogModules());
    VLOG(0) << "Detected base config change";
  }
}

void ConfigLoader::configureFromDaemon(
    time_point<system_clock> now,
    Config& config) {
  const std::string config_str = readOnDemandConfigFromDaemon(now);
  if (config_str.empty()) {
    return;
  }

  LOG(INFO) << "Received config from dyno:\n" << config_str;
  // Untrusted daemon IPC config; restrict trace output path.
  config.setOnDemand(true);
  config.parse(config_str);
  notifyHandlers(config);
}

void ConfigLoader::updateConfigThread() {
  // It's important to hang to this reference until the thread stops.
  // Otherwise, the Config's static members may be destroyed before this
  // function finishes.
  auto handle = Config::getStaticObjectsLifetimeHandle();

  // We're trying to update two configs here:
  // 1. Base config - this is the config that is read from the config file
  // 2. On-demand config - this is the config that is read via IPC channel
  //    from daemon. It's layered on top of the base config.
  // They have different update intervals (on demand is more frequent).
  // Besides, on-demand update frequency can be configured via base config.

  // initialze with some time buffer in the past
  auto prev_config_load_time =
      system_clock::now() - configUpdateIntervalSecs_ * 2;
  auto prev_on_demand_load_time = prev_config_load_time;
  auto onDemandConfig = std::make_unique<Config>();

  // This can potentially sleep for long periods of time, so allow
  // the destructor to wake it to avoid a 5-minute long destruct period.
  for (;;) {
    auto interval =
        std::min(
            configUpdateIntervalSecs_ + prev_config_load_time,
            onDemandConfigUpdateIntervalSecs_ + prev_on_demand_load_time) -
        system_clock::now();
    if (interval.count() > 0) {
      std::unique_lock<std::mutex> lock(updateThreadMutex_);
      updateThreadCondVar_.wait_for(lock, interval);
    }
    if (stopFlag_) {
      break;
    }
    auto now = system_clock::now();
    // This runs on a bare background thread: an escaped exception would
    // terminate the process, so config-update failures are caught and skipped.
    // Advance the load timestamps before the guarded work so a persistent
    // failure backs off to the normal interval instead of hot-looping.
    if (now > prev_config_load_time + configUpdateIntervalSecs_) {
      prev_config_load_time = now;
      try {
        updateBaseConfig();
        onDemandConfigUpdateIntervalSecs_ =
            config_->onDemandConfigUpdateIntervalSecs();
      } catch (const std::exception& e) {
        LOG(ERROR) << "Skipping base config update after error: " << e.what();
      } catch (...) {
        LOG(ERROR) << "Skipping base config update after unknown error";
      }
    }
    if (now > prev_on_demand_load_time + onDemandConfigUpdateIntervalSecs_) {
      prev_on_demand_load_time = now;
      onDemandConfig = std::make_unique<Config>();
      try {
        configureFromDaemon(now, *onDemandConfig);
      } catch (const std::exception& e) {
        LOG(ERROR) << "Skipping on-demand config update after error: "
                   << e.what();
      } catch (...) {
        LOG(ERROR) << "Skipping on-demand config update after unknown error";
      }
    }
    if (onDemandConfig->verboseLogLevel() >= 0) {
      LOG(INFO) << "Setting verbose level to "
                << onDemandConfig->verboseLogLevel()
                << " from on-demand config";
      SET_LOG_VERBOSITY_LEVEL(
          onDemandConfig->verboseLogLevel(),
          onDemandConfig->verboseLogModules());
    }
    // Mark one completed iteration and wake any test waiting for deterministic
    // progression of the real poll thread.
    {
      std::scoped_lock lock(loopCountMutex_);
      updateThreadLoopCount_.fetch_add(1, std::memory_order_release);
    }
    loopCountCondVar_.notify_all();
  }
}

bool ConfigLoader::hasNewConfig(const Config& oldConfig) {
  std::scoped_lock lock(configLock_);
  return config_->timestamp() > oldConfig.timestamp();
}

} // namespace KINETO_NAMESPACE
