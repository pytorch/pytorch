/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef __linux__

#include "IpcFabricConfigClient.h"

#include <cstdlib>
#include <random>
#include <sstream>

// TODO(T90238193)
// @lint-ignore-every CLANGTIDY facebook-hte-RelativeInclude
#include "ThreadUtil.h"

namespace KINETO_NAMESPACE {

namespace uuid {
std::string generate_uuid_v4() {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  static std::uniform_int_distribution<> dis(0, 15);
  static std::uniform_int_distribution<> dis2(8, 11);

  std::stringstream ss;
  int i = 0;
  ss << std::hex;
  for (i = 0; i < 8; i++) {
    ss << dis(gen);
  }
  ss << "-";
  for (i = 0; i < 4; i++) {
    ss << dis(gen);
  }
  ss << "-4";
  for (i = 0; i < 3; i++) {
    ss << dis(gen);
  }
  ss << "-";
  ss << dis2(gen);
  for (i = 0; i < 3; i++) {
    ss << dis(gen);
  }
  ss << "-";
  for (i = 0; i < 12; i++) {
    ss << dis(gen);
  }
  return ss.str();
}
} // namespace uuid

static std::vector<int32_t> getPids() {
  const auto& pids = pidCommandPairsOfAncestors();
  std::vector<int32_t> res;
  res.reserve(pids.size());
  for (const auto& pid_pair : pids) {
    res.push_back(pid_pair.first);
  }
  return res;
}

static int64_t getJobId() {
  /* Look up a job id using env variables from job schedulers
   */
  // SLURM Job ID https://slurm.schedmd.com/sbatch.html#OPT_SLURM_JOB_ID
  const char* id = getenv("SLURM_JOB_ID");

  // Feel free to add other env variable look ups here
  // for different schedulers
  if (id == nullptr) {
    return 0;
  }
  return strtoll(id, nullptr, 10);
}

IpcFabricConfigClient::IpcFabricConfigClient()
    : jobId_(getJobId()), pids_(getPids()), ipcFabricEnabled_(true) {
  // setup IPC Fabric
  std::string ep_name = "dynoconfigclient" + uuid::generate_uuid_v4();

  fabricManager_ = ::dynolog::ipcfabric::FabricManager::factory(ep_name);
#ifdef ENABLE_IPC_FABRIC
  LOG(INFO) << "Setting up IPC Fabric at endpoint: " << ep_name << " status = "
            << (fabricManager_ ? "initialized" : "failed (null)");
#endif
}

// The following only makes sense if IPC Fabric is being used
#ifdef ENABLE_IPC_FABRIC

// Connect to the Dynolog service through Fabric name `dynolog`
constexpr const char* kDynoIpcName = "dynolog";
constexpr int maxIpcRetries = 9;
constexpr int kSleepUs = 10000;

int32_t IpcFabricConfigClient::registerInstance(int32_t gpu) {
  if (!ipcFabricEnabled_) {
    return -1;
  }

  if (!fabricManager_) {
    LOG(ERROR) << "FabricManager not initialized.";
    return -1;
  }

  // Setup message
  ::dynolog::ipcfabric::LibkinetoContext ctxt{gpu, getpid(), jobId_};

  std::unique_ptr<::dynolog::ipcfabric::Message> msg =
      ::dynolog::ipcfabric::Message::constructMessage<decltype(ctxt)>(
          ctxt, "ctxt");

  try {
    if (!fabricManager_->sync_send(*msg, std::string(kDynoIpcName))) {
      LOG(ERROR) << "Failed to register pid " << ctxt.pid
                 << " with dyno: IPC sync_send fail";
      return -1;
    }
    msg = fabricManager_->poll_recv(maxIpcRetries, kSleepUs);
    if (!msg) {
      LOG(ERROR) << "Failed to register pid " << ctxt.pid
                 << " with dyno: IPC recv fail";
      return -1;
    }
  } catch (const std::runtime_error& ex) {
    LOG(ERROR) << "Failed to send/recv registering pic over fabric: "
               << ex.what();
    return -1;
  }

  LOG(INFO) << "Registered instance with daemon";
  return *(int*)msg->buf.get();
}

std::string IpcFabricConfigClient::getLibkinetoBaseConfig() {
  if (!ipcFabricEnabled_) {
    return "";
  }

  LOG(WARNING)
      << "Missing IPC Fabric implementation for getLibkinetoBaseConfig";
  return "";
}

std::string IpcFabricConfigClient::getLibkinetoOndemandConfig(int32_t type) {
  if (!ipcFabricEnabled_) {
    return "";
  }

  if (!fabricManager_) {
    LOG(ERROR) << "FabricManager not initialized.";
    return "";
  }

  int size = pids_.size();
  ::dynolog::ipcfabric::LibkinetoRequest* req =
      (::dynolog::ipcfabric::LibkinetoRequest*)malloc(
          sizeof(::dynolog::ipcfabric::LibkinetoRequest) +
          sizeof(int32_t) * size);
  req->type = type;
  req->n = size;
  req->jobid = jobId_;
  for (int i = 0; i < size; i++) {
    req->pids[i] = pids_[i];
  }
  std::unique_ptr<::dynolog::ipcfabric::Message> msg =
      ::dynolog::ipcfabric::Message::
          constructMessage<::dynolog::ipcfabric::LibkinetoRequest, int32_t>(
              *req, "req", size);

  try {
    if (!fabricManager_->sync_send(*msg, std::string(kDynoIpcName))) {
      LOG(ERROR) << "Failed to send config type=" << type
                 << " to dyno: IPC sync_send fail";
      free(req);
      req = nullptr;
      return "";
    }
    free(req);
    msg = fabricManager_->poll_recv(maxIpcRetries, kSleepUs);
    if (!msg) {
      // LOG(ERROR) << "Failed to receive ondemand config type=" << type
      //  << " from dyno: IPC recv fail";
      return "";
    }
  } catch (const std::runtime_error& ex) {
    LOG(ERROR) << "Failed to recv ondemand config over ipc fabric: "
               << ex.what();
    free(req);
    return "";
  }

  return std::string((char*)msg->buf.get(), msg->metadata.size);
}

#else // ENABLE_IPC_FABRIC

int32_t IpcFabricConfigClient::registerInstance([[maybe_unused]] int32_t gpu) {
  return -1;
}

std::string IpcFabricConfigClient::getLibkinetoBaseConfig() {
  return "";
}
std::string IpcFabricConfigClient::getLibkinetoOndemandConfig(
    [[maybe_unused]] int32_t type) {
  return "";
}

#endif // ENABLE_IPC_FABRIC
} // namespace KINETO_NAMESPACE
#endif // __linux__
