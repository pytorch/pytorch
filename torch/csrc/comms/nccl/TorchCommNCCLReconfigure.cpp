// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <torch/csrc/comms/nccl/TorchCommNCCL.hpp>

#include <cstring>
#include <string>

#include <fmt/core.h>
#include <nccl.h> // @manual
#include <torch/csrc/distributed/c10d/PrefixStore.hpp> // @manual=//caffe2:torch-cpp-cpu
#include <torch/csrc/distributed/c10d/TCPStore.hpp> // @manual=//caffe2:torch-cpp-cpu

#include <torch/csrc/comms/utils/Logging.hpp>
#include <torch/csrc/comms/utils/TracingGuard.hpp>

namespace torch::comms {

InitHandle TorchCommNCCL::getInitHandle() const {
  std::string storeAddr;
  if (reconfigure_store_) {
    auto* tcpStore = dynamic_cast<c10d::TCPStore*>(reconfigure_store_.get());
    if (!tcpStore) {
      auto* prefixStore =
          dynamic_cast<c10d::PrefixStore*>(reconfigure_store_.get());
      if (prefixStore) {
        tcpStore = dynamic_cast<c10d::TCPStore*>(
            prefixStore->getUnderlyingNonPrefixStore().get());
      }
    }
    if (tcpStore) {
      storeAddr =
          fmt::format("{}:{}", tcpStore->getHost(), tcpStore->getPort());
    }
  }
  return fmt::format("nccl:{}:{}:{}", rank_, uuid_, storeAddr);
}

namespace {

struct HandleInfo {
  int rank;
  int64_t uuid;
  std::string storeAddr;
};

HandleInfo parseHandle(const InitHandle& handle) {
  auto first = handle.find(':');
  if (first == std::string::npos) {
    return {-1, -1, ""};
  }
  auto second = handle.find(':', first + 1);
  if (second == std::string::npos) {
    return {std::stoi(handle.substr(first + 1)), -1, ""};
  }
  int rank = std::stoi(handle.substr(first + 1, second - first - 1));
  auto third = handle.find(':', second + 1);
  if (third == std::string::npos) {
    return {rank, std::stoll(handle.substr(second + 1)), ""};
  }
  int64_t uuid = std::stoll(handle.substr(second + 1, third - second - 1));
  std::string storeAddr = handle.substr(third + 1);
  return {rank, uuid, storeAddr};
}

struct QuorumInfo {
  int64_t uuid = -1;
  std::unordered_set<int> ranks;
  size_t newMemberCount = 0;
};

QuorumInfo findQuorum(
    const std::variant<std::unordered_set<InitHandle>, std::vector<InitHandle>>&
        handles) {
  std::unordered_map<int64_t, std::vector<int>> groupByUuid;
  size_t totalHandles = 0;

  auto processHandle = [&](const InitHandle& handle) {
    totalHandles++;
    auto info = parseHandle(handle);
    groupByUuid[info.uuid].push_back(info.rank);
  };

  std::visit(
      [&](const auto& h) {
        for (const auto& handle : h) {
          processHandle(handle);
        }
      },
      handles);

  QuorumInfo quorum;
  for (const auto& [uuid, ranks] : groupByUuid) {
    if (uuid < 0) {
      continue;
    }
    std::unordered_set<int> uniqueRanks(ranks.begin(), ranks.end());
    if (uniqueRanks.size() != ranks.size()) {
      continue;
    }
    if (uniqueRanks.size() > quorum.ranks.size()) {
      quorum.uuid = uuid;
      quorum.ranks = uniqueRanks;
    }
  }

  quorum.newMemberCount = totalHandles - quorum.ranks.size();
  return quorum;
}

std::string extractStoreAddr(
    const std::variant<std::unordered_set<InitHandle>, std::vector<InitHandle>>&
        handles) {
  std::string addr;
  std::visit(
      [&](const auto& h) {
        for (const auto& handle : h) {
          auto info = parseHandle(handle);
          if (!info.storeAddr.empty()) {
            addr = info.storeAddr;
            return;
          }
        }
      },
      handles);
  return addr;
}

int findRankInHandles(
    const std::variant<std::unordered_set<InitHandle>, std::vector<InitHandle>>&
        handles,
    const InitHandle& myHandle) {
  return std::visit(
      [&](const auto& h) -> int {
        int result = -1;
        int idx = 0;
        for (const auto& handle : h) {
          if (handle == myHandle) {
            if (result >= 0) {
              return -1;
            }
            result = idx;
          }
          idx++;
        }
        return result;
      },
      handles);
}

c10::intrusive_ptr<c10d::Store> connectStore(
    const std::string& storeAddr,
    const std::string& prefix,
    std::chrono::milliseconds timeout) {
  auto colonPos = storeAddr.rfind(':');
  TORCH_INTERNAL_ASSERT(
      colonPos != std::string::npos,
      "Invalid store address in handle: ",
      storeAddr);
  std::string host = storeAddr.substr(0, colonPos);
  uint16_t port =
      static_cast<uint16_t>(std::stoi(storeAddr.substr(colonPos + 1)));

  c10d::TCPStoreOptions opts;
  opts.port = port;
  opts.isServer = false;
  opts.waitWorkers = false;
  opts.useLibUV = true;
  opts.timeout = timeout;

  return c10::make_intrusive<c10d::PrefixStore>(
      prefix, c10::make_intrusive<c10d::TCPStore>(host, opts));
}

} // namespace

// Reconfigures the communicator to a new set of peers.
//
// Each rank's init handle encodes its rank, uuid, and store address from the
// previous configuration. findQuorum() groups handles by uuid to find the
// largest set of ranks that share a common previous communicator.
//
// Cases (determined by quorum analysis):
//
// 1. Fresh init (quorum empty): No ranks share a previous communicator (first
//    call, or all ranks are new). Aborts the old comm, bootstraps ranks via
//    store counter, and calls commInitRankConfig.
//
// 2. Identity reconfigure (quorum == new world size, no new members): All
//    ranks were in the same communicator and world size is unchanged. The old
//    comm may be unhealthy (e.g. revoked after abort()), so we treat this the
//    same as case 1: abort the old comm and call commInitRankConfig.
//
// 3. Scale down then up (quorum exists, caller in quorum): Some ranks left
//    and/or new ranks joined. Calls commShrink to remove departed ranks, then
//    commGrow to add new ones. Rank 0 of the quorum shares the grow unique ID
//    via the store.
//
// 4. New rank joining (quorum exists, caller not in quorum): This rank was
//    not part of the previous communicator. Waits for the quorum's rank 0 to
//    publish a unique ID, then calls commGrow with comm=nullptr to join.
//
// Before any reconfiguration, the old communicator (if present) is cleaned up:
// pending work is revoked, memory hooks detached, timeout thread stopped, and
// the work queue finalized. If the caller is not in the quorum, the old comm
// is aborted and nulled out.
c10::intrusive_ptr<TorchWork> TorchCommNCCL::reconfigure(
    const ReconfigureOptions& opts) {
  TC_LOG(INFO, this) << "TorchCommNCCL reconfigure starting";
  TracingGuard tracingGuard(name_, comm_size_, "reconfigure", rank_);

  int new_size = static_cast<int>(
      std::visit([](const auto& h) { return h.size(); }, opts.handles));
  auto reconfigureTimeout = opts.timeout.value_or(options_.timeout);

  auto quorum = findQuorum(opts.handles);
  bool inQuorum = nccl_comm_ && uuid_ >= 0 && uuid_ == quorum.uuid;
  auto storeAddr = extractStoreAddr(opts.handles);

  // Fall back to fresh init when shrink/grow has no advantage:
  // - Single-rank quorum: a 1-rank comm has no bootstrap networking, so
  //   commGrow will fail. Must clear unconditionally (not just when
  //   inQuorum) so all ranks take the same fresh init path.
  // - Identity reconfigure: same world size, no membership change — old comm
  //   may be unhealthy.
  if (quorum.ranks.size() == 1 ||
      (inQuorum && quorum.ranks.size() == static_cast<size_t>(new_size) &&
       quorum.newMemberCount == 0)) {
    inQuorum = false;
    quorum.ranks.clear();
  }

  if (nccl_comm_) {
    auto workStatus = workq_.garbageCollect();

    if (!inQuorum &&
        (workStatus == TorchWorkNCCL::WorkStatus::NOT_STARTED ||
         workStatus == TorchWorkNCCL::WorkStatus::INPROGRESS)) {
      NCCL_CHECK_IGNORE(
          nccl_api_,
          nccl_api_->commRevoke(nccl_comm_),
          "NCCL commRevoke failed during reconfigure");
    }

    detachMemoryHook();

    if (timeout_thread_.joinable()) {
      shutdown_ = true;
      {
        std::lock_guard<std::mutex> lock(timeout_mutex_);
        timeout_cv_.notify_all();
      }
      timeout_thread_.join();
    }

    workq_.finalize();

    if (!inQuorum) {
      NCCL_CHECK_IGNORE(
          nccl_api_,
          nccl_api_->commAbort(nccl_comm_),
          "NCCL commAbort failed during reconfigure");
      nccl_comm_ = nullptr;
    }
  }

  if (quorum.ranks.empty()) {
    comm_state_ = CommState::NORMAL;
    shutdown_ = false;

    CUDA_CHECK(
        cuda_api_,
        cuda_api_->setDevice(device_.index()),
        fmt::format("Failed to set CUDA device to {}", device_.index()));

    ncclUniqueId uniqueId{};
    int myRank = findRankInHandles(opts.handles, getInitHandle());

    if (new_size > 1) {
      auto store = connectStore(
          storeAddr,
          fmt::format("{}/{}", name_, opts.uuid),
          reconfigureTimeout);

      if (myRank < 0) {
        myRank = static_cast<int>(store->add("rank_counter", 1)) - 1;
      }

      if (myRank == 0) {
        NCCL_CHECK(
            nccl_api_,
            nccl_comm_,
            nccl_api_->getUniqueId(&uniqueId),
            "NCCL getUniqueId failed during reconfigure");
        std::vector<uint8_t> vec(
            reinterpret_cast<uint8_t*>(&uniqueId),
            reinterpret_cast<uint8_t*>(&uniqueId) + sizeof(uniqueId));
        store->set("unique_id", vec);
      } else {
        store->wait({"unique_id"}, reconfigureTimeout);
        auto vec = store->get("unique_id");
        std::memcpy(&uniqueId, vec.data(), sizeof(ncclUniqueId));
      }
    } else {
      NCCL_CHECK(
          nccl_api_,
          nccl_comm_,
          nccl_api_->getUniqueId(&uniqueId),
          "NCCL getUniqueId failed during reconfigure");
    }

    ncclComm_t new_comm = nullptr;
    NCCL_CHECK(
        nccl_api_,
        nccl_comm_,
        nccl_api_->commInitRankConfig(
            &new_comm, new_size, uniqueId, myRank, nullptr),
        "NCCL commInitRankConfig failed during reconfigure");
    nccl_comm_ = new_comm;

    initNcclResources();
  } else if (inQuorum) {
    ncclComm_t current = nccl_comm_;

    if (quorum.ranks.size() < static_cast<size_t>(comm_size_)) {
      std::vector<int> excludeRanks;
      for (int r = 0; r < comm_size_; ++r) {
        if (quorum.ranks.find(r) == quorum.ranks.end()) {
          excludeRanks.push_back(r);
        }
      }

      ncclComm_t shrunk = nullptr;
      NCCL_CHECK(
          nccl_api_,
          current,
          nccl_api_->commShrink(
              current,
              excludeRanks.data(),
              static_cast<int>(excludeRanks.size()),
              &shrunk,
              nullptr,
              NCCL_SHRINK_ABORT),
          "NCCL commShrink failed during reconfigure");
      current = shrunk;
    }

    if (quorum.newMemberCount > 0) {
      int currentRank;
      NCCL_CHECK(
          nccl_api_,
          current,
          nccl_api_->commUserRank(current, &currentRank),
          "NCCL commUserRank failed during grow");

      if (currentRank == 0) {
        ncclUniqueId uniqueId{};
        NCCL_CHECK(
            nccl_api_,
            current,
            nccl_api_->commGetUniqueId(current, &uniqueId),
            "NCCL commGetUniqueId failed during grow");

        auto store = connectStore(
            storeAddr,
            fmt::format("{}/{}", name_, opts.uuid),
            reconfigureTimeout);
        std::vector<uint8_t> vec(
            reinterpret_cast<uint8_t*>(&uniqueId),
            reinterpret_cast<uint8_t*>(&uniqueId) + sizeof(uniqueId));
        store->set("unique_id", vec);
      }

      TC_LOG(INFO, this) << "TorchCommNCCL existing commGrow: rank=" << rank_
                         << " new_size=" << new_size;

      ncclComm_t grown = nullptr;
      NCCL_CHECK(
          nccl_api_,
          current,
          nccl_api_->commGrow(current, new_size, nullptr, -1, &grown, nullptr),
          "NCCL commGrow failed during reconfigure");
      current = grown;
    }

    nccl_comm_ = current;
    comm_state_ = CommState::NORMAL;
    shutdown_ = false;
    initNcclResources();
  } else {
    comm_state_ = CommState::NORMAL;
    shutdown_ = false;

    int quorumSize = static_cast<int>(quorum.ranks.size());
    auto store = connectStore(
        storeAddr, fmt::format("{}/{}", name_, opts.uuid), reconfigureTimeout);
    store->wait({"unique_id"}, reconfigureTimeout);

    auto vec = store->get("unique_id");
    ncclUniqueId uniqueId{};
    std::memcpy(&uniqueId, vec.data(), sizeof(ncclUniqueId));

    int growRank =
        quorumSize + static_cast<int>(store->add("rank_counter", 1)) - 1;

    TC_LOG(INFO, this) << "TorchCommNCCL new rank commGrow: rank=" << growRank
                       << " uniqueId=" << vec.size()
                       << " new_size=" << new_size;

    ncclComm_t new_comm = nullptr;
    NCCL_CHECK(
        nccl_api_,
        nccl_comm_,
        nccl_api_->commGrow(
            nullptr, new_size, &uniqueId, growRank, &new_comm, nullptr),
        "NCCL commGrow failed for new rank during reconfigure");

    nccl_comm_ = new_comm;
    initNcclResources();
  }

  init_state_ = InitializationState::INITIALIZED;
  uuid_ = opts.uuid;

  TC_LOG(INFO, this) << "TorchCommNCCL reconfigure completed for rank: "
                     << rank_;

  return c10::make_intrusive<TorchWorkCompleted>();
}

} // namespace torch::comms
