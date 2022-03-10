#include <fmt/format.h>
#include <torch/csrc/distributed/rpc/agent_utils.h>

namespace torch {
namespace distributed {
namespace rpc {

std::unordered_map<std::string, worker_id_t> collectNames(
    ::c10d::PrefixStore store,
    const worker_id_t selfId,
    const std::string& selfName,
    const int worldSize) {
  std::vector<uint8_t> selfNameVector(
      (uint8_t*)selfName.c_str(),
      (uint8_t*)selfName.c_str() + selfName.length());
  store.set(c10::to_string(selfId), selfNameVector);

  std::unordered_map<std::string, worker_id_t> nameToId;
  nameToId.reserve(worldSize);
  nameToId.emplace(selfName, selfId);
  for (worker_id_t workerId = 0; workerId < worldSize; ++workerId) {
    if (workerId == selfId) {
      continue;
    }
    std::vector<uint8_t> workerNameVector = store.get(c10::to_string(workerId));
    std::string workerName(
        (char*)workerNameVector.data(), workerNameVector.size());

    TORCH_CHECK(
        nameToId.find(workerName) == nameToId.end(),
        "RPC worker name ",
        workerName,
        " is not unique. Workers ",
        nameToId.find(workerName)->second,
        " and ",
        workerId,
        " share the same name.");

    nameToId.emplace(workerName, workerId);
  }
  return nameToId;
}

const string storeKeyBarrierId = "_ID_";
const string storeKeyProcessCount = "PROCESS_COUNT";
const string storeKeyActiveCallCount = "ACTIVE_CALLS";
const string storeKeyReady = "READY";
static std::atomic<int> barrierId(0);

std::tuple<std::string, std::string, std::string> getNextKeyIds() {
  barrierId++;
  std::string processCountKey =
      fmt::format("{}{}{}", storeKeyProcessCount, storeKeyBarrierId, barrierId);
  std::string activeCallCountKey = fmt::format(
      "{}{}{}", storeKeyActiveCallCount, storeKeyBarrierId, barrierId);
  std::string barrierKey =
      fmt::format("{}{}{}", storeKeyReady, storeKeyBarrierId, barrierId);
  return std::make_tuple(processCountKey, activeCallCountKey, barrierKey);
}

// Synchronize process with all other agent processes strictly using store
// Block until all ``RpcAgent``s reach this method.
// Returns total number of active calls of all RPC agents in the group
int syncCallCount(
    ::c10d::PrefixStore store,
    const int worldSize,
    int activeCalls) {
  std::string processCountKey, activeCallCountKey, readyKey;
  std::tie(processCountKey, activeCallCountKey, readyKey) = getNextKeyIds();

  // Add to keys which will record the number of processes and active calls
  int totalCallCount = store.add(activeCallCountKey, activeCalls);
  int totalProcessCount = store.add(processCountKey, 1);

  // The last worker will need to set the ready key
  if (totalProcessCount == worldSize) {
    store.set(readyKey, std::vector<uint8_t>());
  }

  // Wait on the ready key to be set
  store.wait(std::vector<std::string>{readyKey});

  // Read count of active calls which may have changed
  auto activeCallCountData = store.get(activeCallCountKey);
  totalCallCount = std::stoi(
      std::string(activeCallCountData.begin(), activeCallCountData.end()));
  return totalCallCount;
}

} // namespace rpc
} // namespace distributed
} // namespace torch
