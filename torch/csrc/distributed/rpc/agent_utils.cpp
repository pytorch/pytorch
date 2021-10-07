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

// A mutex and a cv to guard access to the call counts and watch for changes.
std::mutex barrierMutex_;

std::tuple<std::string, std::string, std::string> getNextKeyIds() {
  barrierId++;
  std::string processCountKey =
      storeKeyProcessCount + storeKeyBarrierId + std::to_string(barrierId);
  std::string activeCallCountKey =
      storeKeyActiveCallCount + storeKeyBarrierId + std::to_string(barrierId);
  std::string barrierKey =
      storeKeyReady + storeKeyBarrierId + std::to_string(barrierId);
  return std::make_tuple(processCountKey, activeCallCountKey, barrierKey);
}

// Parse the key value to return, split based on "_"
std::tuple<int, int> parseKeyValue(string keyValue) {
  std::string delimiter = "_";
  int numProcess = std::stoi(keyValue.substr(0, keyValue.find(delimiter)));
  int totalActiveCalls =
      std::stoi(keyValue.substr(keyValue.find(delimiter) + 1));
  return std::make_tuple(numProcess, totalActiveCalls);
}

// Creates a key value of form "{process_num}_{total_active_client_calls}"
std::string createKeyValue(int nextProcessNum, int totalActiveCalls) {
  return std::to_string(nextProcessNum) + "_" +
      std::to_string(totalActiveCalls);
}

// Synchronize process with all other agent processes strictly using store
// Block until all ``RpcAgent``s reach this method.
// Returns true if there are 0 active calls amongst all rpc agents
// Returns false otherwise
bool syncCallCount(
    ::c10d::PrefixStore store,
    const int worldSize,
    int activeCalls) {
  std::unique_lock<std::mutex> lock(barrierMutex_);
  std::string processCountKey, activeCallCountKey, readyKey;
  std::tie(processCountKey, activeCallCountKey, readyKey) = getNextKeyIds();

  // Add to keys which will record the number of processes and active calls
  int totalProcessCount = store.add(processCountKey, 1);
  int totalCallCount = store.add(activeCallCountKey, activeCalls);

  // The last worker will need to set the ready key
  if (totalProcessCount == worldSize) {
    store.set(readyKey, std::vector<uint8_t>());
  }

  // Wait on the ready key to be set
  store.wait(std::vector<std::string>{readyKey});
  if (totalCallCount == 0) {
    return true;
  }
  return false;
}

} // namespace rpc
} // namespace distributed
} // namespace torch
