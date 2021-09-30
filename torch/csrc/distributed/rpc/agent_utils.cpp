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

const string barrierKeyPlaceholder = "_BARRIER_";
const string barrierReady = "_READY";
static std::atomic<int> barrierId(0);

// A mutex and a cv to guard access to the call counts and watch for changes.
std::mutex barrierMutex_;

string getNextBarrierKey() {
  barrierId++;
  return barrierKeyPlaceholder + std::to_string(barrierId);
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

// Performs a barrier with store
// An optional active_count argument can be used to determine the number
// of total active client calls among all agents
// Returns true if we are checking for calls and there are 0 active calls
// Returns false otherwise
bool barrier(
    ::c10d::PrefixStore store,
    const int worldSize,
    bool checkCalls,
    int activeCalls) {
  std::unique_lock<std::mutex> lock(barrierMutex_);
  VLOG(1) << "Starting store-based barrier";
  std::string barrierKey = getNextBarrierKey();
  bool keyCreated = false;
  std::string currKeyValue, newKeyValue, desiredValue;
  int processNum = 0;
  int totalActiveCalls = 0;
  while (true) {
    // Check if key exists
    keyCreated = store.check(std::vector<std::string>{barrierKey});

    if (!keyCreated) {
      currKeyValue = "";
      processNum = 0;
      desiredValue = createKeyValue(processNum + 1, activeCalls);
    } else {
      std::vector<uint8_t> valueArr = store.get(barrierKey);
      currKeyValue = std::string(valueArr.begin(), valueArr.end());
      std::tie(processNum, totalActiveCalls) = parseKeyValue(currKeyValue);
      desiredValue =
          createKeyValue(processNum + 1, totalActiveCalls + activeCalls);
    }
    newKeyValue = store.compareSet(barrierKey, currKeyValue, desiredValue);

    // if the value was changed successfully, this process has recorded itself
    // in the barrier
    if (newKeyValue == desiredValue) {
      break;
    }
  }

  int newProcessNum = 0;
  std::tie(newProcessNum, totalActiveCalls) = parseKeyValue(newKeyValue);
  VLOG(1) << newProcessNum << "+" << worldSize;
  // The last worker will need to set the ready key
  if (newProcessNum == worldSize) {
    store.set(barrierKey + barrierReady, std::vector<uint8_t>());
  }

  VLOG(1) << "Waiting for " << barrierKey + barrierReady;
  // wait on the ready key to be set
  store.wait(std::vector<std::string>{barrierKey + barrierReady});

  VLOG(1) << "Ending store-based barrier";
  if (checkCalls && totalActiveCalls == 0) {
    return true;
  }
  return false;
}

} // namespace rpc
} // namespace distributed
} // namespace torch
