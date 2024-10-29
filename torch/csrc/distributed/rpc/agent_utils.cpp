#include <fmt/format.h>
#include <torch/csrc/distributed/rpc/agent_utils.h>

namespace torch::distributed::rpc {

std::unordered_map<std::string, worker_id_t> collectNames(
    ::c10d::PrefixStore store,
    const worker_id_t selfId,
    const std::string& selfName,
    const int worldSize) {
  std::vector<uint8_t> selfNameVector(
      (uint8_t*)selfName.c_str(),
      (uint8_t*)selfName.c_str() + selfName.length());
  store.set(std::to_string(selfId), selfNameVector);

  std::unordered_map<std::string, worker_id_t> nameToId;
  nameToId.reserve(worldSize);
  nameToId.emplace(selfName, selfId);
  // NOLINTNEXTLINE(*loop*)
  for (worker_id_t workerId = 0; workerId < worldSize; ++workerId) {
    if (workerId == selfId) {
      continue;
    }
    std::vector<uint8_t> workerNameVector = store.get(std::to_string(workerId));
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

static std::vector<std::string> splitString(
    const std::string& s,
    const std::string& delim) {
  std::vector<std::string> tokens;
  size_t start = 0;
  size_t end = 0;
  // Iterate through each delimiter
  while ((end = s.find(delim, start)) != std::string::npos) {
    tokens.emplace_back(s.substr(start, end - start));
    start = end + delim.length();
  }
  tokens.emplace_back(s.substr(start));
  return tokens;
}

const std::string allWorkerInfosKey = "_ALL_WORKER_INFOS";

std::unordered_map<std::string, worker_id_t> collectCurrentNames(
    ::c10d::PrefixStore store,
    const worker_id_t selfId,
    const std::string& selfName) {
  std::vector<uint8_t> selfNameVector(
      (uint8_t*)selfName.c_str(),
      (uint8_t*)selfName.c_str() + selfName.length());

  // Check that ID does not already exist and set {ID : NAME}
  std::vector<uint8_t> resultVector = store.compareSet(
      std::to_string(selfId), std::vector<uint8_t>(), selfNameVector);
  TORCH_CHECK(
      resultVector == selfNameVector,
      "RPC worker id ",
      selfId,
      " is not unique. Worker ",
      resultVector,
      " and already has ID and ",
      selfNameVector,
      " cannot be added.");

  store.set(std::to_string(selfId), selfNameVector);

  std::unordered_map<std::string, worker_id_t> nameToId;
  nameToId.emplace(selfName, selfId);

  // Check to see if there is list of worker names in the store
  bool worker_names_available =
      store.check(std::vector<std::string>{allWorkerInfosKey});
  std::string allWorkerInfos;
  if (worker_names_available) {
    // Get the current list of workers
    std::vector<uint8_t> allWorkerInfosKeyVector = store.get(allWorkerInfosKey);
    allWorkerInfos = std::string(
        (char*)allWorkerInfosKeyVector.data(), allWorkerInfosKeyVector.size());
    // workerInfos are comma separated with a comma at the end (e.g.
    // "Name1-Rank1,Name2-Rank2,Name3-Rank2,") parse list of workers.
    if (!allWorkerInfos.empty()) {
      for (const std::string& workerInfoString : splitString(
               allWorkerInfos.substr(0, allWorkerInfos.size() - 1), ",")) {
        auto workerInfoVec = splitString(workerInfoString, "-");
        std::string workerName = workerInfoVec.at(0);
        int workerId = std::stoi(workerInfoVec.at(1));

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
    }
  }
  // Add own name to worker list
  allWorkerInfos = fmt::format("{}{}-{},", allWorkerInfos, selfName, selfId);
  std::vector<uint8_t> allWorkerInfosVector(
      (uint8_t*)allWorkerInfos.c_str(),
      (uint8_t*)allWorkerInfos.c_str() + allWorkerInfos.length());
  store.set(allWorkerInfosKey, allWorkerInfosVector);

  return nameToId;
}

void removeCurrentName(
    ::c10d::PrefixStore store,
    const worker_id_t selfId,
    const std::string& selfName) {
  // Get current list of names/ranks
  std::vector<uint8_t> allWorkerInfosKeyVector = store.get(allWorkerInfosKey);
  std::string allWorkerInfos = std::string(
      (char*)allWorkerInfosKeyVector.data(), allWorkerInfosKeyVector.size());

  // Remove the current name and rank
  std::string str_to_erase = fmt::format("{}-{},", selfName, selfId);
  auto start_position_to_erase = allWorkerInfos.find(str_to_erase);
  allWorkerInfos.erase(start_position_to_erase, str_to_erase.length());

  // Set the new data
  std::vector<uint8_t> newAllWorkerInfosVector(
      (uint8_t*)allWorkerInfos.c_str(),
      (uint8_t*)allWorkerInfos.c_str() + allWorkerInfos.length());
  store.set(allWorkerInfosKey, newAllWorkerInfosVector);
}

const string storeKeyBarrierId = "_ID_";
const string storeKeyProcessCount = "PROCESS_COUNT";
const string storeKeyActiveCallCount = "ACTIVE_CALLS";
const string storeKeyReady = "READY";
static std::atomic<int> barrierId(0);

static std::tuple<std::string, std::string, std::string> getNextKeyIds() {
  barrierId++;
  auto newBarrierId = barrierId.load();
  std::string processCountKey = fmt::format(
      "{}{}{}", storeKeyProcessCount, storeKeyBarrierId, newBarrierId);
  std::string activeCallCountKey = fmt::format(
      "{}{}{}", storeKeyActiveCallCount, storeKeyBarrierId, newBarrierId);
  std::string barrierKey =
      fmt::format("{}{}{}", storeKeyReady, storeKeyBarrierId, newBarrierId);
  return std::make_tuple(
      std::move(processCountKey),
      std::move(activeCallCountKey),
      std::move(barrierKey));
}

// Synchronize process with all other agent processes strictly using store
// Block until all ``RpcAgent``s reach this method.
// Returns total number of active calls of all RPC agents in the group
int syncCallCount(
    ::c10d::PrefixStore store,
    const int worldSize,
    int activeCalls) {
  auto [processCountKey, activeCallCountKey, readyKey] = getNextKeyIds();

  // Add to keys which will record the number of processes and active calls
  store.add(activeCallCountKey, activeCalls);
  auto totalProcessCount = store.add(processCountKey, 1);

  // The last worker will need to set the ready key
  if (totalProcessCount == worldSize) {
    store.set(readyKey, std::vector<uint8_t>());
  }

  // Wait on the ready key to be set
  store.wait(std::vector<std::string>{readyKey});

  // Read count of active calls which may have changed
  auto activeCallCountData = store.get(activeCallCountKey);
  int totalCallCount = std::stoi(
      std::string(activeCallCountData.begin(), activeCallCountData.end()));
  return totalCallCount;
}

} // namespace torch::distributed::rpc
