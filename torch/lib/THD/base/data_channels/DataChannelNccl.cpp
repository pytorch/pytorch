#include "DataChannelNccl.hpp"
#include "../Cuda.hpp"
#include "DataChannelUtils.hpp"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAGuard.h>

#include <THC/THC.h>
#include <cuda.h>

#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace thd {

namespace {

std::unordered_map<THDReduceOp, ncclRedOp_t> ncclOp = {
    {THDReduceOp::THDReduceMIN, ncclMin},
    {THDReduceOp::THDReduceMAX, ncclMax},
    {THDReduceOp::THDReduceSUM, ncclSum},
    {THDReduceOp::THDReducePRODUCT, ncclProd},
};

std::unordered_map<at::ScalarType, ncclDataType_t> ncclDatatype = {
    {at::kChar, ncclInt8},
    {at::kByte, ncclUint8},
    {at::kFloat, ncclFloat},
    {at::kDouble, ncclDouble},
    {at::kInt, ncclInt32},
    {at::kLong, ncclInt64},
    {at::kHalf, ncclHalf},
};

// Helper function that gets the data type and issues error if not supported
static ncclDataType_t _getNcclDataType(at::ScalarType type) {
  try {
    return ncclDatatype.at(type);
  } catch (std::out_of_range& e) {
    throw std::runtime_error("Unsupported data type for NCCL backend");
  }
}

// Helper function that gets the device list to determine the CUDA devices
std::vector<int> getDevicesList(const std::string& deviceSeq) {
  std::stringstream ss(deviceSeq);
  std::string device;
  std::vector<int> devices;
  while (std::getline(ss, device, ',')) {
    devices.push_back(stoi(device));
  }
  return devices;
}

} // namespace

// DataChannelNccl
DataChannelNccl::DataChannelNccl(InitMethod::Config config, int timeout)
    : _rank(config.rank),
      _numProcesses(config.world_size),
      _timeout(timeout),
      _masterListeningSocket(-1),
      _slaveSocket(-1) {
  // Establish the socket connections from rank 0 to all others
  if (_rank == 0) {
    _masterListeningSocket = config.master.listen_socket;
    _masterSendingSockets = std::vector<int>(_numProcesses - 1, -1);

    try {
      for (rank_type i = 0; i < _numProcesses - 1; ++i) {
        std::tie(_masterSendingSockets[i], std::ignore) =
            accept(_masterListeningSocket, _timeout);
      }
    } catch (...) {
      // Destroy the created sockets
      _destroySockets();
      throw std::runtime_error("Rank 0 cannot establish thelistening socket");
    }

  } else {
    _masterAddr = config.worker.master_addr;
    _masterPort = config.worker.master_port;

    try {
      _slaveSocket = connect(_masterAddr, _masterPort, true, _timeout);
    } catch (...) {
      // Destroy the created sockets
      _destroySockets();
      std::string errStr = "Rank: " + std::to_string(_rank) +
          " cannot "
          "connect to the master: " +
          _masterAddr + ":" + std::to_string(_masterPort);
      throw std::runtime_error(errStr);
    }
  }
}

// Use the socket to broadcast NCCL ID
void DataChannelNccl::broadcastUniqueNcclId(ncclUniqueId* ncclId) {
  // Send the unique NCCL id to every rank
  if (_rank == 0) {
    for (auto socket : _masterSendingSockets) {
      send_bytes<uint8_t>(
          socket, reinterpret_cast<uint8_t*>(ncclId), NCCL_UNIQUE_ID_BYTES);
    }
  } else {
    recv_bytes<uint8_t>(
        _slaveSocket, reinterpret_cast<uint8_t*>(ncclId), NCCL_UNIQUE_ID_BYTES);
  }
}

// Destructor will only close all the sockets
DataChannelNccl::~DataChannelNccl() {
  /**
   * Note that destructor will be called after cudaruntime being unloaded since
   * DataChannel is a global variable.
   */
  _destroySockets();
}

void DataChannelNccl::_destroySockets() {
  // Destroying all the socket
  if (_masterListeningSocket != -1) {
    ::close(_masterListeningSocket);
    _masterListeningSocket = -1;
  }
  if (_slaveSocket != -1) {
    ::close(_slaveSocket);
    _slaveSocket = -1;
  }
  for (size_t i = 0; i < _masterSendingSockets.size(); ++i) {
    if (_masterSendingSockets[i] != -1) {
      ::close(_masterSendingSockets[i]);
      _masterSendingSockets[i] = -1;
    }
  }
}

// Destroy the data channel
void DataChannelNccl::destroy() {
  std::unique_lock<std::mutex> channelLock(_mutex);

  // Destroying all the socket
  _destroySockets();

  // Guard GPU device
  at::cuda::OptionalCUDAGuard gpuGuard;

  /**
   * Destroy the CUDA and NCCL resources
   * TODO: creating C++ wrappers for CUDA and NCCL resources to do the
   *       cleanup automatically
   */
  for (auto& itemPair : _groupNcclResources) {
    auto groupId = itemPair.first;
    _destroyNcclResources(groupId);
  }

  _groupNcclResources.clear();
  _groupDevices.clear();

  _groups.clear();
}

// Helper function that destroys the CUDA event and NCCL communicator
void DataChannelNccl::_destroyNcclResources(THDGroup groupId) {
  if (_groupNcclResources.find(groupId) != _groupNcclResources.end()) {
    for (int i = 0; i < _groupDevices[groupId].size(); i++) {
      // Devices used for this group ID
      auto devices = getDevicesList(_groupDevices[groupId][i]);
      // Guard GPU device
      at::cuda::OptionalCUDAGuard gpuGuard;
      // Destroy the CUDA events
      size_t idx = 0;
      for (auto& event : *(_groupNcclResources[groupId][i].ncclCudaEvents())) {
        gpuGuard.set_index(devices[idx++]);
        THCudaCheck(cudaEventSynchronize(event));
        THCudaCheck(cudaEventDestroy(event));
      }
      // Destroy the communicators
      for (auto& comm : *(_groupNcclResources[groupId][i].ncclComms())) {
        NCCL_CHECK(ncclCommDestroy(comm));
      }
    }
  }
}

// Destroy the cached NCCL resource associated with a given group
void DataChannelNccl::clearGroupCache(THDGroup groupId) {
  std::unique_lock<std::mutex> channelLock(_mutex);

  _destroyNcclResources(groupId);

  _groupNcclResources.erase(groupId);
  _groupDevices.erase(groupId);
}

// Initialization function
bool DataChannelNccl::init() {
  std::vector<rank_type> ranks;
  ranks.reserve(_numProcesses);

  for (rank_type rank = 0; rank < _numProcesses; ++rank) {
    ranks.push_back(rank);
  }

  // Insert the current group
  _groups.insert({THDGroupWORLD, DataChannel::Group(ranks, _numProcesses - 1)});

  // Get the GPU count
  THCudaCheck(cudaGetDeviceCount(&_numGPUs));

  return true;
}

rank_type DataChannelNccl::getRank() {
  return _rank;
}

rank_type DataChannelNccl::getNumProcesses() {
  return _numProcesses;
}

NcclResourcePair DataChannelNccl::_getNcclResourcePair(
    std::vector<at::Tensor>& input,
    THDGroup groupId) {
  if (input.empty()) {
    throw std::runtime_error(
        "Not able to create/get the Nccl Comm since "
        "input tensor is empty");
  }
  // Get the deviceList String
  std::string deviceList;
  for (auto tensor : input) {
    if (deviceList.empty()) {
      deviceList = std::to_string(tensor.get_device());
    } else {
      deviceList += "," + std::to_string(tensor.get_device());
    }
  }

  int index = -1;

  if (_groupDevices.find(groupId) != _groupDevices.end()) {
    auto pos = std::find(
        _groupDevices[groupId].begin(),
        _groupDevices[groupId].end(),
        deviceList);
    if (pos != _groupDevices[groupId].end())
      index = pos - _groupDevices[groupId].begin();
  }

  if (index >= 0) {
    return std::make_pair(
        _groupNcclResources[groupId][index].ncclComms(),
        _groupNcclResources[groupId][index].ncclCudaEvents());
  }

  // Add in the device list of the group
  _groupDevices[groupId].push_back(deviceList);

  // NCCL communicator
  auto comms =
      std::unique_ptr<std::vector<ncclComm_t>>(new std::vector<ncclComm_t>());

  comms->resize(input.size());

  // Corresponding CUDA events
  auto events =
      std::unique_ptr<std::vector<cudaEvent_t>>(new std::vector<cudaEvent_t>());

  events->resize(input.size());

  // Create the unique NCCL ID and broadcast it
  ncclUniqueId ncclId;
  NCCL_CHECK(ncclGetUniqueId(&ncclId));

  // Broadcast so that each process can have a unique NCCL ID
  broadcastUniqueNcclId(&ncclId);

  // Guard GPU device
  at::cuda::OptionalCUDAGuard gpuGuard;

  // Now creating the CUDA events
  for (size_t i = 0; i < input.size(); ++i) {
    gpuGuard.set_index(input[i].get_device());
    THCudaCheck(cudaEventCreate(&((*events)[i])));
  }
  // Create the communicator on each device of the input
  NCCL_CHECK(ncclGroupStart());
  for (size_t i = 0; i < input.size(); ++i) {
    int nRanks = int(_numProcesses) * input.size();
    gpuGuard.set_index(input[i].get_device());
    NCCL_CHECK(ncclCommInitRank(
        &((*comms)[i]), nRanks, ncclId, _rank * input.size() + i));
  }
  NCCL_CHECK(ncclGroupEnd());

  // Move into the hash table
  if (_groupNcclResources.find(groupId) == _groupNcclResources.end())
    _groupNcclResources.emplace(
        std::make_pair(groupId, std::vector<NcclResources>()));

  _groupNcclResources[groupId].push_back(
      NcclResources(std::move(comms), std::move(events)));

  return std::make_pair(
      _groupNcclResources[groupId].back().ncclComms(),
      _groupNcclResources[groupId].back().ncclCudaEvents());
}

// Helper function that checks the input and output tensors for validity
bool DataChannelNccl::_tensorCheckHelper(
    const std::vector<at::Tensor>& input,
    const std::vector<at::Tensor>& output,
    size_t outputOverInput) {
  if (input.size() != output.size()) {
    throw std::runtime_error(
        "Input tensor sequence should have the same "
        "number of tensors as the output tensor sequence");
  }

  if (input.size() == 0) {
    // Return false saying this is a no-op
    return false;
  }

  if (input.size() > _numGPUs) {
    throw std::runtime_error(
        "The number of input tensors is larger than "
        "the number of available GPUs");
  }

  // To make sure each tensor is on separate devices
  std::unordered_set<int> usedDevices;
  usedDevices.reserve(input.size());

  uint64_t inputNumElement = input[0].numel();
  auto elementType = input[0].type().scalarType();

  for (size_t i = 0; i < input.size(); ++i) {
    //  Check to make sure it's a GPU dense tensor
    if (!(input[i].is_cuda() && !input[i].is_sparse() &&
          output[i].is_cuda() && !output[i].is_sparse())) {
      throw std::runtime_error(
          "Only CUDA dense tensor is supported for NCCL "
          "collective operations");
    }
    // Check the tensor type is identical
    if (input[i].type().scalarType() != elementType ||
        output[i].type().scalarType() != elementType) {
      throw std::runtime_error(
          "Expecting all GPU tensors to have identical "
          "type");
    }
    // Check the input tensor size is identical
    if (input[i].numel() != inputNumElement) {
      throw std::runtime_error(
          "Expecting all input tensors to have identical "
          "number of elements");
    }
    // Check the output tensor size equals to input tensor size
    if (output[i].numel() != inputNumElement * outputOverInput) {
      throw std::runtime_error(
          "The number of elements of output tensor does "
          "not match the number of elements of the input "
          "tensor");
    }
    // Contiguous verification
    if (!input[i].is_contiguous() || !output[i].is_contiguous()) {
      throw std::runtime_error("Expecting all GPU tensors to be contiguous");
    }

    bool inserted;
    std::tie(std::ignore, inserted) = usedDevices.insert(input[i].get_device());
    // Device verification, if the insertion didn't take place
    if (!inserted) {
      throw std::runtime_error("Expecting inputs on different GPU devices");
    }

    // Now check the output device
    if (input[i].get_device() != output[i].get_device()) {
      throw std::runtime_error(
          "Expecting input and output tensors to be on "
          "the same device");
    }
  }
  return true;
}

void DataChannelNccl::allReduce(
    std::vector<at::Tensor>& data,
    THDReduceOp operation,
    THDGroup groupId) {
  std::unique_lock<std::mutex> channelLock(_mutex);
  // Check the tensor vector for consistency
  if (!_tensorCheckHelper(data, data)) {
    return;
  }
  _checkGroupIdValid(groupId);

  auto ncclResourcePair = _getNcclResourcePair(data, groupId);
  auto comms = ncclResourcePair.first;
  auto events = ncclResourcePair.second;

  // Guard GPU device
  at::cuda::OptionalCUDAGuard gpuGuard;

  std::unique_lock<std::mutex> cudaFreeMutexLock(
      *(THCCachingAllocator_getCudaFreeMutex()));

  NCCL_CHECK(ncclGroupStart());
  for (size_t i = 0; i < data.size(); ++i) {
    gpuGuard.set_index(data[i].get_device());
    auto stream = THCState_getCurrentStream(THDGetCudaState());

    NCCL_CHECK(ncclAllReduce(
        data[i].data_ptr(),
        data[i].data_ptr(),
        data[i].numel(),
        _getNcclDataType(data[i].type().scalarType()),
        ncclOp[operation],
        (*comms)[i],
        stream));
    THCudaCheck(cudaEventRecord((*events)[i], stream));
  }
  NCCL_CHECK(ncclGroupEnd());

  cudaFreeMutexLock.unlock();
}

void DataChannelNccl::allReduce(
    at::Tensor& data,
    THDReduceOp operation,
    THDGroup groupId) {
  std::vector<at::Tensor> dataVec = {data};
  allReduce(dataVec, operation, groupId);
}

void DataChannelNccl::allGather(
    std::vector<at::Tensor>& output,
    std::vector<at::Tensor>& input,
    THDGroup groupId) {
  std::unique_lock<std::mutex> channelLock(_mutex);

  if (!_tensorCheckHelper(input, output, _numProcesses * input.size())) {
    return;
  }
  _checkGroupIdValid(groupId);

  auto ncclResourcePair = _getNcclResourcePair(input, groupId);
  auto comms = ncclResourcePair.first;
  auto events = ncclResourcePair.second;

  // Guard GPU device
  at::cuda::OptionalCUDAGuard gpuGuard;

  std::unique_lock<std::mutex> cudaFreeMutexLock(
      *(THCCachingAllocator_getCudaFreeMutex()));

  NCCL_CHECK(ncclGroupStart());
  for (size_t i = 0; i < input.size(); ++i) {
    gpuGuard.set_index(input[i].get_device());
    auto stream = THCState_getCurrentStream(THDGetCudaState());

    NCCL_CHECK(ncclAllGather(
        input[i].data_ptr(),
        output[i].data_ptr(),
        input[i].numel(),
        _getNcclDataType(input[i].type().scalarType()),
        (*comms)[i],
        stream));
    THCudaCheck(cudaEventRecord((*events)[i], stream));
  }
  NCCL_CHECK(ncclGroupEnd());

  cudaFreeMutexLock.unlock();
}

void DataChannelNccl::allGather(
    std::vector<at::Tensor>& output,
    at::Tensor& input,
    THDGroup groupId) {
  std::vector<at::Tensor> inputDataVec = {input};
  allGather(output, inputDataVec, groupId);
}

void DataChannelNccl::reduce(
    std::vector<at::Tensor>& data,
    THDReduceOp operation,
    rank_type dstRank,
    THDGroup groupId) {
  std::unique_lock<std::mutex> channelLock(_mutex);

  // Check the tensor vector for consistency
  if (!_tensorCheckHelper(data, data)) {
    return;
  }
  _checkGroupIdValid(groupId);

  auto ncclResourcePair = _getNcclResourcePair(data, groupId);
  auto comms = ncclResourcePair.first;
  auto events = ncclResourcePair.second;

  // Guard GPU device
  at::cuda::OptionalCUDAGuard gpuGuard;

  std::unique_lock<std::mutex> cudaFreeMutexLock(
      *(THCCachingAllocator_getCudaFreeMutex()));

  NCCL_CHECK(ncclGroupStart());
  for (size_t i = 0; i < data.size(); ++i) {
    gpuGuard.set_index(data[i].get_device());
    auto stream = THCState_getCurrentStream(THDGetCudaState());

    NCCL_CHECK(ncclReduce(
        data[i].data_ptr(),
        data[i].data_ptr(),
        data[i].numel(),
        _getNcclDataType(data[i].type().scalarType()),
        ncclOp[operation],
        dstRank * data.size(),
        (*comms)[i],
        stream));
    THCudaCheck(cudaEventRecord((*events)[i], stream));
  }
  NCCL_CHECK(ncclGroupEnd());

  cudaFreeMutexLock.unlock();
}

void DataChannelNccl::reduce(
    at::Tensor& data,
    THDReduceOp operation,
    rank_type dstRank,
    THDGroup groupId) {
  std::vector<at::Tensor> dataVec = {data};
  reduce(dataVec, operation, dstRank, groupId);
}

void DataChannelNccl::broadcast(
    std::vector<at::Tensor>& data,
    rank_type srcRank,
    THDGroup groupId) {
  std::unique_lock<std::mutex> channelLock(_mutex);

  // Check the tensor vector for consistency
  if (!_tensorCheckHelper(data, data)) {
    return;
  }
  _checkGroupIdValid(groupId);

  auto ncclResourcePair = _getNcclResourcePair(data, groupId);
  auto comms = ncclResourcePair.first;
  auto events = ncclResourcePair.second;

  // Guard GPU device
  at::cuda::OptionalCUDAGuard gpuGuard;

  std::unique_lock<std::mutex> cudaFreeMutexLock(
      *(THCCachingAllocator_getCudaFreeMutex()));

  NCCL_CHECK(ncclGroupStart());
  for (size_t i = 0; i < data.size(); ++i) {
    gpuGuard.set_index(data[i].get_device());
    auto stream = THCState_getCurrentStream(THDGetCudaState());

    NCCL_CHECK(ncclBcast(
        data[i].data_ptr(),
        data[i].numel(),
        _getNcclDataType(data[i].type().scalarType()),
        srcRank * data.size(),
        (*comms)[i],
        stream));
    THCudaCheck(cudaEventRecord((*events)[i], stream));
  }
  NCCL_CHECK(ncclGroupEnd());

  cudaFreeMutexLock.unlock();
}

void DataChannelNccl::broadcast(
    at::Tensor& data,
    rank_type srcRank,
    THDGroup groupId) {
  std::vector<at::Tensor> dataVec = {data};
  broadcast(dataVec, srcRank, groupId);
}

void DataChannelNccl::barrier(THDGroup groupId) {
  throw std::runtime_error("DataChannelNccl does not support barrier");
}

THDGroup DataChannelNccl::newGroup(const std::vector<rank_type>& ranks) {
  /**
   * Check if the input rank is a full group since
   * NCCL data channel currently doesn't support sub-group creation
   */
  std::vector<rank_type> ranksToCompare = std::vector<rank_type>(ranks);
  std::sort(ranksToCompare.begin(), ranksToCompare.end());
  for (size_t i = 0; i < ranksToCompare.size(); ++i) {
    if (ranksToCompare[i] != static_cast<rank_type>(i)) {
      throw std::runtime_error(
          "NCCL backend currently only supports fullgroup "
          "creation. In other words, every rank in the "
          "process group needs to be a member of the new "
          "group to be created and sub-group creation is "
          "currently not supported.");
    }
  }

  std::unique_lock<std::mutex> channelLock(_mutex);

  auto newGroup = DataChannel::Group(ranks, _numProcesses - 1);
  THDGroup newGroupId = static_cast<THDGroup>(_groups.size());

  // Insert the current group
  _groups.insert({newGroupId, newGroup});

  return newGroupId;
}

// Helper function that checks if the given groupId is valid
void DataChannelNccl::_checkGroupIdValid(THDGroup groupId) {
  if (_groups.find(groupId) == _groups.end()) {
    std::string errMsg =
        "Group ID: " + std::to_string(groupId) + " is not valid";
    throw std::runtime_error(errMsg);
  }
}

void DataChannelNccl::gather(
    std::vector<at::Tensor>& output,
    at::Tensor& input,
    rank_type dstRank,
    THDGroup groupId) {
  throw std::runtime_error("DataChannelNccl does not support gather");
}

void DataChannelNccl::scatter(
    std::vector<at::Tensor>& input,
    at::Tensor& output,
    rank_type srcRank,
    THDGroup groupId) {
  throw std::runtime_error("DataChannelNccl does not support scatter");
}

void DataChannelNccl::send(Scalar& data, rank_type dstRank) {
  throw std::runtime_error("DataChannelNccl does not support send");
}

void DataChannelNccl::send(at::Tensor& data, rank_type dstRank) {
  throw std::runtime_error("DataChannelNccl does not support send");
}

void DataChannelNccl::receive(Scalar& data, rank_type srcRank) {
  throw std::runtime_error("DataChannelNccl does not support receive");
}

rank_type DataChannelNccl::receive(at::Tensor& data) {
  throw std::runtime_error(
      "DataChannelNccl does not support receive "
      "from any source");
}

void DataChannelNccl::receive(at::Tensor& data, rank_type srcRank) {
  throw std::runtime_error("DataChannelNccl does not support receive");
}

DataChannelNccl::RequestNccl* DataChannelNccl::isend(
    at::Tensor& data,
    rank_type dstRank) {
  throw std::runtime_error("DataChannelNccl does not support isend");
}

DataChannelNccl::RequestNccl* DataChannelNccl::ireceive(
    at::Tensor& data,
    rank_type srcRank) {
  throw std::runtime_error("DataChannelNccl does not support ireceive");
}

} // namespace thd
