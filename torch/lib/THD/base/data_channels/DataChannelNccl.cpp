#include "../Cuda.hpp"
#include "DataChannelNccl.hpp"
#include "DataChannelUtils.hpp"

#include <cuda.h>
#include <THC/THC.h>

#include <unistd.h>

#include <cstdint>
#include <stdexcept>
#include <unordered_set>
#include <sstream>

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
};


// Helper function that gets the data type and issues error if not supported
static ncclDataType_t _getNcclDataType(at::ScalarType type) {
  if (ncclDatatype.find(type) == ncclDatatype.end()) {
    throw std::runtime_error("Unsupported data type for NCCL");
  }
  return ncclDatatype[type];
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


// RequestNccl
DataChannelNccl::RequestNccl::RequestNccl(QueueWorker::Request&& request)
 : _request(std::move(request)) {
}


DataChannelNccl::RequestNccl::~RequestNccl() {}


bool DataChannelNccl::RequestNccl::isCompleted() {
  return _request.isCompleted();
}

void DataChannelNccl::RequestNccl::wait() {
  _request.wait();
}

// End of RequestNccl

// DataChannelNccl
DataChannelNccl::DataChannelNccl(InitMethod::Config config, int timeout)
  : _rank(config.rank)
  , _numProcesses(config.world_size)
  , _timeout(timeout)
  , _masterListeningSocket(-1)
{
  if (_rank == 0) {
    _masterListeningSocket = config.master.listen_socket;
  } else {
    _masterAddr = config.worker.master_addr;
    _masterPort = config.worker.master_port;
  }
}


// Use the socket to broadcast NCCL ID
void DataChannelNccl::broadcastUniqueNcclId(ncclUniqueId* srcNcclId,
                                            ncclUniqueId* dstNcclId) {
  // Send the unique NCCL id to every rank
  if (_rank == 0) {
    std::vector<int> sockets(_numProcesses - 1);
    for (rank_type i = 0; i < _numProcesses - 1; ++i) {
      std::tie(sockets[i], std::ignore) = accept(_masterListeningSocket,
                                                 _timeout);
    }
    for (auto socket : sockets) {
      ResourceGuard socket_guard([socket]() { ::close(socket); });
      send_bytes<uint8_t>(socket,
                          reinterpret_cast<uint8_t*>(srcNcclId),
                          NCCL_UNIQUE_ID_BYTES);
    }
  } else {
    int socket;
    try {
      socket = connect(_masterAddr, _masterPort, true, _timeout);
    } catch (...) {
      std::string errStr = "Rank: " + std::to_string(_rank) + " cannot "
                           "connect to the master: " + _masterAddr + ":" +
                           std::to_string(_masterPort);
      throw std::runtime_error(errStr);
    }
    ResourceGuard socket_guard([socket]() { ::close(socket); });
    recv_bytes<uint8_t>(socket,
                        reinterpret_cast<uint8_t*>(dstNcclId),
                        NCCL_UNIQUE_ID_BYTES);
  }
}


// Destructor that closes the master's listening socket
DataChannelNccl::~DataChannelNccl() {
  // Destroy the master listening socket
  if (_masterListeningSocket != -1) {
    ::close(_masterListeningSocket);
    _masterListeningSocket = -1;
  }
  /**
   * Note that destructor will be called after cudaruntime being unloaded since
   * DataChannel is a global variable.
   */
}


// Destroy the data channel
void DataChannelNccl::destroy() {

  std::unique_lock<std::mutex> channelLock(_mutex);

  // Destroy the master listening socket
  if (_masterListeningSocket != -1) {
    ::close(_masterListeningSocket);
    _masterListeningSocket = -1;
  }
  int curDevice = 0;
  THCudaCheck(cudaGetDevice(&curDevice));

  // Destroy the CUDA and NCCL resources
  for (auto& itemPair : _ncclCommsAndEvents) {

    auto devices = getDevicesList(_groupDevices[itemPair.first]);

    // Destroy the CUDA events
    size_t idx = 0;
    for (auto& event : *(itemPair.second.second)) {
      THCudaCheck(cudaSetDevice(devices[idx++]));
      THCudaCheck(cudaEventSynchronize(event));
      THCudaCheck(cudaEventDestroy(event));
    }
    // Destroy the communicators
    for (auto& comm : *(itemPair.second.first)) {
      NCCL_CHECK(ncclCommDestroy(comm));
    }

  }
  _ncclCommsAndEvents.clear();
  _groups.clear();
  _groupDevices.clear();

  // Restore to previous device
  THCudaCheck(cudaSetDevice(curDevice));
}


// Destroy the resource for a single thread group
void DataChannelNccl::destroyGroup(THDGroup groupId) {

  std::unique_lock<std::mutex> channelLock(_mutex);

  if (_ncclCommsAndEvents.find(groupId) != _ncclCommsAndEvents.end()) {
    int curDevice = 0;
    THCudaCheck(cudaGetDevice(&curDevice));

    // Destroy the CUDA events
    size_t idx = 0;
    for (auto& event : *(_ncclCommsAndEvents[groupId].second)) {
      auto devices = getDevicesList(_groupDevices[groupId]);
      THCudaCheck(cudaSetDevice(devices[idx++]));
      THCudaCheck(cudaEventSynchronize(event));
      THCudaCheck(cudaEventDestroy(event));
    }
    // Destroy the communicators
    for (auto& comm : *(_ncclCommsAndEvents[groupId].first)) {
      NCCL_CHECK(ncclCommDestroy(comm));
    }
    // Restore to previous device
    THCudaCheck(cudaSetDevice(curDevice));
    _ncclCommsAndEvents.erase(groupId);
  }
  if (_groups.find(groupId) != _groups.end()) {
    _groups.erase(groupId);
  }
  if (_groupDevices.find(groupId) != _groupDevices.end()) {
    _groupDevices.erase(groupId);
  }
}


// Initialization function
bool DataChannelNccl::init() {

  std::vector<rank_type> ranks;
  ranks.reserve(_numProcesses);

  for (rank_type rank = 0; rank < _numProcesses; ++rank) {
    ranks.push_back(rank);
  }

  // Insert the current group
  _groups.insert({
    THDGroupWORLD,
    DataChannel::Group(ranks, _numProcesses - 1)
  });

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


std::pair<std::vector<ncclComm_t>*, std::vector<cudaEvent_t>*>
DataChannelNccl::_getNcclCommsAndEvents(
    std::vector<at::Tensor>& input,
    THDGroup groupId) {

  if (input.empty()) {
    throw std::runtime_error("Not able to create/get the Nccl Comm since "
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

  if (_groupDevices.find(groupId) != _groupDevices.end() &&
      deviceList != _groupDevices[groupId]) {
    std::string errMsg;
    errMsg = "The current group: " + std::to_string(groupId) +
            " has already got a GPU device list: " + _groupDevices[groupId]
            + " associated with. Each group should only be associated with a "
            + "given device list. Please create a new group for your provided "
            + "device list: " + deviceList;
    throw std::runtime_error(errMsg);
  }

  if (_ncclCommsAndEvents.find(groupId) != _ncclCommsAndEvents.end()) {
    return std::make_pair(_ncclCommsAndEvents[groupId].first.get(),
                          _ncclCommsAndEvents[groupId].second.get());
  }

  // Add in the device list of the group
  _groupDevices[groupId] = deviceList;

  // NCCL communication world
  std::unique_ptr<std::vector<ncclComm_t>> comms =
    std::unique_ptr<std::vector<ncclComm_t>>(new std::vector<ncclComm_t>());

  comms->resize(input.size());

  // Corresponding CUDA events
  std::unique_ptr<std::vector<cudaEvent_t>> events =
    std::unique_ptr<std::vector<cudaEvent_t>>(new std::vector<cudaEvent_t>());

  events->resize(input.size());

  // Create the unique NCCL ID and broadcast it
  ncclUniqueId ncclId;
  NCCL_CHECK(ncclGetUniqueId(&ncclId));

  // Broadcast so that each process can have a unique NCCL ID
  broadcastUniqueNcclId(&ncclId, &ncclId);

  int curDevice = 0;
  THCudaCheck(cudaGetDevice(&curDevice));

  // Now creating the CUDA events
  for (size_t i = 0; i < input.size(); ++i) {
    THCudaCheck(cudaSetDevice(input[i].get_device()));
    THCudaCheck(cudaEventCreate(&((*events)[i])));
  }
  // Create the communicator on each device of the input
  NCCL_CHECK(ncclGroupStart());
  for (size_t i = 0; i < input.size(); ++i) {
    int nRanks = int(_numProcesses) * input.size();
    THCudaCheck(cudaSetDevice(input[i].get_device()));
    NCCL_CHECK(ncclCommInitRank(&((*comms)[i]),
                                nRanks,
                                ncclId,
                                _rank * input.size() + i));
  }
  NCCL_CHECK(ncclGroupEnd());

  // Restore to previous device
  THCudaCheck(cudaSetDevice(curDevice));

  // Move into the hash table
  _ncclCommsAndEvents[groupId] = std::move(std::make_pair(std::move(comms),
                                                          std::move(events)));

  return std::make_pair(_ncclCommsAndEvents[groupId].first.get(),
                        _ncclCommsAndEvents[groupId].second.get());
}


// Helper function that checks the input and output tensors for validity
void DataChannelNccl::_tensorCheckHelper(
    const std::vector<at::Tensor>& input,
    const std::vector<at::Tensor>& output,
    size_t outputOverInput) {

  if (input.size() <= 0) {
    throw std::runtime_error("Input tensor sequence cannot be empty");
  }
  if (input.size() != output.size()) {
    throw std::runtime_error("Input tensor sequence should have the same size "
                             "as output tensor sequence");
  }
  if (input.size() > _numGPUs) {
    throw std::runtime_error("The number of input tensors is larger than "
                             "the number of available GPUs");
  }

  // To make sure each tensor is on separate devices
  std::unordered_set<int> usedDevices;
  usedDevices.reserve(input.size());

  uint64_t inputNumElement = input[0].numel();
  auto elementType = input[0].type().scalarType();

  for (size_t i = 0; i < input.size(); ++i) {

    //  Check to make sure it's a GPU dense tensor
    if (!(input[i].type().isCuda() && !input[i].type().isSparse() &&
          output[i].type().isCuda()  && !output[i].type().isSparse())) {
      throw std::runtime_error("Only CUDA dense tensor is supported for NCCL "
                               "collective operations");
    }
    // Check the tensor type is identical
    if (input[i].type().scalarType() != elementType ||
        output[i].type().scalarType() != elementType) {
      throw std::runtime_error("Expecting all GPU tensors to have identical "
                               "type");
    }
    // Check the input tensor size is identical
    if (input[i].numel() != inputNumElement) {
      throw std::runtime_error("Expecting all input tensors to have identical "
                               "number of elements");
    }
    // Check the output tensor size equals to input tensor size
    if (output[i].numel() != inputNumElement * outputOverInput) {
      throw std::runtime_error("The number of elements of output tensor does "
                               "not match the number of elements of the input "
                               "tensor");
    }
    // Contiguous verification
    if (!input[i].is_contiguous() || !output[i].is_contiguous()) {
      throw std::runtime_error("Expecting all GPU tensors to be contiguous");
    }
    // Device verification
    if (usedDevices.find(input[i].get_device()) != usedDevices.end()) {
      throw std::runtime_error("Expecting inputs on different GPU devices");
    }

    usedDevices.insert(input[i].get_device());

    // Now check the output device
    if (input[i].get_device() != output[i].get_device()) {
      throw std::runtime_error("Expecting input and output tensors to be on "
                               "the same device");
    }
  }
}


void DataChannelNccl::allReduce(std::vector<at::Tensor>& input,
                                std::vector<at::Tensor>& output,
                                THDReduceOp operation,
                                THDGroup groupId) {

  std::unique_lock<std::mutex> channelLock(_mutex);
  // Check the tensor vector for consistency
  _tensorCheckHelper(input, output);
  _checkGroupIdValid(groupId);

  auto commsAndEvents = _getNcclCommsAndEvents(input, groupId);
  auto comms = commsAndEvents.first;
  auto events = commsAndEvents.second;

  int curDevice = 0;
  THCudaCheck(cudaGetDevice(&curDevice));

  std::unique_lock<std::mutex> cudaFreeMutexLock(
      *(THCCachingAllocator_getCudaFreeMutex()));

  NCCL_CHECK(ncclGroupStart());
  for (size_t i = 0; i < input.size(); ++i) {

    THCudaCheck(cudaSetDevice(input[i].get_device()));
    auto stream = THCState_getCurrentStream(THDGetCudaState());

    NCCL_CHECK(ncclAllReduce(input[i].data_ptr(),
                             output[i].data_ptr(),
                             input[i].numel(),
                             _getNcclDataType(input[i].type().scalarType()),
                             ncclOp[operation],
                             (*comms)[i],
                             stream));
    THCudaCheck(cudaEventRecord((*events)[i], stream));
  }
  NCCL_CHECK(ncclGroupEnd());

  cudaFreeMutexLock.unlock();

  // Restore to previous device
  THCudaCheck(cudaSetDevice(curDevice));
}


void DataChannelNccl::allReduce(at::Tensor& data,
                                THDReduceOp operation,
                                THDGroup groupId) {

  std::vector<at::Tensor> dataVec;
  dataVec.push_back(data);
  allReduce(dataVec, dataVec, operation, groupId);
}


void DataChannelNccl::allGather(std::vector<at::Tensor>& input,
                                std::vector<at::Tensor>& output,
                                THDGroup groupId) {

  std::unique_lock<std::mutex> channelLock(_mutex);

  _tensorCheckHelper(input, output, _numProcesses * input.size());
  _checkGroupIdValid(groupId);

  auto commsAndEvents = _getNcclCommsAndEvents(input, groupId);
  auto comms = commsAndEvents.first;
  auto events = commsAndEvents.second;

  int curDevice = 0;
  THCudaCheck(cudaGetDevice(&curDevice));

  std::unique_lock<std::mutex> cudaFreeMutexLock(
      *(THCCachingAllocator_getCudaFreeMutex()));

  NCCL_CHECK(ncclGroupStart());
  for (size_t i = 0; i < input.size(); ++i) {

    THCudaCheck(cudaSetDevice(input[i].get_device()));
    auto stream = THCState_getCurrentStream(THDGetCudaState());

    NCCL_CHECK(ncclAllGather(input[i].data_ptr(),
                             output[i].data_ptr(),
                             input[i].numel(),
                             _getNcclDataType(input[i].type().scalarType()),
                             (*comms)[i],
                             stream));
    THCudaCheck(cudaEventRecord((*events)[i], stream));
  }
  NCCL_CHECK(ncclGroupEnd());

  cudaFreeMutexLock.unlock();

  // Restore to previous device
  THCudaCheck(cudaSetDevice(curDevice));
}


void DataChannelNccl::allGather(std::vector<at::Tensor>& output,
                                at::Tensor& input,
                                THDGroup groupId) {

  std::vector<at::Tensor> inputDataVec;
  inputDataVec.push_back(input);
  allGather(inputDataVec, output, groupId);
}


void DataChannelNccl::reduce(std::vector<at::Tensor>& data,
                             THDReduceOp operation,
                             rank_type dstRank,
                             THDGroup groupId) {

  std::unique_lock<std::mutex> channelLock(_mutex);

  // Check the tensor vector for consistency
  _tensorCheckHelper(data, data);
  _checkGroupIdValid(groupId);

  auto commsAndEvents = _getNcclCommsAndEvents(data, groupId);
  auto comms = commsAndEvents.first;
  auto events = commsAndEvents.second;

  int curDevice = 0;
  THCudaCheck(cudaGetDevice(&curDevice));

  std::unique_lock<std::mutex> cudaFreeMutexLock(
      *(THCCachingAllocator_getCudaFreeMutex()));

  NCCL_CHECK(ncclGroupStart());
  for (size_t i = 0; i < data.size(); ++i) {

    THCudaCheck(cudaSetDevice(data[i].get_device()));
    auto stream = THCState_getCurrentStream(THDGetCudaState());

    NCCL_CHECK(ncclReduce(data[i].data_ptr(),
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

  // Restore to previous device
  THCudaCheck(cudaSetDevice(curDevice));
}

void DataChannelNccl::reduce(at::Tensor& data,
                             THDReduceOp operation,
                             rank_type dstRank,
                             THDGroup groupId) {

  std::vector<at::Tensor> dataVec;
  dataVec.push_back(data);
  reduce(dataVec, operation, dstRank, groupId);
}

void DataChannelNccl::broadcast(std::vector<at::Tensor>& data,
                                rank_type srcRank,
                                THDGroup groupId) {

  std::unique_lock<std::mutex> channelLock(_mutex);

  // Check the tensor vector for consistency
  _tensorCheckHelper(data, data);
  _checkGroupIdValid(groupId);

  auto commsAndEvents = _getNcclCommsAndEvents(data, groupId);
  auto comms = commsAndEvents.first;
  auto events = commsAndEvents.second;

  int curDevice = 0;
  THCudaCheck(cudaGetDevice(&curDevice));

  std::unique_lock<std::mutex> cudaFreeMutexLock(
      *(THCCachingAllocator_getCudaFreeMutex()));

  NCCL_CHECK(ncclGroupStart());
  for (size_t i = 0; i < data.size(); ++i) {

    THCudaCheck(cudaSetDevice(data[i].get_device()));
    auto stream = THCState_getCurrentStream(THDGetCudaState());

    NCCL_CHECK(ncclBcast(data[i].data_ptr(),
                         data[i].numel(),
                         _getNcclDataType(data[i].type().scalarType()),
                         srcRank * data.size(),
                         (*comms)[i],
                         stream));
    THCudaCheck(cudaEventRecord((*events)[i], stream));
  }
  NCCL_CHECK(ncclGroupEnd());

  cudaFreeMutexLock.unlock();

  // Restore to previous device
  THCudaCheck(cudaSetDevice(curDevice));
}


void DataChannelNccl::broadcast(at::Tensor& data,
                                rank_type srcRank,
                                THDGroup groupId) {

  std::vector<at::Tensor> dataVec;
  dataVec.push_back(data);
  broadcast(dataVec, srcRank, groupId);
}


void DataChannelNccl::barrier(THDGroup groupId) {

  std::unique_lock<std::mutex> channelLock(_mutex);

  _checkGroupIdValid(groupId);

  if (_ncclCommsAndEvents.find(groupId) == _ncclCommsAndEvents.end() ||
      _groupDevices.find(groupId) == _groupDevices.end()) {
    return;
  }

  auto devices = getDevicesList(_groupDevices[groupId]);

  int curDevice = 0;
  THCudaCheck(cudaGetDevice(&curDevice));

  int idx = 0;
  // Synchronize on the CUDA events
  for (auto& event : *(_ncclCommsAndEvents[groupId].second)) {
    THCudaCheck(cudaSetDevice(devices[idx++]));
    THCudaCheck(cudaEventSynchronize(event));
  }

  // Restore to previous device
  THCudaCheck(cudaSetDevice(curDevice));
}


THDGroup DataChannelNccl::newGroup(const std::vector<rank_type>& ranks) {

  std::unique_lock<std::mutex> channelLock(_mutex);

  auto newGroup = DataChannel::Group(ranks, _numProcesses - 1);
  THDGroup newGroupId = static_cast<THDGroup>(_groups.size());

  // Insert the current group
  _groups.insert({
    newGroupId,
    newGroup
  });

  return newGroupId;
}


// Helper function that checks if the given groupId is valid
void DataChannelNccl::_checkGroupIdValid(THDGroup groupId) {

  if (_groups.find(groupId) == _groups.end()) {
    std::string errMsg = "Group ID: " + std::to_string(groupId) +
                         " is not valid";
    throw std::runtime_error(errMsg);
  }
}


void DataChannelNccl::gather(std::vector<at::Tensor>& output,
                             at::Tensor& input,
                             rank_type dstRank,
                             THDGroup groupId) {

  throw std::runtime_error("DataChannelNccl does not support gather");
}


void DataChannelNccl::scatter(std::vector<at::Tensor>& input,
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
  throw std::runtime_error("DataChannelNccl does not support receive "
                           "from any source");
}


void DataChannelNccl::receive(at::Tensor& data, rank_type srcRank) {
  throw std::runtime_error("DataChannelNccl does not support receive");
}



DataChannelNccl::RequestNccl* DataChannelNccl::isend(at::Tensor& data,
                                                     rank_type dstRank) {

  throw std::runtime_error("DataChannelNccl does not support isend");
}


DataChannelNccl::RequestNccl* DataChannelNccl::ireceive(at::Tensor& data,
                                                        rank_type srcRank) {

  throw std::runtime_error("DataChannelNccl does not support ireceive");
}


} // namespace thd
