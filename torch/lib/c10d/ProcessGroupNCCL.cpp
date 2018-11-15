#include "ProcessGroupNCCL.hpp"

#include <map>
#include <tuple>
#include <unordered_set>

#include <THC.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGuard.h>

#include <c10d/Utils.hpp>

namespace c10d {

namespace {

// NCCL op mapping
std::map<ReduceOp, ncclRedOp_t> ncclOp = {
    {ReduceOp::MIN, ncclMin},
    {ReduceOp::MAX, ncclMax},
    {ReduceOp::SUM, ncclSum},
    {ReduceOp::PRODUCT, ncclProd},
};

// NCCL type typing
std::map<at::ScalarType, ncclDataType_t> ncclDataType = {
    {at::kChar, ncclInt8},
    {at::kByte, ncclUint8},
    {at::kFloat, ncclFloat},
    {at::kDouble, ncclDouble},
    {at::kInt, ncclInt32},
    {at::kLong, ncclInt64},
    {at::kHalf, ncclHalf},
};

// Helper function that gets the data type and issues error if not supported
ncclDataType_t getNcclDataType(at::ScalarType type) {
  try {
    return ncclDataType.at(type);
  } catch (std::out_of_range& e) {
    throw std::runtime_error("Unsupported data type for NCCL process group");
  }
}

// Get the deviceList String from the list of devices
std::string getKeyFromDevices(const std::vector<at::Device>& devices) {
  std::string deviceList;
  for (auto& device : devices) {
    if (deviceList.empty()) {
      deviceList = std::to_string(device.index());
    } else {
      deviceList += "," + std::to_string(device.index());
    }
  }
  return deviceList;
}

// Get the list of devices from list of tensors
std::vector<at::Device> getDeviceList(const std::vector<at::Tensor>& tensors) {
  std::vector<at::Device> res;
  res.reserve(tensors.size());
  for (auto& tensor : tensors) {
    res.push_back(tensor.device());
  }
  return res;
}

// Helper that lets the input ncclStreams to wait for the current stream
void syncStreams(
    const std::vector<at::Device>& devices,
    std::vector<at::cuda::CUDAEvent>& ncclEvents,
    std::vector<at::cuda::CUDAStream>& ncclStreams) {
  for (size_t i = 0; i < devices.size(); ++i) {
    at::cuda::CUDAStream& ncclStream = ncclStreams[i];
    at::cuda::CUDAEvent& ncclEvent = ncclEvents[i];
    ncclEvent.record(at::cuda::getCurrentCUDAStream(devices[i].index()));
    ncclEvent.block(ncclStream);
  }
}

} // namespace

ProcessGroupNCCL::WorkNCCL::WorkNCCL(const std::vector<at::Device>& devices)
    : devices_(devices) {
  // Creates the CUDA event wrappers
  // Note: The actual events are lazily created when first recorded to with
  // DEFAULT_FLAGS = cudaEventDisableTiming.
  cudaEvents_.resize(devices.size());
}

ProcessGroupNCCL::WorkNCCL::~WorkNCCL() {}

// Check if the NCCL kernels are queued on the GPUs
bool ProcessGroupNCCL::WorkNCCL::isCompleted() {
  return true;
}

// Helper that checks if the NCCL kernels are completed on the GPUs
bool ProcessGroupNCCL::WorkNCCL::finishedGPUExecution() const {
  for (size_t i = 0; i < devices_.size(); ++i) {
    // Checking the work's corresponding CUDA events' status
    auto ret = cudaEventQuery(cudaEvents_[i]);
    if (ret != cudaSuccess && ret != cudaErrorNotReady) {
      AT_CUDA_CHECK(ret);
    }
    if (ret == cudaErrorNotReady) {
      return false;
    }
  }
  return true;
}

// Same as synchronize(), and will always return true
bool ProcessGroupNCCL::WorkNCCL::wait() {
  synchronize();
  return true;
}

// Waiting on the work's corresponding CUDA events
void ProcessGroupNCCL::WorkNCCL::synchronize() {
  for (size_t i = 0; i < devices_.size(); ++i) {
    auto currentStream = at::cuda::getCurrentCUDAStream(devices_[i].index());
    // Block the current stream on the NCCL stream
    cudaEvents_[i].block(currentStream);
  }
}

bool ProcessGroupNCCL::WorkNCCL::isSuccess() const {
  return true;
}

const std::exception& ProcessGroupNCCL::WorkNCCL::exception() const {
  throw std::runtime_error(
      "exception() is not supported by NCCL process "
      "group's work, since isSuccess() will always return true, and "
      "isCompleted() and wait() will either succeed or throw");
}

std::unordered_map<ssize_t, ssize_t> ProcessGroupNCCL::pgUniqueNCCLIDCnt_;
ssize_t ProcessGroupNCCL::processGroupCounter_ = -1;
std::mutex ProcessGroupNCCL::pgTrackingLock_;

ProcessGroupNCCL::ProcessGroupNCCL(
    const std::shared_ptr<Store>& store,
    int rank,
    int size)
    : ProcessGroup(rank, size), store_(store) {
  // Generate the Process Group ID for current PG, this needs to be identical
  // for all processes
  std::unique_lock<std::mutex> lock(pgTrackingLock_);
  ++processGroupCounter_;
  pgUniqueNCCLIDCnt_[processGroupCounter_] = -1;
  processGroupID_ = std::to_string(processGroupCounter_);
}

ProcessGroupNCCL::~ProcessGroupNCCL() {
  std::unique_lock<std::mutex> lock(pgTrackingLock_);
  pgUniqueNCCLIDCnt_.erase(std::stoull(processGroupID_));
}

void ProcessGroupNCCL::broadcastUniqueNCCLID(ncclUniqueId* ncclID) {
  // Every time when we create a new unique NCCL ID, we need to use a new
  // global key to access/update the store.
  // The key is a combination of processGroupID_ and the current count of
  // NCCL unique ID created
  std::unique_lock<std::mutex> lock(pgTrackingLock_);
  auto processGroupIDKey = std::stoull(processGroupID_);
  auto uniqueNCCLIDCnt = pgUniqueNCCLIDCnt_[processGroupIDKey] + 1;
  pgUniqueNCCLIDCnt_[processGroupIDKey] = uniqueNCCLIDCnt;

  lock.unlock();

  std::string storeKey =
      processGroupID_ + "_" + std::to_string(uniqueNCCLIDCnt);

  // Rank 0 writes to the store as bcast
  if (rank_ == 0) {
    auto ncclIDVal = std::vector<uint8_t>(
        reinterpret_cast<uint8_t*>(ncclID),
        reinterpret_cast<uint8_t*>(ncclID) + NCCL_UNIQUE_ID_BYTES);
    store_->set(storeKey, ncclIDVal);
    // Other ranks get to the store
  } else {
    auto ncclIDVal = store_->get(storeKey);
    // Just a sanity check
    if (ncclIDVal.size() != NCCL_UNIQUE_ID_BYTES) {
      throw std::runtime_error(
          "Unexpected NCCL unique ID length received "
          "from the store");
    }
    // Now put the data back to the input pointer
    memcpy(ncclID, ncclIDVal.data(), NCCL_UNIQUE_ID_BYTES);
  }
}

std::vector<std::shared_ptr<NCCLComm>>& ProcessGroupNCCL::getNCCLComm(
    const std::string& devicesKey,
    const std::vector<at::Device>& devices) {
  // Sanity check
  if (devicesKey.empty()) {
    throw std::runtime_error(
        "Not able to create/get the NCCL Communicator since "
        "the GPU devices are not known");
  }
  if (devNCCLCommMap_.find(devicesKey) != devNCCLCommMap_.end()) {
    // Reuse the cached communicator if there is one.
    return devNCCLCommMap_[devicesKey];
  }
  // NCCL communicator not cached, create a new entry
  std::vector<std::shared_ptr<NCCLComm>> ncclComms;
  ncclComms.resize(devices.size());

  // Create the unique NCCL ID and broadcast it
  ncclUniqueId ncclID;

  if (rank_ == 0) {
    C10D_NCCL_CHECK(ncclGetUniqueId(&ncclID));
  }

  // Broadcast so that each process can have a unique NCCL ID
  broadcastUniqueNCCLID(&ncclID);

  at::cuda::OptionalCUDAGuard gpuGuard;

  std::vector<at::cuda::CUDAStream> streamVal;
  streamVal.reserve(devices.size());

  // Create the NCCL communicators for each GPU
  C10D_NCCL_CHECK(ncclGroupStart());

  for (size_t i = 0; i < devices.size(); ++i) {
    // GPU world size and GPU rank
    int numRanks = getSize() * devices.size();
    int rank = getRank() * devices.size() + i;

    gpuGuard.set_index(devices[i].index());
    ncclComms[i] = NCCLComm::create(numRanks, rank, ncclID);

    // Creates the NCCL streams
    streamVal.push_back(at::cuda::getStreamFromPool());
  }

  C10D_NCCL_CHECK(ncclGroupEnd());

  // Move the NCCL resource to cache
  devNCCLCommMap_.emplace(devicesKey, std::move(ncclComms));
  ncclStreams_.emplace(devicesKey, std::move(streamVal));

  // Note: these events are created with the (default) cudaEventDisableTiming
  // flag This flag provides the best performance when used with
  // cudaStreamWaitEvent() and cudaEventQuery(). Since we here don't measure the
  // performance using cudaEvent, this should be set.
  ncclEvents_.emplace(
      std::piecewise_construct,
      std::make_tuple(devicesKey),
      std::make_tuple(devices.size()));

  return devNCCLCommMap_[devicesKey];
}

// Helper function that checks the input and output tensors for validity
void ProcessGroupNCCL::tensorCheckHelper(
    const std::vector<at::Tensor>& input,
    const std::vector<at::Tensor>& output,
    int outputOverInput) {
  if (input.size() * outputOverInput != output.size()) {
    throw std::runtime_error(
        "Input tensor sequence should have the same "
        "number of tensors as the output tensor sequence");
  }

  if (input.size() == 0) {
    throw std::runtime_error("The number of input tensors should not be zero");
  }

  if (input.size() > static_cast<size_t>(at::cuda::getNumGPUs())) {
    throw std::runtime_error(
        "The number of input tensors is larger than "
        "the number of available GPUs");
  }

  // To make sure each tensor is on separate devices
  std::unordered_set<int> usedDevices;
  usedDevices.reserve(input.size());

  auto inputNumElement = input[0].numel();
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
    if (output[i].numel() != inputNumElement) {
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
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupNCCL::allreduce(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts) {
  tensorCheckHelper(tensors, tensors);

  auto devices = getDeviceList(tensors);
  auto key = getKeyFromDevices(devices);
  auto& ncclComms = getNCCLComm(key, devices);

  // First let NCCL streams wait for THC stream
  syncStreams(devices, ncclEvents_[key], ncclStreams_[key]);

  // Work itself will create the CUDA events on all GPUs of tensors
  auto work = std::make_shared<ProcessGroupNCCL::WorkNCCL>(devices);

  at::cuda::OptionalCUDAGuard gpuGuard;

  std::unique_lock<std::mutex> cudaFreeMutexLock(
      *(THCCachingAllocator_getCudaFreeMutex()));

  C10D_NCCL_CHECK(ncclGroupStart());

  for (size_t i = 0; i < tensors.size(); ++i) {
    gpuGuard.set_index(devices[i].index());
    at::cuda::CUDAStream& ncclStream = ncclStreams_[key][i];

    C10D_NCCL_CHECK(ncclAllReduce(
        tensors[i].data_ptr(),
        tensors[i].data_ptr(),
        tensors[i].numel(),
        getNcclDataType(tensors[i].type().scalarType()),
        ncclOp[opts.reduceOp],
        ncclComms[i]->getNcclComm(),
        ncclStream.stream()));
  }

  C10D_NCCL_CHECK(ncclGroupEnd());

  // Event should only be recorded after the ncclGroupEnd()
  for (size_t i = 0; i < tensors.size(); ++i) {
    at::cuda::CUDAStream& ncclStream = ncclStreams_[key][i];
    work->cudaEvents_[i].record(ncclStream);
  }

  return work;
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupNCCL::broadcast(
    std::vector<at::Tensor>& tensors,
    const BroadcastOptions& opts) {
  tensorCheckHelper(tensors, tensors);

  auto devices = getDeviceList(tensors);
  auto key = getKeyFromDevices(devices);
  auto& ncclComms = getNCCLComm(key, devices);

  // First let NCCL streams wait for current streams
  syncStreams(devices, ncclEvents_[key], ncclStreams_[key]);

  // Work itself will create the CUDA events on all GPUs of tensors
  auto work = std::make_shared<ProcessGroupNCCL::WorkNCCL>(devices);

  at::cuda::OptionalCUDAGuard gpuGuard;

  std::unique_lock<std::mutex> cudaFreeMutexLock(
      *(THCCachingAllocator_getCudaFreeMutex()));

  C10D_NCCL_CHECK(ncclGroupStart());

  for (size_t i = 0; i < tensors.size(); ++i) {
    gpuGuard.set_index(devices[i].index());
    at::cuda::CUDAStream& ncclStream = ncclStreams_[key][i];
    // root rank of the the GPU
    int root = opts.rootRank * tensors.size() + opts.rootTensor;

    C10D_NCCL_CHECK(ncclBcast(
        tensors[i].data_ptr(),
        tensors[i].numel(),
        getNcclDataType(tensors[i].type().scalarType()),
        root,
        ncclComms[i]->getNcclComm(),
        ncclStream.stream()));
  }

  C10D_NCCL_CHECK(ncclGroupEnd());

  // Event should only be recorded after the ncclGroupEnd()
  for (size_t i = 0; i < tensors.size(); ++i) {
    at::cuda::CUDAStream& ncclStream = ncclStreams_[key][i];
    work->cudaEvents_[i].record(ncclStream);
  }

  return work;
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupNCCL::reduce(
    std::vector<at::Tensor>& tensors,
    const ReduceOptions& opts) {
  tensorCheckHelper(tensors, tensors);

  auto devices = getDeviceList(tensors);
  auto key = getKeyFromDevices(devices);
  auto& ncclComms = getNCCLComm(key, devices);

  // First let NCCL streams wait for current streams
  syncStreams(devices, ncclEvents_[key], ncclStreams_[key]);

  // Work itself will create the CUDA events on all GPUs of tensors
  auto work = std::make_shared<ProcessGroupNCCL::WorkNCCL>(devices);

  at::cuda::OptionalCUDAGuard gpuGuard;

  std::unique_lock<std::mutex> cudaFreeMutexLock(
      *(THCCachingAllocator_getCudaFreeMutex()));

  C10D_NCCL_CHECK(ncclGroupStart());

  for (size_t i = 0; i < tensors.size(); ++i) {
    gpuGuard.set_index(devices[i].index());
    at::cuda::CUDAStream& ncclStream = ncclStreams_[key][i];
    // root rank of the the GPU
    int root = opts.rootRank * tensors.size() + opts.rootTensor;

    C10D_NCCL_CHECK(ncclReduce(
        tensors[i].data_ptr(),
        tensors[i].data_ptr(),
        tensors[i].numel(),
        getNcclDataType(tensors[i].type().scalarType()),
        ncclOp[opts.reduceOp],
        root,
        ncclComms[i]->getNcclComm(),
        ncclStream.stream()));
  }

  C10D_NCCL_CHECK(ncclGroupEnd());

  // Event should only be recorded after the ncclGroupEnd()
  for (size_t i = 0; i < tensors.size(); ++i) {
    at::cuda::CUDAStream& ncclStream = ncclStreams_[key][i];
    work->cudaEvents_[i].record(ncclStream);
  }

  return work;
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupNCCL::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors) {
  if (outputTensors.size() != inputTensors.size()) {
    throw std::runtime_error("allgather: input and output size mismatch");
  }
  std::vector<at::Tensor> flattenOutputTensors;
  flattenOutputTensors.resize(outputTensors.size());

  for (size_t i = 0; i < outputTensors.size(); ++i) {
    tensorCheckHelper(
        std::vector<at::Tensor>{inputTensors[i]},
        outputTensors[i],
        size_ * inputTensors.size());
    // Flatten the output tensors (from all ranks) to a single big tensor
    flattenOutputTensors[i] = newLikeFlat(outputTensors, i);

    if (static_cast<size_t>(flattenOutputTensors[i].numel()) !=
        inputTensors[i].numel() * size_ * inputTensors.size()) {
      throw std::runtime_error("Unexpected size for flatten tensor");
    }
  }

  auto devices = getDeviceList(inputTensors);
  auto key = getKeyFromDevices(devices);
  auto& ncclComms = getNCCLComm(key, devices);

  // First let NCCL streams wait for current streams
  syncStreams(devices, ncclEvents_[key], ncclStreams_[key]);

  // Work itself will create the CUDA events on all GPUs of tensors
  auto work = std::make_shared<ProcessGroupNCCL::WorkNCCL>(devices);

  at::cuda::OptionalCUDAGuard gpuGuard;

  std::unique_lock<std::mutex> cudaFreeMutexLock(
      *(THCCachingAllocator_getCudaFreeMutex()));

  C10D_NCCL_CHECK(ncclGroupStart());

  for (size_t i = 0; i < inputTensors.size(); ++i) {
    gpuGuard.set_index(devices[i].index());

    at::cuda::CUDAStream& ncclStream = ncclStreams_[key][i];

    C10D_NCCL_CHECK(ncclAllGather(
        inputTensors[i].data_ptr(),
        flattenOutputTensors[i].data_ptr(),
        inputTensors[i].numel(),
        getNcclDataType(inputTensors[i].type().scalarType()),
        ncclComms[i]->getNcclComm(),
        ncclStream.stream()));
  }

  C10D_NCCL_CHECK(ncclGroupEnd());

  // Copy the flattened output tensors to the outputs
  for (size_t i = 0; i < outputTensors.size(); ++i) {
    at::cuda::CUDAStream& ncclStream = ncclStreams_[key][i];
    at::cuda::CUDAStreamGuard guard(ncclStream);
    for (size_t j = 0; j < outputTensors[0].size(); ++j) {
      outputTensors[i][j].copy_(flattenOutputTensors[i][j], true);
    }
  }

  // Event should only be recorded after the ncclGroupEnd()
  for (size_t i = 0; i < inputTensors.size(); ++i) {
    at::cuda::CUDAStream& ncclStream = ncclStreams_[key][i];
    work->cudaEvents_[i].record(ncclStream);
  }
  return work;
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupNCCL::gather(
    std::vector<std::vector<at::Tensor>>& /* unused */,
    std::vector<at::Tensor>& /* unused */,
    const GatherOptions& /* unused */) {
  throw std::runtime_error("ProcessGroupNCCL does not support gather");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupNCCL::scatter(
    std::vector<at::Tensor>& /* unused */,
    std::vector<std::vector<at::Tensor>>& /* unused */,
    const ScatterOptions& /* unused */) {
  throw std::runtime_error("ProcessGroupNCCL does not support scatter");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupNCCL::send(
    std::vector<at::Tensor>& /* unused */,
    int /* unused */,
    int /* unused */) {
  throw std::runtime_error("ProcessGroupNCCL does not support send");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupNCCL::recv(
    std::vector<at::Tensor>& /* unused */,
    int /* unused */,
    int /* unused */) {
  throw std::runtime_error("ProcessGroupNCCL does not support recv");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupNCCL::recvAnysource(
    std::vector<at::Tensor>& /* unused */,
    int* /* unused */,
    int /* unused */) {
  throw std::runtime_error("ProcessGroupNCCL does not support recv");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupNCCL::barrier() {
  throw std::runtime_error("ProcessGroupNCCL does not support barrier");
}

std::unordered_map<int, int> ProcessGroupNCCL::getGroupRank() {
  throw std::runtime_error("ProcessGroupNCCL doest not support getGroupRank");
}

} // namespace c10d
