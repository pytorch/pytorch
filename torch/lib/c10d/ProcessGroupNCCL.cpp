#include <c10d/ProcessGroupNCCL.hpp>

#include <map>
#include <tuple>
#include <unordered_set>

#include <THC/THC.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

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

// [Sync Streams] Helper that lets the input ncclStreams to wait for the current
// stream. NCCL communications run on ncclStreams, but input tensors are
// allocated on different streams (i.e., current streams). Communications on
// ncclStreams cannot start before pending input tensor ops on current streams
// finish. Otherwise, ops on two streams might read/write same tensors
// concurrently.
//
// The synchronization above alone is not enough. We also need to make sure
// input tensors are not freed before their usages on ncclStreams finish. This
// can be achieved by calling c10::cuda::CUDACachingAllocator::recordStream,
// which remembers the usage stream (ncclStream), creates an event on the usage
// stream when GC attempts to free the input tensor, and delays GC until that
// event is done.
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

bool ProcessGroupNCCL::WorkNCCL::isCompleted() {
  return finishedGPUExecution();
}

bool ProcessGroupNCCL::WorkNCCL::isSuccess() const {
  return true;
}

std::exception_ptr ProcessGroupNCCL::WorkNCCL::exception() const {
  throw std::runtime_error(
      "exception() is not supported by NCCL process "
      "group's work, since isSuccess() will always return true, and "
      "isCompleted() and wait() will either succeed or throw");
}

// Helper that checks if the NCCL kernels are completed on the GPUs
bool ProcessGroupNCCL::WorkNCCL::finishedGPUExecution() {
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

// Waiting on the work's corresponding CUDA events
void ProcessGroupNCCL::WorkNCCL::synchronize() {
  for (size_t i = 0; i < devices_.size(); ++i) {
    auto currentStream = at::cuda::getCurrentCUDAStream(devices_[i].index());
    // Block the current stream on the NCCL stream
    cudaEvents_[i].block(currentStream);
    // If we use the work to do barrier, we should block here
    if (!barrierTensors_.empty()) {
      at::cuda::CUDAGuard gpuGuard(devices_[i]);
      AT_CUDA_CHECK(cudaDeviceSynchronize());
    }
  }
}

// Same as calling synchronize().
void ProcessGroupNCCL::WorkNCCL::wait() {
  synchronize();
}

std::unordered_map<std::string, ssize_t> ProcessGroupNCCL::pgUniqueNCCLIDCnt_;
std::unordered_map<std::string, ssize_t>
    ProcessGroupNCCL::processGroupCounterMap_;

std::mutex ProcessGroupNCCL::pgTrackingLock_;

ProcessGroupNCCL::ProcessGroupNCCL(
    const std::shared_ptr<Store>& store,
    int rank,
    int size,
    const std::string& groupName)
    : ProcessGroup(rank, size), store_(store), groupName_(groupName) {
  // Generate the Process Group ID for current PG, this needs to be identical
  // for all processes
  std::unique_lock<std::mutex> lock(pgTrackingLock_);
  // Default group is an empty string
  const auto groupKey = groupName_ + "_";
  if (processGroupCounterMap_.count(groupKey) == 0) {
    processGroupCounterMap_[groupKey] = -1;
  }
  ++processGroupCounterMap_[groupKey];
  processGroupID_ = std::to_string(processGroupCounterMap_[groupKey]);
  groupPgID_ = groupName_ + "_" + processGroupID_;
  pgUniqueNCCLIDCnt_[groupPgID_] = -1;
}

ProcessGroupNCCL::~ProcessGroupNCCL() {
  std::unique_lock<std::mutex> lock(pgTrackingLock_);
  pgUniqueNCCLIDCnt_.erase(groupPgID_);
}

void ProcessGroupNCCL::broadcastUniqueNCCLID(ncclUniqueId* ncclID) {
  // Every time when we create a new unique NCCL ID, we need to use a new
  // global key to access/update the store.
  // The key is a combination of processGroupID_ and the current count of
  // NCCL unique ID created
  std::unique_lock<std::mutex> lock(pgTrackingLock_);
  auto groupPgId = groupName_ + "_" + processGroupID_;
  const auto uniqueNCCLIDCnt = ++pgUniqueNCCLIDCnt_[groupPgID_];

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

  for (auto& device : devices) {
    usedDeviceIdxs_.insert(device.index());
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

namespace {

// Check that all `tensors' have the same type and shape and are distributed
// across distinct GPUs.
void check_gpu_tensors(const std::vector<at::Tensor>& tensors) {
  if (tensors.size() == 0) {
    throw std::runtime_error("Tensor list must be nonempty");
  }
  if (tensors.size() > static_cast<size_t>(at::cuda::getNumGPUs())) {
    throw std::runtime_error(
      "Tensor list mustn't be larger than the number of available GPUs"
    );
  }

  const auto& first = tensors.front();

  // Set for ensuring that tensors are on separate devices.
  std::unordered_set<decltype(first.get_device())> usedDevices;
  usedDevices.reserve(tensors.size());

  for (const auto& t : tensors) {
    if (!t.is_cuda() || t.is_sparse()) {
      throw std::runtime_error("Tensors must be CUDA and dense");
    }
    if (t.scalar_type() != first.scalar_type()) {
      throw std::runtime_error("Tensors must have identical type");
    }
    if (t.sizes() != first.sizes()) {
      throw std::runtime_error("Tensors must have identical size");
    }
    if (!t.is_contiguous()) {
      throw std::runtime_error("Tensors must be contiguous");
    }
    const auto inserted = usedDevices.insert(t.get_device()).second;
    if (!inserted) {
      throw std::runtime_error("Tensors must be on distinct GPU devices");
    }
  }
}

// Flatten each list in `tensor_lists' for a gather or scatter operation, and
// ensure compatibility with the corresponding tensor in `other'.
std::vector<at::Tensor> flatten_for_scatter_gather(
    std::vector<std::vector<at::Tensor>>& tensor_lists,
    std::vector<at::Tensor>& other,
    size_t world_size) {
  if (tensor_lists.size() != other.size()) {
    throw std::runtime_error(
      "Tensor list operands to scatter/gather must have the same length"
    );
  }
  const auto num_devices = tensor_lists.size();

  std::vector<at::Tensor> flattened;
  flattened.resize(num_devices);

  for (auto i = size_t{}; i < num_devices; ++i) {
    if (tensor_lists[i].size() != world_size * num_devices) {
      throw std::runtime_error(
        "Tensor list input to scatter/gather must match number of collective"
        " participants"
      );
    }

    // Only check device match for the first tensor in the list; the call to
    // newLikeFlat() below will check the rest.
    if (tensor_lists[i].front().get_device() != other[i].get_device()) {
      throw std::runtime_error(
        "Corresponding input/output tensors to scatter/gather must all reside"
        " on the same device"
      );
    }

    for (const auto& t : tensor_lists[i]) {
      if (t.numel() != other[i].numel()) {
        throw std::runtime_error(
          "All tensor operands to scatter/gather must have the same size"
        );
      }
    }
    // Flatten the tensors (from all ranks) into a single big tensor.
    flattened[i] = newLikeFlat(tensor_lists, i);
  }
  return flattened;
}

}

template<typename Fn, typename PreProcess, typename PostProcess>
std::shared_ptr<ProcessGroup::Work> ProcessGroupNCCL::collective(
    std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    Fn fn,
    PreProcess pre,
    PostProcess post) {
  const auto devices = getDeviceList(inputs);
  const auto key = getKeyFromDevices(devices);
  auto& ncclComms = getNCCLComm(key, devices);

  // First let NCCL streams wait for input tensors allocation streams
  syncStreams(devices, ncclEvents_[key], ncclStreams_[key]);

  // Work itself will create the CUDA events on all GPUs of tensors
  auto work = std::make_shared<ProcessGroupNCCL::WorkNCCL>(devices);

  at::cuda::OptionalCUDAGuard gpuGuard;

  std::unique_lock<std::mutex> cudaFreeMutexLock(
      *(c10::cuda::CUDACachingAllocator::getFreeMutex()));

  pre(ncclStreams_[key]);

  C10D_NCCL_CHECK(ncclGroupStart());

  for (size_t i = 0; i < inputs.size(); ++i) {
    gpuGuard.set_index(devices[i].index());
    at::cuda::CUDAStream& ncclStream = ncclStreams_[key][i];

    // Both `inputs' and `outputs' are created on a worker stream and used in
    // different ncclStreams.  Hence, both must record the ncclStream to
    // prevent being freed before the collective finishes.
    //
    // We only record `inputs' here, and leave recording `outputs' to `fn' for
    // operations where `inputs' and `outputs' are not the same.
    //
    // See [Sync Streams].
    c10::cuda::CUDACachingAllocator::recordStream(
      inputs[i].storage().data(), ncclStream);

    C10D_NCCL_CHECK(fn(
      inputs[i],
      outputs[i],
      ncclComms[i]->getNcclComm(),
      ncclStream
    ));
  }

  C10D_NCCL_CHECK(ncclGroupEnd());

  post(ncclStreams_[key]);

  // Event should only be recorded after the ncclGroupEnd()
  for (size_t i = 0; i < inputs.size(); ++i) {
    at::cuda::CUDAStream& ncclStream = ncclStreams_[key][i];
    work->cudaEvents_[i].record(ncclStream);
  }

  return work;
}

template<typename Fn>
std::shared_ptr<ProcessGroup::Work> ProcessGroupNCCL::collective(
    std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    Fn fn) {
  return collective(
    inputs, outputs, fn,
    [] (std::vector<at::cuda::CUDAStream>&) {},
    [] (std::vector<at::cuda::CUDAStream>&) {}
  );
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupNCCL::allreduce(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts) {
  check_gpu_tensors(tensors);

  return collective(tensors, tensors,
    [&] (at::Tensor& input, at::Tensor& output,
         ncclComm_t comm, at::cuda::CUDAStream& stream) {
      return ncclAllReduce(
        input.data_ptr(),
        output.data_ptr(),
        input.numel(),
        getNcclDataType(input.scalar_type()),
        ncclOp[opts.reduceOp],
        comm,
        stream.stream()
      );
    }
  );
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupNCCL::broadcast(
    std::vector<at::Tensor>& tensors,
    const BroadcastOptions& opts) {
  check_gpu_tensors(tensors);

  return collective(tensors, tensors,
    [&] (at::Tensor& input, at::Tensor& output,
         ncclComm_t comm, at::cuda::CUDAStream& stream) {
      const auto root = opts.rootRank * tensors.size() + opts.rootTensor;
      return ncclBcast(
        input.data_ptr(),
        input.numel(),
        getNcclDataType(input.scalar_type()),
        root,
        comm,
        stream.stream()
      );
    }
  );
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupNCCL::reduce(
    std::vector<at::Tensor>& tensors,
    const ReduceOptions& opts) {
  check_gpu_tensors(tensors);

  return collective(tensors, tensors,
    [&] (at::Tensor& input, at::Tensor& output,
         ncclComm_t comm, at::cuda::CUDAStream& stream) {
      const auto root = opts.rootRank * tensors.size() + opts.rootTensor;
      return ncclReduce(
        input.data_ptr(),
        output.data_ptr(),
        input.numel(),
        getNcclDataType(input.scalar_type()),
        ncclOp[opts.reduceOp],
        root,
        comm,
        stream.stream()
      );
    }
  );
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupNCCL::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllgatherOptions& opts) {
  check_gpu_tensors(inputTensors);

  auto outputFlattened = flatten_for_scatter_gather(
    outputTensors, inputTensors, size_
  );
  check_gpu_tensors(outputFlattened);

  return collective(inputTensors, outputFlattened,
    [&] (at::Tensor& input, at::Tensor& output,
         ncclComm_t comm, at::cuda::CUDAStream& stream) {
      c10::cuda::CUDACachingAllocator::recordStream(
        output.storage().data(), stream
      );
      return ncclAllGather(
        input.data_ptr(),
        output.data_ptr(),
        input.numel(),
        getNcclDataType(input.scalar_type()),
        comm,
        stream.stream()
      );
    },
    [&] (std::vector<at::cuda::CUDAStream>& ncclStreams) {},
    [&] (std::vector<at::cuda::CUDAStream>& ncclStreams) {
      // Copy the flattened output tensors to the outputs.
      for (size_t i = 0; i < outputTensors.size(); ++i) {
        at::cuda::CUDAStreamGuard guard(ncclStreams[i]);
        for (size_t j = 0; j < outputTensors[0].size(); ++j) {
          // See [Sync Streams].
          c10::cuda::CUDACachingAllocator::recordStream(
            outputTensors[i][j].storage().data(), ncclStreams[i]);

          outputTensors[i][j].copy_(outputFlattened[i][j], true);
        }
      }
    }
  );
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupNCCL::reduce_scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ReduceScatterOptions& opts) {
  check_gpu_tensors(outputTensors);

  auto inputFlattened = flatten_for_scatter_gather(
    inputTensors, outputTensors, size_
  );
  check_gpu_tensors(inputFlattened);

  return collective(inputFlattened, outputTensors,
    [&] (at::Tensor& input, at::Tensor& output,
         ncclComm_t comm, at::cuda::CUDAStream& stream) {
      c10::cuda::CUDACachingAllocator::recordStream(
        output.storage().data(), stream
      );
      return ncclReduceScatter(
        input.data_ptr(),
        output.data_ptr(),
        output.numel(),
        getNcclDataType(input.scalar_type()),
        ncclOp[opts.reduceOp],
        comm,
        stream.stream()
      );
    },
    [&] (std::vector<at::cuda::CUDAStream>& ncclStreams) {
      // Copy the input tensors to the flattened inputs.
      for (size_t i = 0; i < inputTensors.size(); ++i) {
        at::cuda::CUDAStreamGuard guard(ncclStreams[i]);
        for (size_t j = 0; j < inputTensors[0].size(); ++j) {
          // See [Sync Streams].
          c10::cuda::CUDACachingAllocator::recordStream(
            inputTensors[i][j].storage().data(), ncclStreams[i]);

          inputFlattened[i][j].copy_(inputTensors[i][j], true);
        }
      }
    },
    [&] (std::vector<at::cuda::CUDAStream>& ncclStreams) {}
  );
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupNCCL::barrier(
    const BarrierOptions& opts) {
  std::vector<at::Device> devices;
  if (usedDeviceIdxs_.empty()) {
    // This means there is not yet a NCCL collective being called
    // Here we have to use the best guesses and will use a single GPU to call
    // allreduce to achieve barrier.
    // In case the multiple processes fall into the same node, we use rank to
    // ensure that each process is on a different GPU
    auto numGPUs = at::cuda::getNumGPUs();
    int16_t deviceIdx = static_cast<int16_t>(rank_ % numGPUs);
    devices.push_back(at::Device(at::DeviceType::CUDA, deviceIdx));
  } else {
    for (auto usedDeviceIdx : usedDeviceIdxs_) {
      devices.push_back(at::Device(at::DeviceType::CUDA, usedDeviceIdx));
    }
  }

  std::vector<at::Tensor> barrierTensors;
  barrierTensors.reserve(devices.size());

  at::cuda::OptionalCUDAGuard gpuGuard;
  for (auto& device : devices) {
    gpuGuard.set_index(device.index());
    barrierTensors.push_back(at::empty(
        {1},
        at::TensorOptions().device(at::DeviceType::CUDA).dtype(at::kByte)));
  }

  // All reduce to achieve the barrier
  auto work = allreduce(barrierTensors);

  // Work will take over barrierTensors
  auto ncclWork = dynamic_cast<ProcessGroupNCCL::WorkNCCL*>(work.get());
  AT_CHECK(ncclWork);
  ncclWork->barrierTensors_ = std::move(barrierTensors);

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
    int /* unused */) {
  throw std::runtime_error("ProcessGroupNCCL does not support recv");
}

} // namespace c10d
