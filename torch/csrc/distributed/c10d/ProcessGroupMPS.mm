#ifdef USE_C10D_MPS

#include <torch/csrc/distributed/c10d/ProcessGroupMPS.hpp>

#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/OperationUtils.h>
#include <c10/util/irange.h>

#include <jaccl/group.h>
#include <jaccl/jaccl.h>
#include <jaccl/rdma.h>

#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

namespace c10d {

namespace {

constexpr const char* kMpsCoordKey = "mps_pg/jaccl_coord";
constexpr const char* kMpsDevicePrefix = "mps_pg/jaccl_device/";

void* mpsHostPtr(const at::Tensor& tensor) {
  if (!tensor.device().is_mps() || !tensor.is_contiguous()) {
    return nullptr;
  }
  id<MTLBuffer> buf = at::native::mps::getMTLBufferStorage(tensor);
  if (buf == nil) {
    return nullptr;
  }
  void* base = [buf contents];
  if (base == nullptr) {
    return nullptr;
  }
  return static_cast<char*>(base) +
      tensor.storage_offset() * tensor.itemsize();
}

// Pick the first RDMA device for which ibv_alloc_pd succeeds. Honours
// JACCL_DEVICE to pin a specific device name. Returns empty string when no
// usable device exists on this host.
std::string probeRDMADevice() {
  auto& verbs = ::jaccl::ibv();
  if (!verbs.is_available()) {
    return {};
  }
  int numDevices = 0;
  auto devices = verbs.get_device_list(&numDevices);
  if (devices == nullptr) {
    return {};
  }
  const char* deviceOverride = std::getenv("JACCL_DEVICE");
  std::string chosen;
  for (int i = 0; i < numDevices; i++) {
    std::string name = verbs.get_device_name(devices[i]);
    if (deviceOverride && name != deviceOverride) {
      continue;
    }
    auto ctx = verbs.open_device(devices[i]);
    if (!ctx) {
      continue;
    }
    auto pd = verbs.alloc_pd(ctx);
    if (pd) {
      verbs.dealloc_pd(pd);
      verbs.close_device(ctx);
      chosen = name;
      break;
    }
    verbs.close_device(ctx);
  }
  verbs.free_device_list(devices);
  return chosen;
}

int dtypeToJACCL(at::ScalarType dtype) {
  switch (dtype) {
    case at::kBool:
      return ::jaccl::Dtype::Bool;
    case at::kByte:
      return ::jaccl::Dtype::UInt8;
    case at::kChar:
      return ::jaccl::Dtype::Int8;
    case at::kShort:
      return ::jaccl::Dtype::Int16;
    case at::kInt:
      return ::jaccl::Dtype::Int32;
    case at::kLong:
      return ::jaccl::Dtype::Int64;
    case at::kHalf:
      return ::jaccl::Dtype::Float16;
    case at::kBFloat16:
      return ::jaccl::Dtype::BFloat16;
    case at::kFloat:
      return ::jaccl::Dtype::Float32;
    case at::kDouble:
      return ::jaccl::Dtype::Float64;
    case at::kComplexFloat:
      return ::jaccl::Dtype::Complex64;
    default:
      TORCH_CHECK(
          false,
          "ProcessGroupMPS: JACCL does not support dtype ",
          toString(dtype));
  }
}

void jacclAllReduce(
    ::jaccl::Group& group,
    void* data,
    size_t numBytes,
    at::ScalarType dtype,
    ReduceOp::RedOpType op) {
  int jd = dtypeToJACCL(dtype);
  switch (op) {
    case ReduceOp::SUM:
      group.all_sum(data, data, numBytes, jd);
      return;
    case ReduceOp::MIN:
      group.all_min(data, data, numBytes, jd);
      return;
    case ReduceOp::MAX:
      group.all_max(data, data, numBytes, jd);
      return;
    default:
      TORCH_CHECK(
          false,
          "ProcessGroupMPS: JACCL supports SUM/MIN/MAX only; got ReduceOp=",
          static_cast<int>(op));
  }
}

void jacclBroadcast(
    ::jaccl::Group& group,
    void* data,
    size_t numBytes,
    int rootRank) {
  if (group.rank() == rootRank) {
    for (int dst = 0; dst < group.size(); dst++) {
      if (dst != rootRank) {
        group.send(data, numBytes, dst);
      }
    }
  } else {
    group.recv(data, numBytes, rootRank);
  }
}

} // namespace

ProcessGroupMPS::WorkMPS::WorkMPS(
    OpType opType,
    const char* profilingTitle,
    const std::optional<std::vector<at::Tensor>>& inputTensors)
    : Work(-1, opType, profilingTitle, inputTensors) {
  future_ = c10::make_intrusive<at::ivalue::Future>(
      c10::ListType::create(c10::TensorType::get()));
}

bool ProcessGroupMPS::WorkMPS::isCompleted() {
  return Work::isCompleted();
}

bool ProcessGroupMPS::WorkMPS::isSuccess() const {
  return Work::isSuccess();
}

bool ProcessGroupMPS::WorkMPS::wait(std::chrono::milliseconds timeout) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (!completed_) {
    if (timeout == kNoTimeout) {
      cv_.wait(lock, [this] { return completed_; });
    } else {
      cv_.wait_for(lock, timeout, [this] { return completed_; });
    }
  }
  if (exception_) {
    std::rethrow_exception(exception_);
  }
  return true;
}

c10::intrusive_ptr<c10::ivalue::Future>
ProcessGroupMPS::WorkMPS::getFuture() {
  return future_;
}

void ProcessGroupMPS::WorkMPS::finishWork() {
  future_->markCompleted(c10::IValue(outputTensors_));
  finish();
}

void ProcessGroupMPS::WorkMPS::finishWorkError(
    const std::exception_ptr& eptr) {
  future_->setError(eptr);
  finishAndThrow(eptr);
}

ProcessGroupMPS::Options::Options(std::chrono::milliseconds timeout)
    : Backend::Options(MPS_BACKEND_NAME, timeout) {}

ProcessGroupMPS::ProcessGroupMPS(
    const c10::intrusive_ptr<Store>& store,
    int rank,
    int size,
    c10::intrusive_ptr<Options> options)
    : Backend(rank, size),
      store_(store),
      options_(std::move(options)) {
  // Each rank probes its local RDMA device and advertises it via the
  // TCPStore; all ranks then assemble the full device connectivity matrix
  // that jaccl::Config requires. We assume one physical device per rank,
  // used for every peer connection — that matches the single-cable
  // Thunderbolt-5 topology this backend targets.
  std::string localDevice = probeRDMADevice();
  store_->set(
      kMpsDevicePrefix + std::to_string(rank),
      std::vector<uint8_t>(localDevice.begin(), localDevice.end()));

  std::vector<std::string> keys;
  keys.reserve(size);
  for (int r = 0; r < size; r++) {
    keys.push_back(kMpsDevicePrefix + std::to_string(r));
  }
  store_->wait(keys);

  std::vector<std::string> peerDevices(size);
  int missingRank = -1;
  for (int r = 0; r < size; r++) {
    auto data = store_->get(kMpsDevicePrefix + std::to_string(r));
    peerDevices[r].assign(data.begin(), data.end());
    if (peerDevices[r].empty() && missingRank < 0) {
      missingRank = r;
    }
  }
  TORCH_CHECK(
      missingRank < 0,
      "ProcessGroupMPS requires Apple Thunderbolt RDMA on every rank, but "
      "rank ",
      missingRank,
      " could not find a usable RDMA device (ibv_alloc_pd failed on every "
      "rdma_en* device). Check your Thunderbolt 5 cable and network setup, "
      "or use the 'gloo' backend for CPU-based distributed training.");

  std::vector<std::vector<std::vector<std::string>>> deviceMatrix(size);
  for (int src = 0; src < size; src++) {
    deviceMatrix[src].resize(size);
    for (int dst = 0; dst < size; dst++) {
      if (src != dst) {
        deviceMatrix[src][dst].push_back(peerDevices[src]);
      }
    }
  }

  // Rank 0 publishes the JACCL side-channel address on a port adjacent to
  // MASTER_PORT so it cannot collide with the TCPStore.
  std::string coordAddr;
  if (rank == 0) {
    const char* masterAddr = std::getenv("MASTER_ADDR");
    const char* masterPortEnv = std::getenv("MASTER_PORT");
    int basePort = masterPortEnv ? std::atoi(masterPortEnv) : 29500;
    std::string host = masterAddr ? masterAddr : "127.0.0.1";
    coordAddr = host + ":" + std::to_string(basePort + 1);
    store_->set(
        kMpsCoordKey,
        std::vector<uint8_t>(coordAddr.begin(), coordAddr.end()));
  } else {
    auto data = store_->get(kMpsCoordKey);
    coordAddr = std::string(data.begin(), data.end());
  }

  ::jaccl::Config cfg;
  cfg.set_rank(rank).set_coordinator(coordAddr).set_devices(
      std::move(deviceMatrix));
  jacclGroup_ = ::jaccl::init(cfg, /*strict=*/true);
  TORCH_CHECK(
      jacclGroup_,
      "ProcessGroupMPS: jaccl::init returned null for a configuration that "
      "should have been a valid mesh.");

  TORCH_WARN(
      "ProcessGroupMPS: using JACCL RDMA transport on ", localDevice);

  workerThread_ = std::thread(&ProcessGroupMPS::runLoop, this);
}

ProcessGroupMPS::~ProcessGroupMPS() {
  {
    std::lock_guard<std::mutex> lock(workMutex_);
    stop_ = true;
  }
  workCV_.notify_one();
  if (workerThread_.joinable()) {
    workerThread_.join();
  }
}

void ProcessGroupMPS::runLoop() {
  while (true) {
    std::function<void()> fn;
    {
      std::unique_lock<std::mutex> lock(workMutex_);
      workCV_.wait(lock, [this] { return stop_ || !workQueue_.empty(); });
      if (stop_ && workQueue_.empty())
        return;
      fn = std::move(workQueue_.front());
      workQueue_.pop_front();
    }
    fn();
  }
}

void ProcessGroupMPS::enqueue(std::function<void()> fn) {
  {
    std::lock_guard<std::mutex> lock(workMutex_);
    workQueue_.push_back(std::move(fn));
  }
  workCV_.notify_one();
}

at::Tensor ProcessGroupMPS::syncAndCopyToCPU(const at::Tensor& tensor) {
  at::mps::getDefaultMPSStream()->synchronize(
      at::mps::SyncType::COMMIT_AND_WAIT);
  return tensor.to(at::kCPU).contiguous();
}

void ProcessGroupMPS::copyToMPS(
    const at::Tensor& cpuTensor,
    at::Tensor& mpsTensor) {
  mpsTensor.copy_(cpuTensor);
  at::mps::getDefaultMPSStream()->synchronize(
      at::mps::SyncType::COMMIT_AND_WAIT);
}

c10::intrusive_ptr<Work> ProcessGroupMPS::allreduce(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts) {
  TORCH_CHECK(
      tensors.size() == 1,
      "ProcessGroupMPS::allreduce: only single tensor supported");

  auto& tensor = tensors[0];
  auto work = c10::make_intrusive<WorkMPS>(
      OpType::ALLREDUCE,
      "mps:allreduce",
      std::optional<std::vector<at::Tensor>>({tensor}));
  work->outputTensors_ = {tensor};

  auto fn = [this, tensor, reduceOp = opts.reduceOp, work]() mutable {
    try {
      if (void* hostPtr = mpsHostPtr(tensor)) {
        at::mps::getDefaultMPSStream()->synchronize(
            at::mps::SyncType::COMMIT_AND_WAIT);
        jacclAllReduce(
            *jacclGroup_,
            hostPtr,
            static_cast<size_t>(tensor.nbytes()),
            tensor.scalar_type(),
            reduceOp);
      } else {
        auto cpuTensor = syncAndCopyToCPU(tensor);
        jacclAllReduce(
            *jacclGroup_,
            cpuTensor.data_ptr(),
            static_cast<size_t>(cpuTensor.nbytes()),
            cpuTensor.scalar_type(),
            reduceOp);
        copyToMPS(cpuTensor, tensor);
      }
      work->finishWork();
    } catch (...) {
      work->finishWorkError(std::current_exception());
    }
  };

  enqueue(std::move(fn));
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupMPS::broadcast(
    std::vector<at::Tensor>& tensors,
    const BroadcastOptions& opts) {
  TORCH_CHECK(
      tensors.size() == 1,
      "ProcessGroupMPS::broadcast: only single tensor supported");

  auto& tensor = tensors[0];
  auto work = c10::make_intrusive<WorkMPS>(
      OpType::BROADCAST,
      "mps:broadcast",
      std::optional<std::vector<at::Tensor>>({tensor}));
  work->outputTensors_ = {tensor};

  auto fn = [this, tensor, rootRank = opts.rootRank, work]() mutable {
    try {
      if (void* hostPtr = mpsHostPtr(tensor)) {
        at::mps::getDefaultMPSStream()->synchronize(
            at::mps::SyncType::COMMIT_AND_WAIT);
        jacclBroadcast(
            *jacclGroup_,
            hostPtr,
            static_cast<size_t>(tensor.nbytes()),
            rootRank);
      } else {
        auto cpuTensor = syncAndCopyToCPU(tensor);
        jacclBroadcast(
            *jacclGroup_,
            cpuTensor.data_ptr(),
            static_cast<size_t>(cpuTensor.nbytes()),
            rootRank);
        copyToMPS(cpuTensor, tensor);
      }
      work->finishWork();
    } catch (...) {
      work->finishWorkError(std::current_exception());
    }
  };

  enqueue(std::move(fn));
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupMPS::barrier(
    const BarrierOptions& /*opts*/) {
  auto work = c10::make_intrusive<WorkMPS>(OpType::BARRIER, "mps:barrier");

  auto fn = [this, work]() {
    try {
      jacclGroup_->barrier();
      work->finishWork();
    } catch (...) {
      work->finishWorkError(std::current_exception());
    }
  };

  enqueue(std::move(fn));
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupMPS::send(
    std::vector<at::Tensor>& tensors,
    int dstRank,
    int /*tag*/) {
  TORCH_CHECK(
      tensors.size() == 1,
      "ProcessGroupMPS::send: only single tensor supported");

  auto& tensor = tensors[0];
  auto work = c10::make_intrusive<WorkMPS>(OpType::SEND, "mps:send");

  auto fn = [this, tensor, dstRank, work]() mutable {
    try {
      if (void* hostPtr = mpsHostPtr(tensor)) {
        at::mps::getDefaultMPSStream()->synchronize(
            at::mps::SyncType::COMMIT_AND_WAIT);
        jacclGroup_->send(
            hostPtr, static_cast<size_t>(tensor.nbytes()), dstRank);
      } else {
        auto cpuTensor = syncAndCopyToCPU(tensor);
        jacclGroup_->send(
            cpuTensor.data_ptr(),
            static_cast<size_t>(cpuTensor.nbytes()),
            dstRank);
      }
      work->finishWork();
    } catch (...) {
      work->finishWorkError(std::current_exception());
    }
  };

  enqueue(std::move(fn));
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupMPS::recv(
    std::vector<at::Tensor>& tensors,
    int srcRank,
    int /*tag*/) {
  TORCH_CHECK(
      tensors.size() == 1,
      "ProcessGroupMPS::recv: only single tensor supported");

  auto& tensor = tensors[0];
  auto work = c10::make_intrusive<WorkMPS>(OpType::RECV, "mps:recv");
  work->outputTensors_ = {tensor};

  auto fn = [this, tensor, srcRank, work]() mutable {
    try {
      if (void* hostPtr = mpsHostPtr(tensor)) {
        at::mps::getDefaultMPSStream()->synchronize(
            at::mps::SyncType::COMMIT_AND_WAIT);
        jacclGroup_->recv(
            hostPtr, static_cast<size_t>(tensor.nbytes()), srcRank);
      } else {
        auto cpuTensor = syncAndCopyToCPU(tensor);
        jacclGroup_->recv(
            cpuTensor.data_ptr(),
            static_cast<size_t>(cpuTensor.nbytes()),
            srcRank);
        copyToMPS(cpuTensor, tensor);
      }
      work->finishWork();
    } catch (...) {
      work->finishWorkError(std::current_exception());
    }
  };

  enqueue(std::move(fn));
  return work;
}

} // namespace c10d

#endif // USE_C10D_MPS
