#include "ProcessGroupMPI.hpp"

#include <map>

#if defined(OPEN_MPI) && OPEN_MPI
#include <mpi-ext.h> // Needed for CUDA-aware check
#endif

namespace c10d {

#define MPI_CHECK(cmd)                                                   \
  do {                                                                   \
    int mpiStatus = cmd;                                                 \
    if (mpiStatus != MPI_SUCCESS) {                                      \
      std::string err = "MPI error in: " + std::string(__FILE__) + ":" + \
          std::to_string(__LINE__) +                                     \
          ", with error code: " + std::to_string(mpiStatus);             \
      throw std::runtime_error(err);                                     \
    }                                                                    \
  } while (0)

namespace {

// Op mapping
std::map<ReduceOp, MPI_Op> mpiOp = {
    {ReduceOp::MIN, MPI_MIN},
    {ReduceOp::MAX, MPI_MAX},
    {ReduceOp::SUM, MPI_SUM},
    {ReduceOp::PRODUCT, MPI_PROD},
};
// Type mapping
std::map<at::ScalarType, MPI_Datatype> mpiDatatype = {
    {at::kByte, MPI_UNSIGNED_CHAR},
    {at::kChar, MPI_CHAR},
    {at::kDouble, MPI_DOUBLE},
    {at::kFloat, MPI_FLOAT},
    {at::kInt, MPI_INT},
    {at::kLong, MPI_LONG},
    {at::kShort, MPI_SHORT},
};

// Checking CUDA-aware MPI support, currently we only support CUDA aware
// MPI ops through Open MPI
bool cudaAwareMpiCheck() {
// Run time check
#if defined(MPIX_CUDA_AWARE_SUPPORT)
  if (MPIX_Query_cuda_support() == 1) {
    return true;
  } else {
    return false;
  }
#else // !defined(MPIX_CUDA_AWARE_SUPPORT)
  return false;
#endif // MPIX_CUDA_AWARE_SUPPORT
}

// Checking the input tensor's validity
void checkSingleTensorHelper(const at::Tensor& tensor) {
  if (!tensor.is_contiguous()) {
    throw std::runtime_error("input tensor has to be contiguous");
  }
  if (tensor.is_sparse()) {
    throw std::runtime_error("input tensor has to be dense");
  }
  if (tensor.is_cuda() && !cudaAwareMpiCheck()) {
    throw std::runtime_error(
        "CUDA tensor detected and the MPI used doesn't "
        "have CUDA-aware MPI support");
  }
}

void checkSingleTensor(const std::vector<at::Tensor>& tensors) {
  if (tensors.size() != 1) {
    throw std::runtime_error(
        "MPI process group does not support multi-GPU collectives");
  }
  checkSingleTensorHelper(tensors[0]);
}

void checkSameSizeAndType(
    const at::Tensor& tensor,
    const std::vector<at::Tensor>& tensors) {
  for (size_t i = 0; i < tensors.size(); ++i) {
    if ((tensors[i].numel() != tensor.numel()) ||
        (tensors[i].type() != tensor.type())) {
      throw std::runtime_error("Tensors are not equal in size or data type");
    }
    checkSingleTensorHelper(tensors[i]);
  }
}

} // namespace

// ProcessGroupMPI::WorkMPI
ProcessGroupMPI::WorkMPI::WorkMPI() : completed_(false) {}

ProcessGroupMPI::WorkMPI::~WorkMPI() {}

bool ProcessGroupMPI::WorkMPI::isCompleted() {
  return completed_;
}

bool ProcessGroupMPI::WorkMPI::isSuccess() const {
  return !exception_;
}

void ProcessGroupMPI::WorkMPI::synchronize() {}

bool ProcessGroupMPI::WorkMPI::wait() {
  std::unique_lock<std::mutex> lock(workMutex_);
  while (!completed_) {
    workCV_.wait(lock);
  }
  return isSuccess();
}

void ProcessGroupMPI::WorkMPI::finish() {
  {
    std::unique_lock<std::mutex> lock(workMutex_);
    completed_ = true;
  }
  workCV_.notify_all();
}

void ProcessGroupMPI::WorkMPI::finishWithException(
    std::exception_ptr caughtWorkException) {
  {
    std::unique_lock<std::mutex> lock(workMutex_);
    completed_ = true;
    exception_ = caughtWorkException;
  }
  workCV_.notify_all();
}

const std::exception& ProcessGroupMPI::WorkMPI::exception() const {
  try {
    std::rethrow_exception(exception_);
  } catch (const std::exception& e) {
    return e;
  }
}

ProcessGroupMPI::AsyncWork::AsyncWork(
    at::Tensor tensor,
    MPI_Request request,
    int* srcRank)
    : tensor_(std::move(tensor)), request_(request), srcRank_(srcRank) {
  memset(&status_, 0, sizeof(status_));
}

ProcessGroupMPI::AsyncWork::~AsyncWork() {
  if (request_ != MPI_REQUEST_NULL) {
    throw std::runtime_error(
        "Attempted destruction of AsyncWork before work has completed");
  }
}

bool ProcessGroupMPI::AsyncWork::isCompleted() {
  if (request_ == MPI_REQUEST_NULL) {
    return true;
  }

  std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
  int flag = 0;
  MPI_CHECK(MPI_Test(&request_, &flag, &status_));
  if (request_ != MPI_REQUEST_NULL) {
    return false;
  }

  // request_ == MPI_REQUEST_NULL; the work has completed
  if (srcRank_ != nullptr) {
    *srcRank_ = status_.MPI_SOURCE;
  }

  // Populate exception if request was not successful
  if (status_.MPI_ERROR != MPI_SUCCESS) {
    populateException();
  }

  return true;
}

bool ProcessGroupMPI::AsyncWork::isSuccess() const {
  if (request_ != MPI_REQUEST_NULL) {
    throw std::runtime_error(
        "Invalid call to AsyncWork::isSuccess before work has completed");
  }

  return status_.MPI_ERROR == MPI_SUCCESS;
}

void ProcessGroupMPI::AsyncWork::synchronize() {}

bool ProcessGroupMPI::AsyncWork::wait() {
  if (request_ == MPI_REQUEST_NULL) {
    return true;
  }

  std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
  MPI_CHECK(MPI_Wait(&request_, &status_));
  if (srcRank_ != nullptr && status_.MPI_ERROR == MPI_SUCCESS) {
    *srcRank_ = status_.MPI_SOURCE;
  }

  auto ok = (status_.MPI_ERROR == MPI_SUCCESS);

  // Populate exception if request was not successful
  if (!ok) {
    populateException();
  }

  return ok;
}

const std::exception& ProcessGroupMPI::AsyncWork::exception() const {
  try {
    std::rethrow_exception(exception_);
  } catch (const std::exception& e) {
    return e;
  }
}

void ProcessGroupMPI::AsyncWork::populateException() {
  std::array<char, MPI_MAX_ERROR_STRING> buf;
  int len = buf.size();
  MPI_CHECK(MPI_Error_string(status_.MPI_ERROR, buf.data(), &len));
  exception_ =
      std::make_exception_ptr(std::runtime_error(std::string(buf.data(), len)));
}

// Static global states
int ProcessGroupMPI::numProcessGroups_ = 0;
int ProcessGroupMPI::mpiThreadSupport_ = 0;
std::mutex ProcessGroupMPI::pgGlobalMutex_;
// We only want to initialize once
std::once_flag ProcessGroupMPI::onceFlagInitMPI;

void ProcessGroupMPI::mpiExit() {
  std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
  MPI_CHECK(MPI_Finalize());
}

void ProcessGroupMPI::initMPIOnce() {
  // Initialize MPI environment
  std::call_once(onceFlagInitMPI, []() {
    MPI_CHECK(MPI_Init_thread(
        nullptr, nullptr, MPI_THREAD_SERIALIZED, &mpiThreadSupport_));
    if (mpiThreadSupport_ < MPI_THREAD_SERIALIZED) {
      throw std::runtime_error(
          "Used MPI implementation doesn't have the "
          "minimum level of threading support: "
          "MPI_THREAD_SERIALIZED. This is required by "
          "c10d package");
    }
    if (std::atexit(ProcessGroupMPI::mpiExit)) {
      throw std::runtime_error("Fail to register the MPI exit handler");
    }
  });
}

std::shared_ptr<ProcessGroupMPI> ProcessGroupMPI::createProcessGroupMPI(
    std::vector<int> ranks) {
  // Once initialization
  initMPIOnce();

  std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);

  int rank = -1;
  int size = -1;

  // Update the world size and rank
  MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));
  MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

  if (rank < 0 || size < 0) {
    throw std::runtime_error("Failed to get the world_size / rank");
  }

  // If no ranks are specified, assume we're creating the root group
  if (ranks.empty()) {
    globalLock.unlock();
    return std::make_shared<ProcessGroupMPI>(rank, size, MPI_COMM_WORLD);
  }

  MPI_Group worldGroup;
  MPI_CHECK(MPI_Comm_group(MPI_COMM_WORLD, &worldGroup));

  MPI_Group ranksGroup;
  MPI_CHECK(
      MPI_Group_incl(worldGroup, ranks.size(), ranks.data(), &ranksGroup));

  MPI_Comm groupComm;
  MPI_CHECK(MPI_Comm_create(MPI_COMM_WORLD, ranksGroup, &groupComm));

  MPI_CHECK(MPI_Group_free(&worldGroup));
  MPI_CHECK(MPI_Group_free(&ranksGroup));

  globalLock.unlock();
  return std::make_shared<ProcessGroupMPI>(rank, size, groupComm);
}

ProcessGroupMPI::ProcessGroupMPI(int rank, int size, MPI_Comm pgComm)
    : ProcessGroup(rank, size),
      stop_(false),
      pgComm_(pgComm),
      groupRank_(-1),
      groupSize_(-1) {
  std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);

  if (pgComm_ != MPI_COMM_NULL) {
    MPI_CHECK(MPI_Comm_rank(pgComm_, &groupRank_));
    MPI_CHECK(MPI_Comm_size(pgComm_, &groupSize_));
    std::vector<int> rankToGroupRank{rank_, groupRank_};
    std::vector<int> allRankToGroupRank;
    allRankToGroupRank.resize(2 * groupSize_);
    MPI_CHECK(MPI_Allgather(
        rankToGroupRank.data(),
        2,
        MPI_INT,
        allRankToGroupRank.data(),
        2,
        MPI_INT,
        pgComm_));
    for (size_t i = 0; i < allRankToGroupRank.size(); i += 2) {
      groupRankMap_[allRankToGroupRank[i]] = allRankToGroupRank[i + 1];
    }
  }

  // increase the total PG count
  ++numProcessGroups_;
  globalLock.unlock();

  // Start the worker thread accepting MPI calls
  workerThread_ = std::thread(&ProcessGroupMPI::runLoop, this);
}

ProcessGroupMPI::~ProcessGroupMPI() {
  destroy();
}

void ProcessGroupMPI::destroy() {
  std::unique_lock<std::mutex> lock(pgMutex_);

  while (!queue_.empty()) {
    queueConsumeCV_.wait(lock);
  }
  // Queue is empty, signal stop
  stop_ = true;

  // Release lock to allow threads to terminate
  queueProduceCV_.notify_all();

  lock.unlock();

  // Join the single worker thread
  workerThread_.join();

  // Decrease the number of PG created
  std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
  --numProcessGroups_;
}

void ProcessGroupMPI::abort() {
  destroy();
  MPI_Abort(pgComm_, EXIT_FAILURE);
}

void ProcessGroupMPI::runLoop() {
  std::unique_lock<std::mutex> lock(pgMutex_);

  while (!stop_) {
    if (queue_.empty()) {
      queueProduceCV_.wait(lock);
      continue;
    }

    auto workTuple = std::move(queue_.front());

    queue_.pop_front();
    queueConsumeCV_.notify_one();

    auto& workEntry = std::get<0>(workTuple);
    auto& work = std::get<1>(workTuple);

    lock.unlock();

    try {
      workEntry->run(workEntry);
      work->finish();
    } catch (...) {
      work->finishWithException(std::current_exception());
    }

    lock.lock();
  }
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupMPI::enqueue(
    std::unique_ptr<WorkEntry> entry) {
  auto work = std::make_shared<WorkMPI>();
  std::unique_lock<std::mutex> lock(pgMutex_);
  queue_.push_back(std::make_tuple(std::move(entry), work));
  queueProduceCV_.notify_one();
  return work;
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupMPI::broadcast(
    std::vector<at::Tensor>& tensors,
    const BroadcastOptions& opts) {
  if (pgComm_ == MPI_COMM_NULL) {
    return nullptr;
  }
  checkSingleTensor(tensors);
  std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
      [opts, this](std::unique_ptr<WorkEntry>& entry) {
        auto data = (entry->src)[0];
        std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
        MPI_CHECK(MPI_Bcast(
            data.data_ptr(),
            data.numel(),
            mpiDatatype.at(data.type().scalarType()),
            opts.rootRank,
            pgComm_));
      };
  auto entry = std::unique_ptr<WorkEntry>(
      new WorkEntry(&tensors, nullptr, std::move(runFunc)));
  return enqueue(std::move(entry));
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupMPI::allreduce(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts) {
  if (pgComm_ == MPI_COMM_NULL) {
    return nullptr;
  }
  checkSingleTensor(tensors);

  std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
      [opts, this](std::unique_ptr<WorkEntry>& entry) {
        auto data = (entry->src)[0];
        std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
        MPI_CHECK(MPI_Allreduce(
            MPI_IN_PLACE,
            data.data_ptr(),
            data.numel(),
            mpiDatatype.at(data.type().scalarType()),
            mpiOp.at(opts.reduceOp),
            pgComm_));
      };
  auto entry = std::unique_ptr<WorkEntry>(
      new WorkEntry(&tensors, nullptr, std::move(runFunc)));
  return enqueue(std::move(entry));
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupMPI::reduce(
    std::vector<at::Tensor>& tensors,
    const ReduceOptions& opts) {
  if (pgComm_ == MPI_COMM_NULL) {
    return nullptr;
  }
  checkSingleTensor(tensors);

  std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
      [opts, this](std::unique_ptr<WorkEntry>& entry) {
        auto data = (entry->src)[0];
        auto dataPtr = (entry->src)[0].data_ptr();
        void* sendbuf = (groupRank_ == opts.rootRank) ? MPI_IN_PLACE : dataPtr;
        void* recvbuf = (groupRank_ == opts.rootRank) ? dataPtr : nullptr;

        std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
        MPI_CHECK(MPI_Reduce(
            sendbuf,
            recvbuf,
            data.numel(),
            mpiDatatype.at(data.type().scalarType()),
            mpiOp.at(opts.reduceOp),
            opts.rootRank,
            pgComm_));
      };
  auto entry = std::unique_ptr<WorkEntry>(
      new WorkEntry(&tensors, nullptr, std::move(runFunc)));
  return enqueue(std::move(entry));
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupMPI::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors) {
  if (pgComm_ == MPI_COMM_NULL) {
    return nullptr;
  }
  checkSingleTensor(inputTensors);
  if (outputTensors.size() != 1) {
    throw std::runtime_error(
        "MPI process group only supports a single "
        "tensor op");
  }
  if (static_cast<size_t>(groupSize_) != outputTensors[0].size()) {
    throw std::runtime_error(
        "All gather: number of output tensors should equal "
        "to the world size");
  }

  checkSameSizeAndType(inputTensors[0], outputTensors[0]);

  std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
      [this](std::unique_ptr<WorkEntry>& entry) {
        auto data = (entry->src)[0];
        std::vector<at::Tensor>& outputDataVec = entry->dst;
        auto flatOutputTensor = newLikeFlat(outputDataVec);

        std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
        MPI_CHECK(MPI_Allgather(
            data.data_ptr(),
            data.numel(),
            mpiDatatype.at(data.type().scalarType()),
            flatOutputTensor.data_ptr(),
            data.numel(),
            mpiDatatype.at(data.type().scalarType()),
            pgComm_));

        for (size_t i = 0; i < outputDataVec.size(); ++i) {
          outputDataVec[i].copy_(flatOutputTensor[i]);
        }
      };
  auto entry = std::unique_ptr<WorkEntry>(
      new WorkEntry(&inputTensors, &outputTensors[0], std::move(runFunc)));
  return enqueue(std::move(entry));
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupMPI::gather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const GatherOptions& opts) {
  if (pgComm_ == MPI_COMM_NULL) {
    return nullptr;
  }
  checkSingleTensor(inputTensors);

  if (outputTensors.size() != 1) {
    throw std::runtime_error("Gather: multi-GPU collective is not supported");
  }

  if (groupRank_ != opts.rootRank) {
    if (outputTensors[0].size() > 0) {
      throw std::runtime_error(
          "Gather: number of output tensors should be 0 "
          "for non-root");
    }
  } else {
    if (static_cast<size_t>(groupSize_) != outputTensors[0].size()) {
      throw std::runtime_error(
          "Gather: number of output tensors should equal "
          "to the world size");
    }
    checkSameSizeAndType(inputTensors[0], outputTensors[0]);
  }

  std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
      [opts, this](std::unique_ptr<WorkEntry>& entry) {
        auto data = (entry->src)[0];
        void* recvbuf = nullptr;
        at::Tensor flatOutputTensor;

        if (groupRank_ == opts.rootRank) {
          flatOutputTensor = newLikeFlat(entry->dst);
          recvbuf = flatOutputTensor.data_ptr();
        }

        std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
        MPI_CHECK(MPI_Gather(
            data.data_ptr(),
            data.numel(),
            mpiDatatype.at(data.type().scalarType()),
            recvbuf,
            data.numel(),
            mpiDatatype.at(data.type().scalarType()),
            opts.rootRank,
            pgComm_));

        if (groupRank_ == opts.rootRank) {
          std::vector<at::Tensor>& outputDataVec = entry->dst;
          // copy the flattened output tensors to the outputs
          for (size_t i = 0; i < outputDataVec.size(); ++i) {
            outputDataVec.at(i).copy_(flatOutputTensor[i]);
          }
        }
      };

  if (groupRank_ == opts.rootRank) {
    auto entry = std::unique_ptr<WorkEntry>(
        new WorkEntry(&inputTensors, &outputTensors[0], std::move(runFunc)));
    return enqueue(std::move(entry));
  } else {
    auto entry = std::unique_ptr<WorkEntry>(
        new WorkEntry(&inputTensors, nullptr, std::move(runFunc)));
    return enqueue(std::move(entry));
  }
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupMPI::scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ScatterOptions& opts) {
  if (pgComm_ == MPI_COMM_NULL) {
    return nullptr;
  }
  checkSingleTensor(outputTensors);
  if (inputTensors.size() != 1) {
    throw std::runtime_error("Scatter: multi-GPU collective is not supported");
  }

  if (groupRank_ != opts.rootRank) {
    if (inputTensors[0].size() > 0) {
      throw std::runtime_error(
          "Scatter: number of input tensors should be 0 "
          "for non-root");
    }
  } else {
    if (static_cast<size_t>(groupSize_) != inputTensors[0].size()) {
      throw std::runtime_error(
          "Scatter: number of input tensors should equal "
          "to the world size");
    }
    checkSameSizeAndType(outputTensors[0], inputTensors[0]);
  }

  std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
      [opts, this](std::unique_ptr<WorkEntry>& entry) {
        auto data = (entry->dst)[0];
        void* sendbuf = nullptr;
        at::Tensor flatInputTensor;

        if (groupRank_ == opts.rootRank) {
          std::vector<at::Tensor>& inputDataVec = entry->src;
          flatInputTensor = newLikeFlat(inputDataVec);
          sendbuf = flatInputTensor.data_ptr();

          // copy the input tensors to the flatten large send buffer
          for (size_t i = 0; i < inputDataVec.size(); ++i) {
            flatInputTensor[i].copy_(inputDataVec.at(i));
          }
        }

        std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
        MPI_CHECK(MPI_Scatter(
            sendbuf,
            data.numel(),
            mpiDatatype.at(data.type().scalarType()),
            data.data_ptr(),
            data.numel(),
            mpiDatatype.at(data.type().scalarType()),
            opts.rootRank,
            pgComm_));
      };

  if (groupRank_ == opts.rootRank) {
    auto entry = std::unique_ptr<WorkEntry>(
        new WorkEntry(&inputTensors[0], &outputTensors, std::move(runFunc)));
    return enqueue(std::move(entry));
  } else {
    auto entry = std::unique_ptr<WorkEntry>(
        new WorkEntry(nullptr, &outputTensors, std::move(runFunc)));
    return enqueue(std::move(entry));
  }
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupMPI::send(
    std::vector<at::Tensor>& tensors,
    int dstRank,
    int tag) {
  if (pgComm_ == MPI_COMM_NULL) {
    return nullptr;
  }

  checkSingleTensor(tensors);

  auto& tensor = tensors[0];
  MPI_Request request = MPI_REQUEST_NULL;

  {
    std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
    MPI_CHECK(MPI_Isend(
        tensor.data_ptr(),
        tensor.numel(),
        mpiDatatype.at(tensor.type().scalarType()),
        dstRank,
        tag,
        pgComm_,
        &request));
  }

  return std::make_shared<AsyncWork>(tensor, request);
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupMPI::recv(
    std::vector<at::Tensor>& tensors,
    int srcRank,
    int tag) {
  if (pgComm_ == MPI_COMM_NULL) {
    return nullptr;
  }

  checkSingleTensor(tensors);

  auto& tensor = tensors[0];
  MPI_Request request = MPI_REQUEST_NULL;

  {
    std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
    MPI_CHECK(MPI_Irecv(
        tensor.data_ptr(),
        tensor.numel(),
        mpiDatatype.at(tensor.type().scalarType()),
        srcRank,
        tag,
        pgComm_,
        &request));
  }

  return std::make_shared<AsyncWork>(tensor, request);
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupMPI::recvAnysource(
    std::vector<at::Tensor>& tensors,
    int* srcRank,
    int tag) {
  if (pgComm_ == MPI_COMM_NULL) {
    return nullptr;
  }

  checkSingleTensor(tensors);

  auto& tensor = tensors[0];
  MPI_Request request = MPI_REQUEST_NULL;

  {
    std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
    MPI_CHECK(MPI_Irecv(
        tensor.data_ptr(),
        tensor.numel(),
        mpiDatatype.at(tensor.type().scalarType()),
        MPI_ANY_SOURCE,
        tag,
        pgComm_,
        &request));
  }

  return std::make_shared<AsyncWork>(tensor, request, srcRank);
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupMPI::barrier() {
  if (pgComm_ == MPI_COMM_NULL) {
    return nullptr;
  }
  std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
      [this](std::unique_ptr<WorkEntry>& entry) {
        std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
        MPI_CHECK(MPI_Barrier(pgComm_));
      };
  auto entry = std::unique_ptr<WorkEntry>(
      new WorkEntry(nullptr, nullptr, std::move(runFunc)));
  return enqueue(std::move(entry));
}

std::unordered_map<int, int> ProcessGroupMPI::getGroupRank() {
  return groupRankMap_;
}

} // namespace c10d
