#include "ProcessGroupMPI.hpp"

#include <map>

#include <mpi.h>
#include <mpi-ext.h> // Needed for CUDA-aware check

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

// Checking CUDA-aware MPI support
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

void mpiExit() {
  MPI_CHECK(MPI_Finalize());
}

} // namespace

// ProcessGroupMPI::WorkMPI
ProcessGroupMPI::WorkMPI::WorkMPI() : completed_(false) {}

ProcessGroupMPI::WorkMPI::~WorkMPI() {}

bool ProcessGroupMPI::WorkMPI::isCompleted() const {
  return completed_;
}

bool ProcessGroupMPI::WorkMPI::isSuccess() const {
  return !workException_;
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
    workException_ = caughtWorkException;
  }
  workCV_.notify_all();
}

const std::exception& ProcessGroupMPI::WorkMPI::exception() const {
  try {
    std::rethrow_exception(workException_);
  } catch (const std::exception& e) {
    return e;
  }
}

// Static global states
int ProcessGroupMPI::numProcessGroups_ = 0;
int ProcessGroupMPI::mpiThreadSupport_ = 0;
std::mutex ProcessGroupMPI::pgGlobalMutex_;
// We only want to initialize once
std::once_flag ProcessGroupMPI::onceFlagInitMPI;

void ProcessGroupMPI::initMPIOnce() {
  // Initialize MPI environment
  std::call_once(onceFlagInitMPI, []() {
    MPI_CHECK(MPI_Init_thread(
        nullptr, nullptr, MPI_THREAD_MULTIPLE, &mpiThreadSupport_));
    if (mpiThreadSupport_ < MPI_THREAD_SERIALIZED) {
      throw std::runtime_error(
          "Used MPI implementation doesn't have the "
          "minimum level of threading support: "
          "MPI_THREAD_SERIALIZED. This is required by "
          "c10d package");
    }
    if (std::atexit(mpiExit)) {
      throw std::runtime_error("Fail to register the MPI exit handler");
    }
  });
}

std::shared_ptr<ProcessGroupMPI> ProcessGroupMPI::createProcessGroupMPI() {
  // Once initialization
  initMPIOnce();

  int rank = -1;
  int size = -1;
  // Update the world size and rank
  MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));
  MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

  if (rank < 0 || size < 0) {
    throw std::runtime_error("Failed to get the world_size / rank");
  }

  return std::make_shared<ProcessGroupMPI>(rank, size);
}

ProcessGroupMPI::ProcessGroupMPI(int rank, int size)
    : ProcessGroup(rank, size), stop_(false) {
  std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);

  if (mpiThreadSupport_ != MPI_THREAD_MULTIPLE && numProcessGroups_ >= 1) {
    throw std::runtime_error(
        "More than one process group created, "
        "this is not supported due to the used MPI "
        "implementation doesn't provide the full support "
        "of multi-threading");
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
  MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
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
  checkSingleTensor(tensors);
  std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
      [opts](std::unique_ptr<WorkEntry>& entry) {
        auto data = (*entry->src)[0];
        MPI_CHECK(MPI_Bcast(
            data.data_ptr(),
            data.numel(),
            mpiDatatype.at(data.type().scalarType()),
            opts.rootRank,
            MPI_COMM_WORLD));
      };
  auto entry = std::unique_ptr<WorkEntry>(
      new WorkEntry(&tensors, nullptr, std::move(runFunc)));
  return enqueue(std::move(entry));
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupMPI::allreduce(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts) {
  checkSingleTensor(tensors);
  std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
      [opts](std::unique_ptr<WorkEntry>& entry) {
        auto data = (*entry->src)[0];
        MPI_CHECK(MPI_Allreduce(
            MPI_IN_PLACE,
            data.data_ptr(),
            data.numel(),
            mpiDatatype.at(data.type().scalarType()),
            mpiOp.at(opts.reduceOp),
            MPI_COMM_WORLD));
      };
  auto entry = std::unique_ptr<WorkEntry>(
      new WorkEntry(&tensors, nullptr, std::move(runFunc)));
  return enqueue(std::move(entry));
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupMPI::reduce(
    std::vector<at::Tensor>& tensors,
    const ReduceOptions& opts) {
  checkSingleTensor(tensors);

  std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
      [opts, this](std::unique_ptr<WorkEntry>& entry) {
        auto data = (*entry->src)[0];
        auto dataPtr = (*entry->src)[0].data_ptr();
        void* sendbuf = (rank_ == opts.rootRank) ? MPI_IN_PLACE : dataPtr;
        void* recvbuf = (rank_ == opts.rootRank) ? dataPtr : nullptr;

        MPI_CHECK(MPI_Reduce(
            sendbuf,
            recvbuf,
            data.numel(),
            mpiDatatype.at(data.type().scalarType()),
            mpiOp.at(opts.reduceOp),
            opts.rootRank,
            MPI_COMM_WORLD));
      };
  auto entry = std::unique_ptr<WorkEntry>(
      new WorkEntry(&tensors, nullptr, std::move(runFunc)));
  return enqueue(std::move(entry));
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupMPI::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors) {
  checkSingleTensor(inputTensors);
  if (outputTensors.size() != 1) {
    throw std::runtime_error(
        "MPI process group only supports a single "
        "tensor op");
  }
  if (static_cast<size_t>(size_) != outputTensors[0].size()) {
    throw std::runtime_error(
        "All gather: number of output tensors should equal "
        "to the world size");
  }

  checkSameSizeAndType(inputTensors[0], outputTensors[0]);

  std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
      [](std::unique_ptr<WorkEntry>& entry) {
        auto data = (*entry->src)[0];
        std::vector<at::Tensor>& outputDataVec = *(entry->dst);
        auto flatOutputTensor = newLikeFlat(outputDataVec);

        MPI_CHECK(MPI_Allgather(
            data.data_ptr(),
            data.numel(),
            mpiDatatype.at(data.type().scalarType()),
            flatOutputTensor.data_ptr(),
            data.numel(),
            mpiDatatype.at(data.type().scalarType()),
            MPI_COMM_WORLD));

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
  checkSingleTensor(inputTensors);

  if (rank_ != opts.rootRank) {
    if (outputTensors.size() > 0) {
      throw std::runtime_error(
          "Gather: number of output tensors should be 0 "
          "for non-root");
    }
  } else {
    if (outputTensors.size() != 1) {
      throw std::runtime_error("Gather: multi-GPU collective is not supported");
    }
    if (static_cast<size_t>(size_) != outputTensors[0].size()) {
      throw std::runtime_error(
          "Gather: number of output tensors should equal "
          "to the world size");
    }
    checkSameSizeAndType(inputTensors[0], outputTensors[0]);
  }

  std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
      [opts, this](std::unique_ptr<WorkEntry>& entry) {
        auto data = (*entry->src)[0];
        void* recvbuf = nullptr;
        at::Tensor flatOutputTensor;
        std::vector<at::Tensor>* outputDataVec = nullptr;

        if (rank_ == opts.rootRank) {
          outputDataVec = entry->dst;
          flatOutputTensor = newLikeFlat(*outputDataVec);
          recvbuf = flatOutputTensor.data_ptr();
        }
        MPI_CHECK(MPI_Gather(
            data.data_ptr(),
            data.numel(),
            mpiDatatype.at(data.type().scalarType()),
            recvbuf,
            data.numel(),
            mpiDatatype.at(data.type().scalarType()),
            opts.rootRank,
            MPI_COMM_WORLD));

        if (rank_ == opts.rootRank) {
          // copy the flattened output tensors to the outputs
          for (size_t i = 0; i < outputDataVec->size(); ++i) {
            outputDataVec->at(i).copy_(flatOutputTensor[i]);
          }
        }
      };

  if (rank_ == opts.rootRank) {
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
  checkSingleTensor(outputTensors);

  if (rank_ != opts.rootRank) {
    if (inputTensors.size() > 0) {
      throw std::runtime_error(
          "Scatter: number of input tensors should be 0 "
          "for non-root");
    }
  } else {
    if (inputTensors.size() != 1) {
      throw std::runtime_error("Gather: multi-GPU collective is not supported");
    }
    if (static_cast<size_t>(size_) != inputTensors[0].size()) {
      throw std::runtime_error(
          "Scatter: number of input tensors should equal "
          "to the world size");
    }
    checkSameSizeAndType(outputTensors[0], inputTensors[0]);
  }

  std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
      [opts, this](std::unique_ptr<WorkEntry>& entry) {
        auto data = (*entry->dst)[0];
        void* sendbuf = nullptr;
        at::Tensor flatInputTensor;
        std::vector<at::Tensor>* inputDataVec;

        if (rank_ == opts.rootRank) {
          inputDataVec = entry->src;
          flatInputTensor = newLikeFlat(*inputDataVec);
          sendbuf = flatInputTensor.data_ptr();

          // copy the input tensors to the flatten large send buffer
          for (size_t i = 0; i < inputDataVec->size(); ++i) {
            flatInputTensor[i].copy_(inputDataVec->at(i));
          }
        }

        MPI_CHECK(MPI_Scatter(
            sendbuf,
            data.numel(),
            mpiDatatype.at(data.type().scalarType()),
            data.data_ptr(),
            data.numel(),
            mpiDatatype.at(data.type().scalarType()),
            opts.rootRank,
            MPI_COMM_WORLD));
      };

  if (rank_ == opts.rootRank) {
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
    int dstRank) {
  checkSingleTensor(tensors);
  std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
      [dstRank](std::unique_ptr<WorkEntry>& entry) {
        auto data = (*entry->src)[0];
        MPI_CHECK(MPI_Send(
            data.data_ptr(),
            data.numel(),
            mpiDatatype.at(data.type().scalarType()),
            dstRank,
            0,
            MPI_COMM_WORLD));
      };
  auto entry = std::unique_ptr<WorkEntry>(
      new WorkEntry(&tensors, nullptr, std::move(runFunc)));
  return enqueue(std::move(entry));
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupMPI::recv(
    std::vector<at::Tensor>& tensors,
    int srcRank) {
  checkSingleTensor(tensors);
  std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
      [srcRank](std::unique_ptr<WorkEntry>& entry) {
        auto data = (*entry->src)[0];
        MPI_CHECK(MPI_Recv(
            data.data_ptr(),
            data.numel(),
            mpiDatatype.at(data.type().scalarType()),
            srcRank,
            0,
            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE));
      };
  auto entry = std::unique_ptr<WorkEntry>(
      new WorkEntry(&tensors, nullptr, std::move(runFunc)));
  return enqueue(std::move(entry));
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupMPI::recvAnysource(
    std::vector<at::Tensor>& tensors,
    int* srcRank) {
  checkSingleTensor(tensors);
  std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
      [srcRank](std::unique_ptr<WorkEntry>& entry) {
        auto data = (*entry->src)[0];
        MPI_Status status;
        MPI_CHECK(MPI_Recv(
            data.data_ptr(),
            data.numel(),
            mpiDatatype.at(data.type().scalarType()),
            MPI_ANY_SOURCE,
            0,
            MPI_COMM_WORLD,
            &status));
        *(entry->srcRank) = status.MPI_SOURCE;
      };
  auto entry = std::unique_ptr<WorkEntry>(
      new WorkEntry(&tensors, nullptr, std::move(runFunc)));
  entry->srcRank = srcRank;
  return enqueue(std::move(entry));
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupMPI::barrier() {
  std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
      [](std::unique_ptr<WorkEntry>& entry) {
        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
      };
  auto entry = std::unique_ptr<WorkEntry>(
      new WorkEntry(nullptr, nullptr, std::move(runFunc)));
  return enqueue(std::move(entry));
}

} // namespace c10d
