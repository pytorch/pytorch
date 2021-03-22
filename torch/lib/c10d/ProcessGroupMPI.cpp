#include <c10d/ProcessGroupMPI.hpp>

#include <limits>
#include <map>

#include <c10/core/DeviceGuard.h>

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
    const at::Tensor& t_in,
    const std::vector<at::Tensor>& tensors) {
  for (const auto & tensor : tensors) {
    if ((tensor.numel() != t_in.numel()) ||
        (tensor.scalar_type() != t_in.scalar_type())) {
      throw std::runtime_error("Tensors are not equal in size or data type");
    }
    checkSingleTensorHelper(tensor);
  }
}

} // namespace

ProcessGroupMPI::AsyncWork::AsyncWork(
    at::Tensor tensor,
    MPI_Request request,
    const char* profilingTitle,
    const c10::optional<std::vector<at::Tensor>>& inputTensors)
    : ProcessGroup::Work(-1, OpType::UNKNOWN, profilingTitle, inputTensors),
      tensor_(std::move(tensor)),
      request_(request) {
  memset(&status_, 0, sizeof(status_));
}

ProcessGroupMPI::AsyncWork::~AsyncWork() {
  if (request_ != MPI_REQUEST_NULL) {
    std::cerr
        << "Attempted destruction of AsyncWork before work has completed, "
        << "terminating the program." << std::endl;
    std::terminate();
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

int ProcessGroupMPI::AsyncWork::sourceRank() const {
  return status_.MPI_SOURCE;
}

bool ProcessGroupMPI::AsyncWork::wait(std::chrono::milliseconds /* unused */) {
  if (request_ == MPI_REQUEST_NULL) {
    // AsyncWork needs to manually call profiling end callbacks if they are set,
    // since it does not call ProcessGroup::finish().
    if (ProcessGroup::Work::recordFunctionEndCallback_) {
      ProcessGroup::Work::recordFunctionEndCallback_();
      ProcessGroup::Work::recordFunctionEndCallback_ = nullptr;
    }
    return true;
  }

  std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
  MPI_CHECK(MPI_Wait(&request_, &status_));
  auto ok = (status_.MPI_ERROR == MPI_SUCCESS);

  // AsyncWork needs to manually call profiling end callbacks if they are set,
  // since it does not call ProcessGroup::finish().
  if (ProcessGroup::Work::recordFunctionEndCallback_) {
      ProcessGroup::Work::recordFunctionEndCallback_();
      ProcessGroup::Work::recordFunctionEndCallback_ = nullptr;
  }

  if (!ok) {
    populateException();
    std::rethrow_exception(exception_);
  }
  // Always return true, because abort API is not implemented.
  return true;
}

void ProcessGroupMPI::AsyncWork::abort() {
  TORCH_CHECK(false, "ProcessGroupMPI::AsyncWork::abort not implemented.")
}

void ProcessGroupMPI::AsyncWork::populateException() {
  std::array<char, MPI_MAX_ERROR_STRING> buf;
  int len = buf.size();
  MPI_CHECK(MPI_Error_string(status_.MPI_ERROR, buf.data(), &len));
  exception_ =
      std::make_exception_ptr(std::runtime_error(std::string(buf.data(), len)));
}

// Static global states
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

c10::intrusive_ptr<ProcessGroupMPI> ProcessGroupMPI::createProcessGroupMPI(
    std::vector<int> ranks) {
  // Once initialization
  initMPIOnce();

  MPI_Comm groupComm = MPI_COMM_WORLD;
  int rank = -1;
  int size = -1;

  {
    std::lock_guard<std::mutex> globalLock(pgGlobalMutex_);

    // If no ranks are specified, assume we're creating the root group
    if (!ranks.empty()) {
      MPI_Group worldGroup;
      MPI_Group ranksGroup;
      MPI_CHECK(MPI_Comm_group(MPI_COMM_WORLD, &worldGroup));
      MPI_CHECK(
          MPI_Group_incl(worldGroup, ranks.size(), ranks.data(), &ranksGroup));
      MPI_CHECK(MPI_Comm_create(MPI_COMM_WORLD, ranksGroup, &groupComm));
      MPI_CHECK(MPI_Group_free(&worldGroup));
      MPI_CHECK(MPI_Group_free(&ranksGroup));
    }

    // Fetch rank and world size for this group (MPI_COMM_WORLD or new)
    if (groupComm != MPI_COMM_NULL) {
      MPI_CHECK(MPI_Comm_rank(groupComm, &rank));
      MPI_CHECK(MPI_Comm_size(groupComm, &size));

      if (rank < 0 || size < 0) {
        throw std::runtime_error("Failed to get the world_size / rank");
      }
    }
  }

  // If this process is not part of the group, we don't construct a
  // process group instance. This is in line with the semantics of the
  // other process group types.
  if (groupComm == MPI_COMM_NULL) {
    return c10::intrusive_ptr<ProcessGroupMPI>();
  }

  return c10::make_intrusive<ProcessGroupMPI>(rank, size, groupComm);
}

ProcessGroupMPI::ProcessGroupMPI(int rank, int size, MPI_Comm pgComm)
    : ProcessGroup(rank, size), stop_(false), pgComm_(pgComm) {
  if (pgComm_ == MPI_COMM_NULL) {
    throw std::runtime_error("pgComm_ must not be MPI_COMM_NULL");
  }

  // Start the worker thread accepting MPI calls
  workerThread_ = std::thread(&ProcessGroupMPI::runLoop, this);
}

ProcessGroupMPI::~ProcessGroupMPI() {
  destroy();
}

void ProcessGroupMPI::destroy() {
  std::unique_lock<std::mutex> lock(pgMutex_);
  queueConsumeCV_.wait(lock, [&] { return queue_.empty(); });

  // Queue is empty, signal stop
  stop_ = true;

  // Release lock to allow threads to terminate
  lock.unlock();
  queueProduceCV_.notify_all();

  // Join the single worker thread
  workerThread_.join();
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

    auto& workEntry = std::get<0>(workTuple);
    auto& work = std::get<1>(workTuple);

    lock.unlock();
    queueConsumeCV_.notify_one();

    try {
      workEntry->run(workEntry);
      work->finish();
    } catch (...) {
      work->finish(std::current_exception());
    }

    lock.lock();
  }
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupMPI::enqueue(
    std::unique_ptr<WorkEntry> entry,
    const char * profilingTitle,
    const c10::optional<std::vector<at::Tensor>>& inputTensors) {
  auto work = c10::make_intrusive<WorkMPI>(profilingTitle, inputTensors);
  std::unique_lock<std::mutex> lock(pgMutex_);
  queue_.push_back(std::make_tuple(std::move(entry), work));
  lock.unlock();
  queueProduceCV_.notify_one();
  return work;
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupMPI::broadcast(
    std::vector<at::Tensor>& tensors,
    const BroadcastOptions& opts) {
  checkSingleTensor(tensors);
  std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
      [opts, this](std::unique_ptr<WorkEntry>& entry) {
        auto data = (entry->src)[0];
        c10::DeviceGuard guard(data.device());
        std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
        MPI_CHECK(MPI_Bcast(
            data.data_ptr(),
            data.numel(),
            mpiDatatype.at(data.scalar_type()),
            opts.rootRank,
            pgComm_));
      };
  auto entry = std::unique_ptr<WorkEntry>(
      new WorkEntry(&tensors, nullptr, std::move(runFunc)));
  return enqueue(std::move(entry), "mpi:broadcast", c10::optional<std::vector<at::Tensor>>(tensors));
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupMPI::allreduce(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts) {
  checkSingleTensor(tensors);

  std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
      [opts, this](std::unique_ptr<WorkEntry>& entry) {
        auto data = (entry->src)[0];
        c10::DeviceGuard guard(data.device());
        std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
        MPI_CHECK(MPI_Allreduce(
            MPI_IN_PLACE,
            data.data_ptr(),
            data.numel(),
            mpiDatatype.at(data.scalar_type()),
            mpiOp.at(opts.reduceOp),
            pgComm_));
      };
  auto entry = std::unique_ptr<WorkEntry>(
      new WorkEntry(&tensors, nullptr, std::move(runFunc)));
  return enqueue(std::move(entry), "mpi:all_reduce", c10::optional<std::vector<at::Tensor>>(tensors));
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupMPI::allreduce_coalesced(
    std::vector<at::Tensor>& tensors,
    const AllreduceCoalescedOptions& opts) {
  throw std::runtime_error(
      "allreduce_coalesced is currently not supported with MPI");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupMPI::reduce(
    std::vector<at::Tensor>& tensors,
    const ReduceOptions& opts) {
  checkSingleTensor(tensors);

  std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
      [opts, this](std::unique_ptr<WorkEntry>& entry) {
        auto data = (entry->src)[0];
        auto dataPtr = (entry->src)[0].data_ptr();
        void* sendbuf = (rank_ == opts.rootRank) ? MPI_IN_PLACE : dataPtr;
        void* recvbuf = (rank_ == opts.rootRank) ? dataPtr : nullptr;

        c10::DeviceGuard guard(data.device());
        std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
        MPI_CHECK(MPI_Reduce(
            sendbuf,
            recvbuf,
            data.numel(),
            mpiDatatype.at(data.scalar_type()),
            mpiOp.at(opts.reduceOp),
            opts.rootRank,
            pgComm_));
      };
  auto entry = std::unique_ptr<WorkEntry>(
      new WorkEntry(&tensors, nullptr, std::move(runFunc)));
  return enqueue(std::move(entry), "mpi:reduce", c10::optional<std::vector<at::Tensor>>(tensors));
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupMPI::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllgatherOptions& opts) {
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
      [this](std::unique_ptr<WorkEntry>& entry) {
        auto data = (entry->src)[0];
        std::vector<at::Tensor>& outputDataVec = entry->dst;
        auto flatOutputTensor = newLikeFlat(outputDataVec);

        c10::DeviceGuard guard(data.device());
        std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
        MPI_CHECK(MPI_Allgather(
            data.data_ptr(),
            data.numel(),
            mpiDatatype.at(data.scalar_type()),
            flatOutputTensor.data_ptr(),
            data.numel(),
            mpiDatatype.at(data.scalar_type()),
            pgComm_));

        for (size_t i = 0; i < outputDataVec.size(); ++i) {
          outputDataVec[i].copy_(flatOutputTensor[i]);
        }
      };
  auto entry = std::unique_ptr<WorkEntry>(
      new WorkEntry(&inputTensors, &outputTensors[0], std::move(runFunc)));
  return enqueue(std::move(entry), "mpi:all_gather", c10::optional<std::vector<at::Tensor>>(inputTensors));
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupMPI::allgather_coalesced(
    std::vector<std::vector<at::Tensor>>& /* unused */,
    std::vector<at::Tensor>& /* unused */,
    const AllgatherOptions& /* unused */) {
  throw std::runtime_error(
      "ProcessGroupMPI does not support allgather_coalesced");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupMPI::gather(
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
        auto data = (entry->src)[0];
        void* recvbuf = nullptr;
        at::Tensor flatOutputTensor;

        if (rank_ == opts.rootRank) {
          flatOutputTensor = newLikeFlat(entry->dst);
          recvbuf = flatOutputTensor.data_ptr();
        }

        c10::DeviceGuard guard(data.device());
        std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
        MPI_CHECK(MPI_Gather(
            data.data_ptr(),
            data.numel(),
            mpiDatatype.at(data.scalar_type()),
            recvbuf,
            data.numel(),
            mpiDatatype.at(data.scalar_type()),
            opts.rootRank,
            pgComm_));

        if (rank_ == opts.rootRank) {
          std::vector<at::Tensor>& outputDataVec = entry->dst;
          // copy the flattened output tensors to the outputs
          for (size_t i = 0; i < outputDataVec.size(); ++i) {
            outputDataVec.at(i).copy_(flatOutputTensor[i]);
          }
        }
      };

  if (rank_ == opts.rootRank) {
    auto entry = std::unique_ptr<WorkEntry>(
        new WorkEntry(&inputTensors, &outputTensors[0], std::move(runFunc)));
    return enqueue(std::move(entry), "mpi:gather", c10::optional<std::vector<at::Tensor>>(inputTensors));
  } else {
    auto entry = std::unique_ptr<WorkEntry>(
        new WorkEntry(&inputTensors, nullptr, std::move(runFunc)));
    return enqueue(std::move(entry), "mpi:gather", c10::optional<std::vector<at::Tensor>>(inputTensors));
  }
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupMPI::scatter(
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
      throw std::runtime_error(
          "Scatter: multi-GPU collective is not supported");
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
        auto data = (entry->dst)[0];
        void* sendbuf = nullptr;
        at::Tensor flatInputTensor;

        if (rank_ == opts.rootRank) {
          std::vector<at::Tensor>& inputDataVec = entry->src;
          flatInputTensor = newLikeFlat(inputDataVec);
          sendbuf = flatInputTensor.data_ptr();

          // copy the input tensors to the flatten large send buffer
          for (size_t i = 0; i < inputDataVec.size(); ++i) {
            flatInputTensor[i].copy_(inputDataVec.at(i));
          }
        }

        c10::DeviceGuard guard(data.device());
        std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
        MPI_CHECK(MPI_Scatter(
            sendbuf,
            data.numel(),
            mpiDatatype.at(data.scalar_type()),
            data.data_ptr(),
            data.numel(),
            mpiDatatype.at(data.scalar_type()),
            opts.rootRank,
            pgComm_));
      };

  if (rank_ == opts.rootRank) {
    auto entry = std::unique_ptr<WorkEntry>(
        new WorkEntry(&inputTensors[0], &outputTensors, std::move(runFunc)));
    return enqueue(std::move(entry), "mpi:scatter", inputTensors.size() > 0 ? c10::optional<std::vector<at::Tensor>>(inputTensors[0]) : c10::nullopt);
  } else {
    auto entry = std::unique_ptr<WorkEntry>(
        new WorkEntry(nullptr, &outputTensors, std::move(runFunc)));
    return enqueue(std::move(entry), "mpi:scatter", inputTensors.size() > 0 ? c10::optional<std::vector<at::Tensor>>(inputTensors[0]) : c10::nullopt);
  }
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupMPI::reduce_scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ReduceScatterOptions& opts) {
  throw std::runtime_error("ProcessGroupMPI does not support reduce_scatter");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupMPI::alltoall_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    std::vector<int64_t>& outputSplitSizes,
    std::vector<int64_t>& inputSplitSizes,
    const AllToAllOptions& opts) {
  checkSingleTensorHelper(inputTensor);
  checkSingleTensorHelper(outputTensor);

  if (outputSplitSizes.size() == 0 && inputSplitSizes.size() == 0) {
    // We can use alltoall
    TORCH_CHECK(
        outputTensor.numel() == inputTensor.numel() &&
            outputTensor.type() == inputTensor.type(),
        "Tensors are not equal in size or data type");
    TORCH_CHECK(
        outputTensor.size(0) % size_ == 0,
        "Tensor's dim 0 does not divide equally across group size");

    std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
        [opts, this](std::unique_ptr<WorkEntry>& entry) {
          auto srcdata = (entry->src)[0];
          auto dstdata = (entry->dst)[0];
          c10::DeviceGuard guard(srcdata.device());
          std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
          MPI_CHECK(MPI_Alltoall(
              srcdata.data_ptr(),
              srcdata.numel() / size_,
              mpiDatatype.at(srcdata.scalar_type()),
              dstdata.data_ptr(),
              dstdata.numel() / size_,
              mpiDatatype.at(dstdata.scalar_type()),
              pgComm_));
        };
    std::vector<at::Tensor> inputTensors = {inputTensor};
    std::vector<at::Tensor> outputTensors = {outputTensor};
    auto entry = std::unique_ptr<WorkEntry>(
        new WorkEntry(&inputTensors, &outputTensors, std::move(runFunc)));
    return enqueue(std::move(entry), "mpi:all_to_all", c10::optional<std::vector<at::Tensor>>(inputTensors));
  } else {
    // Need alltoallv
    c10d::checkSplitSizes(inputSplitSizes, inputTensor, size_);
    c10d::checkSplitSizes(outputSplitSizes, outputTensor, size_);
    std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
        [opts, this, inputSplitSizes, outputSplitSizes](
            std::unique_ptr<WorkEntry>& entry) {
          auto srcdata = (entry->src)[0];
          auto dstdata = (entry->dst)[0];
          std::vector<int> send_lengths(size_);
          std::vector<int> recv_lengths(size_);
          std::vector<int> send_offsets(size_);
          std::vector<int> recv_offsets(size_);
          c10d::computeLengthsAndOffsets(
              inputSplitSizes, srcdata, &send_lengths, &send_offsets);
          c10d::computeLengthsAndOffsets(
              outputSplitSizes, dstdata, &recv_lengths, &recv_offsets);
          c10::DeviceGuard guard(srcdata.device());
          std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
          MPI_CHECK(MPI_Alltoallv(
              srcdata.data_ptr(),
              send_lengths.data(),
              send_offsets.data(),
              mpiDatatype.at(srcdata.scalar_type()),
              dstdata.data_ptr(),
              recv_lengths.data(),
              recv_offsets.data(),
              mpiDatatype.at(dstdata.scalar_type()),
              pgComm_));
        };
    std::vector<at::Tensor> inputTensors = {inputTensor};
    std::vector<at::Tensor> outputTensors = {outputTensor};
    auto entry = std::unique_ptr<WorkEntry>(
        new WorkEntry(&inputTensors, &outputTensors, std::move(runFunc)));
    return enqueue(std::move(entry), "mpi:all_to_all", c10::optional<std::vector<at::Tensor>>(inputTensors));
  }
}
c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupMPI::alltoall(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllToAllOptions& opts) {
  TORCH_CHECK(
      inputTensors.size() == size_,
      "Number of input tensors are not equal to group size");
  TORCH_CHECK(
      outputTensors.size() == size_,
      "Number of output tensors are not equal to group size");
  std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
      [opts, this](std::unique_ptr<WorkEntry>& entry) {
        std::vector<int> send_lengths(size_);
        std::vector<int> recv_lengths(size_);
        std::vector<int> send_offsets(size_);
        std::vector<int> recv_offsets(size_);
        auto srcdata = entry->src;
        auto dstdata = entry->dst;
        int64_t src_len =
            c10d::computeLengthsAndOffsets(srcdata, &send_lengths, &send_offsets);
        int64_t dst_len =
            c10d::computeLengthsAndOffsets(dstdata, &recv_lengths, &recv_offsets);
        std::vector<int64_t> send_lengthsL(
            send_lengths.begin(), send_lengths.end());
        std::vector<int64_t> recv_lengthsL(
            recv_lengths.begin(), recv_lengths.end());
        at::Tensor srcFlatData = at::empty({src_len}, srcdata[0].options());
        at::Tensor dstFlatData = at::empty({dst_len}, dstdata[0].options());
        auto srcFlatDataSplits =
            srcFlatData.split_with_sizes(c10::IntArrayRef(send_lengthsL), 0);
        for (int i = 0; i < size_; i++) {
          srcFlatDataSplits[i].copy_(srcdata[i].view({-1}));
        }
        c10::DeviceGuard guard1(srcdata[0].device());
        std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
        MPI_CHECK(MPI_Alltoallv(
            srcFlatData.data_ptr(),
            send_lengths.data(),
            send_offsets.data(),
            mpiDatatype.at(srcdata[0].scalar_type()),
            dstFlatData.data_ptr(),
            recv_lengths.data(),
            recv_offsets.data(),
            mpiDatatype.at(dstdata[0].scalar_type()),
            pgComm_));

        auto dstFlatDataSplits =
            dstFlatData.split_with_sizes(c10::IntArrayRef(recv_lengthsL), 0);
        for (int i = 0; i < size_; i++) {
          dstdata[i].view({-1}).copy_(dstFlatDataSplits[i]);
        }
      };
  auto entry = std::unique_ptr<WorkEntry>(
      new WorkEntry(&inputTensors, &outputTensors, std::move(runFunc)));
  return enqueue(std::move(entry), "mpi:all_to_all", c10::optional<std::vector<at::Tensor>>(inputTensors));
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupMPI::send(
    std::vector<at::Tensor>& tensors,
    int dstRank,
    int tag) {
  checkSingleTensor(tensors);

  auto& tensor = tensors[0];
  MPI_Request request = MPI_REQUEST_NULL;

  {
    c10::DeviceGuard guard(tensor.device());
    std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
    MPI_CHECK(MPI_Isend(
        tensor.data_ptr(),
        tensor.numel(),
        mpiDatatype.at(tensor.scalar_type()),
        dstRank,
        tag,
        pgComm_,
        &request));
  }

  return c10::make_intrusive<AsyncWork>(tensor, request, "mpi:send", c10::optional<std::vector<at::Tensor>>(tensors));
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupMPI::recv(
    std::vector<at::Tensor>& tensors,
    int srcRank,
    int tag) {
  checkSingleTensor(tensors);

  auto& tensor = tensors[0];
  MPI_Request request = MPI_REQUEST_NULL;

  {
    c10::DeviceGuard guard(tensor.device());
    std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
    MPI_CHECK(MPI_Irecv(
        tensor.data_ptr(),
        tensor.numel(),
        mpiDatatype.at(tensor.scalar_type()),
        srcRank,
        tag,
        pgComm_,
        &request));
  }

  return c10::make_intrusive<AsyncWork>(tensor, request, "mpi:recv", c10::optional<std::vector<at::Tensor>>(tensors));
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupMPI::recvAnysource(
    std::vector<at::Tensor>& tensors,
    int tag) {
  checkSingleTensor(tensors);

  auto& tensor = tensors[0];
  MPI_Request request = MPI_REQUEST_NULL;

  {
    c10::DeviceGuard guard(tensor.device());
    std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
    MPI_CHECK(MPI_Irecv(
        tensor.data_ptr(),
        tensor.numel(),
        mpiDatatype.at(tensor.scalar_type()),
        MPI_ANY_SOURCE,
        tag,
        pgComm_,
        &request));
  }

  return c10::make_intrusive<AsyncWork>(tensor, request, "mpi:recvAnySource", c10::optional<std::vector<at::Tensor>>(tensors));
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupMPI::barrier(
    const BarrierOptions& opts) {
  std::function<void(std::unique_ptr<WorkEntry>&)> runFunc =
      [this](std::unique_ptr<WorkEntry>& entry) {
        std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
        MPI_CHECK(MPI_Barrier(pgComm_));
      };
  auto entry = std::unique_ptr<WorkEntry>(
      new WorkEntry(nullptr, nullptr, std::move(runFunc)));
  return enqueue(std::move(entry), "mpi:barrier", c10::nullopt);
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupMPI::allgather_base(
    at::Tensor& /*unused */,
    at::Tensor& /*unused */,
    const AllgatherOptions& /*unused */) {
  throw std::runtime_error(
      "no support for allgather_base in MPI process group");
}

} // namespace c10d
