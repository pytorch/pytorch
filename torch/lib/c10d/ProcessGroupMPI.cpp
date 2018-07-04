#include "ProcessGroupMPI.hpp"

#include <mpi-ext.h> // Needed for CUDA-aware check
#include <map>

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
void checkSingleTensor(const std::vector<at::Tensor>& tensors) {
  if (tensors.size() != 1) {
    throw std::runtime_error(
        "MPI process group only supports a single "
        "tensor op");
  }
  if (!tensors[0].is_contiguous()) {
    throw std::runtime_error("input tensor has to be contiguous");
  }
  if (tensors[0].is_cuda() && !cudaAwareMpiCheck()) {
    throw std::runtime_error(
        "CUDA tensor detected and the MPI used doesn't "
        "have CUDA-aware MPI support");
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

} // namespace c10d
