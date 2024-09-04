#pragma once

#include <ATen/ATen.h>
#include <chrono>
#include <mutex>
#include <vector>

constexpr auto kNoTimeout = std::chrono::milliseconds(0);

namespace c10d {

constexpr const char* const kSeqNumStoreKey = "SEQ_NUM_STORE_KEY";

enum class OpType : std::uint8_t {
  BROADCAST = 0,
  ALLREDUCE = 1,
  ALLREDUCE_COALESCED = 2,
  REDUCE = 3,
  ALLGATHER = 4,
  _ALLGATHER_BASE = 5,
  ALLGATHER_COALESCED = 6,
  GATHER = 7,
  SCATTER = 8,
  REDUCE_SCATTER = 9,
  ALLTOALL_BASE = 10,
  ALLTOALL = 11,
  SEND = 12,
  RECV = 13,
  RECVANYSOURCE = 14,
  BARRIER = 15,
  _REDUCE_SCATTER_BASE = 16,
  COALESCED = 17,
  _ALLREDUCE_SPARSE = 18,
  UNKNOWN = 100,
};

// Converts OpType to human readable string.
TORCH_API std::string opTypeToString(OpType opType);

// Whether or not an OP is an p2p op (SEND, RECV, RECVANYSOURCE)
TORCH_API bool isP2POp(OpType opType, bool batchP2P = false);

// Please do not use Work API, it is going away, to be
// replaced by ivalue::Future.
// Python binding for this class might change, please do not assume
// this will be bound using pybind.
class TORCH_API Work : public torch::CustomClassHolder {
 public:
  Work(
      int rank = -1,
      OpType opType = OpType::UNKNOWN,
      const char* profilingTitle = nullptr,
      const std::optional<std::vector<at::Tensor>>& inputTensors =
          std::nullopt);

  ~Work() override;

  // Checks if request has completed. Non-blocking operation.
  virtual bool isCompleted();

  // Returns if the work completed successfully.
  // If false, the exception function can be called to get details.
  virtual bool isSuccess() const;

  // Returns exception if isSuccess() returned false.
  virtual std::exception_ptr exception() const;

  // Returns source rank if this objects represents a recv-from-any.
  virtual int sourceRank() const;

  // Returns result tensors, if applicable.
  // If work is not supposed to have result, we return empty list.
  virtual std::vector<at::Tensor> result();

  // Ensures that operations on the output tensors that are invoked
  // after this function returns are correctly sequenced after the
  // asynchronous completion of this work.
  //
  // For CUDA tensors, it inserts stream synchronization such that
  // the streams of the caller wait for completion of the
  // asynchronous operations on the destination tensors.
  //
  // For CPU tensors, it is currently a nop.
  //
  // This function should only be used if the caller polls for
  // completion through the `isCompleted` function, it has returned
  // true, and the `isSuccess` function also has returned true.
  //
  virtual void synchronize();

  // Waits until request completes. Blocking operation.
  // Throws if the work completed with an exception.
  // Returns false if the work is aborted.
  // Otherwise, it always returns true, indicating the work is completed.
  //
  // Functionally equivalent to:
  //
  //   while (!isCompleted()) { /* nop */ }
  //   auto success = isSuccess();
  //   if (!success) { std::rethrow_exception(exception()); }
  //   return success;
  //
  virtual bool wait(std::chrono::milliseconds timeout = kNoTimeout);

  virtual void abort();

  // Returns a Future object that will be associated with the completion of
  // work. Only NCCL backend is currently supported.
  virtual c10::intrusive_ptr<c10::ivalue::Future> getFuture();

  virtual float getDuration() const;

  virtual uint64_t getSequencenumber() const;

  OpType retrieveOpType() const;

  static c10::intrusive_ptr<Work> create_from_future(
      const c10::intrusive_ptr<c10::ivalue::Future>&);

 protected:
  // Completes the work object and optionally sets the exception in a
  // thread-safe manner. Notifies all waiting condition variables as well.
  void finish(std::exception_ptr exception = nullptr);

  // Similar to finish, but throws an exception if one is already set or
  // provided by the user.
  void finishAndThrow(std::exception_ptr exception);

  mutable std::timed_mutex mutex_;
  std::condition_variable_any cv_;
  bool completed_ = false;
  std::exception_ptr exception_;

  // Current rank of the node.
  const int rank_;

  // Operation type that this work object refers to.
  OpType opType_;

  // When profiling, the callback to record end of operation event. This
  // callback needs to be called when collective operation is complete.
  std::function<void()> recordFunctionEndCallback_;
};

struct TORCH_API WorkInfo {
  WorkInfo(
      const OpType& opType,
      const uint64_t seq,
      const std::chrono::time_point<std::chrono::system_clock>& timeStarted,
      const std::chrono::time_point<std::chrono::system_clock>& timeFinished,
      const std::chrono::duration<float>& activeDuration)
      : opType(opType),
        seq(seq),
        timeStarted(timeStarted),
        timeFinished(timeFinished),
        activeDuration(activeDuration) {}

  OpType opType;
  uint64_t seq;
  std::chrono::time_point<std::chrono::system_clock> timeStarted;
  std::chrono::time_point<std::chrono::system_clock> timeFinished;
  std::chrono::duration<float> activeDuration;
};

} // namespace c10d
