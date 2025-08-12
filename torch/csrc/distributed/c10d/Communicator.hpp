
#pragma once

#include <vector>

#include <ATen/ATen.h>
#include <c10/core/Allocator.h>
#include <c10/macros/Macros.h>

#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>


constexpr auto kCommDefaultTimeout =
    std::chrono::milliseconds(30 * 60 * 1000);

namespace c10d {

class TORCH_API Communicator : public torch::CustomClassHolder {
 public:
  // Backend Options is a base struct that defines the basic options
  // when constructing a Backend. Each Backend subclass should
  // extend this struct and define its options if it wants to provide more
  // config options (beyond basic ones defined here) to end user.
  struct TORCH_API Options : torch::CustomClassHolder {
    explicit Options(
        std::chrono::milliseconds timeout = kCommDefaultTimeout)
        : timeout(timeout) {}
    ~Options() override = default;

    std::chrono::milliseconds timeout;
    std::string group_name;
  };

  explicit Communicator(int rank, int size);
  ~Communicator() override = 0;

  int getRank() const {
    return rank_;
  }

  int getSize() const {
    return size_;
  }

  // Subclasses must override this method to return the backend name
  virtual const std::string getCommunicatorName() const {
    TORCH_INTERNAL_ASSERT(false, "getCommunicatorName is not implemented.");
  }

  c10::intrusive_ptr<c10d::Work> allreduce(
      std::vector<at::Tensor>& tensors,
      c10d::ReduceOp reduceOp = c10d::ReduceOp::SUM,
      bool asyncOp = false,
      std::chrono::milliseconds timeout = kCommDefaultTimeout,
      std::optional<at::Tensor> sparseIndices = std::nullopt);

  c10::intrusive_ptr<c10d::Work> allreduce_dispatched(
      std::vector<at::Tensor>& tensors,
      c10d::ReduceOp reduceOp = c10d::ReduceOp::SUM,
      bool asyncOp = false,
      std::chrono::milliseconds timeout = kCommDefaultTimeout,
      std::optional<at::Tensor> sparseIndices = std::nullopt);

  virtual c10::intrusive_ptr<c10d::Work> allreduceImpl(
      std::vector<at::Tensor>& tensors,
      c10d::ReduceOp reduceOp = c10d::ReduceOp::SUM,
      bool asyncOp = false,
      std::chrono::milliseconds timeout = kCommDefaultTimeout,
      std::optional<at::Tensor> sparseIndices = std::nullopt) = 0;

  protected:
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
    const int rank_;
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
    const int size_;
    std::string pg_uid_;
    std::string pg_desc_;
    size_t local_id_;
    uint64_t seqCollective_{0};
    uint64_t seqP2P_{0};
    uint64_t op_id_{0};

    std::optional<at::Device> bound_device_id_;
};

} // namespace c10d
