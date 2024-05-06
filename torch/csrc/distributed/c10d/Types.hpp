#pragma once

#include <torch/csrc/distributed/c10d/Store.hpp>

#include <chrono>
#include <cstdint>

#include <ATen/core/Tensor.h>
#include <ATen/core/ivalue.h>

#include <c10/macros/Macros.h>
#include <c10/util/intrusive_ptr.h>

namespace c10d {

// Base class for supplementary data potentially needed by ReduceOps
struct TORCH_API _SupplementBase : torch::CustomClassHolder {
  ~_SupplementBase() override = default;
};

// Supplementary data specific to NCCL PREMUL_SUM
// The point of use in ProcessGroupNCCL knows how to unpack it.
struct NCCLPreMulSumSupplement : _SupplementBase {
  double double_factor{0.0};
  at::Tensor tensor_factor;
  NCCLPreMulSumSupplement(double f) : double_factor{f} {}
  NCCLPreMulSumSupplement(at::Tensor t) : tensor_factor{std::move(t)} {
    TORCH_CHECK_EQ(tensor_factor.numel(), 1);
  }
};

// Other ReduceOps that need different supplementary data can also
// derive from _SupplementBase.
struct TORCH_API ReduceOp : torch::CustomClassHolder {
  // note(crcrpar): RedOpType could be defined outside of `ReduceOp`
  enum RedOpType : uint8_t {
    SUM = 0,
    AVG = 1,
    PRODUCT = 2,
    MIN = 3,
    MAX = 4,
    BAND = 5, // Bitwise AND
    BOR = 6, // Bitwise OR
    BXOR = 7, // Bitwise XOR
    PREMUL_SUM = 8, // Multiply by a user-supplied constant before summing.
    UNUSED = 9
  };

  ReduceOp() = default;

  ReduceOp(RedOpType op) : op_(op) {
    TORCH_INTERNAL_ASSERT(
        op_ != PREMUL_SUM,
        "Use `torch.distributed._make_nccl_premul_sum` to create an instance of ReduceOp with PREMUL_SUM");
  }

  ReduceOp(
      RedOpType op,
      const c10::intrusive_ptr<_SupplementBase>& optional_supplement) {
    if (optional_supplement) {
      op_ = op;
    } else {
      supplement_ = optional_supplement;
    }
  }

  // The heap resource supplement_, if it exists, is managed by a
  // c10::intrusive_ptr, so constructors and operator= can be simple
  ReduceOp(const ReduceOp& other) = default;
  ReduceOp& operator=(const ReduceOp& other) = default;

  ReduceOp(ReduceOp&& other) = default;
  ReduceOp& operator=(ReduceOp&& other) = default;

  operator RedOpType() const {
    return op_;
  }

  bool operator==(const std::uint8_t other) {
    TORCH_INTERNAL_ASSERT(other < 9, "Invalid other op value");
    return other == op_;
  }

  bool operator==(const ReduceOp::RedOpType other) {
    return *this == static_cast<std::uint8_t>(other);
  }

  // todo(crcrpar): Handle `RedOpType::PREMUL_SUM` with its scaling factor.
  bool operator==(const ReduceOp& other) {
    return *this == other.op_;
  }

  RedOpType op_ = SUM;
  // supplement_ is "type-erased" storage for optional supplementary
  // data the op might need.
  // The point of use will know the derived type supplement_ really is,
  // and downcast its pointer to extract the data as the needed type(s).
  // Right now, only PREMUL_SUM needs supplementary data, but the same
  // mechanism could extend to support other nontrivial reduce ops with
  // different supplementary payloads.
  c10::intrusive_ptr<_SupplementBase> supplement_;
};

template <typename T>
ReduceOp makeNCCLPreMulSum(const T& factor) {
  ReduceOp rop;
  rop.op_ = ReduceOp::PREMUL_SUM;
  rop.supplement_ = c10::make_intrusive<NCCLPreMulSumSupplement>(factor);
  return rop;
}

constexpr auto kUnsetTimeout = std::chrono::milliseconds(-1);

struct BroadcastOptions {
  int64_t rootRank = 0;
  int64_t rootTensor = 0;
  std::chrono::milliseconds timeout = kUnsetTimeout;
  bool asyncOp = true;
};

struct AllreduceOptions {
  ReduceOp reduceOp = ReduceOp::SUM;
  std::chrono::milliseconds timeout = kUnsetTimeout;
  c10::optional<at::Tensor> sparseIndices = c10::nullopt;
};

struct AllreduceCoalescedOptions : AllreduceOptions {};

struct ReduceOptions {
  ReduceOp reduceOp = ReduceOp::SUM;
  int64_t rootRank = 0;
  int64_t rootTensor = 0;
  std::chrono::milliseconds timeout = kUnsetTimeout;
};

struct AllgatherOptions {
  std::chrono::milliseconds timeout = kUnsetTimeout;
  bool asyncOp = true;
};

struct GatherOptions {
  int64_t rootRank = 0;
  std::chrono::milliseconds timeout = kUnsetTimeout;
};

struct ScatterOptions {
  int64_t rootRank = 0;
  std::chrono::milliseconds timeout = kUnsetTimeout;
  bool asyncOp = true;
};

struct ReduceScatterOptions {
  ReduceOp reduceOp = ReduceOp::SUM;
  std::chrono::milliseconds timeout = kUnsetTimeout;
  bool asyncOp = true;
};

struct AllToAllOptions {
  std::chrono::milliseconds timeout = kUnsetTimeout;
};

struct BarrierOptions {
  std::vector<int64_t> device_ids;
  std::chrono::milliseconds timeout = kUnsetTimeout;
  c10::optional<at::Device> device;
};

struct DistributedBackendOptions {
  c10::intrusive_ptr<::c10d::Store> store;
  int group_rank;
  int group_size;
  std::chrono::duration<float> timeout;
  std::string group_id;
  std::vector<int64_t> global_ranks_in_group;
};

} // namespace c10d
