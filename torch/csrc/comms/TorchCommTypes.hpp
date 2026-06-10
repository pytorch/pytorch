// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <ATen/ATen.h>
#include <c10/core/Device.h>
#include <c10/util/intrusive_ptr.h>
#include <chrono>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

namespace torch::comms {

// Forward declaration
class TorchComm;
class TorchWork;

using PreMulSumFactorT = std::variant<at::Tensor, double>;

class ReduceOp {
 public:
  // ReduceOp enum for reduction operations
  enum class RedOpType {
    SUM = 0,
    PRODUCT,
    MIN,
    MAX,
    BAND,
    BOR,
    BXOR,
    PREMUL_SUM,
    AVG,
  };

  /* implicit */ ReduceOp(RedOpType type) : type_(type), factor_(std::nullopt) {
    TORCH_INTERNAL_ASSERT(
        type != RedOpType::PREMUL_SUM, "PREMUL_SUM needs a factor");
  }

  static ReduceOp make_nccl_premul_sum(const PreMulSumFactorT& factor) {
    return ReduceOp(RedOpType::PREMUL_SUM, factor);
  }

  // The const static ReduceOp objects are for python bindings, for torchcomms
  // internal *static* function, it is better to use the RedOpType enum directly
  // to avoid static initialization order fiasco.
  // @lint-ignore-every CLANGTIDY NonPodStaticDeclaration
  static const ReduceOp SUM;
  static const ReduceOp PRODUCT;
  static const ReduceOp MIN;
  static const ReduceOp MAX;
  static const ReduceOp BAND;
  static const ReduceOp BOR;
  static const ReduceOp BXOR;
  static const ReduceOp AVG;

  // Copy/move constructors are allowed for creating new ReduceOp instances,
  // but assignment operators are deleted to prevent accidental modification
  // of existing ReduceOp objects (particularly the static const instances).
  // This ensures ReduceOp objects remain immutable after construction.
  ReduceOp(const ReduceOp& other) = default;
  ReduceOp& operator=(const ReduceOp& other) = delete;

  ReduceOp(ReduceOp&& other) noexcept = default;
  ReduceOp& operator=(ReduceOp&& other) noexcept = delete;
  ~ReduceOp() = default;

  operator RedOpType() const {
    return type_;
  }

  RedOpType type() const {
    return type_;
  }

  const std::optional<const PreMulSumFactorT>& factor() const {
    return factor_;
  }

 private:
  ReduceOp() = default;
  ReduceOp(RedOpType type, const PreMulSumFactorT& factor)
      : type_(type), factor_(factor) {}

  RedOpType type_{RedOpType::SUM};
  std::optional<const PreMulSumFactorT> factor_{std::nullopt};
};

// Default timeout for collective operations.  It can be overridden during
// TorchComm creation or during each collective operation.
constexpr std::chrono::milliseconds kDefaultTimeout = std::chrono::seconds(600);
constexpr std::chrono::milliseconds kNoTimeout = std::chrono::milliseconds(0);

// InitHandle type for fault tolerance API.
// An InitHandle encodes information required
// by the backend to complete the initialization process via reconfigure().
using InitHandle = std::string;

/**
 * Options for the reconfigure() fault tolerance API.
 *
 * The reconfigure call initializes the communicator with a user-provided set
 * of peers. After a successful reconfigure call, the communicator is fully
 * initialized and collective operations are permitted.
 */
struct ReconfigureOptions {
  /**
   * Uniquely identifies this instance of the communicator.
   * The uuid must not have been used previously on this communicator.
   * Every time a communicator is initialized, pass in a new UUID to identify
   * this new instance of the communicator.
   */
  int64_t uuid;

  /**
   * Represents the members that will participate in this communicator.
   * Each URL/handle represents a rank in the communicator.
   *
   * Two regimes are supported:
   * - vector<InitHandle>: Guarantees that assigned ranks correspond to
   *   position of handle in the vector (ordered assignment)
   * - unordered_set<InitHandle>: The backend will determine the rank
   *   assignment based on internal considerations (no external rank order
   *   is guaranteed)
   */
  std::variant<std::unordered_set<InitHandle>, std::vector<InitHandle>> handles;

  /**
   * How long to allow reconfiguration to take before failing with an error.
   * If nullopt, uses the backend's default timeout.
   */
  std::optional<std::chrono::milliseconds> timeout{std::nullopt};

  /**
   * Additional configuration key-value pairs, implementation-specific.
   * These hints can be used to pass backend-specific options.
   */
  std::unordered_map<std::string, std::string> hints;
};

} // namespace torch::comms
