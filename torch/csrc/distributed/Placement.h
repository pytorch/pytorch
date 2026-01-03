#pragma once
/**
 * The implementations in this file are coupled with
 * torch/distributed/tensor/placement_types.py.
 */

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace torch::distributed {

class Placement {
 public:
  Placement() = default;
  virtual ~Placement() = default;

  Placement(const Placement&) = default;
  Placement& operator=(const Placement&) = default;
  Placement(Placement&&) noexcept = default;
  Placement& operator=(Placement&&) noexcept = default;

  virtual bool is_shard(std::optional<std::int64_t> dim) const {
    return false;
  }

  virtual bool is_replicate() const {
    return false;
  }

  virtual bool is_partial(
      std::optional<std::string_view> reduce_op = std::nullopt) const {
    return false;
  }
};

class Shard : public Placement {
 public:
  std::int64_t dim;
  explicit Shard(std::int64_t dim) : dim(dim) {}

  bool is_shard(std::optional<std::int64_t> dim_) const override {
    if (typeid(*this) != typeid(Shard)) {
      return false;
    }
    return !dim_.has_value() || *dim_ == dim;
  }

  bool operator==(const Shard& rhs) const {
    return dim == rhs.dim;
  }

  bool operator!=(const Shard& rhs) const {
    return !operator==(rhs);
  }
};

class StridedShard : public Placement {
 public:
  std::int64_t dim;
  std::int64_t split_factor;
  explicit StridedShard(std::int64_t dim, std::int64_t split_factor_)
      : dim(dim), split_factor(split_factor_) {}

  bool operator==(const StridedShard& rhs) const {
    return dim == rhs.dim && split_factor == rhs.split_factor;
  }

  bool operator!=(const StridedShard& rhs) const {
    return !operator==(rhs);
  }
};

class Replicate : public Placement {
 public:
  bool is_replicate() const override {
    return true;
  }

  bool operator==(const Replicate& rhs) const {
    return true;
  }

  bool operator!=(const Replicate& rhs) const {
    return false;
  }
};

class Partial : public Placement {
 public:
  std::string reduce_op;

  Partial() : Partial("sum") {}

  explicit Partial(std::optional<std::string> reduce_op_)
      : reduce_op(
            reduce_op_.has_value() ? std::move(*reduce_op_)
                                   : std::string("sum")) {}

  bool is_partial(
      std::optional<std::string_view> op = std::nullopt) const override {
    return !op.has_value() || *op == reduce_op;
  }

  bool operator==(const Partial& rhs) const {
    return reduce_op == rhs.reduce_op;
  }

  bool operator!=(const Partial& rhs) const {
    return !operator==(rhs);
  }
};

} // namespace torch::distributed
