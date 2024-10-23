#pragma once
#include <torch/csrc/profiler/unwind/unwind_error.h>
#include <algorithm>
#include <memory>
#include <optional>
#include <vector>

namespace torch::unwind {
template <typename T>
struct RangeTable {
  RangeTable() {
    // guarentee that lower_bound[-1] is always valid
    addresses_.push_back(0);
    payloads_.emplace_back(std::nullopt);
  }
  void add(uint64_t address, std::optional<T> payload, bool sorted) {
    if (addresses_.back() > address) {
      UNWIND_CHECK(!sorted, "expected addresses to be sorted");
      sorted_ = false;
    }
    addresses_.push_back(address);
    payloads_.emplace_back(std::move(payload));
  }
  std::optional<T> find(uint64_t address) {
    maybeSort();
    auto it = std::upper_bound(addresses_.begin(), addresses_.end(), address);
    return payloads_.at(it - addresses_.begin() - 1);
  }
  void dump() {
    for (size_t i = 0; i < addresses_.size(); i++) {
      fmt::print("{} {:x}: {}\n", i, addresses_[i], payloads_[i] ? "" : "END");
    }
  }
  size_t size() const {
    return addresses_.size();
  }
  uint64_t back() {
    maybeSort();
    return addresses_.back();
  }

 private:
  void maybeSort() {
    if (sorted_) {
      return;
    }
    std::vector<uint64_t> indices;
    indices.reserve(addresses_.size());
    for (size_t i = 0; i < addresses_.size(); i++) {
      indices.push_back(i);
    }
    std::sort(indices.begin(), indices.end(), [&](uint64_t a, uint64_t b) {
      return addresses_[a] < addresses_[b] ||
          (addresses_[a] == addresses_[b] &&
           bool(payloads_[a]) < bool(payloads_[b]));
    });
    std::vector<uint64_t> addresses;
    std::vector<std::optional<T>> payloads;
    addresses.reserve(addresses_.size());
    payloads.reserve(addresses_.size());
    for (auto i : indices) {
      addresses.push_back(addresses_[i]);
      payloads.push_back(payloads_[i]);
    }
    addresses_ = std::move(addresses);
    payloads_ = std::move(payloads);
    sorted_ = true;
  }
  bool sorted_ = true;
  std::vector<uint64_t> addresses_;
  std::vector<std::optional<T>> payloads_;
};
} // namespace torch::unwind
