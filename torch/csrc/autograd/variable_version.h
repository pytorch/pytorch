#pragma once

#include <atomic>
#include <cstdint>
#include <memory>

// Every Variable has a version counter. Version counters are incremented
// whenever the data or shape of a tensor changes through Variable operations.
// These are typicallly in-place operations. Version counters are used to
// detect modifications to saved variables which would result in incorrect
// gradient calculations. Version counters may be shared between Variables:
//
// 1. A view shares the version counter of the base Variable,
// 2. Detached variables share the version counter of the source,
// 3. Unpacked saved variables share the version counter of the source.

namespace torch { namespace autograd {

struct VariableVersion {
 public:
  // NOTE: As of C++11 and 14, default-constructing a std::atomic variable
  // leaves it in a persistently undefined state. See
  // https://cplusplus.github.io/LWG/issue2334.
  VariableVersion(uint32_t version = 0)
      : version_block_(std::make_shared<std::atomic<uint32_t>>(version)) {}

  void bump() noexcept {
    version_block_->fetch_add(1);
  }

  uint32_t current_version() const noexcept {
    return version_block_->load();
  }

 private:
  std::shared_ptr<std::atomic<uint32_t>> version_block_;
};
}} // namespace torch::autograd
