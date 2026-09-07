#include <ATen/core/dispatch/DispatchKeyExtractor.h>
#include <c10/util/irange.h>

#include <algorithm>
#include <sstream>
#include <string>

namespace c10 {

void DispatchKeyExtractor::setOperatorHasFallthroughForKey(DispatchKey k, bool has_fallthrough) {
  // (1) update nonFallthroughKeys_
  if (has_fallthrough) {
    nonFallthroughKeys_ = nonFallthroughKeys_.remove(k);
  } else {
    nonFallthroughKeys_ = nonFallthroughKeys_.add(k);
  }
  // (2) update nonFallthroughKeysPerBackend_
  if (isPerBackendFunctionalityKey(toFunctionalityKey(k))) {
    // This is a per-backend functionality key.
    // We need to figure out what the current backend is,
    // and only update the bitset for that backend.
    // subtracting 1 because the first backend should have index 0 (CPU),
    // But the enum starts with BackendComponent::InvalidBit.
    auto backend_idx = static_cast<uint8_t>(toBackendComponent(k)) - 1;
    TORCH_INTERNAL_ASSERT(backend_idx >= 0 && static_cast<uint8_t>(backend_idx) < nonFallthroughKeysPerBackend_.size());
    if (has_fallthrough) {
      nonFallthroughKeysPerBackend_[backend_idx] = nonFallthroughKeysPerBackend_[backend_idx].remove(k);
    } else {
      nonFallthroughKeysPerBackend_[backend_idx] = nonFallthroughKeysPerBackend_[backend_idx].add(k);
    }

    // Set requiresBitsetPerBackend_ accordingly
    for (const auto i : c10::irange(nonFallthroughKeysPerBackend_.size() - 1)) {
      if (nonFallthroughKeysPerBackend_[i] != nonFallthroughKeysPerBackend_[i+1]) {
        requiresBitsetPerBackend_ = true;
        return;
      }
    }
    requiresBitsetPerBackend_ = false;
    return;
  } else {
    // Otherwise, if a fallthrough is set for a functionality that isn't per backend,
    // Then we update the fallthrough bitset for EVERY backend.
    // TODO: we could probably optimize this by only lazily updating these values
    // the first time that we see requiresBitsetPerBackend_ = true
    // (which should almost never happen)
    if (has_fallthrough) {
      for (auto& nonFallthroughKeysPerBackend_elem : nonFallthroughKeysPerBackend_) {
        nonFallthroughKeysPerBackend_elem = nonFallthroughKeysPerBackend_elem.remove(k);
      }
    } else {
      for (auto& nonFallthroughKeysPerBackend_elem : nonFallthroughKeysPerBackend_) {
        nonFallthroughKeysPerBackend_elem = nonFallthroughKeysPerBackend_elem.add(k);
      }
    }
  }
}

std::string DispatchKeyExtractor::dumpState() const {
  std::string bits = dispatch_arg_indices_reverse_.to_string();
  std::reverse(bits.begin(), bits.end());
  std::ostringstream oss;
  oss << bits << ' ' << nonFallthroughKeys_ << '\n';
  return std::move(oss).str();
}

void DispatchKeyExtractor::checkInvariants(const FunctionSchema& schema) const {
  TORCH_INTERNAL_ASSERT(makeBitsetForDispatchArgs(schema) == dispatch_arg_indices_reverse_);
}

} // namespace c10
