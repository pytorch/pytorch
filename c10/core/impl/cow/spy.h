#pragma once

#include <cstdint>

#include <c10/core/impl/cow/simulator.h>
#include <c10/macros/Macros.h>

namespace c10 {
struct Storage;
struct TensorImpl;
} // namespace c10

namespace c10::impl::cow {

// Allows for introspection into TensorImpl and StorageImpl's
// copy-on-write state.
class Spy {
 public:
  // Gets the generation number from the storage.
  static C10_API auto get_generation(Storage const& tensor) -> std::uint64_t;
  // Gets the copy-on-write simulator instance from the tensor.
  static C10_API auto get_simulator(TensorImpl const& tensor)
      -> Simulator const*;
};

} // namespace c10::impl::cow
