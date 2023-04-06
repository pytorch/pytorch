#pragma once

#include <cstdint>

#include <c10/core/impl/cow/shadow_storage.h>
#include <c10/macros/Macros.h>
#include <c10/util/Optional.h>

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
  static C10_API auto get_generation(Storage const& tensor)
      -> c10::optional<cow::ShadowStorage::Generation>;
  // Gets the shadow storage instance from the tensor.
  static C10_API auto get_shadow_storage(TensorImpl const& tensor)
      -> ShadowStorage const*;
};

} // namespace c10::impl::cow
