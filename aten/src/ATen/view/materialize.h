#pragma once

namespace at { class Tensor; }

namespace at::view {

// Materializes all layers of composite view tensors.
//
// Requires: tensor.key_set.has(c10::DispatchKey::CompositeView)
auto materialize(Tensor const& tensor) -> Tensor;

} // namespace at::view
