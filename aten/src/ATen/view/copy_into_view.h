#pragma once

namespace at { class Tensor; }

namespace at::view {

// Copies from the source tensor into the view tensor.
//
// The purpose of this function is to be the inverse of the
// materialize function. This is intended to be used when exiting an
// operator with a mutated composite view input and thus should
// reverse the operations used to materialize it.
//
// Requires: tensor.key_set.has(c10::DispatchKey::CompositeView)
auto copy_into_view(Tensor& view_tensor, Tensor const& source) -> void;

} // namespace at::view
