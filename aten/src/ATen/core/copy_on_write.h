#pragma once

namespace at {

class TensorBase;

// Materializes any copy on write tensors that are getting written to.
auto simulate_materialize_copy_on_write(TensorBase const& tensor) -> void;

} // namespace at
