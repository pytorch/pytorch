#pragma once

#include <ATen/DefaultTensorOptions.h>
#include <ATen/TensorOptions.h>

namespace at {

/// RAII guard that stores the current default options upon construction, sets
/// the current default options to the ones given to its constructor, and
/// finally resets the options back to the original ones in the destructor.
struct OptionsGuard {
 public:
  /// Stores the current default options and sets them to the given ones.
  explicit OptionsGuard(const TensorOptions& options)
      : original_(DefaultTensorOptions::exchange(options)) {}

  /// Restores the original default options.
  ~OptionsGuard() {
    DefaultTensorOptions::assign(original_);
  }

  /// Returns the original options that were in place at the time of
  /// construction of this object.
  const TensorOptions& original() {
    return original_;
  }

 private:
  /// The original options that were in place at the time of construction of
  /// this object.
  TensorOptions original_;
};

} // namespace at
