#pragma once

#include <memory>

#include <c10/util/string_view.h>

namespace caffe2 {

/**
 * Patch the value of a knob during a unit test.
 *
 * This forces the knob to the specified value for as long as the KnobPatcher
 * object exists.  When the KnobPatcher object is destroyed the knob will revert
 * to its previous value.
 */
class KnobPatcher {
 public:
  KnobPatcher(c10::string_view name, bool value);
  ~KnobPatcher();

  KnobPatcher(KnobPatcher&&) noexcept;
  KnobPatcher& operator=(KnobPatcher&&) noexcept;
  KnobPatcher(const KnobPatcher&) = delete;
  KnobPatcher& operator=(const KnobPatcher&) = delete;

 private:
  class PatchState;

  std::unique_ptr<PatchState> state_;
};

} // namespace caffe2
