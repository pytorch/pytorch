#pragma once

// This file contains functions for checking rollout knobs to enable staged
// roll out of specific code functionality.

#include <memory>

#include <c10/util/string_view.h>

namespace caffe2 {

/**
 * Check an arbitrary knob by name.
 */
bool CheckKnob(c10::string_view name);

/*
 * The following are functions for checking specific known knob values.
 *
 * These APIs are more efficient than checking by name.
 */

// An example knob, just for use in unit tests.
bool CheckKnobExampleKnob();

} // namespace caffe2
