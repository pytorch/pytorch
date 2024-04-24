// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This is a very basic knob implementation that purely uses command line flags.
// This can be replaced with a more sophisticated implementation for use in
// other production environments.

#include <map>

#include <c10/util/string_view.h>
#include <c10/util/Flags.h>

#include "caffe2/utils/knobs.h"

namespace caffe2 {

namespace detail {
// Get the map of knob names to pointers to their command-line controlled
// boolean value.
std::map<c10::string_view, bool*>& getRegisteredKnobs() {
  // It's safe to store the keys as string_view, since DEFINE_KNOB() ensures
  // that these views always point to string literals.
  static std::map<c10::string_view, bool*> registeredKnobs;
  return registeredKnobs;
}
} // namespace detail

bool CheckKnob(c10::string_view name) {
  const auto& knobs = detail::getRegisteredKnobs();
  auto iter = knobs.find(name);
  if (iter == knobs.end()) {
      throw std::invalid_argument(
          "attempted to check unknown knob \"" + std::string(name) + "\"");
  }
  return *iter->second;
}

namespace {
class RegisterKnob {
 public:
  RegisterKnob(c10::string_view name, bool* cmdlineFlag) {
    auto ret = caffe2::detail::getRegisteredKnobs().emplace(name, cmdlineFlag);
    if (!ret.second) {
      throw std::runtime_error("duplicate knob name: " + std::string(name));
    }
  }
};
} // namespace
} // namespace caffe2

/**
 * Define a knob.
 *
 * This will define a --caffe2_knob_<name> command line flag to control the
 * knob.
 *
 * The knob can be checked in code by calling CheckKnob(name)
 * or CheckKnob<check_fn_name>()
 */
#define DEFINE_KNOB(name, check_fn_name, default_value, docstring) \
  C10_DEFINE_bool(caffe2_knob_##name, default_value, docstring);   \
  namespace caffe2 {                                               \
  bool CheckKnob##check_fn_name() {                                \
    return FLAGS_caffe2_knob_##name;                               \
  }                                                                \
  }                                                                \
  static caffe2::RegisterKnob _knob_##name(#name, &FLAGS_caffe2_knob_##name)

/*
 * Definitions of well-known knobs.
 */

DEFINE_KNOB(
    example_knob,
    ExampleKnob,
    false,
    "An example knob, mainly intended for use in unit tests");
