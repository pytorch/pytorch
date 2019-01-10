/**
 * @file flags.h
 * @brief Commandline flags support for Caffe2.
 *
 * This is a portable commandline flags tool for caffe2, so we can optionally
 * choose to use gflags or a lightweighted custom implementation if gflags is
 * not possible on a certain platform. If you have gflags installed, set the
 * macro CAFFE2_USE_GFLAGS will seamlessly route everything to gflags.
 *
 * To define a flag foo of type bool default to true, do the following in the
 * *global* namespace:
 *     CAFFE2_DEFINE_bool(foo, true, "An example.");
 *
 * To use it in another .cc file, you can use CAFFE2_DECLARE_* as follows:
 *     CAFFE2_DECLARE_bool(foo);
 *
 * In both cases, you can then access the flag via caffe2::FLAGS_foo.
 */

#ifndef CAFFE2_CORE_FLAGS_H_
#define CAFFE2_CORE_FLAGS_H_

#include "caffe2/core/registry.h"

namespace caffe2 {
/**
 * Sets the usage message when a commandline tool is called with "--help".
 */
void SetUsageMessage(const string& str);

/**
 * Returns the usage message for the commandline tool set by SetUsageMessage.
 */
const char* UsageMessage();

/**
 * Parses the commandline flags.
 *
 * This command parses all the commandline arguments passed in via pargc
 * and argv. Once it is finished, partc and argv will contain the remaining
 * commandline args that caffe2 does not deal with. Note that following
 * convention, argv[0] contains the binary name and is not parsed.
 */
bool ParseCaffeCommandLineFlags(int* pargc, char*** pargv);
/**
 * Checks if the commandline flags has already been passed.
 */
bool CommandLineFlagsHasBeenParsed();

}  // namespace caffe2


////////////////////////////////////////////////////////////////////////////////
// Below are gflags and non-gflags specific implementations.
////////////////////////////////////////////////////////////////////////////////

#ifdef CAFFE2_USE_GFLAGS

#include <gflags/gflags.h>

// gflags before 2.0 uses namespace google and after 2.1 uses namespace gflags.
// Using GFLAGS_GFLAGS_H_ to capture this change.
#ifndef GFLAGS_GFLAGS_H_
namespace gflags = google;
#endif  // GFLAGS_GFLAGS_H_

#define CAFFE2_GFLAGS_DEF_WRAPPER(type, name, default_value, help_str)         \
  DEFINE_##type(name, default_value, help_str);                                \
  namespace caffe2 {                                                           \
    using ::FLAGS_##name;                                                      \
  }

#define CAFFE2_DEFINE_int(name, default_value, help_str)                       \
  CAFFE2_GFLAGS_DEF_WRAPPER(int32, name, default_value, help_str)
#define CAFFE2_DEFINE_int64(name, default_value, help_str)                     \
  CAFFE2_GFLAGS_DEF_WRAPPER(int64, name, default_value, help_str)              
#define CAFFE2_DEFINE_double(name, default_value, help_str)                    \
  CAFFE2_GFLAGS_DEF_WRAPPER(double, name, default_value, help_str)
#define CAFFE2_DEFINE_bool(name, default_value, help_str)                      \
  CAFFE2_GFLAGS_DEF_WRAPPER(bool, name, default_value, help_str)
#define CAFFE2_DEFINE_string(name, default_value, help_str) \
  CAFFE2_GFLAGS_DEF_WRAPPER(string, name, default_value, help_str)

// DECLARE_typed_var should be used in header files and in the global namespace.
#define CAFFE2_GFLAGS_DECLARE_WRAPPER(type, name)                             \
  DECLARE_##type(name);                                                       \
  namespace caffe2 {                                                          \
    using ::FLAGS_##name;                                                     \
  }  // namespace caffe2

#define CAFFE2_DECLARE_int(name) CAFFE2_GFLAGS_DECLARE_WRAPPER(int32, name)
#define CAFFE2_DECLARE_int64(name) CAFFE2_GFLAGS_DECLARE_WRAPPER(int64, name)
#define CAFFE2_DECLARE_double(name) CAFFE2_GFLAGS_DECLARE_WRAPPER(double, name)
#define CAFFE2_DECLARE_bool(name) CAFFE2_GFLAGS_DECLARE_WRAPPER(bool, name)
#define CAFFE2_DECLARE_string(name) CAFFE2_GFLAGS_DECLARE_WRAPPER(string, name)

#else   // CAFFE2_USE_GFLAGS

namespace caffe2 {

class Caffe2FlagParser {
 public:
  Caffe2FlagParser() {}
  bool success() { return success_; }

 protected:
  template <typename T>
  bool Parse(const string& content, T* value);
  bool success_;
};

CAFFE_DECLARE_REGISTRY(Caffe2FlagsRegistry, Caffe2FlagParser, const string&);

}  // namespace caffe2

// The macros are defined outside the caffe2 namespace. In your code, you should
// write the CAFFE2_DEFINE_* and CAFFE2_DECLARE_* macros outside any namespace
// as well.

#define CAFFE2_DEFINE_typed_var(type, name, default_value, help_str)          \
  namespace caffe2 {                                                          \
  CAFFE2_EXPORT type FLAGS_##name = default_value;                            \
  namespace {                                                                 \
  class Caffe2FlagParser_##name : public Caffe2FlagParser {                   \
   public:                                                                    \
    explicit Caffe2FlagParser_##name(const string& content) {                 \
      success_ = Caffe2FlagParser::Parse<type>(content, &FLAGS_##name);       \
    }                                                                         \
  };                                                                          \
  }                                                                           \
  RegistererCaffe2FlagsRegistry g_Caffe2FlagsRegistry_##name(                 \
      #name,                                                                  \
      Caffe2FlagsRegistry(),                                                  \
      RegistererCaffe2FlagsRegistry::DefaultCreator<Caffe2FlagParser_##name>, \
      "(" #type ", default " #default_value ") " help_str);                   \
  }

#define CAFFE2_DEFINE_int(name, default_value, help_str)                       \
  CAFFE2_DEFINE_typed_var(int, name, default_value, help_str)
#define CAFFE2_DEFINE_int64(name, default_value, help_str) \
  CAFFE2_DEFINE_typed_var(int64_t, name, default_value, help_str)
#define CAFFE2_DEFINE_double(name, default_value, help_str) \
  CAFFE2_DEFINE_typed_var(double, name, default_value, help_str)
#define CAFFE2_DEFINE_bool(name, default_value, help_str)                      \
  CAFFE2_DEFINE_typed_var(bool, name, default_value, help_str)
#define CAFFE2_DEFINE_string(name, default_value, help_str)                    \
  CAFFE2_DEFINE_typed_var(string, name, default_value, help_str)

// DECLARE_typed_var should be used in header files and in the global namespace.
#define CAFFE2_DECLARE_typed_var(type, name) \
  namespace caffe2 {                         \
  CAFFE2_IMPORT extern type FLAGS_##name;    \
  } // namespace caffe2

#define CAFFE2_DECLARE_int(name) CAFFE2_DECLARE_typed_var(int, name)
#define CAFFE2_DECLARE_int64(name) CAFFE2_DECLARE_typed_var(int64_t, name)
#define CAFFE2_DECLARE_double(name) CAFFE2_DECLARE_typed_var(double, name)
#define CAFFE2_DECLARE_bool(name) CAFFE2_DECLARE_typed_var(bool, name)
#define CAFFE2_DECLARE_string(name) CAFFE2_DECLARE_typed_var(string, name)

#endif  // CAFFE2_USE_GFLAGS

#endif  // CAFFE2_CORE_FLAGS_H_
