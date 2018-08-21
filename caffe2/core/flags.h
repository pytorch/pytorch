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
CAFFE2_API void SetUsageMessage(const string& str);

/**
 * Returns the usage message for the commandline tool set by SetUsageMessage.
 */
CAFFE2_API const char* UsageMessage();

/**
 * Parses the commandline flags.
 *
 * This command parses all the commandline arguments passed in via pargc
 * and argv. Once it is finished, partc and argv will contain the remaining
 * commandline args that caffe2 does not deal with. Note that following
 * convention, argv[0] contains the binary name and is not parsed.
 */
CAFFE2_API bool ParseCaffeCommandLineFlags(int* pargc, char*** pargv);
/**
 * Checks if the commandline flags has already been passed.
 */
CAFFE2_API bool CommandLineFlagsHasBeenParsed();

}  // namespace caffe2


////////////////////////////////////////////////////////////////////////////////
// Below are gflags and non-gflags specific implementations.
////////////////////////////////////////////////////////////////////////////////

#ifdef CAFFE2_USE_GFLAGS

////////////////////////////////////////////////////////////////////////////////
// Begin gflags section: most functions are basically rerouted to gflags.
////////////////////////////////////////////////////////////////////////////////

#include <gflags/gflags.h>

// gflags before 2.0 uses namespace google and after 2.1 uses namespace gflags.
// Using GFLAGS_GFLAGS_H_ to capture this change.
#ifndef GFLAGS_GFLAGS_H_
namespace gflags = google;
#endif  // GFLAGS_GFLAGS_H_

// Motivation about the gflags wrapper:
// (1) We would need to make sure that the gflags version and the non-gflags
// version of Caffe2 are going to expose the same flags abstraction. One should
// explicitly use caffe2::FLAGS_flag_name to access the flags.
// (2) For flag names, it is recommended to start with caffe2_ to distinguish it
// from regular gflags flags. For example, do
//    CAFFE2_DEFINE_BOOL(caffe2_my_flag, true, "An example");
// to allow one to use caffe2::FLAGS_caffe2_my_flag.
// (3) Gflags has a design issue that does not properly expose the global flags,
// if one builds the library with -fvisibility=hidden. The current gflags (as of
// Aug 2018) only deals with the Windows case using dllexport, and not the Linux
// counterparts. As a result, we will explciitly use CAFFE2_EXPORT to export the
// flags defined in Caffe2. This is done via a global reference, so the flag
// itself is not duplicated - under the hood it is the same global gflags flag.
#define CAFFE2_GFLAGS_DEF_WRAPPER(                                             \
    type, real_type, name, default_value, help_str)                            \
  DEFINE_##type(name, default_value, help_str);                                \
  namespace caffe2 {                                                           \
    CAFFE2_EXPORT real_type& FLAGS_##name = ::FLAGS_##name;                    \
  }

#define CAFFE2_DEFINE_int(name, default_value, help_str)                       \
  CAFFE2_GFLAGS_DEF_WRAPPER(int32, gflags::int32, name, default_value, help_str)
#define CAFFE2_DEFINE_int64(name, default_value, help_str)                     \
  CAFFE2_GFLAGS_DEF_WRAPPER(int64, gflags::int64, name, default_value, help_str)              
#define CAFFE2_DEFINE_double(name, default_value, help_str)                    \
  CAFFE2_GFLAGS_DEF_WRAPPER(double, double, name, default_value, help_str)
#define CAFFE2_DEFINE_bool(name, default_value, help_str)                      \
  CAFFE2_GFLAGS_DEF_WRAPPER(bool, bool, name, default_value, help_str)
#define CAFFE2_DEFINE_string(name, default_value, help_str)                    \
  CAFFE2_GFLAGS_DEF_WRAPPER(                                                   \
      string, ::fLS::clstring, name, default_value, help_str)

// DECLARE_typed_var should be used in header files and in the global namespace.
#define CAFFE2_GFLAGS_DECLARE_WRAPPER(type, real_type, name)                   \
  DECLARE_##type(name);                                                        \
  namespace caffe2 {                                                           \
    CAFFE2_IMPORT extern real_type& FLAGS_##name;                              \
  }  // namespace caffe2

#define CAFFE2_DECLARE_int(name)                                               \
  CAFFE2_GFLAGS_DECLARE_WRAPPER(int32, gflags::int32, name)
#define CAFFE2_DECLARE_int64(name)                                             \
  CAFFE2_GFLAGS_DECLARE_WRAPPER(int64, gflags::int64, name)
#define CAFFE2_DECLARE_double(name)                                            \
  CAFFE2_GFLAGS_DECLARE_WRAPPER(double, double, name)
#define CAFFE2_DECLARE_bool(name)                                              \
  CAFFE2_GFLAGS_DECLARE_WRAPPER(bool, bool, name)
#define CAFFE2_DECLARE_string(name)                                            \
  CAFFE2_GFLAGS_DECLARE_WRAPPER(string, ::fLS::clstring, name)

////////////////////////////////////////////////////////////////////////////////
// End gflags section.
////////////////////////////////////////////////////////////////////////////////

#else   // CAFFE2_USE_GFLAGS

////////////////////////////////////////////////////////////////////////////////
// Begin non-gflags section: providing equivalent functionality.
////////////////////////////////////////////////////////////////////////////////

namespace caffe2 {

class CAFFE2_API Caffe2FlagParser {
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

#define CAFFE2_DEFINE_typed_var(type, name, default_value, help_str)           \
  namespace caffe2 {                                                           \
  CAFFE2_EXPORT type FLAGS_##name = default_value;                             \
  namespace {                                                                  \
  class Caffe2FlagParser_##name : public Caffe2FlagParser {                    \
   public:                                                                     \
    explicit Caffe2FlagParser_##name(const string& content) {                  \
      success_ = Caffe2FlagParser::Parse<type>(content, &FLAGS_##name);        \
    }                                                                          \
  };                                                                           \
  }                                                                            \
  RegistererCaffe2FlagsRegistry g_Caffe2FlagsRegistry_##name(                  \
      #name,                                                                   \
      Caffe2FlagsRegistry(),                                                   \
      RegistererCaffe2FlagsRegistry::DefaultCreator<Caffe2FlagParser_##name>,  \
      "(" #type ", default " #default_value ") " help_str);                    \
  }

#define CAFFE2_DEFINE_int(name, default_value, help_str)                       \
  CAFFE2_DEFINE_typed_var(int, name, default_value, help_str)
#define CAFFE2_DEFINE_int64(name, default_value, help_str)                     \
  CAFFE2_DEFINE_typed_var(int64_t, name, default_value, help_str)
#define CAFFE2_DEFINE_double(name, default_value, help_str)                    \
  CAFFE2_DEFINE_typed_var(double, name, default_value, help_str)
#define CAFFE2_DEFINE_bool(name, default_value, help_str)                      \
  CAFFE2_DEFINE_typed_var(bool, name, default_value, help_str)
#define CAFFE2_DEFINE_string(name, default_value, help_str)                    \
  CAFFE2_DEFINE_typed_var(string, name, default_value, help_str)

// DECLARE_typed_var should be used in header files and in the global namespace.
#define CAFFE2_DECLARE_typed_var(type, name)                                   \
  namespace caffe2 {                                                           \
    CAFFE2_IMPORT extern type FLAGS_##name;                                    \
  } // namespace caffe2

#define CAFFE2_DECLARE_int(name) CAFFE2_DECLARE_typed_var(int, name)
#define CAFFE2_DECLARE_int64(name) CAFFE2_DECLARE_typed_var(int64_t, name)
#define CAFFE2_DECLARE_double(name) CAFFE2_DECLARE_typed_var(double, name)
#define CAFFE2_DECLARE_bool(name) CAFFE2_DECLARE_typed_var(bool, name)
#define CAFFE2_DECLARE_string(name) CAFFE2_DECLARE_typed_var(string, name)

////////////////////////////////////////////////////////////////////////////////
// End non-gflags section.
////////////////////////////////////////////////////////////////////////////////

#endif  // CAFFE2_USE_GFLAGS

#endif  // CAFFE2_CORE_FLAGS_H_
