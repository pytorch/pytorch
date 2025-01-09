#ifndef C10_UTIL_FLAGS_H_
#define C10_UTIL_FLAGS_H_

/* Commandline flags support for C10.
 *
 * This is a portable commandline flags tool for c10, so we can optionally
 * choose to use gflags or a lightweight custom implementation if gflags is
 * not possible on a certain platform. If you have gflags installed, set the
 * macro C10_USE_GFLAGS will seamlessly route everything to gflags.
 *
 * To define a flag foo of type bool default to true, do the following in the
 * *global* namespace:
 *     C10_DEFINE_bool(foo, true, "An example.");
 *
 * To use it in another .cc file, you can use C10_DECLARE_* as follows:
 *     C10_DECLARE_bool(foo);
 *
 * In both cases, you can then access the flag via FLAGS_foo.
 *
 * It is recommended that you build with gflags. To learn more about the flags
 * usage, refer to the gflags page here:
 *
 * https://gflags.github.io/gflags/
 *
 * Note about Python users / devs: gflags is initiated from a C++ function
 * ParseCommandLineFlags, and is usually done in native binaries in the main
 * function. As Python does not have a modifiable main function, it is usually
 * difficult to change the flags after Python starts. Hence, it is recommended
 * that one sets the default value of the flags to one that's acceptable in
 * general - that will allow Python to run without wrong flags.
 */

#include <c10/macros/Export.h>
#include <string>

#include <c10/util/Registry.h>

namespace c10 {
/**
 * Sets the usage message when a commandline tool is called with "--help".
 */
C10_API void SetUsageMessage(const std::string& str);

/**
 * Returns the usage message for the commandline tool set by SetUsageMessage.
 */
C10_API const char* UsageMessage();

/**
 * Parses the commandline flags.
 *
 * This command parses all the commandline arguments passed in via pargc
 * and argv. Once it is finished, partc and argv will contain the remaining
 * commandline args that c10 does not deal with. Note that following
 * convention, argv[0] contains the binary name and is not parsed.
 */
C10_API bool ParseCommandLineFlags(int* pargc, char*** pargv);

/**
 * Checks if the commandline flags has already been passed.
 */
C10_API bool CommandLineFlagsHasBeenParsed();

} // namespace c10

////////////////////////////////////////////////////////////////////////////////
// Below are gflags and non-gflags specific implementations.
// In general, they define the following macros for one to declare (use
// C10_DECLARE) or define (use C10_DEFINE) flags:
// C10_{DECLARE,DEFINE}_{int,int64,double,bool,string}
////////////////////////////////////////////////////////////////////////////////

#ifdef C10_USE_GFLAGS

////////////////////////////////////////////////////////////////////////////////
// Begin gflags section: most functions are basically rerouted to gflags.
////////////////////////////////////////////////////////////////////////////////
#include <gflags/gflags.h>

// C10 uses hidden visibility by default. However, in gflags, it only uses
// export on Windows platform (with dllexport) but not on linux/mac (with
// default visibility). As a result, to ensure that we are always exporting
// global variables, we will redefine the GFLAGS_DLL_DEFINE_FLAG macro if we
// are building C10 as a shared library.
// This has to be done after the inclusion of gflags, because some early
// versions of gflags.h (e.g. 2.0 on ubuntu 14.04) directly defines the
// macros, so we need to do definition after gflags is done.
#ifdef GFLAGS_DLL_DEFINE_FLAG
#undef GFLAGS_DLL_DEFINE_FLAG
#endif // GFLAGS_DLL_DEFINE_FLAG
#ifdef GFLAGS_DLL_DECLARE_FLAG
#undef GFLAGS_DLL_DECLARE_FLAG
#endif // GFLAGS_DLL_DECLARE_FLAG
#define GFLAGS_DLL_DEFINE_FLAG C10_EXPORT
#define GFLAGS_DLL_DECLARE_FLAG C10_IMPORT

// gflags before 2.0 uses namespace google and after 2.1 uses namespace gflags.
// Using GFLAGS_GFLAGS_H_ to capture this change.
#ifndef GFLAGS_GFLAGS_H_
namespace gflags = google;
#endif // GFLAGS_GFLAGS_H_

// Motivation about the gflags wrapper:
// (1) We would need to make sure that the gflags version and the non-gflags
// version of C10 are going to expose the same flags abstraction. One should
// explicitly use FLAGS_flag_name to access the flags.
// (2) For flag names, it is recommended to start with c10_ to distinguish it
// from regular gflags flags. For example, do
//    C10_DEFINE_BOOL(c10_my_flag, true, "An example");
// to allow one to use FLAGS_c10_my_flag.
// (3) Gflags has a design issue that does not properly expose the global flags,
// if one builds the library with -fvisibility=hidden. The current gflags (as of
// Aug 2018) only deals with the Windows case using dllexport, and not the Linux
// counterparts. As a result, we will explicitly use C10_EXPORT to export the
// flags defined in C10. This is done via a global reference, so the flag
// itself is not duplicated - under the hood it is the same global gflags flag.
#define C10_GFLAGS_DEF_WRAPPER(type, real_type, name, default_value, help_str) \
  DEFINE_##type(name, default_value, help_str);

#define C10_DEFINE_int(name, default_value, help_str) \
  C10_GFLAGS_DEF_WRAPPER(int32, gflags::int32, name, default_value, help_str)
#define C10_DEFINE_int32(name, default_value, help_str) \
  C10_DEFINE_int(name, default_value, help_str)
#define C10_DEFINE_int64(name, default_value, help_str) \
  C10_GFLAGS_DEF_WRAPPER(int64, gflags::int64, name, default_value, help_str)
#define C10_DEFINE_double(name, default_value, help_str) \
  C10_GFLAGS_DEF_WRAPPER(double, double, name, default_value, help_str)
#define C10_DEFINE_bool(name, default_value, help_str) \
  C10_GFLAGS_DEF_WRAPPER(bool, bool, name, default_value, help_str)
#define C10_DEFINE_string(name, default_value, help_str) \
  C10_GFLAGS_DEF_WRAPPER(string, ::fLS::clstring, name, default_value, help_str)

// DECLARE_typed_var should be used in header files and in the global namespace.
#define C10_GFLAGS_DECLARE_WRAPPER(type, real_type, name) DECLARE_##type(name);

#define C10_DECLARE_int(name) \
  C10_GFLAGS_DECLARE_WRAPPER(int32, gflags::int32, name)
#define C10_DECLARE_int32(name) C10_DECLARE_int(name)
#define C10_DECLARE_int64(name) \
  C10_GFLAGS_DECLARE_WRAPPER(int64, gflags::int64, name)
#define C10_DECLARE_double(name) \
  C10_GFLAGS_DECLARE_WRAPPER(double, double, name)
#define C10_DECLARE_bool(name) C10_GFLAGS_DECLARE_WRAPPER(bool, bool, name)
#define C10_DECLARE_string(name) \
  C10_GFLAGS_DECLARE_WRAPPER(string, ::fLS::clstring, name)

#define TORCH_DECLARE_int(name) C10_DECLARE_int(name)
#define TORCH_DECLARE_int32(name) C10_DECLARE_int32(name)
#define TORCH_DECLARE_int64(name) C10_DECLARE_int64(name)
#define TORCH_DECLARE_double(name) C10_DECLARE_double(name)
#define TORCH_DECLARE_bool(name) C10_DECLARE_bool(name)
#define TORCH_DECLARE_string(name) C10_DECLARE_string(name)

////////////////////////////////////////////////////////////////////////////////
// End gflags section.
////////////////////////////////////////////////////////////////////////////////

#else // C10_USE_GFLAGS

////////////////////////////////////////////////////////////////////////////////
// Begin non-gflags section: providing equivalent functionality.
////////////////////////////////////////////////////////////////////////////////

namespace c10 {

class C10_API C10FlagParser {
 public:
  bool success() {
    return success_;
  }

 protected:
  template <typename T>
  bool Parse(const std::string& content, T* value);
  bool success_{false};
};

C10_DECLARE_REGISTRY(C10FlagsRegistry, C10FlagParser, const std::string&);

} // namespace c10

// The macros are defined outside the c10 namespace. In your code, you should
// write the C10_DEFINE_* and C10_DECLARE_* macros outside any namespace
// as well.

#define C10_DEFINE_typed_var(type, name, default_value, help_str)       \
  C10_EXPORT type FLAGS_##name = default_value;                         \
  namespace c10 {                                                       \
  namespace {                                                           \
  class C10FlagParser_##name : public C10FlagParser {                   \
   public:                                                              \
    explicit C10FlagParser_##name(const std::string& content) {         \
      success_ = C10FlagParser::Parse<type>(content, &FLAGS_##name);    \
    }                                                                   \
  };                                                                    \
  }                                                                     \
  RegistererC10FlagsRegistry g_C10FlagsRegistry_##name(                 \
      #name,                                                            \
      C10FlagsRegistry(),                                               \
      RegistererC10FlagsRegistry::DefaultCreator<C10FlagParser_##name>, \
      "(" #type ", default " #default_value ") " help_str);             \
  }

#define C10_DEFINE_int(name, default_value, help_str) \
  C10_DEFINE_typed_var(int, name, default_value, help_str)
#define C10_DEFINE_int32(name, default_value, help_str) \
  C10_DEFINE_int(name, default_value, help_str)
#define C10_DEFINE_int64(name, default_value, help_str) \
  C10_DEFINE_typed_var(int64_t, name, default_value, help_str)
#define C10_DEFINE_double(name, default_value, help_str) \
  C10_DEFINE_typed_var(double, name, default_value, help_str)
#define C10_DEFINE_bool(name, default_value, help_str) \
  C10_DEFINE_typed_var(bool, name, default_value, help_str)
#define C10_DEFINE_string(name, default_value, help_str) \
  C10_DEFINE_typed_var(std::string, name, default_value, help_str)

// DECLARE_typed_var should be used in header files and in the global namespace.
#define C10_DECLARE_typed_var(type, name) C10_API extern type FLAGS_##name

#define C10_DECLARE_int(name) C10_DECLARE_typed_var(int, name)
#define C10_DECLARE_int32(name) C10_DECLARE_int(name)
#define C10_DECLARE_int64(name) C10_DECLARE_typed_var(int64_t, name)
#define C10_DECLARE_double(name) C10_DECLARE_typed_var(double, name)
#define C10_DECLARE_bool(name) C10_DECLARE_typed_var(bool, name)
#define C10_DECLARE_string(name) C10_DECLARE_typed_var(std::string, name)

#define TORCH_DECLARE_typed_var(type, name) TORCH_API extern type FLAGS_##name

#define TORCH_DECLARE_int(name) TORCH_DECLARE_typed_var(int, name)
#define TORCH_DECLARE_int32(name) TORCH_DECLARE_int(name)
#define TORCH_DECLARE_int64(name) TORCH_DECLARE_typed_var(int64_t, name)
#define TORCH_DECLARE_double(name) TORCH_DECLARE_typed_var(double, name)
#define TORCH_DECLARE_bool(name) TORCH_DECLARE_typed_var(bool, name)
#define TORCH_DECLARE_string(name) TORCH_DECLARE_typed_var(std::string, name)

////////////////////////////////////////////////////////////////////////////////
// End non-gflags section.
////////////////////////////////////////////////////////////////////////////////

#endif // C10_USE_GFLAGS

#endif // C10_UTIL_FLAGS_H_
