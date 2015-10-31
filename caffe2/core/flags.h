#ifndef CAFFE2_CORE_FLAGS_H_
#define CAFFE2_CORE_FLAGS_H_
// A lightweighted commandline flags tool for caffe2, so we do not need to rely
// on gflags.

#include "caffe2/core/registry.h"

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

DECLARE_REGISTRY(Caffe2FlagsRegistry, Caffe2FlagParser, const string&);

bool ParseCaffeCommandLineFlags(int* pargc, char** argv);
bool CommandLineFlagsHasBeenParsed();

}  // namespace caffe2


// The macros are defined outside the caffe2 namespace. In your code, you should
// write the CAFFE2_DEFINE_* and CAFFE2_DECLARE_* macros outside any namespace
// as well.

#define CAFFE2_DEFINE_typed_var(type, name, default_value, unused_help_str)    \
  namespace caffe2 {                                                           \
    type FLAGS_##name = default_value;                                         \
    namespace {                                                                \
      class Caffe2FlagParser_##name : public Caffe2FlagParser {                \
       public:                                                                 \
        explicit Caffe2FlagParser_##name(const string& content) {              \
          success_ = Caffe2FlagParser::Parse<type>(content, &FLAGS_##name);    \
        }                                                                      \
      };                                                                       \
    }                                                                          \
    REGISTER_CLASS(Caffe2FlagsRegistry, name, Caffe2FlagParser_##name);        \
  }

#define CAFFE2_DEFINE_int(name, default_value, unused_help_str)                \
  CAFFE2_DEFINE_typed_var(int, name, default_value, unused_help_str)
#define CAFFE2_DEFINE_double(name, default_value, unused_help_str)             \
  CAFFE2_DEFINE_typed_var(double, name, default_value, unused_help_str)
#define CAFFE2_DEFINE_bool(name, default_value, unused_help_str)               \
  CAFFE2_DEFINE_typed_var(bool, name, default_value, unused_help_str)
#define CAFFE2_DEFINE_string(name, default_value, unused_help_str)             \
  CAFFE2_DEFINE_typed_var(string, name, default_value, unused_help_str)

// DECLARE_typed_var should be used in header files and in the global namespace.
#define CAFFE2_DECLARE_typed_var(type, name)                                   \
  namespace caffe2 {                                                           \
    extern type FLAGS_##name;                                                  \
  }  // namespace caffe2

#define CAFFE2_DECLARE_int(name) CAFFE2_DECLARE_typed_var(int, name)
#define CAFFE2_DECLARE_double(name) CAFFE2_DECLARE_typed_var(double, name)
#define CAFFE2_DECLARE_bool(name) CAFFE2_DECLARE_typed_var(bool, name)
#define CAFFE2_DECLARE_string(name) CAFFE2_DECLARE_typed_var(string, name)

#endif  // CAFFE2_CORE_FLAGS_H_
