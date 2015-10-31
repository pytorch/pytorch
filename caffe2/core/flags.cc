#include <cstdlib>

#include "caffe2/core/flags.h"
#include "caffe2/core/logging.h"

namespace caffe2 {
DEFINE_REGISTRY(Caffe2FlagsRegistry, Caffe2FlagParser, const string&);

static bool gCommandLineFlagsParsed = false;

// The function is defined in init.cc.
extern std::stringstream& GlobalInitStream();

bool ParseCaffeCommandLineFlags(int* pargc, char** argv) {
  bool success = true;
  GlobalInitStream() << "Parsing commandline arguments for caffe2."
                     << std::endl;
  // write_head is the location we write the unused arguments to.
  int write_head = 1;
  for (int i = 1; i < *pargc; ++i) {
    string arg(argv[i]);
    int prefix_idx = arg.find('=');
    if (prefix_idx == string::npos) {
      prefix_idx = arg.size();
    }
    // If the arg does not start with "--", and we will ignore it.
    if (arg[0] != '-' || arg[1] != '-') {
      GlobalInitStream()
          << "Caffe2 flag: commandline argument does not match --name=var "
             "or --name format:"
          << arg << std::endl;
      argv[write_head++] = argv[i];
      continue;
    }
    string key = arg.substr(2, prefix_idx - 2);
    string value = (prefix_idx == arg.size() ? ""
                    : arg.substr(prefix_idx + 1, string::npos));
    // If the flag is not registered, we will ignore it.
    if (!Caffe2FlagsRegistry()->Has(key)) {
      GlobalInitStream() << "Caffe2 flag: unrecognized commandline argument: "
                         << arg << std::endl;
      argv[write_head++] = argv[i];
      continue;
    }
    std::unique_ptr<Caffe2FlagParser> parser(
        Caffe2FlagsRegistry()->Create(key, value));
    // Since we have checked that the key is in Caffe2FlagsRegistry, this
    // should not happen.
    CAFFE_CHECK(parser != nullptr) << "This should not happen with key=" << key
                                   << " and value=" << value;
    if (!parser->success()) {
      // TODO: quit elegantly.
      GlobalInitStream() << "Caffe2 flag fatal: illegal argument: "
                         << arg << std::endl;
      success = false;
    }
  }
  *pargc = write_head;
  gCommandLineFlagsParsed = true;
  return success;
}

bool CommandLineFlagsHasBeenParsed() {
  return gCommandLineFlagsParsed;
}

template <>
bool Caffe2FlagParser::Parse<string>(const string& content, string* value) {
  *value = content;
  return true;
}

template <>
bool Caffe2FlagParser::Parse<int>(const string& content, int* value) {
  try {
    *value = std::atoi(content.c_str());
    return true;
  } catch(...) {
    GlobalInitStream() << "Caffe2 flag error: Cannot convert argument to int: "
                       << content << std::endl;
    return false;
  }
}

template <>
bool Caffe2FlagParser::Parse<double>(const string& content, double* value) {
  try {
    *value = std::atof(content.c_str());
    return true;
  } catch(...) {
    GlobalInitStream()
        << "Caffe2 flag error: Cannot convert argument to double: "
        << content << std::endl;
    return false;
  }
}

template <>
bool Caffe2FlagParser::Parse<bool>(const string& content, bool* value) {
  if (content == "false" || content == "False" || content == "FALSE" ||
      content == "0") {
    *value = false;
    return true;
  } else if (content == "true" || content == "True" || content == "TRUE" ||
      content == "1" || content == "") {
    *value = true;
    return true;
  } else {
    GlobalInitStream()
        << "Caffe2 flag error: Cannot convert argument to bool: " << content
        << std::endl;
    return false;
  }
}


}  // namespace caffe2