#include "caffe2/core/flags.h"

#include <cstdlib>
#include <sstream>

#include "caffe2/core/logging.h"

namespace caffe2 {

#ifdef CAFFE2_USE_GFLAGS

void SetUsageMessage(const string& str) {
  if (UsageMessage() != nullptr) {
    // Usage message has already been set, so we will simply return.
    return;
  }
  gflags::SetUsageMessage(str);
}

const char* UsageMessage() {
  return gflags::ProgramUsage();
}

bool ParseCaffeCommandLineFlags(int* pargc, char*** pargv) {
  if (*pargc == 0) return true;
  return gflags::ParseCommandLineFlags(pargc, pargv, true);
}

bool CommandLineFlagsHasBeenParsed() {
  // There is no way we query gflags right now, so we will simply return true.
  return true;
}

#else  // CAFFE2_USE_GFLAGS


CAFFE_DEFINE_REGISTRY(Caffe2FlagsRegistry, Caffe2FlagParser, const string&);

namespace {
static bool gCommandLineFlagsParsed = false;
// Since caffe flags is going to be loaded before caffe logging, we would
// need to have a stringstream to hold the messages instead of directly
// using caffe logging.
std::stringstream& GlobalInitStream() {
  static std::stringstream ss;
  return ss;
}
static string gUsageMessage = "(Usage message not set.)";
}


void SetUsageMessage(const string& str) { gUsageMessage = str; }
const char* UsageMessage() { return gUsageMessage.c_str(); }

bool ParseCaffeCommandLineFlags(int* pargc, char*** pargv) {
  if (*pargc == 0) return true;
  char** argv = *pargv;
  bool success = true;
  GlobalInitStream() << "Parsing commandline arguments for caffe2."
                     << std::endl;
  // write_head is the location we write the unused arguments to.
  int write_head = 1;
  for (int i = 1; i < *pargc; ++i) {
    string arg(argv[i]);

    if (arg.find("--help") != string::npos) {
      // Print the help message, and quit.
      std::cout << UsageMessage() << std::endl;
      std::cout << "Arguments: " << std::endl;
      for (const auto& help_msg : Caffe2FlagsRegistry()->HelpMessage()) {
        std::cout << "    " << help_msg.first << ": " << help_msg.second
                  << std::endl;
      }
      exit(0);
    }
    // If the arg does not start with "--", we will ignore it.
    if (arg[0] != '-' || arg[1] != '-') {
      GlobalInitStream()
          << "Caffe2 flag: commandline argument does not match --name=var "
             "or --name format: "
          << arg << ". Ignoring this argument." << std::endl;
      argv[write_head++] = argv[i];
      continue;
    }

    string key;
    string value;
    int prefix_idx = arg.find('=');
    if (prefix_idx == string::npos) {
      // If there is no equality char in the arg, it means that the
      // arg is specified in the next argument.
      key = arg.substr(2, arg.size() - 2);
      ++i;
      if (i == *pargc) {
        GlobalInitStream()
            << "Caffe2 flag: reached the last commandline argument, but "
               "I am expecting a value for " << arg;
        success = false;
        break;
      }
      value = string(argv[i]);
    } else {
      // If there is an equality character, we will basically use the value
      // after the "=".
      key = arg.substr(2, prefix_idx - 2);
      value = arg.substr(prefix_idx + 1, string::npos);
    }
    // If the flag is not registered, we will ignore it.
    if (!Caffe2FlagsRegistry()->Has(key)) {
      GlobalInitStream() << "Caffe2 flag: unrecognized commandline argument: "
                         << arg << std::endl;
      success = false;
      break;
    }
    std::unique_ptr<Caffe2FlagParser> parser(
        Caffe2FlagsRegistry()->Create(key, value));
    if (!parser->success()) {
      GlobalInitStream() << "Caffe2 flag: illegal argument: "
                         << arg << std::endl;
      success = false;
      break;
    }
  }
  *pargc = write_head;
  gCommandLineFlagsParsed = true;
  // TODO: when we fail commandline flag parsing, shall we continue, or
  // shall we just quit loudly? Right now we carry on the computation, but
  // since there are failures in parsing, it is very likely that some
  // downstream things will break, in which case it makes sense to quit loud
  // and early.
  if (!success) {
    std::cerr << GlobalInitStream().str();
  }
  // Clear the global init stream.
  GlobalInitStream().str(std::string());
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
bool Caffe2FlagParser::Parse<int64_t>(const string& content, int64_t* value) {
  try {
    static_assert(sizeof(long long) == sizeof(int64_t), "");
#ifdef __ANDROID__
    // Android does not have std::atoll.
    *value = atoll(content.c_str());
#else
    *value = std::atoll(content.c_str());
#endif
    return true;
  } catch (...) {
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
      content == "1") {
    *value = true;
    return true;
  } else {
    GlobalInitStream()
        << "Caffe2 flag error: Cannot convert argument to bool: "
        << content << std::endl
        << "Note that if you are passing in a bool flag, you need to "
           "explicitly specify it, like --arg=True or --arg True. Otherwise, "
           "the next argument may be inadvertently used as the argument, "
           "causing the above error."
        << std::endl;
    return false;
  }
}

#endif  // CAFFE2_USE_GFLAGS

}  // namespace caffe2
