#include <c10/macros/Macros.h>
#include <c10/util/Flags.h>

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

#ifndef C10_USE_GFLAGS

namespace c10 {

using std::string;

C10_DEFINE_REGISTRY(C10FlagsRegistry, C10FlagParser, const string&);

namespace {
static bool gCommandLineFlagsParsed = false;
// Since flags is going to be loaded before logging, we would
// need to have a stringstream to hold the messages instead of directly
// using caffe logging.
std::stringstream& GlobalInitStream() {
  static std::stringstream ss;
  return ss;
}
static const char* gUsageMessage = "(Usage message not set.)";
} // namespace

C10_EXPORT void SetUsageMessage(const string& str) {
  static string usage_message_safe_copy = str;
  gUsageMessage = usage_message_safe_copy.c_str();
}

C10_EXPORT const char* UsageMessage() {
  return gUsageMessage;
}

C10_EXPORT bool ParseCommandLineFlags(int* pargc, char*** pargv) {
  if (*pargc == 0)
    return true;
  char** argv = *pargv;
  bool success = true;
  GlobalInitStream() << "Parsing commandline arguments for c10." << std::endl;
  // write_head is the location we write the unused arguments to.
  int write_head = 1;
  for (int i = 1; i < *pargc; ++i) {
    string arg(argv[i]);

    if (arg.find("--help") != string::npos) {
      // Print the help message, and quit.
      std::cout << UsageMessage() << std::endl;
      std::cout << "Arguments: " << std::endl;
      for (const auto& help_msg : C10FlagsRegistry()->HelpMessage()) {
        std::cout << "    " << help_msg.first << ": " << help_msg.second
                  << std::endl;
      }
      exit(0);
    }
    // If the arg does not start with "--", we will ignore it.
    if (arg[0] != '-' || arg[1] != '-') {
      GlobalInitStream()
          << "C10 flag: commandline argument does not match --name=var "
             "or --name format: "
          << arg << ". Ignoring this argument." << std::endl;
      argv[write_head++] = argv[i];
      continue;
    }

    string key;
    string value;
    size_t prefix_idx = arg.find('=');
    if (prefix_idx == string::npos) {
      // If there is no equality char in the arg, it means that the
      // arg is specified in the next argument.
      key = arg.substr(2, arg.size() - 2);
      ++i;
      if (i == *pargc) {
        GlobalInitStream()
            << "C10 flag: reached the last commandline argument, but "
               "I am expecting a value for "
            << arg;
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
    if (!C10FlagsRegistry()->Has(key)) {
      GlobalInitStream() << "C10 flag: unrecognized commandline argument: "
                         << arg << std::endl;
      success = false;
      break;
    }
    std::unique_ptr<C10FlagParser> parser(
        C10FlagsRegistry()->Create(key, value));
    if (!parser->success()) {
      GlobalInitStream() << "C10 flag: illegal argument: " << arg << std::endl;
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

C10_EXPORT bool CommandLineFlagsHasBeenParsed() {
  return gCommandLineFlagsParsed;
}

template <>
C10_EXPORT bool C10FlagParser::Parse<string>(
    const string& content,
    string* value) {
  *value = content;
  return true;
}

template <>
C10_EXPORT bool C10FlagParser::Parse<int>(const string& content, int* value) {
  try {
    *value = std::atoi(content.c_str());
    return true;
  } catch (...) {
    GlobalInitStream() << "C10 flag error: Cannot convert argument to int: "
                       << content << std::endl;
    return false;
  }
}

template <>
C10_EXPORT bool C10FlagParser::Parse<int64_t>(
    const string& content,
    int64_t* value) {
  try {
    static_assert(sizeof(long long) == sizeof(int64_t));
#ifdef __ANDROID__
    // Android does not have std::atoll.
    *value = atoll(content.c_str());
#else
    *value = std::atoll(content.c_str());
#endif
    return true;
  } catch (...) {
    GlobalInitStream() << "C10 flag error: Cannot convert argument to int: "
                       << content << std::endl;
    return false;
  }
}

template <>
C10_EXPORT bool C10FlagParser::Parse<double>(
    const string& content,
    double* value) {
  try {
    *value = std::atof(content.c_str());
    return true;
  } catch (...) {
    GlobalInitStream() << "C10 flag error: Cannot convert argument to double: "
                       << content << std::endl;
    return false;
  }
}

template <>
C10_EXPORT bool C10FlagParser::Parse<bool>(const string& content, bool* value) {
  if (content == "false" || content == "False" || content == "FALSE" ||
      content == "0") {
    *value = false;
    return true;
  } else if (
      content == "true" || content == "True" || content == "TRUE" ||
      content == "1") {
    *value = true;
    return true;
  } else {
    GlobalInitStream()
        << "C10 flag error: Cannot convert argument to bool: " << content
        << std::endl
        << "Note that if you are passing in a bool flag, you need to "
           "explicitly specify it, like --arg=True or --arg True. Otherwise, "
           "the next argument may be inadvertently used as the argument, "
           "causing the above error."
        << std::endl;
    return false;
  }
}

} // namespace c10

#endif // C10_USE_GFLAGS
