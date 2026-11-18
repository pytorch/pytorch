#include <c10/macros/Macros.h>

#ifdef C10_USE_GFLAGS

#include <c10/util/Flags.h>
#include <string>

namespace c10 {

using std::string;

C10_EXPORT void SetUsageMessage(const string& str) {
  if (UsageMessage() != nullptr) {
    // Usage message has already been set, so we will simply return.
    return;
  }
  gflags::SetUsageMessage(str);
}

C10_EXPORT const char* UsageMessage() {
  return gflags::ProgramUsage();
}

C10_EXPORT bool ParseCommandLineFlags(int* pargc, char*** pargv) {
  // In case there is no commandline flags to parse, simply return.
  if (*pargc == 0)
    return true;
  return gflags::ParseCommandLineFlags(pargc, pargv, true);
}

C10_EXPORT bool CommandLineFlagsHasBeenParsed() {
  // There is no way we query gflags right now, so we will simply return true.
  return true;
}

} // namespace c10
#endif // C10_USE_GFLAGS
