#include <c10/util/Exception.h>
#include <c10/util/Knobs.h>
#include <cstdlib>
#include <string>

namespace c10 {
static std::function<bool(const char*, const char*, bool)> resolver =
    justknobs_check_default;
bool justknobs_check(const char* the_namespace, const char* feature_name, bool default_value) {
  return resolver(the_namespace, feature_name, default_value);
}
bool justknobs_check_default(const char* the_namespace, const char* feature_name, bool default_value) {
  const char* env_val = std::getenv(feature_name);
  if (env_val == nullptr) {
    return default_value;
  }
  std::string val = std::string(env_val);
  for (auto& x : val) {
    // NOLINTNEXTLINE(*-narrowing-conversions)
    x = std::tolower(x);
  }

  if (val == "y" || val == "yes" || val == "1" || val == "t" || val == "true") {
    return true;
  } else if (
      val == "n" || val == "no" || val == "0" || val == "f" || val == "false") {
    return false;
  } else {
    TORCH_CHECK(
        false,
        "Invalid value ",
        val,
        " for boolean environment variable ",
        feature_name);
  }
}
void set_justknobs_check_resolver(std::function<bool(const char*, const char*, bool)> new_resolver) {
  resolver = std::move(new_resolver);
}
} // namespace c10
