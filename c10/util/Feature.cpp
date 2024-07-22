#include <c10/util/Exception.h>
#include <c10/util/Feature.h>
#include <cstdlib>
#include <string>

namespace c10 {
static std::function<bool(const char*, const char*)> feature_resolver =
    FeatureEnabledDefaultResolver;
bool FeatureEnabled(const char* the_namespace, const char* feature_name) {
  return feature_resolver(the_namespace, feature_name);
}
bool FeatureEnabledDefaultResolver(const char* the_namespace, const char* feature_name) {
  const char* env_val = std::getenv(feature_name);
  if (env_val == nullptr) {
    // FeatureEnabled is used for killswitches so the
    // default is true
    return true;
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
void SetFeatureResolver(std::function<bool(const char*, const char*)> resolver) {
  feature_resolver = std::move(resolver);
}
} // namespace c10
