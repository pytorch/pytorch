#include <c10/util/Exception.h>
#include <c10/util/env.h>
#include <fmt/format.h>
#include <cstdlib>
#include <mutex>
#include <shared_mutex>

namespace c10::utils {

static std::shared_mutex env_mutex;

// Set an environment variable.
void set_env(const char* name, const char* value, bool overwrite) {
  std::lock_guard lk(env_mutex);
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)
  if (!overwrite) {
    // NOLINTNEXTLINE(concurrency-mt-unsafe)
    if (std::getenv(name) != nullptr) {
      return;
    }
  }
  auto full_env_variable = fmt::format("{}={}", name, value);
  // NOLINTNEXTLINE(concurrency-mt-unsafe)
  auto err = putenv(full_env_variable.c_str());
  TORCH_INTERNAL_ASSERT(
      err == 0,
      "putenv failed for environment \"",
      name,
      "\", the error is: ",
      err);
#pragma warning(pop)
#else
  // NOLINTNEXTLINE(concurrency-mt-unsafe)
  auto err = setenv(name, value, static_cast<int>(overwrite));
  TORCH_INTERNAL_ASSERT(
      err == 0,
      "setenv failed for environment \"",
      name,
      "\", the error is: ",
      err);
#endif
  return;
}

// Reads an environment variable and returns the content if it is set
std::optional<std::string> get_env(const char* name) noexcept {
  std::shared_lock lk(env_mutex);
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
  // NOLINTNEXTLINE(concurrency-mt-unsafe)
  auto envar = std::getenv(name);
#ifdef _MSC_VER
#pragma warning(pop)
#endif
  if (envar != nullptr) {
    return std::string(envar);
  }
  return std::nullopt;
}

// Checks an environment variable is set.
bool has_env(const char* name) noexcept {
  return get_env(name).has_value();
}

// Reads an environment variable and returns
// - optional<true>,              if set equal to "1"
// - optional<false>,             if set equal to "0"
// - nullopt,   otherwise
//
// NB:
// Issues a warning if the value of the environment variable is not 0 or 1.
std::optional<bool> check_env(const char* name) {
  auto env_opt = get_env(name);
  if (env_opt.has_value()) {
    if (env_opt == "0") {
      return false;
    }
    if (env_opt == "1") {
      return true;
    }
    TORCH_WARN(
        "Ignoring invalid value for boolean flag ",
        name,
        ": ",
        *env_opt,
        "valid values are 0 or 1.");
  }
  return std::nullopt;
}
} // namespace c10::utils
