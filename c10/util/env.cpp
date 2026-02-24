#include <c10/util/Exception.h>
#include <c10/util/env.h>
#include <fmt/format.h>
#include <cstdlib>
#include <mutex>
#include <shared_mutex>

namespace c10::utils {

namespace {
// This is used to coordinate access to getenv/setenv calls,
// which may not be thread-safe on certain platforms.
//
// Because get_env may be called in global destructors, we CANNOT use
// std::mutex here, as it may get destructed before get_env is called.
// Instead, we use a pointer to a mutex allocated on the heap.
struct StaticEnvMutex {
  std::shared_timed_mutex val;
};
StaticEnvMutex* get_static_env_mutex() {
  static StaticEnvMutex* env_mutex = new StaticEnvMutex();
  return env_mutex;
}
} // namespace

// Set an environment variable.
void set_env(const char* name, const char* value, bool overwrite) {
  std::lock_guard lk(get_static_env_mutex()->val);
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
  std::shared_lock lk(get_static_env_mutex()->val);
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
