#pragma once

#include <c10/macros/Export.h>
#include <c10/util/Exception.h>

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <mutex>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace c10::CachingAllocator {

using SettingKey = std::variant<int64_t, std::string>;
using SettingScalar =
    std::variant<std::monostate, bool, int64_t, double, std::string>;
using SettingDict = std::map<SettingKey, SettingScalar>;
using SettingValue = std::
    variant<std::monostate, bool, int64_t, double, std::string, SettingDict>;
using Settings = std::map<std::string, SettingValue>;

[[noreturn]] C10_API void settingTypeError(
    const std::string& name,
    const char* expected);
C10_API int64_t
settingInteger(const SettingValue& value, const std::string& name);
C10_API double settingDouble(
    const SettingValue& value,
    const std::string& name);
C10_API const SettingDict& settingDict(
    const SettingValue& value,
    const std::string& name);
C10_API std::recursive_mutex& allocatorSettingsMutex();
C10_API Settings getAllocatorSettingsDict();
C10_API void setAllocatorSettingsDict(const Settings& settings);

namespace detail {
template <typename T>
T readSettingField(const T& field) {
  return field;
}

template <typename T>
T readSettingField(const std::atomic<T>& field) {
  return field.load();
}

template <typename T>
SettingValue settingScalarValue(T value, const std::string& name) {
  if constexpr (std::is_same_v<T, bool> || std::is_floating_point_v<T>) {
    return value;
  } else {
    static_assert(std::is_integral_v<T>);
    TORCH_CHECK_VALUE(
        std::in_range<int64_t>(value),
        name,
        " exceeds the supported integer range");
    return static_cast<int64_t>(value);
  }
}

template <typename T>
T settingScalar(const SettingValue& value, const std::string& name) {
  if constexpr (std::is_same_v<T, bool>) {
    const auto* result = std::get_if<bool>(&value);
    if (result == nullptr) {
      settingTypeError(name, "a bool");
    }
    return *result;
  } else if constexpr (std::is_floating_point_v<T>) {
    return static_cast<T>(settingDouble(value, name));
  } else {
    static_assert(std::is_integral_v<T>);
    const auto result = settingInteger(value, name);
    TORCH_CHECK_VALUE(
        static_cast<uint64_t>(result) <= std::numeric_limits<T>::max(),
        name,
        " exceeds the supported integer range");
    return static_cast<T>(result);
  }
}
} // namespace detail

class C10_API AllocatorSettingsRegistry {
 public:
  // Preparing an update validates and owns its input without changing allocator
  // state. Applying a prepared update must not throw.
  using Update = std::function<void()>;
  using Getter = std::function<SettingValue()>;
  using Prepare = std::function<Update(const SettingValue&)>;

  void add(std::string name, Getter getter, Prepare prepare);
  void addValidator(std::function<void(const Settings&)> validator);

  template <typename Field, typename Validator = std::nullptr_t>
  void addScalar(
      const std::string& name,
      Field& field,
      Validator validator = nullptr) {
    using T = decltype(detail::readSettingField(field));
    add(
        name,
        [&field, name] {
          return detail::settingScalarValue(
              detail::readSettingField(field), name);
        },
        [&field, name, validator](const SettingValue& value) -> Update {
          const T parsed = detail::settingScalar<T>(value, name);
          if constexpr (!std::is_same_v<Validator, std::nullptr_t>) {
            validator(parsed);
          }
          return [&field, parsed] { field = parsed; };
        });
  }

  template <typename Field, typename Validator = std::nullptr_t>
  void addSize(
      const std::string& name,
      Field& field,
      size_t unit,
      bool unlimited,
      Validator validator = nullptr) {
    add(
        name,
        [&field, name, unit, unlimited]() -> SettingValue {
          const size_t value = detail::readSettingField(field);
          if (unlimited && value == std::numeric_limits<size_t>::max()) {
            return std::monostate{};
          }
          return detail::settingScalarValue(value / unit, name);
        },
        [&field, name, unit, unlimited, validator](
            const SettingValue& value) -> Update {
          size_t parsed = std::numeric_limits<size_t>::max();
          if (!(unlimited && std::holds_alternative<std::monostate>(value))) {
            const auto integer = settingInteger(value, name);
            TORCH_CHECK_VALUE(
                static_cast<uint64_t>(integer) <=
                    std::numeric_limits<size_t>::max() / unit,
                name,
                " exceeds the supported size range");
            parsed = static_cast<size_t>(integer) * unit;
          }
          if constexpr (!std::is_same_v<Validator, std::nullptr_t>) {
            validator(parsed);
          }
          return [&field, parsed] { field = parsed; };
        });
  }

 private:
  friend Settings getAllocatorSettingsDict();
  friend void setAllocatorSettingsDict(const Settings& settings);

  struct Entry {
    Getter getter;
    Prepare prepare;
  };

  Settings snapshot() const;
  std::vector<Update> prepare(const Settings& settings) const;

  std::map<std::string, Entry> entries_;
  std::vector<std::function<void(const Settings&)>> validators_;
};

// Backend callbacks add descriptors for their own storage. They must not retain
// the registry, which is rebuilt for each operation to include late extensions.
C10_API void registerAllocatorSettings(
    std::function<void(AllocatorSettingsRegistry&)> registration);

} // namespace c10::CachingAllocator
