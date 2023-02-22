#pragma once

#include <c10/core/SafePyObject.h>
#include <c10/macros/Macros.h>
#include <unordered_map>

namespace at {
namespace impl {

struct TORCH_API ThreadLocalPythonObjects {
  static void set(std::string key, std::shared_ptr<SafePyObject> value);
  static const std::shared_ptr<SafePyObject>& get(std::string key);
  static bool contains(std::string key);

  static const ThreadLocalPythonObjects& get_state();
  static void set_state(const ThreadLocalPythonObjects& state);

 private:
  std::unordered_map<std::string, std::shared_ptr<c10::SafePyObject>> obj_dict_;
};

} // namespace impl
} // namespace at
