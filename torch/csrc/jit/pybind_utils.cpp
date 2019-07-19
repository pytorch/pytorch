#include <torch/csrc/utils/pybind.h>
#include <unordered_map>

namespace torch {
namespace jit {
std::unordered_map<std::string, std::function<py::object(void*)>>&
getClassConverter() {
  static std::unordered_map<std::string, std::function<py::object(void*)>>
      classConverter;
  return classConverter;
}
} // namespace jit
} // namespace torch