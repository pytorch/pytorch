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

c10::optional<py::object> tryToConvertToCustomClass(
    c10::intrusive_ptr<c10::ivalue::Object> obj) {
  if (obj->name().find("__torch__.torch.classes") == 0) {
    auto objPtr = (void*)obj->getSlot(0).toCapsule().release();
    auto classConverter = getClassConverter()[obj->name()];
    return classConverter(objPtr);
  }
  return c10::nullopt;
}
} // namespace jit
} // namespace torch