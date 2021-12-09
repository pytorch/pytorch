#include <torch/csrc/jit/mobile/model_tracer/CustomClassTracer.h>

namespace torch {
namespace jit {
namespace mobile {
CustomClassTracer::custom_classes_type& CustomClassTracer::getLoadedClasses() {
  static custom_classes_type loaded_classes;
  return loaded_classes;
}

} // namespace mobile
} // namespace jit
} // namespace torch
