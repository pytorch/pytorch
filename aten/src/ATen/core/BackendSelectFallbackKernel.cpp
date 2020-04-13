#include <ATen/core/op_registration/op_registration.h>

namespace {

TORCH_LIBRARY_IMPL(_, BackendSelect, m) {
  m.fallback(c10::CppFunction::makeFallthrough());
}

}
