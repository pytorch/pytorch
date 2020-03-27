#include <ATen/core/op_registration/op_registration.h>

namespace {

static auto registry = c10::import()
  .fallback(c10::dispatch(c10::DispatchKey::BackendSelect, c10::CppFunction::makeFallthrough()))
  .fallback(c10::dispatch(c10::DispatchKey::BackendGeneric, c10::CppFunction::makeFallthrough()))
;

}
