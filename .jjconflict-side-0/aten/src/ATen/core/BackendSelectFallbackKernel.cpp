#include <torch/library.h>

TORCH_LIBRARY_IMPL(_, BackendSelect, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}
