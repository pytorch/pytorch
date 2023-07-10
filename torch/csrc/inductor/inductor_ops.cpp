#include <ATen/ops/mm.h>
#include <torch/csrc/inductor/inductor_ops.h>
#include <torch/library.h>

namespace torch {
namespace inductor {
using namespace at;

Tensor _mm_plus_mm(
    const Tensor& a,
    const Tensor& b,
    const Tensor& c,
    const Tensor& d,
    Tensor& out) {
  at::mm_out(out, a, b);
  out.addmm_(c, d);
  return out;
}

TORCH_LIBRARY_FRAGMENT(inductor, m) {
  m.def(
      "_mm_plus_mm(Tensor a, Tensor b, Tensor c, Tensor d, Tensor(t!) out) -> Tensor(t!)",
      _mm_plus_mm);
}

} // namespace inductor
} // namespace torch
