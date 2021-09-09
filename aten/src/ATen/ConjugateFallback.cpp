#include <ATen/native/MathBitsFallback.h>

namespace at {

struct ConjFallback : MathOpFallback {
  ConjFallback() : MathOpFallback(DispatchKey::Conjugate, "conjugate") {}
  bool is_bit_set(const Tensor& tensor) override {
    return tensor.is_conj();
  }
  void _set_bit(const Tensor& tensor, bool value) override {
    return tensor._set_conj(value);
  }
  Tensor resolve_bit(const Tensor& tensor) override {
    return at::resolve_conj(tensor);
  }
  Tensor& math_op_(Tensor& tensor) override {
    return at::conj_physical_(tensor);
  }
};

void conjugateFallback(const c10::OperatorHandle& op, DispatchKeySet dispatch_keys, torch::jit::Stack* stack) {
  ConjFallback object;
  object.fallback_impl(op, dispatch_keys, stack);
}

TORCH_LIBRARY_IMPL(_, Conjugate, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&conjugateFallback>());
}

TORCH_LIBRARY_IMPL(aten, Conjugate, m) {
  m.impl("requires_grad_", torch::CppFunction::makeFallthrough());
  m.impl("set_.source_Storage_storage_offset", torch::CppFunction::makeFallthrough());
  m.impl("set_.source_Tensor", torch::CppFunction::makeFallthrough());
  m.impl("set_", torch::CppFunction::makeFallthrough());
  m.impl("copy_", torch::CppFunction::makeFallthrough());
  m.impl("clone", torch::CppFunction::makeFallthrough());
  m.impl("conj", torch::CppFunction::makeFallthrough());
  m.impl("_conj", torch::CppFunction::makeFallthrough());
  m.impl("_conj_physical", torch::CppFunction::makeFallthrough());
  m.impl("conj_physical", torch::CppFunction::makeFallthrough());
  m.impl("conj_physical_", torch::CppFunction::makeFallthrough());
  m.impl("resolve_conj", torch::CppFunction::makeFallthrough());
  m.impl("empty_like", torch::CppFunction::makeFallthrough());
  m.impl("empty.memory_format", torch::CppFunction::makeFallthrough());
  m.impl("empty.out", torch::CppFunction::makeFallthrough());
  m.impl("empty_strided", torch::CppFunction::makeFallthrough());
  m.impl("full_like", torch::CppFunction::makeFallthrough());
  m.impl("stride.int", torch::CppFunction::makeFallthrough());
  m.impl("stride.Dimname", torch::CppFunction::makeFallthrough());
  m.impl("size.int", torch::CppFunction::makeFallthrough());
  m.impl("size.Dimname", torch::CppFunction::makeFallthrough());
  m.impl("is_complex", torch::CppFunction::makeFallthrough());
  m.impl("_unsafe_view", torch::CppFunction::makeFallthrough());
  m.impl("dot", torch::CppFunction::makeFallthrough());
  m.impl("vdot", torch::CppFunction::makeFallthrough());
  m.impl("dot.out", torch::CppFunction::makeFallthrough());
  m.impl("vdot.out", torch::CppFunction::makeFallthrough());
  m.impl("alias", torch::CppFunction::makeFallthrough());
  m.impl("mm", torch::CppFunction::makeFallthrough());
  m.impl("mm.out", torch::CppFunction::makeFallthrough());
  m.impl("addmm", torch::CppFunction::makeFallthrough());
  m.impl("addmm_", torch::CppFunction::makeFallthrough());
  m.impl("addmm.out", torch::CppFunction::makeFallthrough());
  m.impl("bmm", torch::CppFunction::makeFallthrough());
  m.impl("bmm.out", torch::CppFunction::makeFallthrough());
  m.impl("baddbmm", torch::CppFunction::makeFallthrough());
  m.impl("baddbmm_", torch::CppFunction::makeFallthrough());
  m.impl("baddbmm.out", torch::CppFunction::makeFallthrough());
    // views and their in-place version ops
  m.impl("as_strided", torch::CppFunction::makeFallthrough());
  m.impl("as_strided_", torch::CppFunction::makeFallthrough());
  m.impl("detach", torch::CppFunction::makeFallthrough());
  m.impl("detach_", torch::CppFunction::makeFallthrough());
  m.impl("diagonal", torch::CppFunction::makeFallthrough());
  m.impl("expand", torch::CppFunction::makeFallthrough());
  m.impl("expand_as", torch::CppFunction::makeFallthrough());
  m.impl("movedim.int", torch::CppFunction::makeFallthrough());
  m.impl("movedim.intlist", torch::CppFunction::makeFallthrough());
  m.impl("narrow", torch::CppFunction::makeFallthrough());
  m.impl("permute", torch::CppFunction::makeFallthrough());
  m.impl("select.Dimname", torch::CppFunction::makeFallthrough());
  m.impl("select.int", torch::CppFunction::makeFallthrough());
  m.impl("squeeze", torch::CppFunction::makeFallthrough());
  m.impl("squeeze_", torch::CppFunction::makeFallthrough());
  m.impl("transpose.int", torch::CppFunction::makeFallthrough());
  m.impl("transpose.Dimname", torch::CppFunction::makeFallthrough());
  m.impl("transpose_", torch::CppFunction::makeFallthrough());
  m.impl("t", torch::CppFunction::makeFallthrough());
  m.impl("t_", torch::CppFunction::makeFallthrough());
  m.impl("real", torch::CppFunction::makeFallthrough());
  m.impl("imag", torch::CppFunction::makeFallthrough());
  m.impl("view_as_real", torch::CppFunction::makeFallthrough());
  m.impl("unflatten.int", torch::CppFunction::makeFallthrough());
  m.impl("unflatten.Dimname", torch::CppFunction::makeFallthrough());
  m.impl("unfold", torch::CppFunction::makeFallthrough());
  m.impl("unsqueeze", torch::CppFunction::makeFallthrough());
  m.impl("unsqueeze_", torch::CppFunction::makeFallthrough());
  m.impl("view", torch::CppFunction::makeFallthrough());
  m.impl("view_as", torch::CppFunction::makeFallthrough());
  m.impl("unbind.int", torch::CppFunction::makeFallthrough());
  m.impl("unbind.Dimname", torch::CppFunction::makeFallthrough());
  m.impl("split.Tensor", torch::CppFunction::makeFallthrough());
  m.impl("split_with_sizes", torch::CppFunction::makeFallthrough());
  m.impl("swapaxes", torch::CppFunction::makeFallthrough());
  m.impl("swapdims", torch::CppFunction::makeFallthrough());
  m.impl("chunk", torch::CppFunction::makeFallthrough());
  m.impl("reshape", torch::CppFunction::makeFallthrough());
  m.impl("hsplit.int", torch::CppFunction::makeFallthrough());
  m.impl("hsplit.array", torch::CppFunction::makeFallthrough());
  m.impl("dsplit.int", torch::CppFunction::makeFallthrough());
  m.impl("dsplit.array", torch::CppFunction::makeFallthrough());
  m.impl("vsplit.int", torch::CppFunction::makeFallthrough());
  m.impl("vsplit.array", torch::CppFunction::makeFallthrough());
}

} // namespace at
