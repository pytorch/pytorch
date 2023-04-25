#pragma once

namespace at {
// views and their in-place version ops
#define TORCH_VIEW_FNS(m) \
  m.impl("as_strided_", torch::CppFunction::makeFallthrough()); \
  m.impl("detach", torch::CppFunction::makeFallthrough()); \
  m.impl("detach_", torch::CppFunction::makeFallthrough()); \
  m.impl("diagonal", torch::CppFunction::makeFallthrough()); \
  m.impl("expand", torch::CppFunction::makeFallthrough()); \
  m.impl("expand_as", torch::CppFunction::makeFallthrough()); \
  m.impl("movedim.int", torch::CppFunction::makeFallthrough()); \
  m.impl("movedim.intlist", torch::CppFunction::makeFallthrough()); \
  m.impl("narrow", torch::CppFunction::makeFallthrough()); \
  m.impl("permute", torch::CppFunction::makeFallthrough()); \
  m.impl("select.Dimname", torch::CppFunction::makeFallthrough()); \
  m.impl("select.int", torch::CppFunction::makeFallthrough()); \
  m.impl("squeeze", torch::CppFunction::makeFallthrough()); \
  m.impl("squeeze_", torch::CppFunction::makeFallthrough()); \
  m.impl("transpose.int", torch::CppFunction::makeFallthrough()); \
  m.impl("transpose.Dimname", torch::CppFunction::makeFallthrough()); \
  m.impl("transpose_", torch::CppFunction::makeFallthrough()); \
  m.impl("t", torch::CppFunction::makeFallthrough()); \
  m.impl("t_", torch::CppFunction::makeFallthrough()); \
  m.impl("real", torch::CppFunction::makeFallthrough()); \
  m.impl("imag", torch::CppFunction::makeFallthrough()); \
  m.impl("view_as_real", torch::CppFunction::makeFallthrough()); \
  m.impl("unflatten.int", torch::CppFunction::makeFallthrough()); \
  m.impl("unflatten.Dimname", torch::CppFunction::makeFallthrough()); \
  m.impl("unfold", torch::CppFunction::makeFallthrough()); \
  m.impl("unsqueeze", torch::CppFunction::makeFallthrough()); \
  m.impl("unsqueeze_", torch::CppFunction::makeFallthrough()); \
  m.impl("view_as", torch::CppFunction::makeFallthrough()); \
  m.impl("unbind.int", torch::CppFunction::makeFallthrough()); \
  m.impl("unbind.Dimname", torch::CppFunction::makeFallthrough()); \
  m.impl("split.Tensor", torch::CppFunction::makeFallthrough()); \
  m.impl("split_with_sizes", torch::CppFunction::makeFallthrough()); \
  m.impl("swapaxes", torch::CppFunction::makeFallthrough()); \
  m.impl("swapdims", torch::CppFunction::makeFallthrough()); \
  m.impl("chunk", torch::CppFunction::makeFallthrough()); \
  m.impl("reshape", torch::CppFunction::makeFallthrough()); \
  m.impl("alias", torch::CppFunction::makeFallthrough()); \
  m.impl("hsplit.int", torch::CppFunction::makeFallthrough()); \
  m.impl("hsplit.array", torch::CppFunction::makeFallthrough()); \
  m.impl("dsplit.int", torch::CppFunction::makeFallthrough()); \
  m.impl("dsplit.array", torch::CppFunction::makeFallthrough()); \
  m.impl("vsplit.int", torch::CppFunction::makeFallthrough()); \
  m.impl("vsplit.array", torch::CppFunction::makeFallthrough()); \
  m.impl("conj", torch::CppFunction::makeFallthrough()); \
  m.impl("_conj", torch::CppFunction::makeFallthrough()); \
  m.impl("_unsafe_view", torch::CppFunction::makeFallthrough()); \
  m.impl("resize_", torch::CppFunction::makeFallthrough());

#define TENSOR_UTILITIES_AND_CONSTRUCTORS(m) \
  m.impl("empty_like", torch::CppFunction::makeFallthrough()); \
  m.impl("empty.memory_format", torch::CppFunction::makeFallthrough()); \
  m.impl("empty.out", torch::CppFunction::makeFallthrough()); \
  m.impl("empty_strided", torch::CppFunction::makeFallthrough()); \
  m.impl("full_like", torch::CppFunction::makeFallthrough()); \
  m.impl("stride.int", torch::CppFunction::makeFallthrough()); \
  m.impl("stride.Dimname", torch::CppFunction::makeFallthrough()); \
  m.impl("size.int", torch::CppFunction::makeFallthrough()); \
  m.impl("size.Dimname", torch::CppFunction::makeFallthrough()); \
  m.impl("is_complex", torch::CppFunction::makeFallthrough()); \
  m.impl("is_floating_point", torch::CppFunction::makeFallthrough()); \
  m.impl("requires_grad_", torch::CppFunction::makeFallthrough());
}

#define TORCH_VIEW_FNS_NATIVE_FN_REGISTRATION(m) \
  m.impl("as_strided", torch::CppFunction::makeFallthrough()); \
  m.impl("view", torch::CppFunction::makeFallthrough());
