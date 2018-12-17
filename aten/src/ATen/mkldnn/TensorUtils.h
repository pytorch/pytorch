#pragma once

#include <ATen/mkldnn/Runtime.h>

namespace at { namespace native {

using namespace mkldnn;

inline memory::desc _format_md(const memory::dims& _dims, const memory::format& _format) {
  return memory::desc(_dims, memory::data_type::f32, _format);
}

inline memory::desc _generic_md(const memory::dims& _dims) {
  return _format_md(_dims, memory::format::any);
}

inline memory::primitive_desc _primitive_md(const memory::dims& _dims, const memory::format& _format) {
  auto _engine = MKLDNNEngine::Instance().get_engine(); 
  return memory::primitive_desc(_format_md(_dims, _format), _engine);
}

}} // namespace at::native
