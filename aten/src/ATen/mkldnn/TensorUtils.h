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

inline memory null_memory() {
  auto _engine = MKLDNNEngine::Instance().get_engine();
  return null_memory(_engine);
}

inline memory _new_mkldnn_memory(const memory::primitive_desc& pd, const Tensor& tensor) {
  return memory(pd, tensor.data_ptr());
}

inline memory _new_mkldnn_memory(const memory::primitive_desc& pd) {
  return memory(pd);
}

inline memory _reorder(const memory& from, const memory::primitive_desc& pd) {
  if (pd == from.get_primitive_desc()) {
    return from;
  }
  auto to = memory(pd);
  MKLDNN_EXEC(reorder(from, to));
  return to;
}

inline void  _reorder(const memory& from, const memory& to) {
  if (from != to) {
    MKLDNN_EXEC(reorder(from, to));
  }
}

}} // namespace at::native
