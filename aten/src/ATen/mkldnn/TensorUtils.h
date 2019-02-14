#pragma once

#include <ATen/native/utils/ParamsHash.h>
#include <ATen/mkldnn/Runtime.h>

namespace at { namespace native {

using namespace mkldnn;

/* mkldnn memory descriptor */
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

/* mkldnn memory handle */
template <typename T>
inline void _set_memory_handle(const T& self, const T& other) {
  self.set_data_handle(other.get_data_handle());
}

template <typename T, typename ... Ts>
inline void _set_memory_handle(const T& self, const T& other, const Ts& ... rest) {
  self.set_data_handle(other.get_data_handle());
  _set_memory_handle(rest ...);
}

/* mkldnn memory create/reorder method */
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

/* mkldnn primitve cache */
template <typename key_t, typename prim_t>
struct PrimitiveCache {
  using prim_handle_t = std::shared_ptr<prim_t>;
  std::unordered_map<key_t, prim_handle_t, ParamsHash<key_t>, ParamsEqual<key_t>> map;
  prim_handle_t prim_;

  bool find(const key_t& params, prim_handle_t& results) {
    auto it = map.find(params);
    if (it == map.end()) {
      return false;
    }
    results = it->second;
    return true;
  }

  void insert(const key_t& params, const prim_handle_t& results) {
    map.insert(std::pair<key_t, prim_handle_t>(params, results));
  }

  template <typename F1, typename F2>
  prim_t& get(const key_t& params, const F1& fn_new, const F2& fn_set) {
    if (find(params, prim_)) {
      fn_set(prim_);
    } else {
      fn_new(prim_);
      fn_set(prim_);
      insert(params, prim_);
    }
    return *prim_;
  }
};

}} // namespace at::native
