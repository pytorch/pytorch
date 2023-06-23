#pragma once
#include <c10/core/impl/TorchDispatchModeTLS.h>
#include <torch/csrc/autograd/engine.h>
#include <torch/csrc/utils/python_stub.h>
#include <iostream>
#include <typeindex>
#include <vector>

namespace torch {
namespace autograd {
namespace generated {
struct TypeAndSize;
}

struct CacheKeyBuffer {
  CacheKeyBuffer() : data(nullptr) {}
  CacheKeyBuffer(const char* key, uint16_t len) : data(new char[len]) {
    memcpy(data, key, len);
  }
  CacheKeyBuffer(CacheKeyBuffer&& other)
      : data(std::exchange(other.data, nullptr)) {}
  ~CacheKeyBuffer() {
    if (data != nullptr) {
      delete[] data;
    }
  }
  CacheKeyBuffer(const CacheKeyBuffer& other) = delete;
  CacheKeyBuffer& operator=(const CacheKeyBuffer&) = delete;

  char* data;
};

struct CacheKey {
  CacheKey(const std::type_index& ntype, const char* key, uint16_t len)
      : node_type(ntype), key_size(len), key(key) {}

  bool operator<(const CacheKey& other) const {
    if (node_type != other.node_type)
      return node_type < other.node_type;
    if (key_size != other.key_size)
      return key_size < other.key_size;
    return memcmp(key, other.key, key_size) < 0;
  }

  bool operator==(const CacheKey& other) const {
    return node_type == other.node_type && key_size == other.key_size &&
        memcmp(key, other.key, key_size) == 0;
  }

  size_t hash() const {
    // don't bother hashing the key data, common case 1 cache entry per node
    return std::hash<std::type_index>()(node_type) ^ key_size;
  }

  std::type_index node_type;
  uint16_t key_size;
  const char* key;
};

struct OutputRef {
  OutputRef() : node_id(-1), index(-1) {}
  OutputRef(int node_, int index_) : node_id(node_), index(index_) {}

  bool is_set() const {
    return this->node_id >= 0;
  }

  int node_id;
  int index;
};

struct NodeCall {
  NodeCall(const std::shared_ptr<Node>& node_)
      : node(node_), input_refs(node_->num_inputs()) {}

  NodeCall(const std::shared_ptr<Node>& node_, int num_inputs_)
      : node(node_), input_refs(num_inputs_) {}

  OutputRef& operator[](size_t pos) {
    return input_refs[pos];
  }

  size_t size() const {
    return input_refs.size();
  }

  void mark_output(int input_nr, int output_idx) {
    graph_output.emplace_back(std::make_pair(input_nr, output_idx));
  }

  std::shared_ptr<Node> node;
  // at::SmallVector??
  std::vector<OutputRef> input_refs;

  // borrowed references
  std::vector<std::pair<int, int>> tensor_pre_hooks;
  std::vector<int> pre_hooks;
  std::vector<int> post_hooks;
  std::vector<std::pair<int, int>> graph_output;
};

struct AutogradCompilerCall {
  AutogradCompilerCall(bool accumulate_grad_)
      : accumulate_grad(accumulate_grad_) {}

  void add_tensor_input(const at::Tensor& tensor) {
    inputs.emplace_back(tensor);
  }

  void add_set_grad_target(const at::Tensor& tensor) {
    set_grad_targets.emplace_back(tensor);
  }

  void add_size_input(const c10::SymInt& s) {
    all_size_inputs.emplace_back(s.expect_int());
  }

  int emplace_hook(c10::SafePyObject&& fn) {
    hooks.emplace_back(std::move(fn));
    return hooks.size() - 1;
  }

  std::vector<int64_t> all_size_inputs;
  std::vector<int64_t> dyn_size_inputs;
  std::vector<at::Tensor> inputs;
  std::vector<at::Tensor> set_grad_targets;
  std::vector<c10::SafePyObject> hooks;
  bool accumulate_grad;
};

class CompiledNodeArgs {
 public:
  CompiledNodeArgs(AutogradCompilerCall& compiler, NodeCall& node_call)
      : _compiler(compiler),
        _node_call(node_call),
        _specialization_key_size(0) {}

  void collect(const at::Tensor& t) {
    collect(t.defined());
    if (t.defined()) {
      _compiler.add_tensor_input(t);
    }
  }

  void collect(const SavedVariable& t) {
    collect(t.unpack(_node_call.node));
  }

  void collect(const c10::SymInt& t) {
    _compiler.add_size_input(t);
  }

  template <typename T>
  void collect(const std::vector<T>& t) {
    collect_size(t.size());
    for (size_t i = 0; i < t.size(); ++i) {
      collect(t[i]);
    }
  }

  template <typename T>
  void collect(const c10::ArrayRef<T>& t) {
    collect_size(t.size());
    for (size_t i = 0; i < t.size(); ++i) {
      collect(t[i]);
    }
  }

  template <typename T>
  void collect(const c10::OptionalArray<T>& t) {
    collect(t.list);
  }

  template <typename T>
  void collect(const c10::optional<T>& t) {
    specialize_on_bytes(t.has_value());
    if (t.has_value()) {
      collect(*t);
    }
  }

  template <typename A, typename B>
  void collect(const std::pair<A, B>& t) {
    collect(t.first);
    collect(t.second);
  }

  void collect(const c10::ScalarType& t) {
    specialize_on_bytes(t);
  }

  void collect(const c10::Scalar& t) {
    auto type = t.type();
    specialize_on_bytes(type);
    if (type == c10::ScalarType::Double) {
      collect(t.toDouble());
    } else if (type == c10::ScalarType::Long) {
      collect(t.toLong());
    } else if (type == c10::ScalarType::Bool) {
      collect(t.toBool());
    } else if (type == c10::ScalarType::ComplexDouble) {
      auto c = t.toComplexDouble();
      collect(c.real());
      collect(c.imag());
    } else {
      TORCH_CHECK(false);
    }
  }

  void collect(const c10::TensorOptions& t) {
    // TODO(jansel): are there any pointers in this type we shouldn't memcmp
    // I think this one is wrong.... should fix it
    specialize_on_bytes(t);
  }

  void collect(const at::TensorGeometry& t) {
    collect(t.sym_sizes());
    collect(t.sym_strides());
    collect(t.sym_storage_offset());
  }

  void collect(torch::autograd::generated::TypeAndSize& t);

  void collect(const OutputRef& t) {
    // +1 is for undefined encoded as -1
    collect_size(t.node_id + 1);
    collect_size(t.index + 1);
  }

  void collect(const NodeCall& t) {
    collect(t.input_refs);
    collect(t.graph_output);
  }

  void collect(int8_t t) {
    specialize_on_bytes(t);
  }

  void collect(int16_t t) {
    specialize_on_bytes(t);
  }

  void collect(int32_t t) {
    specialize_on_bytes(t);
  }

  void collect(int64_t t) {
    specialize_on_bytes(t);
  }

  void collect(uint8_t t) {
    specialize_on_bytes(t);
  }

  void collect(uint16_t t) {
    specialize_on_bytes(t);
  }

  void collect(uint32_t t) {
    specialize_on_bytes(t);
  }

  void collect(uint64_t t) {
    specialize_on_bytes(t);
  }

  void collect(bool t) {
    specialize_on_bytes(t);
  }

  void collect(float t) {
    specialize_on_bytes(t);
  }

  void collect(double t) {
    specialize_on_bytes(t);
  }

  void set_grad_target(const at::Tensor& tensor) {
    collect(_compiler.accumulate_grad);
    if (_compiler.accumulate_grad)
      _compiler.add_set_grad_target(tensor);
  }

  void collect_hooks_from(Node* fn) {
    for (auto& i : fn->tensor_pre_hooks()) {
      i->compiled_args(*this);
    }
    for (auto& i : fn->retains_grad_hooks()) {
      i.second->compiled_args(*this);
    }
    for (auto& i : fn->pre_hooks()) {
      i->compiled_args(*this);
    }
    for (auto& i : fn->post_hooks()) {
      i->compiled_args(*this);
    }
    specialize_on_hook_counts();
  }

  void specialize_on_hook_counts() {
    collect_size(_node_call.tensor_pre_hooks.size());
    for (const auto& h : _node_call.tensor_pre_hooks) {
      collect_size(h.second);
    }
    collect_size(_node_call.pre_hooks.size());
    collect_size(_node_call.post_hooks.size());
  }

  CacheKey key() const {
    Node* node = _node_call.node.get();
    return CacheKey(
        typeid(*node), _specialization_key, _specialization_key_size);
  }

  void add_tensor_pre_hook(c10::SafePyObject&& obj, int index) {
    auto fn_id = _compiler.emplace_hook(std::move(obj));
    collect_size(fn_id);
    _node_call.tensor_pre_hooks.emplace_back(std::make_pair(fn_id, index));
  }

  void add_pre_hook(c10::SafePyObject&& obj) {
    auto fn_id = _compiler.emplace_hook(std::move(obj));
    collect_size(fn_id);
    _node_call.pre_hooks.emplace_back(fn_id);
  }

  void add_post_hook(c10::SafePyObject&& obj) {
    auto fn_id = _compiler.emplace_hook(std::move(obj));
    collect_size(fn_id);
    _node_call.post_hooks.emplace_back(fn_id);
  }

  void collect_size(size_t s) {
    // we expect sizes to be small, so try to cram them into a single byte
    // shorter keys, means faster caches
    constexpr auto max = std::numeric_limits<uint8_t>::max();
    if (C10_UNLIKELY(s >= max - 2)) {
      if (s <= std::numeric_limits<uint16_t>::max()) {
        // 3 bytes
        specialize_on_bytes(static_cast<uint8_t>(max - 2));
        specialize_on_bytes(static_cast<uint16_t>(s));
      } else if (s <= std::numeric_limits<uint32_t>::max()) {
        // 5 bytes
        specialize_on_bytes(static_cast<uint8_t>(max - 1));
        specialize_on_bytes(static_cast<uint32_t>(s));
      } else {
        // 9 bytes
        specialize_on_bytes(static_cast<uint8_t>(max));
        specialize_on_bytes(s);
      }
    } else {
      // happy case, 1 byte
      specialize_on_bytes(static_cast<uint8_t>(s));
    }
  }

 protected:
  template <typename T>
  void specialize_on_bytes(const T& t) {
    TORCH_CHECK(
        _specialization_key_size + sizeof(T) <= sizeof(_specialization_key));
    memcpy(_specialization_key + _specialization_key_size, &t, sizeof(T));
    _specialization_key_size += sizeof(T);
  }

 private:
  AutogradCompilerCall& _compiler;
  NodeCall& _node_call;
  uint16_t _specialization_key_size;
  char _specialization_key[512];
};

struct TraceState {
  TraceState(
      const variable_list& pi,
      const std::vector<c10::optional<c10::SymInt>>& ss,
      bool accumulate_grad_,
      size_t num_outputs)
      : proxy_inputs_index(0),
        sym_sizes_index(0),
        proxy_inputs(pi),
        sym_sizes(ss),
        outputs(num_outputs),
        accumulate_grad(accumulate_grad_) {}

  ~TraceState() {
    if (C10_UNLIKELY(proxy_inputs_index != proxy_inputs.size())) {
      TORCH_WARN("not all proxy_inputs consumed")
    }
    if (C10_UNLIKELY(sym_sizes_index != sym_sizes.size())) {
      TORCH_WARN("not all sym_sizes consumed")
    }
  }

  const at::Tensor& next_proxy_input() {
    TORCH_CHECK(proxy_inputs_index < proxy_inputs.size());
    return proxy_inputs[proxy_inputs_index++];
  }

  c10::optional<c10::SymInt> next_sym_size() {
    TORCH_CHECK(sym_sizes_index < sym_sizes.size());
    return sym_sizes[sym_sizes_index++];
  }

  size_t proxy_inputs_index;
  size_t sym_sizes_index;
  variable_list proxy_inputs;
  std::vector<c10::optional<c10::SymInt>> sym_sizes;
  variable_list outputs;
  bool accumulate_grad;
};

class SwapSavedVariables {
 public:
  void before(at::Tensor& t) {
    stashed_tensors.emplace_back(t);
    if (t.defined()) {
      t = state.next_proxy_input();
    }
  }

  void after(at::Tensor& t) {
    t = stashed_tensors[stashed_tensors_index++];
  }

  void before(SavedVariable& t) {
    c10::impl::DisableTorchDispatchModeGuard no_modes; // for unpack
    bool defined = t.unpack(node).defined();
    stashed_variables.emplace_back(std::move(t));
    if (defined) {
      t = SavedVariable(state.next_proxy_input(), false);
    }
  }

  void after(SavedVariable& t) {
    t = std::move(stashed_variables[stashed_variables_index++]);
  }

  void before(c10::SymInt& t) {
    stashed_symints.emplace_back(t);
    auto opt_value = state.next_sym_size();
    if (opt_value.has_value()) {
      t = *opt_value; // dynamic shape
    }
  }
  void after(c10::SymInt& t) {
    t = std::move(stashed_symints[stashed_symints_index++]);
  }

  void before(at::TensorGeometry& t) {
    before(t.mutable_sizes());
    before(t.mutable_strides());
    before(t.mutable_storage_offset());
    t.set_symbolic_sizes_strides(true);
    t.recompute_numel();
  }
  void after(at::TensorGeometry& t) {
    after(t.mutable_sizes());
    after(t.mutable_strides());
    after(t.mutable_storage_offset());
    t.set_symbolic_sizes_strides(false);
    t.recompute_numel();
  }
  void before(torch::autograd::generated::TypeAndSize& t);
  void after(torch::autograd::generated::TypeAndSize& t);

  template <typename T>
  void before(std::vector<T>& t) {
    for (size_t i = 0; i < t.size(); ++i) {
      before(t[i]);
    }
  }

  template <typename T>
  void after(std::vector<T>& t) {
    for (size_t i = 0; i < t.size(); ++i) {
      after(t[i]);
    }
  }

  template <typename T>
  void before(c10::ArrayRef<T>& t) {
    for (size_t i = 0; i < t.size(); ++i) {
      before(t[i]);
    }
  }

  template <typename T>
  void after(c10::ArrayRef<T>& t) {
    for (size_t i = 0; i < t.size(); ++i) {
      after(t[i]);
    }
  }

  template <typename T>
  void before(c10::OptionalArray<T>& t) {
    before(t.list);
  }

  template <typename T>
  void after(c10::OptionalArray<T>& t) {
    after(t.list);
  }

  template <typename T>
  void before(c10::optional<T>& t) {
    if (t.has_value()) {
      before(*t);
    }
  }

  template <typename T>
  void after(c10::optional<T>& t) {
    if (t.has_value()) {
      after(*t);
    }
  }

  void before(const c10::ScalarType& t) {}
  void before(const c10::Scalar& t) {}
  void before(const c10::TensorOptions& t) {}
  void before(int64_t t) {}
  void before(bool t) {}
  void before(double t) {}

  void after(const c10::ScalarType& t) {}
  void after(const c10::Scalar& t) {}
  void after(const c10::TensorOptions& t) {}
  void after(int64_t t) {}
  void after(bool t) {}
  void after(double t) {}

  void set_grad_value(const at::Tensor& tensor) {
    if (state.accumulate_grad)
      state.outputs.emplace_back(tensor);
  }

  SwapSavedVariables(TraceState& s, const std::shared_ptr<Node> n)
      : state(s),
        node(n),
        stashed_tensors_index(0),
        stashed_variables_index(0),
        stashed_symints_index(0) {}

  ~SwapSavedVariables() {
    if (C10_UNLIKELY(
            stashed_tensors_index != stashed_tensors.size() ||
            stashed_variables_index != stashed_variables.size())) {
      TORCH_WARN("not all stashed_tensors consumed")
    }
  }

  TraceState& state;
  std::shared_ptr<Node> node;

  size_t stashed_tensors_index;
  size_t stashed_variables_index;
  size_t stashed_symints_index;
  std::vector<at::Tensor> stashed_tensors;
  std::vector<SavedVariable> stashed_variables;
  std::vector<c10::SymInt> stashed_symints;
};

} // namespace autograd
} // namespace torch

template <>
struct std::hash<torch::autograd::CacheKey> {
  size_t operator()(const torch::autograd::CacheKey& k) const {
    return k.hash();
  }
};
