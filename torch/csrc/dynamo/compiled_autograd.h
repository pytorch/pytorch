#pragma once
#include <c10/core/impl/TorchDispatchModeTLS.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/csrc/autograd/engine.h>
#include <torch/csrc/utils/python_stub.h>
#include <torch/csrc/utils/torch_dispatch_mode.h>
#include <iostream>
#include <typeindex>
#include <vector>

// see [Note: Compiled Autograd]

namespace torch::dynamo::autograd {
using namespace torch::autograd;

struct SizeInput {
  // Note: int value is still needed when dynamic to pass as an arg
  enum DynType : uint8_t { STATIC = 0, DYNAMIC = 1 };
  SizeInput(DynType dt, int64_t v) : dyn_type(dt), value(v) {}
  DynType dyn_type;
  int64_t value;
};

struct CacheKeyBuffer {
  CacheKeyBuffer(const uint8_t* key, uint16_t len) : data(new uint8_t[len]) {
    std::memcpy(data.get(), key, len);
  }
  const uint8_t* get() const {
    return data.get();
  }

 private:
  std::unique_ptr<uint8_t[]> data;
};

struct CacheKey {
  // Key to find the next node in the shadow graph.  We use C++ RTTI for the
  // type of the node (ntype), then a key generated with a visitor pattern.
  CacheKey(const std::type_index& ntype, const uint8_t* key, uint16_t len)
      : node_type(ntype), key_size(len), key(key) {}

  bool operator<(const CacheKey& other) const {
    if (node_type != other.node_type) {
      return node_type < other.node_type;
    }
    if (key_size != other.key_size) {
      return key_size < other.key_size;
    }
    return std::memcmp(key, other.key, key_size) < 0;
  }

  bool operator==(const CacheKey& other) const {
    return node_type == other.node_type && key_size == other.key_size &&
        std::memcmp(key, other.key, key_size) == 0;
  }

  size_t hash() const {
    // don't bother hashing the key data, common case 1 cache entry per node
    return std::hash<std::type_index>()(node_type) ^ key_size;
  }

  std::type_index node_type;
  uint16_t key_size;
  const uint8_t* key;
};

struct NodeCall {
  NodeCall(uint32_t id_, std::shared_ptr<Node> node_)
      : id(id_), node(std::move(node_)) {}

  void mark_output(int input_nr, int output_idx) {
    graph_output.emplace_back(std::make_pair(input_nr, output_idx));
  }

  uint32_t id;
  std::shared_ptr<Node> node;
  std::vector<std::pair<int, int>> tensor_pre_hooks;
  std::vector<int> pre_hooks;
  std::vector<int> post_hooks;
  std::vector<std::pair<int, int>> graph_output;
  bool needed = true;
};

struct NodeCalls : public std::unordered_map<Node*, NodeCall> {
  NodeCall& lookup(const std::shared_ptr<Node>& function) {
    auto it = find(function.get());
    if (it == end()) {
      it = emplace(function.get(), NodeCall(_next_id++, function)).first;
    }
    return it->second;
  }

 private:
  uint32_t _next_id = 0;
};

struct AutogradCompilerCall {
  AutogradCompilerCall(bool accumulate_grad_)
      : accumulate_grad(accumulate_grad_),
        default_dyn_type(SizeInput::STATIC) {}

  void add_tensor_input(const at::Tensor& tensor) {
    inputs.emplace_back(tensor);
  }

  void add_set_grad_target(const at::Tensor& tensor) {
    set_grad_targets.emplace_back(tensor);
  }

  void add_size_input(const c10::SymInt& s) {
    all_size_inputs.emplace_back(SizeInput(default_dyn_type, s.expect_int()));
  }

  int emplace_hook(c10::SafePyObject&& fn) {
    hooks.emplace_back(std::move(fn));
    return hooks.size() - 1;
  }

  std::vector<SizeInput> all_size_inputs;
  std::vector<int64_t> dyn_size_inputs;
  std::vector<at::Tensor> inputs;
  std::vector<at::Tensor> set_grad_targets;
  std::vector<c10::SafePyObject> hooks;
  NodeCalls node_calls;
  bool accumulate_grad;
  SizeInput::DynType default_dyn_type;
};

class CompiledNodeArgs {
  // CompiledNodeArgs builds a representation of the constant values found
  // across all the nodes in the compiled graph, via 'collect' overloads. The
  // collected constants are specialized on by concatenation into a cache key.
  // Tensor, symint arguments (which are lifted to become graph inputs rather
  // than specialized on) are forwarded to the compiler and not included in the
  // key.
 public:
  void collect(const at::Tensor& t) {
    if (cond(t.defined())) {
      _compiler.add_tensor_input(t);
    }
  }
  void collect(const SavedVariable& t) {
    TORCH_CHECK(
        !t.has_hooks(),
        "SavedVariable hooks not implemented in compiled autograd")
    collect(t.unpack(_node_call.node));
  }
  void collect(const c10::SymInt& t) {
    _compiler.add_size_input(t);
  }
  template <typename T>
  void collect(const std::vector<T>& t) {
    collect_size(t.size());
    for (const T& i : t) {
      collect(i);
    }
  }
  template <typename T>
  void collect(const c10::ArrayRef<T>& t) {
    collect_size(t.size());
    for (const T& i : t) {
      collect(i);
    }
  }
  template <typename T>
  void collect(const c10::OptionalArray<T>& t) {
    collect(t.list);
  }
  template <typename T>
  void collect(const c10::optional<T>& t) {
    if (cond(t.has_value())) {
      collect(*t);
    }
  }
  template <typename A, typename B>
  void collect(const std::pair<A, B>& t) {
    collect(t.first);
    collect(t.second);
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
      TORCH_INTERNAL_ASSERT(false);
    }
  }
  void collect(const c10::TensorOptions& t) {
    collect(t.device());
    collect(t.dtype());
    collect(t.layout());
    collect(t.requires_grad());
    collect(t.pinned_memory());
    collect(t.memory_format_opt());
  }
  void collect(const at::TensorGeometry& t) {
    collect(t.sym_sizes());
    collect(t.sym_strides());
    collect(t.sym_storage_offset());
  }
  void collect(const torch::autograd::TypeAndSize& t) {
    collect(t.sym_sizes);
    collect(t.options);
  }
  void collect(const c10::Device& t) {
    collect(t.type());
    collect(t.index());
  }
  void collect(const std::string& t) {
    collect_size(t.size());
    for (char c : t) {
      collect(c);
    }
  }
  void collect(const caffe2::TypeMeta& t) {
    specialize_on_bytes(t.id());
  }
  void collect(const NodeCall& t) {
    collect_size(t.id);
    collect(t.graph_output);
    collect_hooks_from(t.node.get());
  }
  void collect(const Edge& t) {
    if (cond(t.is_valid())) {
      collect_size(_compiler.node_calls.lookup(t.function).id);
      collect_size(t.input_nr);
      collect(t.function->input_metadata(t.input_nr)); // for validate_outputs
    }
  }
  void collect(const InputMetadata& t) {
    TORCH_CHECK(!t.is_nested_tensor(), "NestedTensor not implemented");
    collect(t.options());
    collect(t.is_tensor_subclass());
    collect(t.shape_as_dim_vector());
  }
  void collect(const VariableInfo& t) {
    collect(t.layout);
    collect(t.device);
    collect(t.scalar_type);
    collect(t.size);
    collect(t.requires_grad);
    collect(t.is_empty);
  }
  bool cond(bool cond) {
    collect(cond);
    return cond;
  }

#define COLLECT_AS_BYTES(T) \
  void collect(T t) {       \
    specialize_on_bytes(t); \
  }
  COLLECT_AS_BYTES(c10::ScalarType);
  COLLECT_AS_BYTES(c10::DeviceType);
  COLLECT_AS_BYTES(c10::Layout);
  COLLECT_AS_BYTES(c10::MemoryFormat);
  COLLECT_AS_BYTES(int8_t);
  COLLECT_AS_BYTES(int16_t);
  COLLECT_AS_BYTES(int32_t);
  COLLECT_AS_BYTES(int64_t);
  COLLECT_AS_BYTES(uint8_t);
  COLLECT_AS_BYTES(uint16_t);
  COLLECT_AS_BYTES(uint32_t);
  COLLECT_AS_BYTES(uint64_t);
  COLLECT_AS_BYTES(bool);
  COLLECT_AS_BYTES(float);
  COLLECT_AS_BYTES(double);
#undef COLLECT_AS_BYTES

  void set_grad_target(const at::Tensor& tensor) {
    collect(_compiler.accumulate_grad);
    if (_compiler.accumulate_grad)
      _compiler.add_set_grad_target(tensor);
  }

  void collect_hooks_from(Node* fn) {
    TORCH_CHECK(
        fn->retains_grad_hooks().empty(),
        "retains_grad_hooks not implemented for compiled autograd");
    for (auto& i : fn->tensor_pre_hooks()) {
      i->compiled_args(*this);
    }
    for (auto& i : fn->pre_hooks()) {
      i->compiled_args(*this);
    }
    for (auto& i : fn->post_hooks()) {
      i->compiled_args(*this);
    }
    collect_size(_node_call.tensor_pre_hooks.size());
    collect_size(_node_call.pre_hooks.size());
    collect_size(_node_call.post_hooks.size());
    for (const auto& h : _node_call.tensor_pre_hooks) {
      collect_size(h.second); // index
    }
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
    constexpr uint8_t encode_as_u64 = std::numeric_limits<uint8_t>::max();
    constexpr uint8_t encode_as_u32 = encode_as_u64 - 1;
    constexpr uint8_t encode_as_u16 = encode_as_u64 - 2;
    if (C10_UNLIKELY(s >= encode_as_u16)) {
      // first write a byte indicating the path we followed, then the data
      if (s <= std::numeric_limits<uint16_t>::max()) {
        // 3 bytes
        specialize_on_bytes(encode_as_u16);
        specialize_on_bytes(static_cast<uint16_t>(s));
      } else if (s <= std::numeric_limits<uint32_t>::max()) {
        // 5 bytes
        specialize_on_bytes(encode_as_u32);
        specialize_on_bytes(static_cast<uint32_t>(s));
      } else {
        // 9 bytes
        specialize_on_bytes(encode_as_u64);
        specialize_on_bytes(s);
      }
    } else {
      // happy case, 1 byte
      specialize_on_bytes(static_cast<uint8_t>(s));
    }
  }

  SizeInput::DynType set_default_dyn_type(SizeInput::DynType default_dyn_type) {
    return std::exchange(_compiler.default_dyn_type, default_dyn_type);
  }

  CompiledNodeArgs(AutogradCompilerCall& compiler, NodeCall& node_call)
      : _compiler(compiler),
        _node_call(node_call),
        _specialization_key_size(0),
        _specialization_key_storage(1024),
        _specialization_key(
            (uint8_t*)std::malloc(_specialization_key_storage)) {}
  ~CompiledNodeArgs() {
    std::free(_specialization_key);
  }
  CompiledNodeArgs(const CompiledNodeArgs&) = delete;

 private:
  template <typename T>
  void specialize_on_bytes(const T& t) {
    while (C10_UNLIKELY(
        _specialization_key_size + sizeof(T) > _specialization_key_storage)) {
      _specialization_key_storage *= 2;
      _specialization_key = (uint8_t*)std::realloc(
          _specialization_key, _specialization_key_storage);
    }
    std::memcpy(_specialization_key + _specialization_key_size, &t, sizeof(T));
    _specialization_key_size += sizeof(T);
  }

  AutogradCompilerCall& _compiler;
  NodeCall& _node_call;
  size_t _specialization_key_size;
  size_t _specialization_key_storage;
  uint8_t* _specialization_key;
};

struct TraceState {
  TraceState(
      const variable_list& proxy_inputs_,
      const std::vector<c10::optional<c10::SymInt>>& ss,
      bool accumulate_grad_,
      size_t num_outputs)
      : proxy_inputs_index(0),
        sym_sizes_index(0),
        proxy_inputs(proxy_inputs_),
        sym_sizes(ss),
        outputs(num_outputs),
        accumulate_grad(accumulate_grad_) {}
  void debug_asserts() {
    TORCH_INTERNAL_ASSERT(proxy_inputs_index == proxy_inputs.size());
    TORCH_INTERNAL_ASSERT(sym_sizes_index == sym_sizes.size());
  }
  const at::Tensor& next_proxy_input() {
    TORCH_INTERNAL_ASSERT(proxy_inputs_index < proxy_inputs.size());
    return proxy_inputs[proxy_inputs_index++];
  }
  c10::optional<c10::SymInt> next_sym_size() {
    TORCH_INTERNAL_ASSERT(sym_sizes_index < sym_sizes.size());
    return sym_sizes[sym_sizes_index++];
  }

  size_t proxy_inputs_index;
  size_t sym_sizes_index;
  variable_list proxy_inputs;
  std::vector<c10::optional<c10::SymInt>> sym_sizes;
  variable_list outputs;
  bool accumulate_grad;
};

#define SWAP_SAVED_VARIABLES_SAVE(mapping, var, move) \
  bool inserted = mapping.emplace(&var, move).second; \
  TORCH_INTERNAL_ASSERT(inserted, "duplicate before()");
#define SWAP_SAVED_VARIABLES_RESTORE(mapping, var)                 \
  auto it = mapping.find(&var);                                    \
  TORCH_INTERNAL_ASSERT(it != mapping.end(), "duplicate after()"); \
  var = std::move(it->second);                                     \
  mapping.erase(it);

class SwapSavedVariables {
  // SwapSavedVariables is used during the tracing/compilation phase after a
  // cache-miss. It swaps any 'lifted' inputs (tensors, symints) to proxy nodes,
  // allows tracing to happen, then swaps them back afterwards.
 public:
  void before(at::Tensor& t) {
    SWAP_SAVED_VARIABLES_SAVE(stashed_tensors, t, t);
    if (t.defined()) {
      t = state.next_proxy_input();
    }
  }
  void after(at::Tensor& t) {
    SWAP_SAVED_VARIABLES_RESTORE(stashed_tensors, t);
  }

  void before(SavedVariable& t) {
    torch::torch_dispatch_mode::StashTorchDispatchStackGuard
        no_modes; // for unpack
    bool defined = t.unpack(node).defined();
    SWAP_SAVED_VARIABLES_SAVE(stashed_variables, t, std::move(t));
    if (defined) {
      t = SavedVariable(state.next_proxy_input(), false);
    }
  }
  void after(SavedVariable& t) {
    SWAP_SAVED_VARIABLES_RESTORE(stashed_variables, t);
  }

  void before(c10::SymInt& t) {
    SWAP_SAVED_VARIABLES_SAVE(stashed_symints, t, t);
    auto opt_value = state.next_sym_size();
    if (opt_value.has_value()) {
      t = *opt_value; // dynamic shape
    }
  }
  void after(c10::SymInt& t) {
    SWAP_SAVED_VARIABLES_RESTORE(stashed_symints, t);
  }

  void before(Edge& t) {
    if (t.is_valid()) {
      // need for symints used by validate_outputs
      before(t.function->mutable_input_metadata(t.input_nr));
    }
  }
  void after(Edge& t) {
    if (t.is_valid()) {
      after(t.function->mutable_input_metadata(t.input_nr));
    }
  }
  void before(InputMetadata& t) {
    before(t.mutable_shape_as_dim_vector());
  }
  void after(InputMetadata& t) {
    after(t.mutable_shape_as_dim_vector());
  }
  void before(at::TensorGeometry& t) {
    before(t.mutable_sizes());
    before(t.mutable_strides());
    before(t.mutable_storage_offset());
    t.recompute();
  }
  void after(at::TensorGeometry& t) {
    after(t.mutable_sizes());
    after(t.mutable_strides());
    after(t.mutable_storage_offset());
    t.recompute();
  }
  void before(torch::autograd::TypeAndSize& t) {
    before(t.sym_sizes);
    before(t.options);
  }
  void after(torch::autograd::TypeAndSize& t) {
    after(t.sym_sizes);
    after(t.options);
  }
  void before(VariableInfo& t) {
    before(t.size);
  }
  void after(VariableInfo& t) {
    after(t.size);
  }

  template <typename T>
  void before(std::vector<T>& t) {
    for (T& i : t) {
      before(i);
    }
  }
  template <typename T>
  void after(std::vector<T>& t) {
    for (T& i : t) {
      after(i);
    }
  }
  template <typename T, unsigned N>
  void before(c10::SmallVector<T, N>& t) {
    for (T& i : t) {
      before(i);
    }
  }
  template <typename T, unsigned N>
  void after(c10::SmallVector<T, N>& t) {
    for (T& i : t) {
      after(i);
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

#define NO_OP_VISIT(T)     \
  void before(const T&) {} \
  void after(const T&) {}
  NO_OP_VISIT(caffe2::TypeMeta);
  NO_OP_VISIT(c10::Device);
  NO_OP_VISIT(c10::DeviceType);
  NO_OP_VISIT(c10::Layout);
  NO_OP_VISIT(c10::MemoryFormat);
  NO_OP_VISIT(c10::ScalarType);
  NO_OP_VISIT(c10::Scalar);
  NO_OP_VISIT(c10::TensorOptions);
  NO_OP_VISIT(std::string);
  NO_OP_VISIT(int64_t);
  NO_OP_VISIT(bool);
  NO_OP_VISIT(double);
#undef NO_OP_VISIT

  void set_grad_value(const at::Tensor& tensor) {
    if (state.accumulate_grad)
      state.outputs.emplace_back(tensor);
  }

  SwapSavedVariables(
      AutogradCompilerCall& c,
      TraceState& s,
      std::shared_ptr<Node> n)
      : compiler(c), state(s), node(std::move(n)) {}

  AutogradCompilerCall& compiler;
  TraceState& state;
  std::unordered_map<SavedVariable*, SavedVariable> stashed_variables;
  std::unordered_map<at::Tensor*, at::Tensor> stashed_tensors;
  std::unordered_map<c10::SymInt*, c10::SymInt> stashed_symints;
  std::shared_ptr<Node> node;
};

} // namespace torch::dynamo::autograd

template <>
struct std::hash<torch::dynamo::autograd::CacheKey> {
  size_t operator()(const torch::dynamo::autograd::CacheKey& k) const {
    return k.hash();
  }
};
