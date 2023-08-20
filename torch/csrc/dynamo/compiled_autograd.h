#pragma once
#include <c10/core/impl/TorchDispatchModeTLS.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/csrc/autograd/engine.h>
#include <torch/csrc/utils/python_stub.h>
#include <torch/csrc/utils/torch_dispatch_mode.h>
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

struct SizeArgs {
  // manage a collection of SizeInput args corresponding to all SymInt in the
  // autograd graph

  size_t add(const c10::SymInt* s) {
    // add a symint and return its index
    // if we have already seen this one before, de-duplicate it
    auto it = symint_to_index.find(s);
    if (it != symint_to_index.end()) {
      return it->second;
    } else {
      auto index = all_size_inputs.size();
      all_size_inputs.emplace_back(
          SizeInput(default_dyn_type, s->expect_int()));
      symint_to_index.emplace(s, index);
      return index;
    }
  }

  size_t get_index(const c10::SymInt* key) {
    // lookup something we already inserted and return index in all_size_inputs
    auto it = symint_to_index.find(key);
    TORCH_INTERNAL_ASSERT(it != symint_to_index.end());
    return it->second;
  }

  SizeInput::DynType set_default_dyn_type(SizeInput::DynType value) {
    return std::exchange(default_dyn_type, value);
  }

  // populated by collect, this tracks all of the SymInts we visited
  std::vector<SizeInput> all_size_inputs;

  // Populated after a cache lookup. The concrete sizes we must pass to the
  // compiled graph.
  std::vector<int64_t> dyn_size_inputs;

 private:
  // allows de-duplicating SymInts we visit multiple times and out-of-order
  // swapping
  std::unordered_map<const c10::SymInt*, size_t> symint_to_index;

  // We use this for AotAutograd SymInt to make them default to dynamic handing.
  SizeInput::DynType default_dyn_type = SizeInput::STATIC;
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

struct TensorArg {
  // Represents a de-duplicated tensor that will be passed into the graph
  TensorArg(uint32_t i = 0) : id(i) {}
  uint32_t index() const {
    TORCH_INTERNAL_ASSERT(defined());
    return id - 1;
  }
  bool defined() const {
    return id != 0;
  }
  uint32_t id;
  at::Tensor proxy_tensor;
};

struct TensorArgs {
  // Manages a collection of TensorArgs and mappings from Tensors/SavedVariables
  // to them.  This also allows us to unpack SavedVariable exactly once and
  // store the unpacked Tensor.

  TensorArg& lookup(const at::Tensor& tensor, bool create = false) {
    if (!tensor.defined()) {
      return _undefined;
    }
    auto impl = tensor.unsafeGetTensorImpl();
    auto it = _args.find(impl);
    if (it == _args.end()) {
      TORCH_INTERNAL_ASSERT(create && inputs.size() == _next_id - 1);
      it = _args.emplace(impl, TensorArg(_next_id++)).first;
      inputs.emplace_back(tensor);
    }
    return it->second;
  }

  TensorArg& lookup(const SavedVariable& sv) {
    auto it = _saved_variables.find(&sv);
    TORCH_INTERNAL_ASSERT(it != _saved_variables.end());
    return *it->second;
  }

  TensorArg& add(const at::Tensor& tensor) {
    return lookup(tensor, true);
  }

  TensorArg& add(const SavedVariable& sv, const std::shared_ptr<Node>& node) {
    // TODO(jansel): Here we unpack the SavedVariable exactly once.  This might
    // fire SavedTensor hooks.  In the future we should try to put saved tensor
    // hooks into the graph.
    at::Tensor tensor = sv.unpack(node);
    TensorArg& arg = add(tensor);
    _saved_variables.emplace(&sv, &arg);
    return arg;
  }

  // the concrete tensors that will get passed into the graph as inputs
  std::vector<at::Tensor> inputs;

 private:
  std::unordered_map<const c10::TensorImpl*, TensorArg> _args;
  // Every TensorArg from this is actually owned by _args (or _undefined) and
  // that's why we have an un-owned pointer here.
  std::unordered_map<const SavedVariable*, TensorArg*> _saved_variables;
  TensorArg _undefined;
  uint32_t _next_id = 1; // id=0 used by _undefined
};

struct AutogradCompilerCall {
  int emplace_hook(c10::SafePyObject&& fn) {
    hooks.emplace_back(std::move(fn));
    return hooks.size() - 1;
  }

  TensorArgs tensor_args;
  SizeArgs size_args;
  std::vector<c10::SafePyObject> hooks;
  NodeCalls node_calls;
};

class CompiledNodeArgs {
  // CompiledNodeArgs builds a representation of the constant values found
  // across all the nodes in the compiled graph, via 'collect' overloads. The
  // collected constants are specialized on by concatenation into a cache key.
  // Tensor, symint arguments (which are lifted to become graph inputs rather
  // than specialized on) are forwarded to the compiler and not included in the
  // key.
 public:
  void collect(const TensorArg& t) {
    collect_size(t.id);
    if (t.defined()) {
      const at::Tensor& tensor = _compiler.tensor_args.inputs[t.index()];
      // including these in the cache key means dynamo-level tensor guards can
      // be skipped
      collect(tensor.device());
      collect(tensor.dtype());
      collect(tensor.requires_grad());
    }
  }

  void collect(const at::Tensor& t) {
    collect(_compiler.tensor_args.add(t));
  }
  void collect(const SavedVariable& t) {
    collect(_compiler.tensor_args.add(t, _node_call.node));
  }
  void collect(const c10::SymInt& t) {
    collect_size(_compiler.size_args.add(&t));
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
  void collect(const std::shared_ptr<Node>& t) {
    // Note: this is only capturing the ID of the node not everything
    // contained inside it.  This is used for tracking connections between
    // nodes and the actual details of the node itself must be handled by
    // a seperate call to `node->compiled_args()`.
    if (cond((bool)t)) {
      collect(_compiler.node_calls.lookup(t));
    }
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

  SizeInput::DynType set_default_dyn_type(SizeInput::DynType value) {
    return _compiler.size_args.set_default_dyn_type(value);
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
  TraceState(std::unordered_map<size_t, c10::SymInt>&& ss, size_t num_outputs)
      : sym_sizes(std::move(ss)), outputs(num_outputs) {}

  std::unordered_map<size_t, c10::SymInt> sym_sizes;
  variable_list outputs;
  std::vector<size_t> output_grad_targets;
};

class SwapSavedVariables {
  // SwapSavedVariables is used during the tracing/compilation phase after a
  // cache-miss. It swaps any 'lifted' inputs (tensors, symints) to proxy nodes,
  // allows tracing to happen, then swaps them back afterwards.
 public:
  void before(at::Tensor& t) {
    TensorArg& arg = compiler.tensor_args.lookup(t);
    stashed_tensors.save(&t, std::move(t));
    if (arg.defined()) {
      TORCH_INTERNAL_ASSERT(arg.proxy_tensor.defined());
      t = arg.proxy_tensor;
    }
  }
  void after(at::Tensor& t) {
    stashed_tensors.restore(&t);
  }

  void before(SavedVariable& t) {
    TensorArg& arg = compiler.tensor_args.lookup(t);
    stashed_variables.save(&t, std::move(t));
    if (arg.defined()) {
      TORCH_INTERNAL_ASSERT(arg.proxy_tensor.defined());
      t = SavedVariable(arg.proxy_tensor, false);
    }
  }
  void after(SavedVariable& t) {
    stashed_variables.restore(&t);
  }

  void before(c10::SymInt& t) {
    auto it = state.sym_sizes.find(compiler.size_args.get_index(&t));
    stashed_symints.save(&t, c10::SymInt(t));
    if (it != state.sym_sizes.end()) {
      t = it->second; // dynamic shape
    }
  }
  void after(c10::SymInt& t) {
    stashed_symints.restore(&t);
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

  // record the need to run `dst.mutable_grad() = src` after the graph
  // dst is a real tensor, src is a fake tensor
  void assign_mutable_grad(const at::Tensor& dst, const at::Tensor& src) {
    const TensorArg& arg = compiler.tensor_args.lookup(dst);
    TORCH_INTERNAL_ASSERT(arg.defined());
    TORCH_INTERNAL_ASSERT(
        state.outputs.size() == state.output_grad_targets.size());
    state.outputs.emplace_back(src);
    state.output_grad_targets.emplace_back(arg.index());
  }

  SwapSavedVariables(AutogradCompilerCall& c, TraceState& s)
      : compiler(c), state(s) {}

  void debug_asserts() {
    stashed_variables.debug_assert();
    stashed_tensors.debug_assert();
    stashed_symints.debug_assert();
  }

 private:
  template <typename T>
  struct Stashed {
    Stashed(T&& v) : prior_value(std::move(v)) {}
    T prior_value;
    // Note: we need count here to support duplicate calls to before()
    // which happen when we have multiple autograd::Edge objects pointing
    // to the same autograd::Node
    int count = 1;
  };

  template <typename T>
  struct StashedVars : public std::unordered_map<const T*, Stashed<T>> {
    void save(const T* key, T&& value) {
      auto it = this->find(key);
      if (it == this->end()) {
        this->emplace(key, std::move(value));
      } else {
        // keep the value from the prior save()
        it->second.count++;
      }
    }
    void restore(T* var) {
      auto it = this->find(var);
      TORCH_INTERNAL_ASSERT(it != this->end(), "missing before())");
      if (--it->second.count == 0) {
        // restore the value on the last restore()
        *var = std::move(it->second.prior_value);
        this->erase(it);
      }
    }
    void debug_assert() {
      TORCH_INTERNAL_ASSERT(this->empty(), "missing call to after()");
    }
  };

  AutogradCompilerCall& compiler;
  TraceState& state;

  // These mappings are used to save the prior values when we overwrite things
  // in before(). In after(), we use these to cleanup after ourselves.
  StashedVars<SavedVariable> stashed_variables;
  StashedVars<at::Tensor> stashed_tensors;
  StashedVars<c10::SymInt> stashed_symints;
};

} // namespace torch::dynamo::autograd

template <>
struct std::hash<torch::dynamo::autograd::CacheKey> {
  size_t operator()(const torch::dynamo::autograd::CacheKey& k) const {
    return k.hash();
  }
};
