#pragma once
#include <ATen/TensorGeometry.h>
#include <ATen/core/ivalue.h>
#include <c10/core/impl/TorchDispatchModeTLS.h>
#include <c10/util/flat_hash_map.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/input_metadata.h>
#include <torch/csrc/autograd/saved_variable.h>
#include <torch/csrc/autograd/variable_info.h>
#include <torch/csrc/utils/python_stub.h>
#include <torch/csrc/utils/torch_dispatch_mode.h>
#include <typeindex>
#include <vector>

// see [Note: Compiled Autograd]

namespace torch::dynamo::autograd {
using namespace torch::autograd;

// This is a layer of indirection for calling methods on the Python
// AutogradCompilerInstance (referred to as the "py_compiler") from
// libtorch_cpu (where Python is not available).
// A PyCompilerInterfaceImpl in libtorch_python subclasses it and
// overrides the methods to do the actual calls back to Python.
struct TORCH_API PyCompilerInterface {
  PyCompilerInterface() = default;
  PyCompilerInterface(const PyCompilerInterface&) = delete;
  PyCompilerInterface& operator=(const PyCompilerInterface&) = delete;
  PyCompilerInterface(PyCompilerInterface&&) = delete;
  PyCompilerInterface& operator=(PyCompilerInterface&&) = delete;
  virtual ~PyCompilerInterface() = default;

  // Invokes py_compiler.bind_function
  virtual std::string bind_function(
      PyObject* py_compiler,
      const std::string& fn_name,
      // NOLINTNEXTLINE(performance-unnecessary-value-param)
      functional_apply_t fn,
      // NOLINTNEXTLINE(performance-unnecessary-value-param)
      std::vector<at::TypePtr> packed_args_schema,
      bool is_custom_function = false,
      bool is_traceable = true) const {
    TORCH_INTERNAL_ASSERT(false, "Needs to be overridden");
  }

  // Invokes py_compiler.method_name(fn_name, inputs, packed_args,
  // output_metadata)
  virtual variable_list call_function(
      PyObject* py_compiler,
      const char* method_name,
      const std::string& fn_name,
      const variable_list& inputs,
      const ivalue_list& packed_args,
      const c10::IValue& output_metadata) const {
    TORCH_INTERNAL_ASSERT(false, "Needs to be overridden");
  }
  virtual variable_list call_copy_slices_prologue(
      PyObject* py_compiler,
      const variable_list& inputs,
      const at::TensorGeometry& base,
      const at::TensorGeometry& view) const {
    TORCH_INTERNAL_ASSERT(false, "Needs to be overridden");
  }
  virtual variable_list call_copy_slices_epilogue(
      PyObject* py_compiler,
      const std::vector<bool>& needs_input_grad,
      const at::Tensor& result,
      const variable_list& res,
      const at::Tensor& grad_slice) const {
    TORCH_INTERNAL_ASSERT(false, "Needs to be overridden");
  }
  virtual at::Tensor call_unpack(
      PyObject* py_compiler,
      std::optional<size_t> hook_id,
      size_t hook_input_id) const {
    TORCH_INTERNAL_ASSERT(false, "Needs to be overridden");
  }
  virtual void call_accumulate_grad(
      PyObject* py_compiler,
      const at::Tensor& variable,
      const at::Tensor& grad,
      bool has_post_hooks) const {
    TORCH_INTERNAL_ASSERT(false, "Needs to be overridden");
  }
};

TORCH_API const std::unique_ptr<PyCompilerInterface>& getPyCompilerInterface();
struct TORCH_API PyCompilerGuard {
  explicit PyCompilerGuard(std::unique_ptr<PyCompilerInterface>&& impl);
  PyCompilerGuard(const PyCompilerGuard&) = delete;
  PyCompilerGuard& operator=(const PyCompilerGuard&) = delete;
  PyCompilerGuard(PyCompilerGuard&&) = delete;
  PyCompilerGuard& operator=(PyCompilerGuard&&) = delete;

  ~PyCompilerGuard();
};

// including torch/csrc/autograd/engine.h breaks BC by somehow introducing
// symbol resolution issues. Instead requiring downstream users to include
// engine.h to access collect_input_metadata, we provide it here (with a
// different name to avoid ambiguous symbols...)
TORCH_API std::vector<std::optional<InputMetadata>> get_input_metadata(
    const edge_list& edges);

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
  // NOLINTNEXTLINE(*c-array*)
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
    graph_output.emplace_back(input_nr, output_idx);
  }

  uint32_t id;
  std::shared_ptr<Node> node;
  std::vector<std::pair<int, int>> tensor_pre_hooks;
  std::vector<std::pair<int, int>> cpp_tensor_pre_hooks;
  std::vector<int> pre_hooks;
  std::vector<int> post_hooks;
  std::vector<int> post_acc_grad_hooks;
  std::vector<std::pair<int, int>> graph_output;
  bool needed = true;
};

struct NodeCalls : public std::unordered_map<Node*, NodeCall> {
  NodeCall& lookup(const std::shared_ptr<Node>& function) {
    auto it = find(function.get());
    if (it == end()) {
      it = emplace(function.get(), NodeCall(_next_id++, function)).first;
      nodes.emplace_back(function.get());
    }
    return it->second;
  }

  const NodeCall& lookup(uint32_t id) const {
    TORCH_INTERNAL_ASSERT(id < nodes.size());
    auto it = find(nodes[id]);
    TORCH_INTERNAL_ASSERT(it != end());
    return it->second;
  }

  void clear() {
    _next_id = 0;
    std::unordered_map<Node*, NodeCall>::clear();
    nodes.clear();
  }

 private:
  uint32_t _next_id = 0;
  std::vector<Node*> nodes;
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
  TensorArgs(const std::optional<size_t>& active_node_call_idx)
      : active_node_call_idx(active_node_call_idx) {}

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
      if (active_node_call_idx.has_value()) {
        input_origins.emplace_back(active_node_call_idx.value());
      }
    }
    return it->second;
  }

  TensorArg& lookup(const SavedVariable& sv) {
    if (auto it = _saved_variables.find(&sv); it != _saved_variables.end()) {
      // unpacked before graph
      return *it->second;
    }
    // unpacked in graph
    auto it2 = _saved_variables_proxies.find(&sv);
    TORCH_INTERNAL_ASSERT(it2 != _saved_variables_proxies.end());
    return *it2->second;
  }

  TensorArg& add(const at::Tensor& tensor) {
    return lookup(tensor, true);
  }

  TensorArg& add(const SavedVariable& sv, const std::shared_ptr<Node>& node) {
    // no unpack hooks in this codepath
    at::Tensor tensor = sv.unpack(node);
    TensorArg& arg = add(tensor);
    _saved_variables.emplace(&sv, &arg);
    return arg;
  }

  // the concrete tensors that will get passed into the graph as inputs
  std::vector<at::Tensor> inputs;
  // NodeCall id of each input, only when verbose logging is enabled
  std::vector<uint32_t> input_origins;

 private:
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const std::optional<size_t>& active_node_call_idx;
  std::unordered_map<const c10::TensorImpl*, TensorArg> _args;
  // Every TensorArg from this is actually owned by _args (or _undefined) and
  // that's why we have an un-owned pointer here.
  std::unordered_map<const SavedVariable*, TensorArg*> _saved_variables;
  std::unordered_map<const SavedVariable*, TensorArg*> _saved_variables_proxies;
  TensorArg _undefined;
  uint32_t _next_id = 1; // id=0 used by _undefined
};

struct LiftedIValueArg {
  LiftedIValueArg() = delete;
  LiftedIValueArg(const at::IValue* ptr)
      : actual_ptr(ptr), proxy(at::IValue::uninitialized()) {}

  const at::IValue* actual_ptr; // lifetime handled by autograd node
  at::IValue proxy;
};

struct LiftedIValueArgs {
  LiftedIValueArgs(const std::optional<size_t>& active_node_call_idx)
      : active_node_call_idx(active_node_call_idx) {}

  at::IValue& next_proxy(const at::IValue* actual_ptr) {
    TORCH_INTERNAL_ASSERT(next < args.size());
    auto& iv_arg = args.at(next++);
    TORCH_INTERNAL_ASSERT(iv_arg.actual_ptr == actual_ptr);
    return iv_arg.proxy;
  }

  void add(const at::IValue* iv) {
    args.emplace_back(iv);
    if (active_node_call_idx.has_value()) {
      args_origins.emplace_back(active_node_call_idx.value());
    }
  }

  std::vector<LiftedIValueArg> args;
  size_t next = 0;
  // NodeCall id of each arg, only when verbose logging is enabled
  std::vector<uint32_t> args_origins;

 private:
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const std::optional<size_t>& active_node_call_idx;
};

struct AutogradCompilerCall {
  AutogradCompilerCall(SizeInput::DynType default_dyn_type)
      : active_node_call_idx(std::nullopt),
        tensor_args(active_node_call_idx),
        lifted_ivalue_args(active_node_call_idx),
        default_dyn_type(default_dyn_type) {}
  void add_size_input(const c10::SymInt& s) {
    all_size_inputs.emplace_back(
        default_dyn_type, s.guard_int(__FILE__, __LINE__));
    if (active_node_call_idx.has_value()) {
      size_input_origins.emplace_back(active_node_call_idx.value());
    }
  }

  size_t emplace_hook(c10::SafePyObject&& fn) {
    hooks.emplace_back(std::move(fn));
    return hooks.size() - 1;
  }

  size_t emplace_cpp_tensor_pre_hook(
      std::function<at::TensorBase(const at::TensorBase&)>&& fn) {
    cpp_tensor_pre_hooks.emplace_back(std::move(fn));
    return cpp_tensor_pre_hooks.size() - 1;
  }

  size_t emplace_packed_input(c10::SafePyObject&& input) {
    packed_inputs.emplace_back(std::move(input));
    return packed_inputs.size() - 1;
  }

  void set_active_node_call_idx(size_t node_call_idx) {
    active_node_call_idx = node_call_idx;
  }

  std::optional<size_t> active_node_call_idx;
  TensorArgs tensor_args;
  std::vector<SizeInput> all_size_inputs;
  LiftedIValueArgs lifted_ivalue_args;
  std::vector<int64_t> dyn_size_inputs;
  std::vector<c10::SafePyObject> hooks;
  std::vector<std::function<at::TensorBase(const at::TensorBase&)>>
      cpp_tensor_pre_hooks;
  std::vector<c10::SafePyObject> packed_inputs;
  NodeCalls node_calls;
  SizeInput::DynType default_dyn_type;
  // NodeCall id of each size, only when verbose logging is enabled
  std::vector<uint32_t> size_input_origins;
  std::unordered_map<const SavedVariable*, std::pair<size_t, size_t>>
      sv_to_hooks;
  // pynode -> backward and backward state idx
  std::unordered_map<const Node*, std::pair<size_t, std::optional<size_t>>>
      pynode_objs;
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
  void collect(const SavedVariable& sv, bool is_output) {
    if (auto hook_data = sv.retrieve_unpack_hook_data();
        hook_data.has_value()) {
      // hooks, unpack in graph
      auto& [hook, packed_input] = hook_data.value();
      size_t hook_id = _compiler.emplace_hook(std::move(hook));
      // rely on dynamo to dedup packed tensors from unpacked tensors
      size_t input_id = _compiler.emplace_packed_input(std::move(packed_input));
      _compiler.sv_to_hooks.emplace(&sv, std::make_pair(hook_id, input_id));
    } else {
      // no hooks, unpack now
      collect(
          _compiler.tensor_args.add(sv, is_output ? _node_call.node : nullptr));
    }
  }
  void collect(const c10::SymInt& t) {
    _compiler.add_size_input(t);
  }
  void collect(const std::vector<SavedVariable>& t, bool is_output) {
    collect_size(t.size());
    for (const SavedVariable& i : t) {
      collect(i, is_output);
    }
  }
  template <typename T>
  void collect(const std::vector<T>& t) {
    collect_size(t.size());
    for (const T& i : t) {
      collect(i);
    }
  }
  void collect(const c10::ArrayRef<SavedVariable>& t, bool is_output) {
    collect_size(t.size());
    for (const SavedVariable& i : t) {
      collect(i, is_output);
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
  void collect(const std::optional<T>& t) {
    if (cond(t.has_value())) {
      // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
      collect(*t);
    }
  }
  template <typename A, typename B>
  void collect(const std::pair<A, B>& t) {
    collect(t.first);
    collect(t.second);
  }
  template <typename V>
  void collect(const ska::flat_hash_map<std::string, V>& m) {
    collect_size(m.size());

    std::vector<std::string> keys;
    keys.reserve(m.size());
    std::transform(
        m.begin(), m.end(), std::back_inserter(keys), [](const auto& entry) {
          return entry.first;
        });
    std::sort(keys.begin(), keys.end());
    for (const auto& k : keys) {
      collect(k);
      collect(m.at(k));
    }
  }
  void collect(const at::IValue& iv, bool nested = false) {
    // used by AutogradContext::saved_data from CppNode
    if (iv.isList()) {
      c10::List<at::IValue> list = iv.toList();
      collect_size(list.size());
      for (auto&& value : list) {
        collect(value, true);
      }
    } else if (iv.isGenericDict()) {
      c10::Dict<at::IValue, at::IValue> ordered_dict = iv.toGenericDict();
      collect_size(ordered_dict.size());
      // NOLINTNEXTLINE(modernize-loop-convert)
      for (auto it = ordered_dict.begin(); it != ordered_dict.end(); it++) {
        collect(it->key());
        collect(it->value(), true);
      }
    } else if (iv.isTensor()) {
      collect(iv.toTensor());
    } else if (
        !nested &&
        (iv.isInt() || iv.isSymInt() || iv.isDouble() || iv.isSymFloat())) {
      // can't lift ivalues nested in collections
      _compiler.lifted_ivalue_args.add(&iv);
    } else {
      try {
        collect(static_cast<uint64_t>(at::IValue::hash(iv)));
      } catch (const std::runtime_error& e) {
        std::string msg =
            "Compiled autograd can not trace unhashable IValues, error: " +
            std::string(e.what());
        TORCH_CHECK_NOT_IMPLEMENTED(false, msg);
      }
    }
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
    // a separate call to `node->compiled_args()`.
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
    TORCH_CHECK_NOT_IMPLEMENTED(
        !t.is_nested_tensor(), "NestedTensor support not implemented. ");
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
  COLLECT_AS_BYTES(c10::ScalarType)
  COLLECT_AS_BYTES(c10::DeviceType)
  COLLECT_AS_BYTES(c10::Layout)
  COLLECT_AS_BYTES(c10::MemoryFormat)
  COLLECT_AS_BYTES(int8_t)
  COLLECT_AS_BYTES(int16_t)
  COLLECT_AS_BYTES(int32_t)
  COLLECT_AS_BYTES(int64_t)
  COLLECT_AS_BYTES(uint8_t)
  COLLECT_AS_BYTES(uint16_t)
  COLLECT_AS_BYTES(uint32_t)
  COLLECT_AS_BYTES(uint64_t)
  COLLECT_AS_BYTES(bool)
  COLLECT_AS_BYTES(float)
  COLLECT_AS_BYTES(double)
#undef COLLECT_AS_BYTES

  void collect_hooks_from(Node* fn) {
    for (auto& i : fn->tensor_pre_hooks()) {
      i->compiled_args(*this);
    }
    for (auto& [_, i] : fn->retains_grad_hooks()) {
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
      collect_size(static_cast<size_t>(h.second));
    }
  }

  CacheKey key() const {
    Node* node = _node_call.node.get();
    return CacheKey(
        typeid(*node), _specialization_key, _specialization_key_size);
  }

  void collect_pynode_objs(
      const Node* pynode,
      c10::SafePyObject&& bwd,
      std::optional<c10::SafePyObject>&& bwd_state) {
    size_t bwd_idx = _compiler.emplace_hook(std::move(bwd));
    std::optional<size_t> bwd_state_idx;
    if (auto state = std::move(bwd_state); state.has_value()) {
      bwd_state_idx = _compiler.emplace_hook(std::move(state.value()));
    }
    _compiler.pynode_objs.emplace(
        pynode, std::make_pair(bwd_idx, bwd_state_idx));
  }

  void add_tensor_pre_hook(c10::SafePyObject&& obj, int index) {
    auto fn_id = _compiler.emplace_hook(std::move(obj));
    collect_size(fn_id);
    _node_call.tensor_pre_hooks.emplace_back(fn_id, index);
  }

  void add_cpp_single_tensor_pre_hook(
      const std::function<at::TensorBase(const at::TensorBase&)>& hook,
      size_t idx) {
    auto wrapper = [hook](const at::TensorBase& grad) {
      // handle when hook returns nothing
      auto out = hook(grad);
      if (!out.defined()) {
        return grad;
      }
      return out;
    };

    auto hook_id = _compiler.emplace_cpp_tensor_pre_hook(std::move(wrapper));
    collect_size(hook_id);
    _node_call.cpp_tensor_pre_hooks.emplace_back(hook_id, idx);
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

  void add_post_acc_grad_hook(c10::SafePyObject&& obj) {
    auto fn_id = _compiler.emplace_hook(std::move(obj));
    collect_size(fn_id);
    _node_call.post_acc_grad_hooks.emplace_back(fn_id);
  }

  // Need to template the size_t to silence internal 32-bit build errors due to
  // a mix of -Werror, -Wtautological-type-limit-compare and
  // -Wunknown-pragmas
  template <typename T>
  std::enable_if_t<std::is_unsigned_v<T>, void> collect_size(T s) {
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
        _specialization_key(
            // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
            (uint8_t*)std::malloc(_specialization_key_storage)) {}
  CompiledNodeArgs(const CompiledNodeArgs&) = delete;
  CompiledNodeArgs(CompiledNodeArgs&&) = delete;
  CompiledNodeArgs& operator=(const CompiledNodeArgs&) = delete;
  CompiledNodeArgs& operator=(CompiledNodeArgs&&) = delete;
  ~CompiledNodeArgs() {
    // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
    std::free(_specialization_key);
  }

 private:
  template <typename T>
  void specialize_on_bytes(const T& t) {
    while (C10_UNLIKELY(
        _specialization_key_size + sizeof(T) > _specialization_key_storage)) {
      _specialization_key_storage *= 2;
      // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
      _specialization_key = (uint8_t*)std::realloc(
          _specialization_key, _specialization_key_storage);
    }
    std::memcpy(_specialization_key + _specialization_key_size, &t, sizeof(T));
    _specialization_key_size += sizeof(T);
  }

  AutogradCompilerCall& _compiler;
  NodeCall& _node_call;
  size_t _specialization_key_size{0};
  size_t _specialization_key_storage{1024};
  uint8_t* _specialization_key;
};

struct TraceState {
  TraceState(std::vector<std::optional<c10::SymInt>>&& ss, size_t num_outputs)
      : sym_sizes(std::move(ss)), outputs(num_outputs) {}

  void debug_asserts() {
    TORCH_INTERNAL_ASSERT(sym_sizes_index == sym_sizes.size());
  }
  std::optional<c10::SymInt> next_sym_size() {
    TORCH_INTERNAL_ASSERT(sym_sizes_index < sym_sizes.size());
    return sym_sizes[sym_sizes_index++];
  }

  size_t sym_sizes_index{0};
  std::vector<std::optional<c10::SymInt>> sym_sizes;
  variable_list outputs;
};

class SwapSavedVariables {
  // SwapSavedVariables is used during the tracing/compilation phase after a
  // cache-miss. It swaps any 'lifted' inputs (tensors, symints) to proxy nodes,
  // allows tracing to happen, then swaps them back afterwards.
 public:
  std::pair<size_t, std::optional<size_t>> retrieve_pynode_objs(
      Node* pynode) const {
    auto it = compiler.pynode_objs.find(pynode);
    TORCH_INTERNAL_ASSERT(it != compiler.pynode_objs.end());
    return it->second;
  }

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
    if (auto it = compiler.sv_to_hooks.find(&t);
        it != compiler.sv_to_hooks.end()) {
      const auto& pyinterface =
          torch::dynamo::autograd::getPyCompilerInterface();
      auto proxy_tensor = pyinterface->call_unpack(
          get_py_compiler(), it->second.first, it->second.second);
      stashed_variables.save(&t, std::move(t));
      bool prior = at::SavedTensorDefaultHooks::set_tracing(true);
      t = SavedVariable(proxy_tensor, false);
      at::SavedTensorDefaultHooks::set_tracing(prior);
    } else {
      // no hooks, was already unpacked
      TensorArg& arg = compiler.tensor_args.lookup(t);
      stashed_variables.save(&t, std::move(t));
      if (arg.defined()) {
        bool prior = at::SavedTensorDefaultHooks::set_tracing(true);
        TORCH_INTERNAL_ASSERT(arg.proxy_tensor.defined());
        t = SavedVariable(arg.proxy_tensor, false);
        at::SavedTensorDefaultHooks::set_tracing(prior);
      }
    }
  }
  void after(SavedVariable& t) {
    stashed_variables.restore(&t);
  }

  void before(c10::SymInt& t) {
    stashed_symints.save(&t, c10::SymInt(t));
    auto opt_value = state.next_sym_size();
    if (opt_value.has_value()) {
      t = *opt_value; // dynamic shape
    }
  }
  void after(c10::SymInt& t) {
    stashed_symints.restore(&t);
  }

  void before(at::IValue& iv) {
    if (iv.isTensor()) {
      before(iv.toTensor());
    } else {
      stashed_ivalues.save(&iv, at::IValue(iv));
      if (iv.isInt() || iv.isSymInt() || iv.isDouble() || iv.isSymFloat()) {
        iv = compiler.lifted_ivalue_args.next_proxy(&iv);
      }
    }
  }

  void after(at::IValue& t) {
    if (t.isTensor()) {
      after(t.toTensor());
    } else {
      stashed_ivalues.restore(&t);
    }
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
  void before(std::optional<T>& t) {
    if (t.has_value()) {
      before(*t);
    }
  }
  template <typename T>
  void after(std::optional<T>& t) {
    if (t.has_value()) {
      after(*t);
    }
  }

  template <typename V>
  void before(ska::flat_hash_map<std::string, V>& m) {
    std::vector<std::string> keys;
    keys.reserve(m.size());
    std::transform(
        m.begin(), m.end(), std::back_inserter(keys), [](const auto& entry) {
          return entry.first;
        });
    std::sort(keys.begin(), keys.end());
    for (auto& k : keys) {
      before(m.at(k));
    }
  }

  template <typename V>
  void after(ska::flat_hash_map<std::string, V>& m) {
    for (auto& [_, v] : m) {
      after(v);
    }
  }

#define NO_OP_VISIT(T)     \
  void before(const T&) {} \
  void after(const T&) {}
  NO_OP_VISIT(caffe2::TypeMeta)
  NO_OP_VISIT(c10::Device)
  NO_OP_VISIT(c10::DeviceType)
  NO_OP_VISIT(c10::Layout)
  NO_OP_VISIT(c10::MemoryFormat)
  NO_OP_VISIT(c10::ScalarType)
  NO_OP_VISIT(c10::Scalar)
  NO_OP_VISIT(c10::TensorOptions)
  NO_OP_VISIT(std::string)
  NO_OP_VISIT(int64_t)
  NO_OP_VISIT(bool)
  NO_OP_VISIT(double)
#undef NO_OP_VISIT

  SwapSavedVariables(
      AutogradCompilerCall& c,
      TraceState& s,
      PyObject* p,
      const NodeCall& n)
      : compiler(c), state(s), py_compiler(p), curr_node_call(n) {}

  PyObject* get_py_compiler() const {
    return py_compiler;
  }

  const NodeCall& get_curr_node_call() {
    return curr_node_call;
  }

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
      auto [it, inserted] = this->try_emplace(key, std::move(value));
      if (!inserted) {
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

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  AutogradCompilerCall& compiler;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  TraceState& state;
  // This is a borrowed reference, we do not increment ownership, or lower it,
  // it's lifecycle is entirely longer than this objects.
  PyObject* py_compiler;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const NodeCall& curr_node_call;

  // These mappings are used to save the prior values when we overwrite things
  // in before(). In after(), we use these to cleanup after ourselves.
  StashedVars<SavedVariable> stashed_variables;
  StashedVars<at::Tensor> stashed_tensors;
  StashedVars<c10::SymInt> stashed_symints;
  StashedVars<at::IValue> stashed_ivalues;
};

// NOTE: [Compiled Autograd and backward functions]
// Built-in autograd nodes have functional apply variants
// (e.g. MulBackward0_apply_functional). Compiled Autograd's initial graph
// capture wants to take a variant of this function and proxy it into the graph.
// Every autograd node defines an apply_with_saved function, that when invoked,
// proxies a call to a function into the Compiled Autograd graph.
//
// Some requirements that we have are:
// - The proxy'ed function must have inputs that are FX-graphable types.
// - Windows has a DLL symbol limit of 65536.
// - Node::apply_with_saved is in libtorch_cpu which does not have direct access
// to Python
//
// There were multiple ways to skin the cat, but what we end up doing is:
// - for e.g. MulBackward0_apply_functional, we create a new C++ function
// MulBackward0_apply_functional_ivalue that accepts vector<IValue>.
// - We define how to pack and unpack arbitrary C++ types into IValues.
// - apply_with_saved passes MulBackward0_apply_functional_ivalue and
// the IValue arguments to Python via an indirection.
// In Python, these get proxy'ed into a graph.

// Helper struct for packing/unpacking an arbitrary C++ type into a single
// IValue. There are various full and partial specializations for IValuePacker
// to handle packing specific types (like TensorOptions) into an IValue.
template <typename T>
struct IValuePacker {
  // Defines how to pack T into an IValue.
  static at::IValue pack(const T& t) {
    return t;
  }
  // Defines how to unpack an IValue into T.
  static T unpack(const at::IValue& t) {
    return t.to<T>();
  }
  // Returns the TypePtr for the IValue (this is like the "type" of the IValue).
  // We use this when passing the packed IValue from Python to C++.
  // In Python, the IValue is just a PyObject* with the native type.
  // For example, it may be a Python int, a Python List[int], etc.
  // When passing this PyObject* into C++, we need to know how to parse it
  // into a C++ type that then gets put into an IValue.
  // That's what the TypePtr is for: it contains the information to do the
  // parsing. See torch::jit::toIValue for more information.
  static at::TypePtr packed_type() {
    // On windows CPU is support compiled autograd.
#if defined(_WIN32) && (defined(USE_CUDA) || defined(USE_ROCM))
    // NB: the if-constexpr usage triggers compilation errors on Windows
    // with certain compiler settings
    // (see https://github.com/pytorch/pytorch/pull/144707 for examples).
    // It's not clear what the problem is, so we're going to ignore it for now.
    TORCH_CHECK_NOT_IMPLEMENTED(
        false, "torch.compile not supported on Windows");
#else
    if constexpr (::std::is_same_v<T, at::Tensor>) {
      return at::TensorType::get();
    } else if constexpr (::std::is_same_v<T, int64_t>) {
      return at::IntType::get();
    } else if constexpr (::std::is_same_v<T, c10::SymInt>) {
      return at::SymIntType::get();
    } else if constexpr (::std::is_same_v<T, bool>) {
      return at::BoolType::get();
    } else if constexpr (::std::is_same_v<T, double>) {
      return at::FloatType::get();
    } else if constexpr (::std::is_same_v<T, c10::SymFloat>) {
      return at::SymFloatType::get();
    } else if constexpr (::std::is_same_v<T, c10::SymBool>) {
      return at::SymBoolType::get();
    } else if constexpr (::std::is_same_v<T, c10::Layout>) {
      return at::LayoutType::get();
    } else if constexpr (::std::is_same_v<T, ::std::string>) {
      return at::StringType::get();
    } else if constexpr (::std::is_same_v<T, at::Device>) {
      return at::DeviceObjType::get();
    } else if constexpr (::std::is_same_v<T, at::Scalar>) {
      return at::NumberType::get();
    } else if constexpr (::std::is_same_v<T, at::MemoryFormat>) {
      return at::MemoryFormatType::get();
    } else if constexpr (::std::is_same_v<T, at::ScalarType>) {
      return at::ScalarTypeType::get();
    } else {
      // If you got here, you have probably added a member of a new type
      // to a built-in C++ autograd node.
      // Unfortunately, we don't know how to handle this type yet.
      // To get this new type to work with Compiled Autograd, please
      // either change it to be an IValue-constructible type, or
      // define how to pack and unpack an object of this time into an IValue
      // by creating a specialization of IValuePacker for this type.
      // See NOTE: [Compiled Autograd and backward functions] for context.
      TORCH_CHECK_NOT_IMPLEMENTED(
          false, "IValuePacker not implemented for type");
      return at::NoneType::get();
    }
#endif
  }
};

template <>
struct IValuePacker<size_t> {
  static at::IValue pack(const size_t& t) {
    // We generally use size_t as the size of a list of Tensors or number of
    // dimensions. The number of dimensions generally do not exceed 64
    // (TensorIterator has that limitation), and lists of Tensors generally do
    // not exceed the int64_t max (you'd probably run out of RAM or run into
    // significant Tensor overhead). If you run into this limitation the fix is
    // to figure out how to pack size_t into int64_t. Note that size_t has some
    // weird behavior on Mac OS.
    uint64_t maximum_value = std::numeric_limits<int64_t>::max();
    TORCH_INTERNAL_ASSERT(
        static_cast<uint64_t>(t) <= maximum_value,
        "size_t too large to pack into IValue");
    return static_cast<int64_t>(t); // pack as int64_t
  }
  static size_t unpack(const at::IValue& t) {
    return static_cast<size_t>(t.toInt());
  }
  static at::TypePtr packed_type() {
    return IValuePacker<int64_t>::packed_type();
  }
};

template <>
struct IValuePacker<std::vector<at::SymInt>> {
  static at::IValue pack(const std::vector<at::SymInt>& t) {
    return t;
  }
  static std::vector<at::SymInt> unpack(const at::IValue& t) {
    // We need this because there's no t.to<std::vector<at::SymInt>>() override?
    return t.toSymIntVector();
  }
  static at::TypePtr packed_type() {
    return at::ListType::create(at::SymIntType::get());
  }
};

template <>
struct IValuePacker<VariableInfo> {
  static at::IValue pack(const VariableInfo& t) {
    auto tuple = std::make_tuple(
        t.layout, t.device, t.scalar_type, t.size, t.requires_grad, t.is_empty);
    return tuple;
  }
  static VariableInfo unpack(const at::IValue& t) {
    auto tuple = t.toTuple();
    const auto& tuple_elements = tuple->elements();
    const auto elements = tuple_elements.asArrayRef();
    TORCH_INTERNAL_ASSERT(elements.size() == 6);
    VariableInfo v;
    v.layout = elements[0].toLayout();
    v.device = elements[1].toDevice();
    v.scalar_type = elements[2].toScalarType();
    v.size = elements[3].toSymIntVector();
    v.requires_grad = elements[4].toBool();
    v.is_empty = elements[5].toBool();
    return v;
  }
  static at::TypePtr packed_type() {
    return at::TupleType::create({
        at::LayoutType::get(),
        at::DeviceObjType::get(),
        at::ScalarTypeType::get(),
        at::ListType::create(at::SymIntType::get()),
        at::BoolType::get(),
        at::BoolType::get(),
    });
  }
};

template <>
struct IValuePacker<caffe2::TypeMeta> {
  static at::IValue pack(const caffe2::TypeMeta& t) {
    return at::typeMetaToScalarType(t); // pack as at::ScalarType
  }
  static caffe2::TypeMeta unpack(const at::IValue& t) {
    return caffe2::TypeMeta::fromScalarType(t.to<at::ScalarType>());
  }
  static at::TypePtr packed_type() {
    return IValuePacker<at::ScalarType>::packed_type();
  }
};

inline std::optional<at::ScalarType> optTypeMetaToScalarType(
    const std::optional<caffe2::TypeMeta>& t) {
  if (t.has_value()) {
    return at::typeMetaToScalarType(t.value());
  } else {
    return std::nullopt;
  }
}

using packed_tensoroptions_t = std::tuple<
    std::optional<bool>,
    std::optional<at::MemoryFormat>,
    std::optional<at::Device>,
    std::optional<at::ScalarType>,
    std::optional<at::Layout>,
    std::optional<bool>>;

inline packed_tensoroptions_t pack_TensorOptions(const at::TensorOptions& t) {
  auto tuple = std::make_tuple(
      t.requires_grad_opt(),
      t.memory_format_opt(),
      t.device_opt(),
      optTypeMetaToScalarType(t.dtype_opt()),
      t.layout_opt(),
      t.pinned_memory_opt());
  return tuple;
}
inline at::TensorOptions unpack_TensorOptions(
    const packed_tensoroptions_t& tuple) {
  at::TensorOptions result;
  auto maybe_requires_grad = std::get<0>(tuple);
  if (maybe_requires_grad.has_value()) {
    result = result.requires_grad(maybe_requires_grad);
  }
  auto maybe_memory_format = std::get<1>(tuple);
  if (maybe_memory_format.has_value()) {
    result = result.memory_format(maybe_memory_format);
  }
  auto maybe_device = std::get<2>(tuple);
  if (maybe_device.has_value()) {
    result = result.device(maybe_device.value());
  }
  auto maybe_dtype = std::get<3>(tuple);
  if (maybe_dtype.has_value()) {
    result =
        result.dtype(caffe2::TypeMeta::fromScalarType(maybe_dtype.value()));
  }
  auto maybe_layout = std::get<4>(tuple);
  if (maybe_layout.has_value()) {
    result = result.layout(maybe_layout);
  }
  auto maybe_pinned_memory = std::get<5>(tuple);
  if (maybe_pinned_memory.has_value()) {
    result = result.pinned_memory(maybe_pinned_memory);
  }
  return result;
}

template <>
struct IValuePacker<at::TensorOptions> {
  static at::IValue pack(const at::TensorOptions& t) {
    return pack_TensorOptions(t);
  }
  static at::TensorOptions unpack(const at::IValue& t) {
    auto tuple = t.to<packed_tensoroptions_t>();
    return unpack_TensorOptions(tuple);
  }
  static at::TypePtr packed_type() {
    return at::TupleType::create(
        {at::OptionalType::create(at::BoolType::get()),
         at::OptionalType::create(at::MemoryFormatType::get()),
         at::OptionalType::create(at::DeviceObjType::get()),
         at::OptionalType::create(at::ScalarTypeType::get()),
         at::OptionalType::create(at::LayoutType::get()),
         at::OptionalType::create(at::BoolType::get())});
  }
};

template <>
struct IValuePacker<TypeAndSize> {
  static at::IValue pack(const TypeAndSize& t) {
    auto tuple = std::make_tuple(t.sym_sizes, pack_TensorOptions(t.options));
    return tuple;
  }
  static TypeAndSize unpack(const at::IValue& t) {
    auto tuple =
        t.to<std::tuple<std::vector<at::SymInt>, packed_tensoroptions_t>>();
    TypeAndSize result;
    result.sym_sizes = std::get<0>(tuple);
    result.options = unpack_TensorOptions(std::get<1>(tuple));
    return result;
  }
  static at::TypePtr packed_type() {
    return at::TupleType::create(
        {IValuePacker<std::vector<at::SymInt>>::packed_type(),
         IValuePacker<at::TensorOptions>::packed_type()});
  }
};

template <typename T>
struct IValuePacker<std::optional<T>> {
  static at::IValue pack(const std::optional<T>& t) {
    if (t.has_value()) {
      return IValuePacker<T>::pack(t.value());
    } else {
      return std::nullopt;
    }
  }
  static std::optional<T> unpack(const at::IValue& t) {
    if (t.isNone()) {
      return std::nullopt;
    } else {
      return IValuePacker<T>::unpack(t);
    }
  }
  static at::TypePtr packed_type() {
    return at::OptionalType::create(IValuePacker<T>::packed_type());
  }
};

template <typename T>
struct IValuePacker<std::vector<T>> {
  static at::IValue pack(const std::vector<T>& t) {
    if constexpr (::std::is_constructible_v<at::IValue, T>) {
      return t;
    }
    if (t.empty()) {
      auto lst = c10::impl::GenericList(at::AnyType::get());
      return lst;
    }
    auto type_ptr = IValuePacker<T>::pack(t[0]).type();
    auto lst = c10::impl::GenericList(type_ptr);
    for (const auto& elt : t) {
      lst.emplace_back(IValuePacker<T>::pack(elt));
    }
    return lst;
  }
  static std::vector<T> unpack(const at::IValue& t) {
    if constexpr (::std::is_constructible_v<at::IValue, T>) {
      return t.to<::std::vector<T>>();
    }
    std::vector<T> result;
    auto lst = t.toList();
    for (const at::IValue& elt : lst) {
      result.emplace_back(IValuePacker<T>::unpack(elt));
    }
    return result;
  }
  static at::TypePtr packed_type() {
    return at::ListType::create(IValuePacker<T>::packed_type());
  }
};

template <typename T>
struct IValuePacker<c10::List<T>> {
  static at::IValue pack(const c10::List<T>& t) {
    return IValuePacker<std::vector<T>>::pack(t.vec());
  }
  static c10::List<T> unpack(const at::IValue& t) {
    return c10::List<T>(IValuePacker<std::vector<T>>::unpack(t));
  }
  static at::TypePtr packed_type() {
    return IValuePacker<std::vector<T>>::packed_type();
  }
};

template <size_t N>
struct IValuePacker<std::array<bool, N>> {
  static at::IValue pack(const std::array<bool, N>& t) {
    std::vector<bool> result(t.begin(), t.end());
    return IValuePacker<std::vector<bool>>::pack(result);
  }
  static std::array<bool, N> unpack(const at::IValue& t) {
    std::array<bool, N> result;
    auto packed = IValuePacker<std::vector<bool>>::unpack(t);
    for (size_t i = 0; i < packed.size(); i++) {
      result[i] = packed[i];
    }
    return result;
  }
  static at::TypePtr packed_type() {
    return IValuePacker<std::vector<bool>>::packed_type();
  }
};

template <>
struct IValuePacker<at::TensorGeometry> {
  static at::IValue pack(const at::TensorGeometry& t) {
    auto tuple = std::make_tuple(
        t.sym_sizes().vec(), t.sym_strides().vec(), t.sym_storage_offset());
    return tuple;
  }
  static at::TensorGeometry unpack(const at::IValue& t) {
    auto tuple = t.to<std::tuple<
        std::vector<at::SymInt>,
        std::vector<at::SymInt>,
        at::SymInt>>();
    return at::TensorGeometry(
        std::get<0>(tuple), std::get<1>(tuple), std::get<2>(tuple));
  }
  static at::TypePtr packed_type() {
    return at::TupleType::create(
        {IValuePacker<std::vector<at::SymInt>>::packed_type(),
         IValuePacker<std::vector<at::SymInt>>::packed_type(),
         at::SymIntType::get()});
  }
};

template <>
struct IValuePacker<InputMetadata> {
  static at::IValue pack(const InputMetadata& t) {
    TORCH_INTERNAL_ASSERT(!t.is_nested_tensor());
    auto tuple = std::make_tuple(
        pack_TensorOptions(t.options()),
        t.shape_as_dim_vector().vec(),
        t.is_tensor_subclass());
    return tuple;
  }
  static InputMetadata unpack(const at::IValue& t) {
    auto tuple = t.to<
        std::tuple<packed_tensoroptions_t, std::vector<at::SymInt>, bool>>();

    return InputMetadata(
        unpack_TensorOptions(std::get<0>(tuple)),
        SymIntSmallVec(std::get<1>(tuple)),
        std::get<2>(tuple),
        false);
  }
  static at::TypePtr packed_type() {
    return at::TupleType::create(
        {IValuePacker<at::TensorOptions>::packed_type(),
         IValuePacker<std::vector<at::SymInt>>::packed_type(),
         at::BoolType::get()});
  }
};

template <typename T>
struct IValuePacker<at::OptionalArray<T>> {
  static at::IValue pack(const at::OptionalArray<T>& t) {
    return IValuePacker<std::optional<std::vector<T>>>::pack(t.list);
  }
  static at::OptionalArray<T> unpack(const at::IValue& t) {
    auto result = IValuePacker<std::optional<std::vector<T>>>::unpack(t);
    if (result.has_value()) {
      return {result.value()};
    } else {
      return {};
    }
  }
  static at::TypePtr packed_type() {
    return IValuePacker<std::optional<std::vector<T>>>::packed_type();
  }
};

// This is a helper struct for packing and unpacking multiple arguments into
// an ivalue_list. It leverages IValuePacker<T>.
struct PackedArgs {
  PackedArgs() = default;

  explicit PackedArgs(std::vector<at::IValue> stack_)
      : stack(std::move(stack_)) {}

  const std::vector<at::IValue>& vec() const {
    return stack;
  }

  template <typename T>
  void pack(const T& t) {
    stack.emplace_back(IValuePacker<T>::pack(t));
  }
  template <typename T>
  T unpack() {
    return IValuePacker<T>::unpack(std::move(stack[idx++]));
  }

  void pack_saved_data(const ska::flat_hash_map<std::string, at::IValue>& dct) {
    std::vector<std::string> keys;
    std::vector<at::IValue> values;
    for (const auto& [key, value] : dct) {
      keys.emplace_back(key);
      values.emplace_back(value);
    }
    pack(keys);
    for (const auto& value : values) {
      pack(value);
    }
  }

  ska::flat_hash_map<std::string, at::IValue> unpack_saved_data() {
    ska::flat_hash_map<std::string, at::IValue> dct;
    auto keys = unpack<std::vector<std::string>>();
    for (const auto& key : keys) {
      dct.insert({key, std::move(stack[idx++])});
    }
    return dct;
  }

 private:
  std::vector<at::IValue> stack;
  int64_t idx = 0;
};

} // namespace torch::dynamo::autograd

template <>
struct std::hash<torch::dynamo::autograd::CacheKey> {
  size_t operator()(const torch::dynamo::autograd::CacheKey& k) const {
    return k.hash();
  }
};
