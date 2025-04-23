#pragma once

#include <torch/csrc/autograd/anomaly_mode.h>
#include <torch/csrc/autograd/edge.h>
#include <torch/csrc/autograd/grad_mode.h>
#include <torch/csrc/autograd/graph_task.h>
#include <torch/csrc/autograd/input_metadata.h>
#include <torch/csrc/autograd/saved_variable.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/utils/python_stub.h>
#include <torch/csrc/utils/variadic.h>

#include <ATen/SequenceNumber.h>
#include <ATen/core/Tensor.h>
#include <ATen/record_function.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace torch::autograd {

struct Edge;
struct FunctionPostHook;
struct FunctionPreHook;

using tensor_list = std::vector<at::Tensor>;
using variable_list = std::vector<Variable>;
using edge_list = std::vector<Edge>;
using saved_variable_list = std::vector<SavedVariable>;
using ivalue_list = std::vector<c10::IValue>;
using functional_apply_t = std::function<
    variable_list(const variable_list&, const std::vector<c10::IValue>&)>;
using IndexRange = std::pair<size_t, size_t>;
using torch::dynamo::autograd::CompiledNodeArgs;
using torch::dynamo::autograd::PackedArgs;
using torch::dynamo::autograd::SwapSavedVariables;

// Custom deleter to prevent stack overflows.
TORCH_API void deleteNode(Node* function);

// Guard that sets and restores the evaluating node
class NodeGuard {
 public:
  explicit NodeGuard(std::shared_ptr<Node> node);
  ~NodeGuard();

 private:
  std::shared_ptr<Node> last_evaluating_node_;
};

// Return the Node currently being evaluated (if any)
// This is only set during the backward pass while a Node is being
// executed.
TORCH_API std::shared_ptr<Node> get_current_node();

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                               Node
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// A `Node` is an abstract class that represents an operation taking zero
// or more input `Variable`s and producing zero or more output `Variable`s. All
// functions in PyTorch's autograd machinery derive from this class and
// override its `apply` method. Instances of such subclasses will then be
// invokable via the call operator.
//
//                    Nodes in the Autograd Graph
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// When viewing the autograd system as a graph, `Node`s are the vertices or
// nodes, connected to each other via (directed) `Edge`s, which themselves are
// represented via (`Node`, input_nr) pairs. `Variable`s are the outputs to
// and inputs of `Node`s, and travel between these edges during execution
// of the graph. When two or more `Edge`s (from different sources) point at the
// same input to a `Node`, the values produced along all of these edges are
// implicitly summed prior to being forwarded to the target `Node`.
//
//                              Hierarchy
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Subclasses usually represent differentiable functions as well as their
// gradient operators. Note, however, that due to the very general definition
// of a `Node` taking *zero* or more inputs and producing *zero* or more
// outputs, uses of `Node`s are flexible and extend beyond purely
// mathematical operations. For example, the `AccumulateGrad` function is a
// *sink*: it takes one input, but produces no outputs, instead accumulating
// the input as a side effect. At the other extreme, the `GraphRoot` function
// receives no inputs from other functions, but produces multiple outputs.
//
//                              Interface
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// The most important method on `Node` is the call operator, which takes in
// a list of variables and produces a list of variables. The precise size of
// these lists can be determined with `num_inputs()` and `num_outputs()`.
// `Node`s are stitched together via their `next_edge` interface, which let
// you manipulate the set of outgoing edges of a `Node`. You can add an
// edge with `add_next_edge()`, retrieve an edge with `next_edge(index)` and
// iterate over them via the `next_edges()` method. Other methods exist for
// integration with the JIT and other parts of PyTorch. Every `Node` has a
// *sequence number* that increases monotonically in the order of `Node`
// construction. It can be retrieved via the `sequence_nr()` method. Note that
// this sequence number is *thread local*. This means that when `Node`s
// `A`, `B` and `C` are created consecutively in the same thread, their
// sequence numbers will be ordered `A` < `B` < `C`. If, however, `A` and `B`
// are created in one thread and `C` is created in a new thread, there are *no
// guarantees* w.r.t. the ordering of `C` relative to `A` or `B`.
// See NOTE [ Sequence Number] for more details on the usages of sequence
// number.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
struct TORCH_API Node : std::enable_shared_from_this<Node> {
 public:
  /// Construct a new `Node` with the given `next_edges`
  explicit Node(uint64_t sequence_nr, edge_list&& next_edges = edge_list())
      : sequence_nr_(sequence_nr), next_edges_(std::move(next_edges)) {
    for (const Edge& edge : next_edges_) {
      update_topological_nr(edge);
    }

    if (AnomalyMode::is_enabled()) {
      metadata()->store_stack();

      // If anomaly mode is enabled and graph is constructed, then assign the
      // currently evaluating node as the parent of this node.
      // A parent is a Node where this Node is created.
      // We are tracking the parents to track multiple backward operations.
      assign_parent();
    }

    // Store the thread_id of the forward operator.
    // See NOTE [ Sequence Numbers ]
    thread_id_ = at::RecordFunction::currentThreadId();
  }

  explicit Node(edge_list&& next_edges = edge_list())
      : Node(
            /*sequence_nr=*/at::sequence_number::get_and_increment(),
            std::move(next_edges)) {}

  /// Nodes are neither copyable nor moveable.
  Node(const Node& other) = delete;
  Node(Node&& other) = delete;
  Node& operator=(const Node& other) = delete;
  Node& operator=(Node&& other) = delete;
  virtual ~Node() = default;

  std::shared_ptr<Node> getptr() {
    return shared_from_this();
  }
  /// Evaluates the function on the given inputs and returns the result of the
  /// function call.
  variable_list operator()(variable_list&& inputs) {
    // In the first iteration of named tensors, autograd ignores names and
    // operates on unnamed tensors. In the long term, autograd should
    // probably operate with names.
    at::NoNamesGuard no_names_guard;

#ifdef USE_ROCM
    // Keep track of backward pass for rocblas.
    at::ROCmBackwardPassGuard in_backward;
#endif

    auto step_callbacks =
        at::getStepCallbacksUnlessEmpty(at::RecordScope::BACKWARD_FUNCTION);
    if (C10_UNLIKELY(step_callbacks.has_value())) {
      at::RecordFunction guard(std::move(*step_callbacks));
      // Using sequence number and thread id to correlate with
      // the forward pass function
      guard.setForwardThreadId(thread_id_);
      if (guard.needsInputs()) {
        std::vector<c10::IValue> inputs_vec(inputs.begin(), inputs.end());
        guard.before(
            name(),
            c10::ArrayRef<const c10::IValue>(
                inputs_vec.data(), inputs_vec.size()),
            static_cast<int64_t>(sequence_nr()));
      } else {
        guard.before(name(), static_cast<int64_t>(sequence_nr()));
      }
      return apply(std::move(inputs));
    } else {
      return apply(std::move(inputs));
    }
  }

  // Graph Connectivity API
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  // Inputs. NOTE: inputs of the grad_fn correspond to Tensor outputs of the
  // forward function.

  // Marker for expected undefined input
  struct undefined_input {};

  /// Adds the type and shape metadata for a new input. Returns the index of
  /// of the new input.
  uint32_t add_input_metadata(
      const at::TensorOptions& options,
      c10::SymIntArrayRef shape,
      bool is_tensor_subclass,
      bool is_nested) noexcept {
    uint32_t input_nr = input_metadata_.size();
    auto meta_shape = MetadataShape{std::in_place_type<SymIntSmallVec>, shape};
    input_metadata_.emplace_back(
        options, meta_shape, is_tensor_subclass, is_nested);
    return input_nr;
  }

  uint32_t add_input_metadata(const at::Tensor& t) noexcept {
    uint32_t input_nr = input_metadata_.size();
    input_metadata_.emplace_back(t);
    return input_nr;
  }

  /// Adds a placeholder for an input that will not be used.
  uint32_t add_input_metadata(undefined_input u) noexcept {
    uint32_t input_nr = input_metadata_.size();
    input_metadata_.emplace_back();
    return input_nr;
  }

  uint32_t num_inputs() const noexcept {
    return input_metadata_.size();
  }

  const InputMetadata& input_metadata(size_t index) const {
    return input_metadata_[index];
  }

  // Danger: not thread safe, caller must protect with lock
  InputMetadata& mutable_input_metadata(size_t index) {
    return input_metadata_[index];
  }

  /**
   * Note: Function Streams
   * A function's stream (for a given device type) is the stream of the first
   * element of its input buffer on a device of that type.
   *
   * If all elements are on the same device they MUST share a stream. If
   * elements are on different devices (across multiple GPUs, for example)
   * they may have different streams.
   */
  std::optional<c10::Stream> stream() {
    auto opt_device_type = at::getAccelerator();
    if (!opt_device_type.has_value()) {
      return std::nullopt;
    }
    for (const auto& metadata : input_metadata_) {
      if (metadata.device().type() == opt_device_type.value())
        return metadata.stream();
    }

    return std::nullopt;
  }

  // Used by the engine to determine what device thread to run on
  at::Device device() {
    // Since we pick the first non-CPU tensor, this won't work with
    // mixed device-type operations (e.g., an op that is both CUDA
    // and XLA).  This is *incredibly* unlikely, so we don't worry
    // about it.
    for (const auto& metadata : input_metadata_) {
      auto device = metadata.device();
      if (device.type() != at::kCPU) {
        return device;
      }
    }
    // Only report to the CPU thread if there really were no tensors
    // from other devices.
    return at::kCPU;
  }

  void clear_input_metadata() {
    input_metadata_.clear();
  }

  // Outputs ("Next Edges")

  void update_topological_nr(const Edge& edge) {
    TORCH_INTERNAL_ASSERT(
        !has_parent_,
        "Cannot update a node's topological_nr after it already has a parent."
        " If we allow this, we can no longer guarantee that a parent's"
        " topo_nr is always greater than those of all its children")
    Node* node = edge.function.get();
    if (node) {
      auto topo_nr = node->topological_nr();
      if (topological_nr_ <= topo_nr) {
        topological_nr_ = topo_nr + 1;
      }
    }
  }

  void set_next_edge(size_t index, Edge edge) {
    update_topological_nr(edge);
    next_edges_[index] = std::move(edge);
  }

  void add_next_edge(Edge edge) {
    update_topological_nr(edge);
    next_edges_.emplace_back(std::move(edge));
  }

  void set_next_edges(edge_list&& next_edges) {
    next_edges_ = std::move(next_edges);
    for (const auto& next_edge : next_edges_) {
      update_topological_nr(next_edge);
    }
  }

  const Edge& next_edge(size_t index) const noexcept {
    return next_edges_[index];
  }

  const edge_list& next_edges() const noexcept {
    return next_edges_;
  }

  edge_list& next_edges() noexcept {
    return next_edges_;
  }

  uint32_t num_outputs() const noexcept {
    return next_edges_.size();
  }

  // Miscellaneous Methods
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  /// NOTE [ Sequence Number]
  ///
  /// The sequence_nr has two main usages in autograd:
  ///
  /// 1) Helps determine the node's execution priority in the engine.
  ///    All else being equal, nodes with higher priority numbers are executed
  ///    first. Thus, nodes corresponding to ops executed later are the first to
  ///    be executed in the backward pass. One caveat is that we prioritize
  ///    AccumulateGrad nodes by explicitly setting its sequence_nr to be
  ///    UINT64_MAX.
  /// 2) The sequence number of this `Node` is paired with with thread_id it was
  /// created in
  ///    as a unique identifier by the profiler to annotate recorded events.
  ///    The purpose of this is to help users (and possibly programs)
  ///    interpreting the profiler's output to correlate backward nodes with its
  ///    forward ops. We need both sequence_nr and thread_id to identify a node
  ///    because sequence_nr is thread_local, i.e., starts counting up from zero
  ///    in a new thread
  uint64_t sequence_nr() const noexcept {
    return sequence_nr_;
  }

  void set_sequence_nr(uint64_t sequence_nr) {
    sequence_nr_ = sequence_nr;
  }

  // NOTE [ Topological Number ]
  //
  // topological_nr is used to prune branches in the DAG during autograd
  // discovery as maintaining topological_nr helps us check in O(1) if there
  // does NOT exist a directed path between two nodes.
  //
  // The topological order number of this `Node` representing the length of the
  // longest possible path from this Node to any leaf node. If you are leaf
  // node, aka AccumulateGrad, this will be zero. This value has the property
  // that For every pair of nodes X, Y in G, existence of a directed path from X
  // to Y implies topo_nr(X) > topo_nr(Y). The converse is not true, however, so
  // we cannot prove existence of a path from X to Y, only non-existence.
  //
  // One assumption we make when using topo_nr is that once a node
  // has been used, i.e., has a parent node, its own topo_nr does not change
  // we have added some checks with the `has_parent_` field to enforce this.
  //
  // What NOT to do:
  //
  //   1) 2 -> 1 -> 0               In this diagram we label nodes with their
  //   topo_nr.
  //      2 -> 1 -> 0               We have two simple graphs that can each
  //      arise from
  //                                `t.exp().exp()`, for example.
  //   2)        2 -> 1 -> 0
  //            /
  //      2 -> 1 -> 0               We add 2 as a next edge to 1 even though 1
  //      already
  //                                has a parent.
  //   3)        2 -> 1 -> 0
  //            /
  //      2 -> 3 -> 0               2 < 3, yet there exists a path from 2 to 3!
  //
  uint64_t topological_nr() const noexcept {
    has_parent_ = true;
    return topological_nr_;
  }

  // assigning a node as a parent to this node
  void assign_parent();

  /// Id of the thread that created Node
  uint64_t thread_id() const noexcept {
    return thread_id_;
  }

  /// Returns the name of the dynamic type of the function, for debugging.
  virtual std::string name() const;

  /// The difference between functions `should_compute_output` and
  /// `task_should_compute_output`:
  /// - `should_compute_output` should only be used during graph construction
  /// and takes into account only requires_grad information
  /// - `task_should_compute_output` should only be called during the backward
  /// pass (unless called directly through grad_fn) and takes into account the
  /// current graph task.  Specifically, the autograd engine trims unnecessary
  /// edges when `inputs` are specified, and during backward untrimmed nodes
  /// left on the graph can/should check `task_should_compute_output` to see if
  /// any outgoing edges have been trimmed by the engine. If that is the case,
  /// gradient computation wrt those edges can be omitted.
  ///
  /// Returns true if the particular output edge is active, and that particular
  /// output of this function should be computed.
  bool should_compute_output(size_t output_edge_index) const {
    TORCH_CHECK(output_edge_index < num_outputs(), "Index out of range");
    return next_edges_[output_edge_index].is_valid();
  }

  /// Returns true if any of the output edges in any of the ranges are active.
  bool should_compute_output(std::initializer_list<IndexRange> idxs) const {
    return std::any_of(idxs.begin(), idxs.end(), [this](IndexRange range) {
      for (const auto i : c10::irange(range.first, range.second)) {
        if (should_compute_output(i))
          return true;
      }
      return false;
    });
  }

  /// Same as the above `should_compute_output` function but will also
  /// check whether this edge is needed within the current graph task.
  bool task_should_compute_output(size_t output_edge_index) const {
    TORCH_CHECK(output_edge_index < num_outputs(), "Index out of range");
    const auto& next = next_edges_[output_edge_index];
    if (next.is_valid()) {
      const auto exec_info = get_current_graph_task_exec_info();
      if (exec_info && !exec_info->empty()) {
        auto it = exec_info->find(next.function.get());
        if (it == exec_info->end() || !it->second.should_execute()) {
          return false; // this edge is not needed for the current graph_task
        }
      }
      return true;
    }
    return false;
  }

  /// Returns true if any of the output edges in any of the ranges are active
  /// and should be computed in the current graph task.
  bool task_should_compute_output(
      std::initializer_list<IndexRange> idxs) const {
    return std::any_of(idxs.begin(), idxs.end(), [this](IndexRange range) {
      for (const auto i : c10::irange(range.first, range.second)) {
        if (task_should_compute_output(i))
          return true;
      }
      return false;
    });
  }

  /// Returns the `PyObject` stored for this `Node` (for Python
  /// interaction).
  PyObject* pyobj() const noexcept {
    return pyobj_;
  }

  /// Sets the `PyObject` stored for this `Node` (for Python interaction).
  void set_pyobj(PyObject* pyobj) noexcept {
    pyobj_ = pyobj;
  }

  /// Returns the anomaly metadata stored for this `Node`.
  /// If none exist, creates a new empty one.
  AnomalyMetadata* metadata() noexcept;

  // Hook API
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  uintptr_t add_post_hook(std::unique_ptr<FunctionPostHook>&& post_hook) {
    post_hooks_.emplace_back(std::move(post_hook));
    // Use the raw pointer as the unique key to identify this hook. This key
    // can then be used in del_post_hook(key) to remove this hook.
    return reinterpret_cast<std::uintptr_t>(post_hooks_.back().get());
  }

  const std::vector<std::unique_ptr<FunctionPostHook>>& post_hooks()
      const noexcept {
    return post_hooks_;
  }

  // delete a post hook matching the key
  bool del_post_hook(const uintptr_t& key) {
    for (auto it = post_hooks_.begin(); it != post_hooks_.end(); ++it) {
      if (key == reinterpret_cast<std::uintptr_t>(it->get())) {
        post_hooks_.erase(it);
        return true;
      }
    }
    return false;
  }

  std::vector<std::unique_ptr<FunctionPostHook>>& post_hooks() noexcept {
    return post_hooks_;
  }

  void add_pre_hook(std::unique_ptr<FunctionPreHook>&& pre_hook) {
    pre_hooks_.emplace_back(std::move(pre_hook));
  }

  void add_tensor_pre_hook(std::unique_ptr<FunctionPreHook>&& pre_hook) {
    tensor_pre_hooks_.emplace_back(std::move(pre_hook));
  }

  void add_retains_grad_hook(
      std::unique_ptr<FunctionPreHook>&& pre_hook,
      size_t output_idx) {
    retains_grad_hooks_[output_idx] = std::move(pre_hook);
  }

  std::unique_ptr<FunctionPreHook> pop_retains_grad_hook(size_t output_idx) {
    auto ret = std::move(retains_grad_hooks_[output_idx]);
    retains_grad_hooks_.erase(output_idx);
    return ret;
  }

  const std::vector<std::unique_ptr<FunctionPreHook>>& pre_hooks()
      const noexcept {
    return pre_hooks_;
  }

  std::vector<std::unique_ptr<FunctionPreHook>>& pre_hooks() noexcept {
    return pre_hooks_;
  }

  virtual std::vector<std::unique_ptr<FunctionPreHook>>&
  tensor_pre_hooks() noexcept {
    return tensor_pre_hooks_;
  }

  virtual std::unique_ptr<PostAccumulateGradHook>& tensor_post_acc_grad_hooks()
      const noexcept {
    static std::unique_ptr<PostAccumulateGradHook> empty = nullptr;
    return empty;
  }

  std::unordered_map<size_t, std::unique_ptr<FunctionPreHook>>&
  retains_grad_hooks() noexcept {
    return retains_grad_hooks_;
  }

  // Customization Points for Subclasses
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  /// Releases saved variables if the operation won't be reused.
  virtual void release_variables() {}

  /// Called before an apply if `release_variables()` is going to be called.
  /// Allows larger ops like `InterpreterAutogradFunction` to incrementally
  /// release variables as they run.
  virtual void will_release_variables() {}

  /// Returns true if this function is traceable. An op is traceable if all
  /// operations happening within `apply()` are performed on autograd
  /// `Variables` (i.e. apply mostly instantiates and applies other functions).
  virtual bool is_traceable() {
    return false;
  }

  /// A `Node` is said to pass state transparently to backward, if the
  /// state consists only of (Saved)Variables and only non-variable objects
  /// that parameterize the operation in some way that defines the graph
  /// structure AND the backward function is traceable. In particular,
  /// parametrization MUST NOT depend on the data of any `Variable`.
  /// TODO: it might be possible to handle cases where backward is
  /// non-traceable but state passing could be considered transparent. This
  /// will probably depend on saved_variable_list being mutable.
  /// NOTE: this value matters only if is_traceable() returns false.
  virtual bool passes_state_transparently() {
    return false;
  }

  // see [Note: Compiled Autograd]
  // Used by compiled autograd to
  //   1) Extract tensors/symint args
  //   2) Collect node information for specialization and caching
  // Implementations in subclasses should call args.collect() with all node
  // attrs. These functions are only called durring backward.
  virtual void compiled_args(CompiledNodeArgs& args) const {
    throw std::runtime_error(
        std::string("compiled_args not implemented: ") + name());
  }

  // Used by compiled autograd to call apply() with different saved tensors
  // Implementations should call saved.before() on all attrs, then apply(), then
  // saved.after() on all attrs in the same order.
  virtual variable_list apply_with_saved(
      const variable_list& inputs,
      SwapSavedVariables& saved) {
    throw std::runtime_error(
        std::string("apply_with_saved not implemented: ") + name());
  }

  // If this node is the AOTBackward node produced by torch.compile.
  // Compiled Autograd special-cases on this information.
  virtual bool is_aot_backward() const {
    return false;
  }

 protected:
  /// Performs the `Node`'s actual operation.
  virtual variable_list apply(variable_list&& inputs) = 0;

  /// Calls `apply()`, but instruments it with tracing machinery.
  variable_list traced_apply(variable_list inputs);

  // Sequence number used to correlate backward nodes with forward ops in the
  // profiler and provide determinism in the engine.
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  uint64_t sequence_nr_;

  // See NOTE [ Topological Number ]
  uint64_t topological_nr_ = 0;

  // Tracks whether this node has been added as the next_edge of another node
  // via set_next_edge(s), which always calls topological_nr() of all its
  // children See NOTE [ Topological Number ] for why we need this.
  mutable bool has_parent_ = false;

  // Id of the thread that created the instance
  uint64_t thread_id_ = 0;

  // Note [Thread Safety on Autograd Node]
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Autograd Engine let the owning thread which calls Engine::execute to drive
  // the GraphTask execution, there might be cases that part of the GraphTask is
  // shared across different `backward()` or `grad()` calls, i.e. fork new
  // threads in the middle of the forward and call `backward()` separately from
  // different threads. We need to protect the thread safety on NodeTask to
  // prevent data racing on shared variables read/write.
  //
  // NB: This is only needed for Autograd Nodes that runs on CPU, technically
  // "CUDA", "XLA" nodes don't need locking because device threads are always
  // single threaded.
  //
  // Here we add a thread mutex to help protect the Node's thread safety, so
  // that different threads cannot race the shared data when executing the same
  // NodeTask from multiple CPU threads. It IS the user/developer responsibility
  // to take advantage of this mutex to protect the thread safety of their
  // autograd Node. The general strategy of thread safety on autograd Node:
  //
  // 1. User should lock the mutex during Node::release_variables() if the Node
  // needs
  //    to release the variables on the fly, this serve the purpose that when we
  //    release saved_variables from one thread, no other threads can release
  //    the saved variables concurrently. call the Node::apply(),
  // 2. User should lock the mutex during Node::apply(), this is to ensure Node
  // that
  //    writing to the shared variable are not racing across threads (i.e.
  //    AccumulateGrad and custom C++ Autograd Node if writing to shared
  //    variables )
  // 3. item 2 and item 3 should work together so that when we release saved
  // variables
  //    from one thread, no other threads can call Node::apply(), this ensures
  //    the variable references from other threads aren't dangling.
  // 4. if the Node don't release any variables and no shared data read/write in
  // the Node
  //    i.e. purely functional, user don't need to lock the mutex
  //
  // This way we could protect the thread safety on Autograd Node, but we could
  // still not protect the thread safety on Node pre/post C++ hooks (python
  // hooks are automatically thread safe), we rely on the user to write thread
  // safe C++ hooks if they want the hook to be correctly applied in
  // multithreading environment.
  std::mutex mutex_;

  edge_list next_edges_;
  PyObject* pyobj_ = nullptr; // weak reference
  std::unique_ptr<AnomalyMetadata> anomaly_metadata_ = nullptr;

  // NOTE [Hooks ordering]
  // We have 3 separate fields for pre hooks registered to the autograd nodes
  // because the conditions under which they execute are different, and we
  // want more fine-grained control over the order in which different types
  // of hooks are executed.
  // - pre_hooks  are only executed when the node itself is executed
  // - tensor_pre_hook is executed as long as the engine traverses over it
  //   even if that node won't be executed.
  // - retains_grad_hook are like tensor_pre_hooks except they are always
  //   ordered after all other tensor pre hooks
  std::vector<std::unique_ptr<FunctionPreHook>> pre_hooks_;
  std::vector<std::unique_ptr<FunctionPreHook>> tensor_pre_hooks_;
  std::unordered_map<size_t, std::unique_ptr<FunctionPreHook>>
      retains_grad_hooks_;
  std::vector<std::unique_ptr<FunctionPostHook>> post_hooks_;
  at::SmallVector<InputMetadata, 2> input_metadata_;
};

/// See Node::is_traceable() for definition.
struct TraceableFunction : public Node {
  using Node::Node;
  bool is_traceable() final {
    return true;
  }
};

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                       Associated Free Nodes
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

namespace detail {
// Implementation of `collect_next_edges` (see below).
struct MakeNextFunctionList : IterArgs<MakeNextFunctionList> {
  edge_list next_edges;
  using IterArgs<MakeNextFunctionList>::operator();
  void operator()(const Variable& variable) {
    if (variable.defined()) {
      next_edges.emplace_back(impl::gradient_edge(variable));
    } else {
      next_edges.emplace_back();
    }
  }
  void operator()(const Variable* variable) {
    operator()(*variable);
  }
  void operator()(const std::optional<Variable>& variable) {
    if (variable.has_value()) {
      operator()(*variable);
    } else {
      next_edges.emplace_back();
    }
  }
};
} // namespace detail

/// Create an `Edge` between the given `variable` and the `function`, which is
/// assumed to be the gradient function of this variable (i.e. the function
/// through which this variable is backpropagated during the backward pass).
/// This sets the `grad_fn` property of the `variable`. This function assumes
/// that the `Variable` is a new input to the gradient function and its
/// `input_nr` thus equal to `function->num_inputs()`. Additionally, it
/// increments the `Node`'s number of inputs by one. Approximately
/// equivalent to `variable.set_gradient_edge(function,
/// function->add_input_metadata(variable.dispatch_type(), variable.sizes()))`.
/// If you don't want the `Node`'s `num_inputs` to be incremented, use
/// `set_gradient_edge` directly.
inline void create_gradient_edge(
    Variable& variable,
    std::shared_ptr<Node> function) {
  // Copy before move.
  const auto input_nr = function->add_input_metadata(variable);
  impl::set_gradient_edge(variable, {std::move(function), input_nr});
}

/// Return true if any of the variables in the list require a gradient.
inline bool any_variable_requires_grad(const variable_list& variables) {
  return std::any_of(
      variables.begin(), variables.end(), [](const Variable& variable) {
        return variable.defined() && variable.requires_grad();
      });
}

/// Return the next edges of all the given variables, or tuples of variables.
template <typename... Variables>
edge_list collect_next_edges(Variables&&... variables) {
  detail::MakeNextFunctionList make;
  make.apply(std::forward<Variables>(variables)...);
  return std::move(make.next_edges);
}

struct TypeAndSize {
  TypeAndSize() = default;
  /* implicit */
  TypeAndSize(const at::Tensor& t)
      : sym_sizes(t.sym_sizes().vec()), options(t.options()) {}

  at::Tensor zeros();

  std::vector<c10::SymInt> sym_sizes;
  at::TensorOptions options;
};

} // namespace torch::autograd
