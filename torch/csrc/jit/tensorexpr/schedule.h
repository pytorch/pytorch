#pragma once

#include <memory>
#include <unordered_map>

#include <c10/util/Logging.h>
#include "torch/csrc/jit/tensorexpr/expr.h"
#include "torch/csrc/jit/tensorexpr/ir.h"
#include "torch/csrc/jit/tensorexpr/tensor.h"

namespace torch {
namespace jit {
namespace tensorexpr {
namespace schedule {

// Schedule basics

// An object owned by a schedule. Objects from subclasses should be created
// through Schedule
// method through "new", and only released with the Schedule is destroyed
// through "delete".
class ScheduleNode;
class ScheduleObject {
 public:
  ScheduleObject() {}
  virtual ~ScheduleObject() {}
  ScheduleNode* schedule() {
    return schedule_;
  }

 protected:
  void AddClonePair(ScheduleObject* new_obj);
  void set_schedule(ScheduleNode* schedule) {
    schedule_ = schedule;
  }

 private:
  friend class ScheduleNode;
  virtual ScheduleObject* Clone() = 0;
  ScheduleObject(const ScheduleObject& other) = delete;
  const ScheduleObject& operator=(const ScheduleObject& other) = delete;

  ScheduleNode* schedule_ = nullptr; // not owned
};

// A CRTP helper class to add Clone support for an object.
template <class Object, class Base>
class Cloneable : public Base {
 public:
  // Forward the constructor to the underlying Base class
  // Note that this does not work for implicit argument conversion.
  // All arguments must be an exact match for their Base class counterpart.
  template <typename... Args>
  explicit Cloneable(Args... args) : Base(std::forward<Args>(args)...) {}

  Cloneable(Cloneable&& other) = delete;

 private:
  // The return type is set to ScheduleObject*. Otherwise, the compiler
  // complains about covariant override.
  ScheduleObject* Clone() override {
    Object* new_object = this->schedule()->template NewObject<Object>();
    this->AddClonePair(new_object);
    new_object->CloneFrom(static_cast<Object*>(this));
    return new_object;
  }
};

/// Loop Axis
class LoopAxisTransform;

// A loop axis in the Tensor ExprHandle trees.
// Even if two loops are identical in shapes, the should have separate loop
// axis. In other words, loop axes should be be shared among differnt loops.
class TORCH_API LoopAxis : public Cloneable<LoopAxis, ScheduleObject> {
 public:
  enum AxisType {
    kRegular, // a regular axis such as appeared in Compute
    kReduction, // a redution axis
  };

  const VarHandle& var() const {
    return loop_var_;
  }
  const Range& range() const {
    return loop_range_;
  }
  AxisType axis_type() const {
    return axis_type_;
  }
  const LoopAxisTransform* loop_axis_transform() const {
    return loop_axis_transform_;
  }
  // Whether this axis is a source axis.
  bool is_source() const {
    return loop_axis_transform_ == nullptr;
  }
  // Whether this axis is a leaf axis. Only leaf axes can be used in other axis
  // transformations. Internal axes are tracked for future computation, but
  // logically they disappear from users' perspective.
  bool is_leaf() const {
    return true;
  }

  void CloneFrom(const LoopAxis* other);

  const LoopOptions& loop_options() const {
    return loop_options_;
  }

 private:
  friend class ScheduleNode;
  friend class LoopAxisTransform;

  LoopAxis(
      const VarHandle& loop_var,
      const Range& loop_range,
      AxisType axis_type,
      LoopAxisTransform* transform)
      : loop_var_(loop_var),
        loop_range_(loop_range),
        axis_type_(axis_type),
        loop_axis_transform_(transform) {}

  LoopAxis() {}

  void mark_as_internal() {
    is_leaf_ = false;
  }

  void set_loop_axis_transform(LoopAxisTransform* transform) {
    loop_axis_transform_ = transform;
  }

  void set_output_group_index(int output_group_index) {
    output_group_index_ = output_group_index;
  }

  void set_gpu_block_index(int block_index) {
    loop_options_.set_gpu_block_index(block_index);
  }

  void set_gpu_thread_index(int thread_index) {
    loop_options_.set_gpu_thread_index(thread_index);
  }

  VarHandle loop_var_;
  Range loop_range_;
  AxisType axis_type_;
  // TODO: check that only leaf axis can be used in axis tranforms.
  bool is_leaf_ = true;
  LoopAxisTransform* loop_axis_transform_ = nullptr;
  int output_group_index_ = -1;
  LoopOptions loop_options_;
};

// Loop Axis transformations
// Base class of loop axis transform. A number of input axes were taken, and
// several output groups are generated. Each output group is responsible for
// producing a subset within the input region. Note that each input axis can be
// used in at most one transform.
class TORCH_API LoopAxisTransform
    : public Cloneable<LoopAxisTransform, ScheduleObject> {
 public:
  LoopAxisTransform() {}

  // One Stmt for each output group
  virtual Stmt* ConvertToNewArgs(Stmt* stmt, int group_index) {
    LOG(FATAL) << "unmiplemented";
    return nullptr;
  }

  virtual ExprHandle ConvertToNewArgs(ExprHandle* stmt, int group_index) {
    LOG(FATAL) << "unmiplemented";
    return ExprHandle();
  }

  int output_group_count() const {
    return outputs_.size();
  }
  int output_group_size(int group_index) const {
    CHECK(group_index >= 0 && group_index < (int)outputs_.size());
    return outputs_[group_index].size();
  }
  LoopAxis* output(int group_index, int index) {
    CHECK(group_index >= 0 && group_index < (int)outputs_.size());
    std::vector<LoopAxis*>& output_group = outputs_[group_index];
    CHECK(index >= 0 && index < (int)output_group.size());
    return output_group[index];
  }

  int input_size() const {
    return inputs_.size();
  }

  LoopAxis* input(int index) {
    CHECK(index >= 0 && index < (int)inputs_.size());
    return inputs_[index];
  }

  void CloneFrom(const LoopAxisTransform* other);

 protected:
  friend class ScheduleNode;
  explicit LoopAxisTransform(const std::vector<LoopAxis*>& inputs)
      : inputs_(inputs) {
    // TODO: find a better way to set schedule.
    if (inputs.size() > 0ULL) {
      this->set_schedule(inputs_[0]->schedule());
    }
  }

  void set_output_group_count(int group_count) {
    outputs_.resize(group_count);
  }

  void set_output_group(
      int group_index,
      const std::vector<LoopAxis*>& outputs) {
    CHECK(group_index >= 0 && group_index < (int)outputs_.size());
    outputs_[group_index] = outputs;
    for (LoopAxis* output : outputs) {
      output->set_output_group_index(group_index);
    }
  }

  void mark_loop_axis_internal(LoopAxis* axis) {
    axis->mark_as_internal();
  }

  // Override Schedule::NewAxis, but also sets current transform as the source.
  LoopAxis* NewAxis(const VarHandle& loop_var, const Range& loop_range);

 private:
  std::vector<LoopAxis*> inputs_; // not owned
  std::vector<std::vector<LoopAxis*>> outputs_; // not owened
};

// Basic class for the Split Axis transforms.
class TORCH_API SplitAxisTransform
    : public Cloneable<SplitAxisTransform, LoopAxisTransform> {
 public:
  using BaseClass = Cloneable<SplitAxisTransform, LoopAxisTransform>;
  void CloneFrom(const SplitAxisTransform* other);
  ExprHandle start() {
    return start_;
  }
  ExprHandle stop() {
    return stop_;
  }
  int factor() {
    return factor_;
  }
  bool factor_on_inner() {
    return factor_on_inner_;
  }
  SplitAxisTransform() {}

 protected:
  friend class ScheduleNode;
  SplitAxisTransform(LoopAxis* loop_axis, int factor, bool factor_on_inner);

 private:
  int factor_ = -1;
  bool factor_on_inner_ = true;
  ExprHandle start_;
  ExprHandle stop_;
};

class SplitAxisWithTail
    : public Cloneable<SplitAxisWithTail, SplitAxisTransform> {
 public:
  using BaseClass = Cloneable<SplitAxisWithTail, SplitAxisTransform>;
  void CloneFrom(const SplitAxisWithTail* other);
  Stmt* ConvertToNewArgs(Stmt* stmt, int output_group) override;
  ExprHandle ConvertToNewArgs(ExprHandle* stmt, int output_group) override;
  SplitAxisWithTail() {}

 private:
  friend class ScheduleNode;
  SplitAxisWithTail(LoopAxis* loop_axis, int factor, bool factor_on_inner);
  ExprHandle combined_loop_index(int output_group);
};

class SplitAxisWithMask
    : public Cloneable<SplitAxisWithMask, SplitAxisTransform> {
 public:
  using BaseClass = Cloneable<SplitAxisWithMask, SplitAxisTransform>;
  void CloneFrom(const SplitAxisWithMask* other);
  Stmt* ConvertToNewArgs(Stmt* stmt, int output_group) override;
  ExprHandle ConvertToNewArgs(ExprHandle* stmt, int output_group) override;
  SplitAxisWithMask() {}
  const ExprHandle& predicate() const {
    return predicate_;
  }

 private:
  friend class ScheduleNode;
  SplitAxisWithMask(LoopAxis* loop_axis, int factor, bool factor_on_inner);
  ExprHandle combined_loop_index(int output_group);

  ExprHandle predicate_; // original predicate
};

class FuseAxisTransform;

// Section: Tensor ExprHandle Tree

// A tensor expr operation within the expression tree.
// This is often a leaf node that corresponds subset of the operations from a
// user-specified tensor expression.
// This operation, combined with all ancestor axis/nodes in the tree, determines
// the semantics of this operation.
class TORCH_API TensorExprOp : public Cloneable<TensorExprOp, ScheduleObject> {
 public:
  const Var* expr_var() const {
    return func_->func_var();
  }

  const Expr* body() const {
    return func_->body();
  }

  Function* func() const {
    return func_;
  }

  void CloneFrom(const TensorExprOp* other) {
    this->func_ = other->func_;
    this->element_stmt_ = other->element_stmt_;
    this->predicates_ = other->predicates_;
  }

  Stmt* ElementStmt() const {
    return this->element_stmt_;
  }

  void ApplyLoopTransform(LoopAxisTransform* loop_transform, int group_index) {
    element_stmt_ =
        loop_transform->ConvertToNewArgs(element_stmt_, group_index);
    for (int i = 0; i < predicates_.size(); i++) {
      predicates_[i] =
          loop_transform->ConvertToNewArgs(&predicates_[i], group_index);
    }
  }

  void AddPredicate(const Expr* predicate) {
    if (predicate) {
      predicates_.push_back(ExprHandle(predicate));
    }
  }

  const std::vector<ExprHandle>& predicates() const {
    return predicates_;
  }

 private:
  friend class ScheduleNode;
  TensorExprOp() {}
  explicit TensorExprOp(Function* func)
      : func_(func), element_stmt_(func_->ElementStmt()) {}

  // TODO: this needs more work.
  // The ancestor-axes mark the region to evaluate expression.
  // We still need to know the buffer this writes to.
  Function* func_;
  Stmt* element_stmt_;
  std::vector<ExprHandle> predicates_;
};

// Part of the recursive node structure in the tensor expr tree.
// This variable type node could contain one of multiple types that follows:
//   * A single loop axis
//   * a tensor expr op.
class TORCH_API TensorExprNode
    : public Cloneable<TensorExprNode, ScheduleObject> {
 public:
  enum NodeType {
    // These could show up in the tensor expression trees.
    kEmptyValue, // The value in this node is empty, but could have siblings and
                 // children.
    kOperation, // this node records an tensor expr op.
    kAxis, // this node records a loop axis
  };

  NodeType node_type() const {
    return node_value_.node_type;
  }

  bool is_empty_value() const {
    return node_value_.node_type == kEmptyValue;
  }
  bool is_tensor_expr_op() const {
    return node_value_.node_type == kOperation;
  }
  bool is_loop_axis() const {
    return node_value_.node_type == kAxis;
  }

  TensorExprOp* tensor_expr_op() {
    DCHECK(is_tensor_expr_op());
    DCHECK(node_value_.tensor_expr_op != nullptr);
    return node_value_.tensor_expr_op;
  }
  const TensorExprOp* tensor_expr_op() const {
    return const_cast<TensorExprNode*>(this)->tensor_expr_op();
  }

  LoopAxis* loop_axis() {
    DCHECK(is_loop_axis());
    DCHECK(node_value_.loop_axis != nullptr);
    return node_value_.loop_axis;
  }
  const LoopAxis* loop_axis() const {
    return const_cast<TensorExprNode*>(this)->loop_axis();
  }

  TensorExprNode* parent() {
    return parent_;
  }
  TensorExprNode* first_child() {
    return first_child_;
  }
  TensorExprNode* next_sibling() {
    return next_sibling_;
  }

  void CloneFrom(const TensorExprNode* other);

 private:
  friend class ScheduleNode;

  TensorExprNode() {}

  // Create a new node under the current node.
  // Initialize the node list if it is still empty.
  // Set the child's parent to this node.
  TensorExprNode* NewNextSibling();
  TensorExprNode* NewFirstChild();

  void SetNextSibling(TensorExprNode* node);
  void SetFirstChild(TensorExprNode* node);
  // Set the parent of this node, and all its siblings
  void SetParent(TensorExprNode* parent);

  // Replace the subtree in "old_node" as the new subtree in "new_node".
  // All relevant sibings and parents links in the "new_node" are updated.
  // "old_node" might contain dangling pointers.
  static void ReplaceSubtree(
      TensorExprNode* old_node,
      TensorExprNode* new_node);

  void set_tensor_expr_op(TensorExprOp* expr_op) {
    DCHECK_EQ(node_value_.node_type, NodeType::kEmptyValue);
    node_value_.node_type = kOperation;
    node_value_.tensor_expr_op = expr_op;
  }

  void set_loop_axis(LoopAxis* loop_axis) {
    DCHECK_EQ(node_value_.node_type, NodeType::kEmptyValue);
    node_value_.node_type = kAxis;
    node_value_.loop_axis = loop_axis;
  }

  // A variable-type that unions different value types for this node.
  // TODO: isolate this into its own class, so different stage can have
  // different value types.
  struct NodeValue {
    // A variable-type payload with this load.
    NodeType node_type = kEmptyValue;
    //   node_type == kOperation,
    TensorExprOp* tensor_expr_op = nullptr;
    //   node_type_ == kAxis,
    LoopAxis* loop_axis = nullptr;

    void CloneFrom(const NodeValue* other);
  };

  // Data structures maintains the tensor expr tree.
  TensorExprNode* next_sibling_ = nullptr; // the next sibling of this node
  TensorExprNode* first_child_ = nullptr; // the first child of this node
  TensorExprNode* parent_ = nullptr; // the parent node of this node

  // Payload multi-type value in this node.
  NodeValue node_value_;
};

class TORCH_API ScheduleNode : public KernelScopedObject {
 public:
  // Section: user-facing functionalities.
  ~ScheduleNode();

  // Section: for schedule related internal functions.
  LoopAxis* NewAxis(const VarHandle& loop_var, const Range& loop_range) {
    return NewObject<LoopAxis>(
        loop_var, loop_range, LoopAxis::kRegular, nullptr);
  }

  SplitAxisWithTail* NewSplitAxisWithTail(
      LoopAxis* loop_axis,
      int factor,
      bool factor_on_inner) {
    return NewObject<SplitAxisWithTail>(loop_axis, factor, factor_on_inner);
  }

  SplitAxisWithMask* NewSplitAxisWithMask(
      LoopAxis* loop_axis,
      int factor,
      bool factor_on_inner) {
    return NewObject<SplitAxisWithMask>(loop_axis, factor, factor_on_inner);
  }

  TensorExprOp* NewTensorExprOp(Function* func) {
    return NewObject<TensorExprOp>(func);
  }

  TensorExprNode* NewTensorExprNode() {
    return NewObject<TensorExprNode>();
  }

  // Create an object
  template <typename T, typename... Args>
  T* NewObject(Args... args) {
    T* p = new T(std::forward<Args>(args)...);
    schedule_objects_.push_back(p);
    p->set_schedule(this);
    return p;
  }

  void SplitWithTail(
      TensorExprNode* expr_node,
      const VarHandle& loop_var,
      int factor,
      bool factor_on_inner,
      VarHandle* outer_var,
      VarHandle* inner_var,
      VarHandle* tail_var,
      TensorExprNode** tail_op);

  void SplitWithMask(
      TensorExprNode* expr_node,
      const VarHandle& loop_var,
      int factor,
      bool factor_on_inner,
      VarHandle* outer_var,
      VarHandle* inner_var);

  void ComputeInline(TensorExprNode* expr_node);

  void GPUExecConfig(
      TensorExprNode* expr_node,
      const std::vector<VarHandle>& blockIdx,
      const std::vector<VarHandle>& threadIdx);

  Stmt* Lower();

  using CloneMap = std::unordered_map<ScheduleObject*, ScheduleObject*>;
  CloneMap& clone_map() {
    return *clone_map_;
  }

  // An RAII object to manage the clone-map for any potential cloning.
  class ScopedCloneMap {
   public:
    ScopedCloneMap(ScheduleNode* schedule) : clone_map_(schedule->clone_map_) {
      if (clone_map_) {
        return;
      }
      clone_map_.reset(new CloneMap());
      map_initialized_ = true;
    }
    ~ScopedCloneMap() {
      if (!map_initialized_) {
        return;
      }
      clone_map_.reset();
    }
    CloneMap& clone_map() {
      return *clone_map_;
    }

   private:
    std::unique_ptr<CloneMap>& clone_map_;
    bool map_initialized_ = false;
  };

  template <class Object>
  friend Object* LookUpCloneObject(Object* object);

  template <class Object>
  friend Object* CloneObject(Object* object);

 private:
  friend class Schedule;
  explicit ScheduleNode(const std::vector<Tensor*>& funcs);
  ScheduleObject* CloneScheduleObject(ScheduleObject* object);
  ScheduleObject* LookUpCloneScheduleObject(ScheduleObject* object);
  Stmt* Lower(TensorExprNode* node);
  Stmt* LowerNoSibling(TensorExprNode* node);

  std::vector<Tensor*> output_tensors_;
  std::vector<Tensor*> internal_tensors_;
  std::vector<Function*> inlined_functions_;
  TensorExprNode* root_node_ = nullptr; // not owned
  std::vector<ScheduleObject*> schedule_objects_; // Owned
  // a mapping between old and new objects during the clone process.
  // whoever creates this map is responsible for releasing it.
  std::unique_ptr<CloneMap> clone_map_;
  class DependencyTracker;
  std::unique_ptr<DependencyTracker> dependency_tracker_;
};

template <class Object>
Object* LookUpCloneObject(Object* object) {
  if (object == nullptr) {
    return nullptr;
  }
  ScheduleNode* schedule = object->schedule();
  // TODO: switch to dynamic_cast
  return static_cast<Object*>(schedule->LookUpCloneScheduleObject(object));
}

template <class Object>
Object* CloneObject(Object* object) {
  if (object == nullptr) {
    return nullptr;
  }
  ScheduleNode* schedule = object->schedule();
  ScheduleObject* new_object = schedule->CloneScheduleObject(object);
  // TODO: switch to dynamic_cast when it becomes available.
  return static_cast<Object*>(new_object);
}

class TORCH_API Schedule {
 public:
  static Schedule make(const std::vector<Tensor*>& funcs) {
    return Schedule(new ScheduleNode(funcs));
  }

  explicit Schedule(const std::vector<Tensor*>& funcs)
      : node_(new ScheduleNode(funcs)) {}

  Stmt* Lower() {
    return node()->Lower();
  }

  Schedule(Schedule&& other) : node_(other.node_) {
    other.node_ = nullptr;
  }

 private:
  // TODO: temporarily disable the copy. We should decide whether the semantics
  // of this object.
  Schedule(const Schedule&) = delete;
  Schedule& operator=(const Schedule&) = delete;
  Schedule(ScheduleNode* node) : node_(node) {}
  ScheduleNode* node() {
    return node_;
  }
  const ScheduleNode* node() const {
    return node_;
  }

  ScheduleNode* node_ = nullptr;
};

} // namespace schedule
} // namespace tensorexpr
} // namespace jit
} // namespace torch
