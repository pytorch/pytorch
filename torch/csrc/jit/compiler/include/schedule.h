#pragma once

#include <memory>
#include <unordered_map>

#include "torch/csrc/jit/compiler/include/expr.h"
#include "torch/csrc/jit/compiler/include/ir.h"
#include "torch/csrc/jit/compiler/include/logging.h"
#include "torch/csrc/jit/compiler/include/refcount.h"
#include "torch/csrc/jit/compiler/include/tensor.h"

namespace torch {
namespace jit {
namespace compiler {
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

// A loop axis in the Tensor Expr trees.
// Even if two loops are identical in shapes, the should have separate loop
// axis. In other words, loop axes should be be shared among differnt loops.
class LoopAxis : public Cloneable<LoopAxis, ScheduleObject> {
 public:
  enum AxisType {
    kRegular, // a regular axis such as appeared in Compute
    kReduction, // a redution axis
  };

  const Var& var() const {
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
  bool is_leaf() const {}

  void CloneFrom(const LoopAxis* other);

 private:
  friend class ScheduleNode;
  friend class LoopAxisTransform;

  LoopAxis(
      const Var& loop_var,
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

  Var loop_var_;
  Range loop_range_;
  AxisType axis_type_;
  // TODO: check that only leaf axis can be used in axis tranforms.
  bool is_leaf_ = true;
  LoopAxisTransform* loop_axis_transform_ = nullptr;
  int output_group_index_ = -1;
};

// Loop Axis transformations
// Base class of loop axis transform. A number of input axes were taken, and
// several output groups are generated. Each output group is responsible for
// producing a subset within the input region. Note that each input axis can be
// used in at most one transform.
class LoopAxisTransform : public Cloneable<LoopAxisTransform, ScheduleObject> {
 public:
  LoopAxisTransform() {}

  // One Stmt for each output group
  virtual Stmt ConvertToNewArgs(const Stmt& stmt, int group_index){};

  int output_group_count() const {
    return outputs_.size();
  }
  int output_group_size(int group_index) const {
    CHECK(group_index >= 0 && group_index < outputs_.size());
    return outputs_[group_index].size();
  }
  LoopAxis* output(int group_index, int index) {
    CHECK(group_index >= 0 && group_index < outputs_.size());
    std::vector<LoopAxis*>& output_group = outputs_[group_index];
    CHECK(index >= 0 && index < output_group.size());
    return output_group[index];
  }

  void CloneFrom(const LoopAxisTransform* other);

 protected:
  friend class ScheduleNode;
  explicit LoopAxisTransform(const std::vector<LoopAxis*>& inputs)
      : inputs_(inputs) {
    // TODO: find a better way to set schedule.
    if (inputs.size() > 0) {
      this->set_schedule(inputs_[0]->schedule());
    }
  }

  void set_output_group_count(int group_count) {
    outputs_.resize(group_count);
  }

  void set_output_group(
      int group_index,
      const std::vector<LoopAxis*>& outputs) {
    CHECK(group_index >= 0 && group_index <= outputs_.size());
    outputs_[group_index] = outputs;
    for (LoopAxis* output : outputs) {
      output->set_output_group_index(group_index);
    }
  }

  void mark_loop_axis_internal(LoopAxis* axis) {
    axis->mark_as_internal();
  }

  // Override Schedule::NewAxis, but also sets current transform as the source.
  LoopAxis* NewAxis(const Var& loop_var, const Range& loop_range);

 private:
  std::vector<LoopAxis*> inputs_; // not owned
  std::vector<std::vector<LoopAxis*>> outputs_; // not owened
};

// Basic class for the Split Axis transforms.
class SplitAxisTransform
    : public Cloneable<SplitAxisTransform, LoopAxisTransform> {
 public:
  using BaseClass = Cloneable<SplitAxisTransform, LoopAxisTransform>;
  void CloneFrom(const SplitAxisTransform* other);
  int start() {
    return start_;
  }
  int stop() {
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
  int start_ = -1;
  int stop_ = -1;
};

class SplitAxisWithTail
    : public Cloneable<SplitAxisWithTail, SplitAxisTransform> {
 public:
  using BaseClass = Cloneable<SplitAxisWithTail, SplitAxisTransform>;
  void CloneFrom(const SplitAxisWithTail* other);
  Stmt ConvertToNewArgs(const Stmt& stmt, int output_group) override;
  SplitAxisWithTail() {}

 private:
  friend class ScheduleNode;
  SplitAxisWithTail(LoopAxis* loop_axis, int factor, bool factor_on_inner);
};

// TODO: Implement the following transforms.
class SplitAxisWithMask;
class FuseAxisTransform;

// Section: Tensor Expr Tree

// A tensor expr operation within the expression tree.
// This is often a leaf node that corresponds subset of the operations from a
// user-specified tensor expression.
// This operation, combined with all ancestor axis/nodes in the tree, determines
// the semantics of this operation.
class TensorExprOp : public Cloneable<TensorExprOp, ScheduleObject> {
 public:
  const Var& expr_var() const {
    return func_.func_var();
  }

  const Expr& body() const {
    return func_.body();
    ;
  }

  void CloneFrom(const TensorExprOp* other) {
    this->func_ = other->func_;
  }

  Stmt ElementStmt() {
    return this->func_.ElementStmt();
  }

 private:
  friend class ScheduleNode;
  TensorExprOp() {}
  explicit TensorExprOp(const Function& func) : func_(func) {}

  // TODO: this needs more work.
  // The ancestor-axes mark the region to evaluate expression.
  // We still need to know the buffer this writes to.
  Function func_;
};

// Part of the recursive node structure in the tensor expr tree.
// This variable type node could contain one of multiple types that follows:
//   * A single loop axis
//   * a tensor expr op.
class TensorExprNode : public Cloneable<TensorExprNode, ScheduleObject> {
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

class ScheduleNode : public RefCounted {
 public:
  // Section: user-facing functionalities.
  ~ScheduleNode();

  // Section: for schedule related internal functions.
  LoopAxis* NewAxis(const Var& loop_var, const Range& loop_range) {
    return NewObject<LoopAxis>(
        loop_var, loop_range, LoopAxis::kRegular, nullptr);
  }

  SplitAxisWithTail* NewSplitAxisWithTail(
      LoopAxis* loop_axis,
      int factor,
      bool factor_on_inner) {
    return NewObject<SplitAxisWithTail>(loop_axis, factor, factor_on_inner);
  }

  TensorExprOp* NewTensorExprOp(const Function& func) {
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
      const Var& loop_var,
      int factor,
      bool factor_on_inner,
      Var* outer_var,
      Var* inner_var,
      Var* tail_var,
      TensorExprNode** tail_op);

  Stmt Lower() {
    return Lower(root_node_);
  }

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
  explicit ScheduleNode(const std::vector<Tensor>& funcs);
  ScheduleObject* CloneScheduleObject(ScheduleObject* object);
  ScheduleObject* LookUpCloneScheduleObject(ScheduleObject* object);
  Stmt Lower(TensorExprNode* node);
  Stmt LowerNoSibling(TensorExprNode* node);

  std::vector<Tensor> tensors_;
  TensorExprNode* root_node_ = nullptr; // not owned
  std::vector<ScheduleObject*> schedule_objects_; // Owned
  // a mapping between old and new objects during the clone process.
  // whoever creates this map is responsible for releasing it.
  std::unique_ptr<CloneMap> clone_map_;
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
  if (object != nullptr) {
    return nullptr;
  }
  ScheduleNode* schedule = object->schedule();
  ScheduleObject* new_object = schedule->CloneScheduleObject(object);
  // TODO: switch to dynamic_cast when it becomes available.
  return static_cast<Object*>(new_object);
}

class Schedule : RefHandle<ScheduleNode> {
 public:
  static Schedule make(const std::vector<Tensor>& funcs) {
    return Schedule(new ScheduleNode(funcs));
  }

  Stmt Lower() {
    return node()->Lower();
  }

 private:
  using BaseClass = RefHandle<ScheduleNode>;
  Schedule(ScheduleNode* node) : BaseClass(node) {}
};

} // namespace schedule
} // namespace compiler
} // namespace jit
} // namespace torch
