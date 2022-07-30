#pragma once

#include <ATen/core/symbol.h>

#include <functional>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <c10/core/ScalarType.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/Flags.h>
#include <torch/csrc/lazy/core/hash.h>
#include <torch/csrc/lazy/core/ir_metadata.h>
#include <torch/csrc/lazy/core/shape.h>

C10_DECLARE_bool(ltc_enable_dynamic_shapes);

namespace torch {
namespace lazy {

static const hash_t kHashSeed(static_cast<uint32_t>(0x5a2d296e9));

class Node;
struct Output;
struct Value;

using NodePtr = std::shared_ptr<Node>;

// The Kind of operation a Node can be associated to.
struct TORCH_API OpKind {
  OpKind() = default;
  explicit OpKind(c10::Symbol op) : op(op) {}

  bool operator==(const OpKind& rhs) const {
    return op == rhs.op;
  }
  bool operator!=(const OpKind& rhs) const {
    return !operator==(rhs);
  }
  bool operator<(const OpKind& rhs) const {
    return c10::unique_t(op) < c10::unique_t(rhs.op);
  }

  hash_t hash() const;

  std::string ToString() const {
    return op.toQualString();
  }

  // Retrieves an existing operation object, or creates a new one. Operations
  // that are specific to lazy tensors, should live within the 'lazy_tensors::'
  // namespace.
  static OpKind Get(const std::string& name);

  c10::Symbol op;
};

inline std::ostream& operator<<(std::ostream& stream, const OpKind& op) {
  stream << op.ToString();
  return stream;
}

using OpList = c10::ArrayRef<Value>;

hash_t OperandHashes(
    const OpList& operands,
    const hash_t& seed,
    bool bakeInSizes);
// A node in the graph. Nodes for operations which require extra data to be
// stored for lowering should inherit from this class and add an operation
// specific member there. For example, a constant might create a new
// NodeConstant class (inheriting from Node) with an extra lazy_tensors::Literal
// field, or a tensor value might create a new NodeTensor with a computation
// client data handle in it.
class TORCH_API Node {
 public:
  static bool enableDynamicShape();

  // Creates a new node with the given op name. The op is a unique identifier
  // for the operation. The num_outputs tells how many outputs a given operation
  // generates.
  //
  // None leaf node's node_hash does not contains shape information always.
  // So we pass in the hash value rather than a function.
  Node(OpKind op, size_t num_outputs);

  // Construct node with operands and shapes
  Node(
      OpKind op,
      OpList operands,
      std::vector<Shape>&& shapes,
      size_t num_outputs = 1);

  // Construct node with operands and shape generated from a function
  Node(
      OpKind op,
      OpList operands,
      const std::function<Shape()>& shape_fn,
      size_t num_outputs = 1);

  // Construct node with operands and no shape
  Node(OpKind op, OpList operands, size_t num_outputs = 1);

  // Construct node with shape and no operands
  Node(OpKind op, Shape shape, size_t num_outputs = 1);

  virtual ~Node();

  const OpKind& op() const {
    return op_;
  }

  size_t num_outputs() const {
    return num_outputs_;
  }

  // Retrieves the full shape of the IR Node.
  virtual c10::ArrayRef<Shape> shapes() const;

  virtual const Shape& shape(size_t output_index = 0) const;

  // Add the shape computed by the shape_fn
  void addComputedShape(const std::function<Shape()>& shape_fn);

  // Compute the shape using the provided shape_fn if not previously cached
  Shape computeShape(const std::function<Shape()>& shape_fn);

  virtual const std::vector<Output>& operands() const;

  virtual const Output& operand(size_t i) const;

  // Gets operand at index i if index is valid, or kNullOutput otherwise.
  virtual const Output& nullable_operand(size_t i) const;

  // Returns the hash of the dag used to look up the compiled graph
  virtual hash_t hash() const = 0;

  // Returns the hash of the dag used to for shape caching
  virtual hash_t shapeHash() const = 0;

  const MetaData& metadata() const {
    return metadata_;
  }

  UserMetaData* user_metadata() const {
    return user_metadata_.get();
  }

  std::shared_ptr<UserMetaData> SetUserMetadata(
      std::shared_ptr<UserMetaData> user_meta) {
    std::swap(user_metadata_, user_meta);
    return user_meta;
  }

  virtual std::string ToString() const;

 private:
  // The ID of the operation captured by this node.
  OpKind op_;
  size_t num_outputs_ = 1;

  // The IR specific metadata attached to the IR node.
  MetaData metadata_;
  // The IR framework user can attach a user defined metadata object deriving
  // from UserMetaData.
  std::shared_ptr<UserMetaData> user_metadata_;

 protected:
  // Adds node's index output number as operand.
  void AddOperand(NodePtr node, size_t index = 0);

  std::vector<Shape> shapes_;
  // A node holds a real reference to its operands.
  std::vector<NodePtr> operands_;
  // Outputs do not hold references on the nodes, and neither do the uses, since
  // otherwise we get into circular reference counting.
  std::vector<Output> operands_as_outputs_;
};

inline std::ostream& operator<<(std::ostream& stream, const Node& node) {
  stream << node.ToString();
  return stream;
}

// Note: Keep this version of NodeCast for smooth PyTorch/XLA migration, and
// clean up once the migration is done.
template <typename T>
const T* NodeCast(const Node* node, OpKind op) {
  if (op != node->op()) {
    return nullptr;
  }
#ifdef NDEBUG
  return static_cast<const T*>(node);
#else
  return &dynamic_cast<const T&>(*node);
#endif
}

template <typename T>
const T* NodeCast(const Node* node) {
  if (T::ClassOpKind() != node->op()) {
    return nullptr;
  }
  // TODO: Some IR classes share the same opkind, such as Mean and MeanDim, so
  // static_cast is not safe here. Unless we have opkind unique for each class,
  // we have to use dynamic_cast here.
  return dynamic_cast<const T*>(node);
}

// Represents a specific output produced by a node. Since the output of a node
// can be composed by multiple outputs, the node+index coordinates fully qualify
// each single output.
struct TORCH_API Output {
  struct Hasher {
    size_t operator()(const Output& output) const;
  };

  Output() = default;
  explicit Output(const Node* node, size_t index = 0)
      : node(node), index(index) {}

  hash_t hash() const;
  hash_t shapeHash() const;

  bool operator==(const Output& rhs) const {
    return node == rhs.node && index == rhs.index;
  }

  // To compare the operands of to-be-constructed node and to-be-reused node
  bool operator==(const Value& rhs) const;

  bool operator!=(const Output& rhs) const {
    return !operator==(rhs);
  }

  const Shape& shape() const {
    return node->shape(index);
  }

  std::string ToString() const;

  // The node providing the output.
  const Node* node{nullptr};
  // The index in the node's output this output refers to.
  size_t index{0};
};

inline std::ostream& operator<<(std::ostream& stream, const Output& output) {
  stream << output.ToString();
  return stream;
}

template <typename T>
using OutputMap = std::unordered_map<Output, T, Output::Hasher>;

// Represents an input/operand for a Node object.
struct TORCH_API Value {
  Value() = default;
  /* implicit */ Value(NodePtr&& node, size_t index = 0)
      : node(std::move(node)), index(index) {}
  /* implicit */ Value(const NodePtr& node, size_t index = 0)
      : node(node), index(index) {}

  hash_t hash() const;
  hash_t shapeHash() const;

  operator bool() const {
    return node != nullptr;
  }

  operator Output() const {
    return Output(node.get(), index);
  }

  const Shape& shape() const {
    return node->shape(index);
  }

  Node* operator->() const {
    return node.get();
  }

  NodePtr node;
  size_t index = 0;
};

} // namespace lazy
} // namespace torch

namespace c10 {
// Explicit template instantiation to make ArrayRef<Value> work
template class at::ArrayRef<torch::lazy::Value>;
} // namespace c10
