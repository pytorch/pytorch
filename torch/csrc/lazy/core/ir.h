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
#include <torch/csrc/lazy/core/hash.h>
#include <torch/csrc/lazy/core/ir_metadata.h>

namespace torch {
namespace lazy {

static const hash_t kHashSeed(static_cast<uint32_t>(0x5a2d296e9));

class Node;

using NodePtr = std::shared_ptr<Node>;

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

  bool operator==(const Output& rhs) const {
    return node == rhs.node && index == rhs.index;
  }
  bool operator!=(const Output& rhs) const {
    return !operator==(rhs);
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
  /* implicit */ Value(NodePtr node, size_t index = 0) : node(std::move(node)), index(index) {}

  hash_t hash() const;

  operator bool() const {
    return node != nullptr;
  }

  operator Output() const {
    return Output(node.get(), index);
  }

  Node* operator->() const {
    return node.get();
  }

  NodePtr node;
  size_t index = 0;
};

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


// A node in the graph. Nodes for operations which requires extra data to be
// stored for lowering, should inherit from this class and add operation
// specific member there. For example, a constant might create a new
// NodeConstant class (inheriting from Node) with an extra lazy_tensors::Literal
// field, or a tensor value might create a new NodeTensor with computation
// client data handle in it.
class TORCH_API Node {
 public:
  // Creates a new node with the given op name. The op is a unique identifier
  // for the operation. The num_outputs tells how many outputs a given operation
  // generates.
  Node(OpKind op, size_t num_outputs, hash_t node_hash, hash_t dag_hash);

  // Contructor used to create leaf nodes.
  Node(OpKind op, size_t num_outputs, hash_t node_hash);

  virtual ~Node();

  const OpKind& op() const {
    return op_;
  }

  size_t num_outputs() const {
    return num_outputs_;
  }

  virtual const std::vector<Output>& operands() const = 0;

  virtual const Output& operand(size_t i) const = 0;

  hash_t node_hash() const {
    return node_hash_;
  }

  hash_t hash() const {
    return dag_hash_;
  }

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

  // The hash value of this node.
  hash_t node_hash_;
  // The hash value of the graph rooted at this node.
  hash_t dag_hash_;
  // The IR specific metadata attached to the IR node.
  MetaData metadata_;
  // The IR framework user can attach a user defined metadata object deriving
  // from UserMetaData.
  std::shared_ptr<UserMetaData> user_metadata_;
};



inline std::ostream& operator<<(std::ostream& stream, const Node& node) {
  stream << node.ToString();
  return stream;
}

// TODO(alanwaketan): Support r-value reference argument type.
template <typename T, typename... Args>
NodePtr MakeNode(Args&&... args) {
  return std::make_shared<T>(std::forward<Args>(args)...);
}

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

} // namespace lazy
} // namespace torch

namespace c10 {
  // Explicit template instantiation to make ArrayRef<Value> work
  template class at::ArrayRef<torch::lazy::Value>;
}
