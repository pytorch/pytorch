#pragma once

#include <ATen/core/interned_strings.h>

#include <functional>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "c10/core/ScalarType.h"
#include "lazy_tensor_core/csrc/python_util.h"
#include "lazy_tensors/computation_client/types.h"
#include "lazy_tensors/span.h"
#include "torch/csrc/lazy/core/hash.h"

namespace torch_lazy_tensors {
namespace ir {

static const torch::lazy::hash_t kHashSeed = static_cast<uint32_t>(0x5a2d296e9);

class Node;

using NodePtr = std::shared_ptr<Node>;

// The base class for user defined metadata which is possible to attach to IR
// nodes.
struct UserMetaData {
  virtual ~UserMetaData() {}
};

struct MetaData {
  std::string scope;
  std::vector<SourceLocation> frame_info;
};

// Represents a use of the output of a given node.
// If use U is within node N, it means that node U.node is using the output
// U.index of the node N.
struct Use {
  Use() = default;
  Use(Node* node, size_t operand_index, size_t index)
      : node(node), operand_index(operand_index), index(index) {}

  bool operator<(const Use& rhs) const;

  std::string ToString() const;

  // The node using the output of the node this use belongs to.
  Node* node = nullptr;
  // The operand index, within node's operands, which this use refers to.
  size_t operand_index = 0;
  // The index within output the user node refers to.
  size_t index = 0;
};

inline std::ostream& operator<<(std::ostream& stream, const Use& use) {
  stream << use.ToString();
  return stream;
}

// Represents a specific output produced by a node. Since the output of a node
// can be composed by multiple outputs, the node+index coordinates fully qualify
// each single output.
struct Output {
  struct Hasher {
    size_t operator()(const Output& output) const;
  };

  Output() = default;
  explicit Output(const Node* node, size_t index = 0)
      : node(node), index(index) {}

  torch::lazy::hash_t hash() const;

  bool operator==(const Output& rhs) const {
    return node == rhs.node && index == rhs.index;
  }
  bool operator!=(const Output& rhs) const { return !operator==(rhs); }

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

using OutputSet = std::unordered_set<Output, Output::Hasher>;

template <typename T>
using OutputMap = std::unordered_map<Output, T, Output::Hasher>;

// Represents an input/operand for a Node object.
struct Value {
  Value() = default;
  Value(NodePtr node, size_t index = 0) : node(std::move(node)), index(index) {}

  torch::lazy::hash_t hash() const;

  operator bool() const { return node != nullptr; }

  operator Output() const { return Output(node.get(), index); }

  Node* operator->() const { return node.get(); }

  NodePtr node;
  size_t index = 0;
};

// The Kind of operation a Node can be associated to.
struct OpKind {
  OpKind() = default;
  explicit OpKind(c10::Symbol op) : op(std::move(op)) {}

  bool operator==(const OpKind& rhs) const { return op == rhs.op; }
  bool operator!=(const OpKind& rhs) const { return !operator==(rhs); }
  bool operator<(const OpKind& rhs) const {
    return c10::unique_t(op) < c10::unique_t(rhs.op);
  }

  torch::lazy::hash_t hash() const;

  std::string ToString() const { return op.toQualString(); }

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

using OpList = lazy_tensors::Span<const Value>;

void EmitShortFrameInfo(std::ostream& stream,
                        const std::vector<SourceLocation>& frames);

// A node in the graph. Nodes for operations which requires extra data to be
// stored for lowering, should inherit from this class and add operation
// specific member there. For example, a constant might create a new
// NodeConstant class (inheriting from Node) with an extra lazy_tensors::Literal
// field, or a tensor value might create a new NodeTensor with computation
// client data handle in it.
class Node {
 public:
  // Creates a new node with the given op name. The op is a unique identifier
  // for the operation. The num_outputs tells how many outputs a given operation
  // generates.
  Node(OpKind op, size_t num_outputs, torch::lazy::hash_t node_hash,
       torch::lazy::hash_t dag_hash);

  // Contructor used to create leaf nodes.
  Node(OpKind op, size_t num_outputs, torch::lazy::hash_t node_hash);

  virtual ~Node();

  const OpKind& op() const { return op_; }

  size_t num_outputs() const { return num_outputs_; }

  virtual const std::vector<Output>& operands() const = 0;

  virtual const Output& operand(size_t i) const = 0;

  torch::lazy::hash_t node_hash() const { return node_hash_; }

  torch::lazy::hash_t hash() const { return dag_hash_; }

  const MetaData& metadata() const { return metadata_; }

  UserMetaData* user_metadata() const { return user_metadata_.get(); }

  std::shared_ptr<UserMetaData> SetUserMetadata(
      std::shared_ptr<UserMetaData> user_meta) {
    std::swap(user_metadata_, user_meta);
    return user_meta;
  }

  virtual std::string ToString() const;

  virtual NodePtr Clone(OpList operands) const;

 private:
  static std::vector<SourceLocation> GetFrameInfo();

  // The ID of the operation captured by this node.
  OpKind op_;
  size_t num_outputs_ = 1;

  // The hash value of this node.
  torch::lazy::hash_t node_hash_ = 0;
  // The hash value of the graph rooted at this node.
  torch::lazy::hash_t dag_hash_ = 0;
  // The IR specific metadata attached to the IR node.
  MetaData metadata_;
  // The IR framework user can attach a user defined metadata object deriving
  // from UserMetaData.
  std::shared_ptr<UserMetaData> user_metadata_;
};

// RAII data structure to be used a stack variable to enter a new IR scope. IR
// scope names will appear in the IR and will help identifying the source of the
// single IR nodes.
struct ScopePusher {
  explicit ScopePusher(const std::string& name);
  ~ScopePusher();

  static void ResetScopes();
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
T* NodeCast(const Node* node, OpKind op) {
  if (op != node->op()) {
    return nullptr;
  }
  const T* casted;
#ifdef NDEBUG
  casted = static_cast<const T*>(node);
#else
  casted = &dynamic_cast<const T&>(*node);
#endif
  return const_cast<T*>(casted);
}

}  // namespace ir
}  // namespace torch_lazy_tensors
