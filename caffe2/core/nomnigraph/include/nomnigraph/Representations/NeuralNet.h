//=== nomnigraph/Representations/NeuralNet.h - NN interface -----*- C++ -*-===//
//
// TODO Licensing.
//
//===----------------------------------------------------------------------===//
//
// This file defines classes that can be used in a
// nom::Graph<nom::repr::NeuralNetOperator, nom::repr::NeuralNetData> graph.
//
//===----------------------------------------------------------------------===//

#ifndef NOM_REPRESENTATIONS_NEURALNET_H
#define NOM_REPRESENTATIONS_NEURALNET_H

#include "nomnigraph/Graph/Graph.h"
#include "nomnigraph/Representations/Compiler.h"
#include "nomnigraph/Representations/ControlFlow.h"
#include "nomnigraph/Support/Casting.h"
#include "nomnigraph/Support/Pointer.h"
#include "nomnigraph/Transformations/SubgraphMatcher.h"

#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include <assert.h>

namespace nom {
namespace repr {

// Expose supported attribute types to this namespace.
using std::string;
using std::vector;

class NeuralNetData;

/// \brief Annotations allow for generic manipulation of
/// neural network operations.  The base class contains
/// a saved void* pointer for external use.  Derived classes
/// add richer semantics to the annotation and it is encouraged
/// to use them.
class Annotation {
 public:
  enum class AnnotationKind { Generic, Caffe2 };

  Annotation(AnnotationKind K) : Kind(K) {}
  Annotation() : Kind(AnnotationKind::Generic) {}
  virtual ~Annotation() {}

  AnnotationKind getKind() const {
    return Kind;
  }

  Annotation(const Annotation&) = delete;
  Annotation& operator=(Annotation&) = delete;

 private:
  const AnnotationKind Kind;
};

class NeuralNetOperator : public Instruction {
 public:
  /// Discriminator for LLVM-style RTTI (isa<>)
  enum class NNKind {
    Undefined,
    GenericOperator,
    NNPhi,
    While,
#include "nomnigraph/Generated/OpEnum.h"
  };

  /// An optional tensor-type specifier.
  enum class NNLayout { Undefined, NCHW, NHWC };

  NeuralNetOperator(NNKind K, Opcode I, NNLayout L)
      : Instruction(I), Kind(K), Layout(L) {}
  NeuralNetOperator(NNKind K, Opcode I)
      : Instruction(I), Kind(K), Layout(NNLayout::Undefined) {}
  NeuralNetOperator(NNKind K, NNLayout L) : Instruction(), Kind(K), Layout(L) {}
  NeuralNetOperator(NNKind K)
      : Instruction(), Kind(K), Layout(NNLayout::Undefined) {}
  NeuralNetOperator()
      : Instruction(), Kind(NNKind::Undefined), Layout(NNLayout::Undefined) {}

  NNKind getKind() const {
    return Kind;
  }

  void setLayout(NNLayout L) {
    Layout = L;
  }

  NNLayout getLayout() const {
    return Layout;
  }

  void setAnnotation(std::unique_ptr<Annotation> extraAnnotation) {
    ExtraAnnotation = std::move(extraAnnotation);
  }

  const Annotation* getAnnotation() const {
    return ExtraAnnotation.get();
  }
  Annotation* getMutableAnnotation() {
    return ExtraAnnotation.get();
  }

  const std::string getName() const;

  /// \brief Validate the inputs and outputs to this operator.
  ///
  /// \p inputs A vector of references to NeuralNetData types that
  /// represent the data being fed into the operator.
  /// \p outputs A vector of references to NeuralNetData types that
  /// represent the data being outputted by the operator.
  /// \return true if the inputs and outputs are compatible with the operator.
  bool checkInputsAndOutputs(
      std::vector<const NeuralNetData*> inputs,
      std::vector<const NeuralNetData*> outputs) {
    return true;
  }

  virtual ~NeuralNetOperator() = 0;

  NeuralNetOperator(const NeuralNetOperator&) = delete;
  NeuralNetOperator& operator=(NeuralNetOperator&) = delete;

 private:
  const NNKind Kind;
  NNLayout Layout; // Mutable attribute, much like a type cast
  std::unique_ptr<Annotation> ExtraAnnotation;
};

class NeuralNetData : public Data {
 public:
  /// Discriminator for LLVM-style RTTI (isa<>)
  enum class NNDataKind { Generic, Tensor };

  NeuralNetData(NNDataKind kind) : Kind(kind) {}

  NeuralNetData() : Kind(NNDataKind::Generic) {}

  NNDataKind getKind() const {
    return Kind;
  }

  virtual NeuralNetData* clone() = 0;

  const std::string getName() const;

  virtual ~NeuralNetData() = 0;

 private:
  NNDataKind Kind;
  size_t Version = 0;
};

class Tensor : public NeuralNetData {
 public:
  enum class DataType { Generic, Float, Half, Int8 };
  enum class Layout { Generic, NCHW, NHWC };

  Tensor(std::string name)
      : NeuralNetData(NNDataKind::Tensor),
        name_(name),
        type_(DataType::Generic) {}
  static bool classof(const NeuralNetData* D) {
    return D->getKind() == NNDataKind::Tensor;
  }

  NeuralNetData* clone() {
    return new Tensor(name_);
  }

  void setType(DataType type) {
    type_ = type;
  }

  DataType getType() const {
    return type_;
  }

  const std::string getName() const {
    return name_;
  }
  ~Tensor() {}

 private:
  std::string name_;
  DataType type_;
};

#define NOMNIGRAPH_DEFINE_NN_RTTI(op)                                 \
  static bool classof(const NeuralNetOperator* N) {                   \
    return N->getKind() == NNKind::op;                                \
  }                                                                   \
  static bool classof(const Value* N) {                               \
    if (isa<NeuralNetOperator>(N)) {                                  \
      return dyn_cast<NeuralNetOperator>(N)->getKind() == NNKind::op; \
    }                                                                 \
    return false;                                                     \
  }

#include "nomnigraph/Generated/OpClasses.h"

class While : public NeuralNetOperator {
 public:
  While() : NeuralNetOperator(NNKind::While, Opcode::Branch) {}
  NOMNIGRAPH_DEFINE_NN_RTTI(While);
  ~While() {}
};

class NNPhi : public NeuralNetOperator {
 public:
  NNPhi() : NeuralNetOperator(NNKind::NNPhi, Opcode::Phi) {}
  NOMNIGRAPH_DEFINE_NN_RTTI(NNPhi);
  ~NNPhi() {}
};

class GenericOperator : public NeuralNetOperator {
 public:
  GenericOperator() : NeuralNetOperator(NNKind::GenericOperator) {}
  GenericOperator(std::string name)
      : NeuralNetOperator(NNKind::GenericOperator), name_(name) {}
  NOMNIGRAPH_DEFINE_NN_RTTI(GenericOperator);
  std::string getName() const {
    return name_;
  }
  void setName(std::string name) {
    name_ = name;
  }
  ~GenericOperator() {}

 private:
  std::string name_;
};

using NNGraph = nom::Graph<std::unique_ptr<nom::repr::Value>>;
using NNSubgraph = nom::Subgraph<std::unique_ptr<nom::repr::Value>>;
using NNCFGraph = nom::repr::ControlFlowGraph<NNGraph>;

struct NNModule {
  NNGraph dataFlow;
  NNCFGraph controlFlow;
  std::unordered_set<NNGraph::NodeRef> inputs;
  std::unordered_set<NNGraph::NodeRef> outputs;
  NNModule(const NNModule&) = delete;
  NNModule(NNModule&&) = default;
  NNModule() {}
};

// Although these seem generic, they make subtle assumptions
// about the structure of the graph that is 100% valid for NNModule graphs
// but not any graph (such as data being a unique_ptr).
namespace nn {

template <bool B, class T = void>
using enable_if_t = typename std::enable_if<B, T>::type;

template <typename T, typename U>
struct inheritedFrom {
  static constexpr bool value =
      std::is_base_of<U, T>::value && !std::is_same<U, T>::value;
};

// This is just a way to fix issues when the isa<> implementation
// can't automatically downcast.
template <typename T, typename N, typename = void>
struct is_impl {
  inline static bool impl(N n) {
    return isa<T>(n->data());
  }
};

template <typename T, typename N>
struct is_impl<T, N, enable_if_t<inheritedFrom<T, NeuralNetOperator>::value>> {
  inline static bool impl(N n) {
    if (!isa<NeuralNetOperator>(n->data().get())) {
      return false;
    }
    auto nno = dyn_cast<NeuralNetOperator>(n->data().get());
    return isa<T>(nno);
  }
};

template <typename T, typename N>
struct is_impl<T, N, enable_if_t<inheritedFrom<T, NeuralNetData>::value>> {
  inline static bool impl(N n) {
    if (!isa<NeuralNetData>(n->data().get())) {
      return false;
    }
    auto nno = dyn_cast<NeuralNetData>(n->data().get());
    return isa<T>(nno);
  }
};

template <typename T, typename N>
inline bool is(N n) {
  return is_impl<T, N>::impl(n);
}

// This is just a way to fix issues when the dyn_cast<> implementation
// can't automatically downcast.
template <typename T, typename N, typename = void>
struct get_impl {
  inline static T* impl(N n) {
    return dyn_cast<T>(n->data().get());
  }
};

template <typename T, typename N>
struct get_impl<T, N, enable_if_t<inheritedFrom<T, NeuralNetOperator>::value>> {
  inline static T* impl(N n) {
    if (!is<T>(n)) {
      assert(0 && "Cannot get type from node");
      return nullptr;
    }
    auto nno = dyn_cast<NeuralNetOperator>(n->data().get());
    return dyn_cast<T>(nno);
  }
};

template <typename T, typename N>
struct get_impl<T, N, enable_if_t<inheritedFrom<T, NeuralNetData>::value>> {
  inline static T* impl(N n) {
    if (!is<T>(n)) {
      assert(0 && "Cannot get type from node");
      return nullptr;
    }
    auto nno = dyn_cast<NeuralNetData>(n->data().get());
    return dyn_cast<T>(nno);
  }
};

template <typename T, typename N>
inline T* get(N n) {
  return get_impl<T, N>::impl(n);
}

template <typename T, typename G>
std::vector<typename G::NodeRef> nodeIterator(G& g) {
  std::vector<typename G::NodeRef> out;
  for (auto node : g.getMutableNodes()) {
    if (!is<T>(node)) {
      continue;
    }
    out.emplace_back(node);
  }
  return out;
}

template <typename T, typename G>
std::vector<std::pair<T*, typename G::NodeRef>> dataIterator(G& g) {
  std::vector<std::pair<T*, typename G::NodeRef>> out;
  for (auto node : g.getMutableNodes()) {
    if (!is<T>(node)) {
      continue;
    }
    auto d = get<T>(node);
    out.emplace_back(std::make_pair(d, node));
  }
  return out;
}

template <typename T, typename... Args>
void insertOp(
    NNGraph& g,
    NNGraph::NodeRef a,
    NNGraph::NodeRef b,
    Args... args) {
  if (is<NeuralNetData>(a) && is<NeuralNetOperator>(b)) {
    auto newNode = g.createNode(util::make_unique<T>(args...));
    auto data = get<NeuralNetData>(a);
    auto newData =
        g.createNode(util::make_unique<Tensor>(data->getName() + "_"));
    g.createEdge(a, newNode);
    g.createEdge(newNode, newData);
    g.createEdge(newData, b);
    return;
  }
  if (is<NeuralNetOperator>(a) && is<NeuralNetData>(b)) {
    auto newNode = g.createNode(util::make_unique<T>(args...));
    auto data = get<NeuralNetData>(b);
    auto newData =
        g.createNode(util::make_unique<Tensor>(data->getName() + "_"));
    g.createEdge(a, newData);
    g.createEdge(newData, newNode);
    g.createEdge(newNode, b);
    return;
  }

  assert(0 && "insertOp takes (DFG, Tensor, Op) or (DFG, Op, Tensor)");
}

template <typename NewT, typename OldT>
NNGraph::NodeRef convertNode(NNGraph& g, NNGraph::NodeRef node) {
  assert(is<OldT>(node) && "Cannot get type from node.");

  NeuralNetOperator* nnOpPtr =
      dyn_cast<NeuralNetOperator>(node->mutableData()->release());

  auto newNode =
      g.createNode(util::make_unique<NewT>(*dyn_cast<OldT>(nnOpPtr)));

  g.replaceNode(node, newNode);
  g.deleteNode(node);

  return newNode;
}

/// NeuralNetData specific helpers.
bool hasProducer(NNGraph::NodeRef n);
NNGraph::NodeRef getProducer(NNGraph::NodeRef n);
bool hasConsumer(NNGraph::NodeRef n);
std::vector<NNGraph::NodeRef> getConsumers(NNGraph::NodeRef n);

bool hasInputs(NNGraph::NodeRef n);
std::vector<NNGraph::NodeRef> getInputs(NNGraph::NodeRef n);
std::vector<NNGraph::NodeRef> getOutputs(NNGraph::NodeRef n);

void coalesceInsertedDataDependencies(repr::NNModule* m);

template <NNGraph* G>
struct NodeHelper {};

struct NNNodeMatchCriteria {
  const std::function<bool(NNGraph::NodeRef)> predicate;
  const std::string debugString;

  NNNodeMatchCriteria(
      const std::function<bool(NNGraph::NodeRef)>& predicate,
      const std::string& debugString = "No debug string specified")
      : predicate(predicate), debugString(debugString){};
};

std::ostream& operator<<(
    std::ostream& oss,
    const NNNodeMatchCriteria& criteria);

using NNMatchGraph = nom::matcher::MatchGraph<NNNodeMatchCriteria>;

bool hasSingleOutputAndConsumer(NNGraph::NodeRef nodeRef);

template <typename NodeType>
NNNodeMatchCriteria matchNodeTypeWithPredicate(
    const std::function<bool(NNGraph::NodeRef, const NodeType&)> predicate,
    bool expectedSingleOutputAndConsumer = false,
    const std::string& debugString = "matchNodeTypeWithPredicate") {
  return NNNodeMatchCriteria(
      [&predicate, expectedSingleOutputAndConsumer](NNGraph::NodeRef nodeRef) {
        NOM_REQUIRE_OR_RET_FALSE(is<NodeType>(nodeRef));
        if (expectedSingleOutputAndConsumer) {
          NOM_REQUIRE_OR_RET_FALSE(hasSingleOutputAndConsumer(nodeRef));
        }
        NodeType* node = get<NodeType>(nodeRef);
        return predicate(nodeRef, *node);
      },
      debugString);
};

template <typename NodeType>
NNNodeMatchCriteria matchNodeType(
    bool expectedSingleOutputAndConsumer = false,
    const std::string& debugString = "matchNodeType") {
  return NNNodeMatchCriteria(
      [expectedSingleOutputAndConsumer](NNGraph::NodeRef nodeRef) {
        if (expectedSingleOutputAndConsumer) {
          NOM_REQUIRE_OR_RET_FALSE(hasSingleOutputAndConsumer(nodeRef));
        }
        return is<NodeType>(nodeRef);
      },
      debugString);
}

NNNodeMatchCriteria matchAnyNode();

struct NNNodeMatch {
  static bool isMatch(
      const NNGraph::NodeRef& node,
      const NNNodeMatchCriteria& criteria) {
    return criteria.predicate(node);
  }
};

using NNSubgraphMatcher =
    nom::matcher::SubgraphMatcher<NNGraph, NNNodeMatchCriteria, NNNodeMatch>;

// This helper method makes it easy to create matching criteria in NNGraph.
// For example, operatorSubgraph(opMatch, ...) will refer to a tree like this:
// ... -> opMatch -> opMatch_Output
NNMatchGraph::NodeRef operatorSubgraph(
    NNMatchGraph& g,
    const NNNodeMatchCriteria& root,
    const std::vector<NNMatchGraph::NodeRef>& childrenCriteria = {},
    int count = 1);

} // namespace nn

} // namespace repr
} // namespace nom

#endif // NOM_REPRESENTATIONS_NEURALNET_H
