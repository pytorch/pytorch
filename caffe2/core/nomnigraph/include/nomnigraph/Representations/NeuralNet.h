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

#include "caffe2/core/common.h"
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
class CAFFE2_API Annotation {
 public:
  enum class AnnotationKind { Generic, Caffe2 };

  Annotation(AnnotationKind kind) : kind_(kind) {}
  Annotation() : kind_(AnnotationKind::Generic) {}
  virtual ~Annotation() {}

  AnnotationKind getKind() const {
    return kind_;
  }

 private:
  const AnnotationKind kind_;
};

class CAFFE2_API NeuralNetOperator : public Instruction {
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
      : Instruction(I), kind_(K), layout_(L) {}
  NeuralNetOperator(NNKind K, Opcode I)
      : Instruction(I), kind_(K), layout_(NNLayout::Undefined) {}
  NeuralNetOperator(NNKind K, NNLayout L)
      : Instruction(), kind_(K), layout_(L) {}
  NeuralNetOperator(NNKind K)
      : Instruction(), kind_(K), layout_(NNLayout::Undefined) {}
  NeuralNetOperator()
      : Instruction(), kind_(NNKind::Undefined), layout_(NNLayout::Undefined) {}

  NNKind getKind() const {
    return kind_;
  }

  void setLayout(NNLayout L) {
    layout_ = L;
  }

  NNLayout getLayout() const {
    return layout_;
  }

  void setAnnotation(std::unique_ptr<Annotation> extraAnnotation) {
    extraAnnotation_ = std::move(extraAnnotation);
  }

  const Annotation* getAnnotation() const {
    return extraAnnotation_.get();
  }

  Annotation* getMutableAnnotation() {
    return extraAnnotation_.get();
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
  const NNKind kind_;
  NNLayout layout_; // Mutable attribute, much like a type cast
  std::unique_ptr<Annotation> extraAnnotation_;
};

class CAFFE2_API NeuralNetData : public Data {
 public:
  /// Discriminator for LLVM-style RTTI (isa<>)
  enum class NNDataKind { Generic, Tensor };

  NeuralNetData(NNDataKind kind) : kind_(kind) {}

  NeuralNetData() : kind_(NNDataKind::Generic) {}

  NNDataKind getKind() const {
    return kind_;
  }

  virtual NeuralNetData* clone() = 0;

  const std::string getName() const;

  virtual ~NeuralNetData() = 0;

 private:
  NNDataKind kind_;
};

class CAFFE2_API Tensor : public NeuralNetData {
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

  void setName(const std::string& name) {
    name_ = name;
  }

  ~Tensor() {}

 private:
  std::string name_;
  DataType type_;
};

#define NOMNIGRAPH_DEFINE_NN_RTTI(op)                                 \
  static bool classof(const NeuralNetOperator* N) {        \
    return N->getKind() == NNKind::op;                                \
  }                                                                   \
  static bool classof(const Value* N) {                    \
    if (isa<NeuralNetOperator>(N)) {                                  \
      return dyn_cast<NeuralNetOperator>(N)->getKind() == NNKind::op; \
    }                                                                 \
    return false;                                                     \
  }

#include "nomnigraph/Generated/OpClasses.h"

class CAFFE2_API While : public NeuralNetOperator {
 public:
  While() : NeuralNetOperator(NNKind::While, Opcode::Branch) {}
  NOMNIGRAPH_DEFINE_NN_RTTI(While);
  ~While() {}
};

class CAFFE2_API NNPhi : public NeuralNetOperator {
 public:
  NNPhi() : NeuralNetOperator(NNKind::NNPhi, Opcode::Phi) {}
  NOMNIGRAPH_DEFINE_NN_RTTI(NNPhi);
  ~NNPhi() {}
};

class CAFFE2_API GenericOperator : public NeuralNetOperator {
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

struct CAFFE2_API NNModule {
  NNGraph dataFlow;
  NNCFGraph controlFlow;
  std::unordered_set<NNGraph::NodeRef> inputs;
  std::unordered_set<NNGraph::NodeRef> outputs;

  NNModule(const NNModule&) = delete;
  NNModule(NNModule&&) = default;
  NNModule() {}

  /* Repalce subgraph sg by node, using the order of
   * node_inputs and node_outputs to determine how to link
   * them to the node.  node_inputs *must* enumerate all the
   * inputs to the subgraph (NeuralNetData that do not
   * have producers inside the subgraph).  Same for node_outputs
   *
   * New output names may be created in the case that an inputs
   * and an output have the same name (to avoid in place ops).
   * This may cause issues with external_output -- be sure to check
   * after running this function (and perhaps inserting a copy/alias op).
   **/
  void replaceSubgraph(
      const NNGraph::SubgraphType& subgraph,
      const NNGraph::NodeRef& node,
      const std::vector<NNGraph::NodeRef>& node_inputs,
      const std::vector<NNGraph::NodeRef>& node_outputs);

  void deleteSubgraph(const NNGraph::SubgraphType& subgraph);
  NNGraph::NodeRef createUniqueDataNode(const std::string& s = "_unique");

  // Simple wrapper of replaceSubgraph where the node is created for you.
  // Returns a NodeRef to the node containing the operator that was created
  template <typename T, typename... Args>
  NNGraph::NodeRef replaceSubgraphWithOperator(
      const NNGraph::SubgraphType&,
      const std::vector<NNGraph::NodeRef>&,
      const std::vector<NNGraph::NodeRef>&,
      Args...);
};

template <typename T, typename... Args>
NNGraph::NodeRef NNModule::replaceSubgraphWithOperator(
    const NNGraph::SubgraphType& sg,
    const std::vector<NNGraph::NodeRef>& subgraph_inputs,
    const std::vector<NNGraph::NodeRef>& subgraph_outputs,
    Args... args) {
  auto node = dataFlow.createNode(util::make_unique<T>(args...));
  replaceSubgraph(sg, node, subgraph_inputs, subgraph_outputs);
  return node;
}

// Although these seem generic, they make subtle assumptions
// about the structure of the graph that is 100% valid for NNModule graphs
// but not any graph (such as data being a unique_ptr).
namespace nn {

template <bool B, class T = void>
using enable_if_t = typename std::enable_if<B, T>::type;

template <typename T, typename U>
struct C10_EXPORT inheritedFrom {
  static constexpr bool value =
      std::is_base_of<U, T>::value && !std::is_same<U, T>::value;
};

// This is just a way to fix issues when the isa<> implementation
// can't automatically downcast.
template <typename T, typename N, typename = void>
struct C10_EXPORT is_impl {
  inline static bool impl(N n) {
    return isa<T>(n->data());
  }
};

template <typename T, typename N>
struct C10_EXPORT
    is_impl<T, N, enable_if_t<inheritedFrom<T, NeuralNetOperator>::value>> {
  inline static bool impl(N n) {
    if (!isa<NeuralNetOperator>(n->data().get())) {
      return false;
    }
    auto nno = dyn_cast<NeuralNetOperator>(n->data().get());
    return isa<T>(nno);
  }
};

template <typename T, typename N>
struct C10_EXPORT
    is_impl<T, N, enable_if_t<inheritedFrom<T, NeuralNetData>::value>> {
  inline static bool impl(N n) {
    if (!isa<NeuralNetData>(n->data().get())) {
      return false;
    }
    auto nno = dyn_cast<NeuralNetData>(n->data().get());
    return isa<T>(nno);
  }
};

template <typename T>
inline bool is(NNGraph::NodeRef n) {
  return is_impl<T, NNGraph::NodeRef>::impl(n);
}

// This is just a way to fix issues when the dyn_cast<> implementation
// can't automatically downcast.
template <typename T, typename N, typename = void>
struct C10_EXPORT get_impl {
  inline static T* impl(N n) {
    return dyn_cast<T>(n->data().get());
  }
};

template <typename T, typename N>
struct C10_EXPORT
    get_impl<T, N, enable_if_t<inheritedFrom<T, NeuralNetOperator>::value>> {
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
struct C10_EXPORT
    get_impl<T, N, enable_if_t<inheritedFrom<T, NeuralNetData>::value>> {
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

template <typename T>
inline std::vector<NNGraph::NodeRef> filter(NNModule& nn) {
  return nodeIterator<T>(nn.dataFlow);
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
CAFFE2_API bool hasProducer(NNGraph::NodeRef n);
CAFFE2_API NNGraph::NodeRef getProducer(NNGraph::NodeRef n);
CAFFE2_API bool hasConsumer(NNGraph::NodeRef n);
CAFFE2_API std::vector<NNGraph::NodeRef> getConsumers(NNGraph::NodeRef n);

CAFFE2_API bool hasInputs(NNGraph::NodeRef n);
CAFFE2_API std::vector<NNGraph::NodeRef> getInputs(NNGraph::NodeRef n);
CAFFE2_API std::vector<NNGraph::NodeRef> getOutputs(NNGraph::NodeRef n);

CAFFE2_API std::set<NNGraph::NodeRef> getInputs(const NNSubgraph& sg);
CAFFE2_API std::set<NNGraph::NodeRef> getOutputs(const NNSubgraph& sg);

// Get the name of the node regardless of underlying type.
CAFFE2_API std::string getName(NNGraph::NodeRef n);

// Replace the producer of the first argument with the second argument
CAFFE2_API void replaceProducer(
    NNGraph::NodeRef tensorNode,
    NNGraph::NodeRef newProducer);
// Set all consumers of first argument to consume the second argument
CAFFE2_API void replaceAllUsesWith(
    NNGraph::NodeRef oldTensorNode,
    NNGraph::NodeRef newTensorNode);
// Set the second argument to consume the inputs of the first argument
CAFFE2_API void replaceAsConsumer(
    NNGraph::NodeRef oldConsumer,
    NNGraph::NodeRef newConsumer);

// Create an output tensor node
CAFFE2_API NNGraph::NodeRef
createOutput(NNModule* nn, NNGraph::NodeRef producer, std::string name);

// Hack for windows compiler.
template <typename T, typename... Args>
CAFFE2_API NNGraph::NodeRef createOperator(NNModule* nn, Args... args);

// Create an operator
template <typename T, typename... Args>
NNGraph::NodeRef createOperator(NNModule* nn, Args... args) {
  return nn->dataFlow.createNode(util::make_unique<T>(args...));
}

CAFFE2_API void coalesceInsertedDataDependencies(repr::NNModule* m);

template <NNGraph* G>
struct C10_EXPORT NodeHelper {};

using NNMatchGraph = nom::matcher::MatchGraph<NNGraph>;
using NNMatchPredicate = nom::matcher::MatchPredicate<NNGraph>;

// Commonly used node predicate.

// The node has a single output and the output has a single consumer.
CAFFE2_API bool hasSingleOutputAndConsumer(NNGraph::NodeRef nodeRef);
// The node has a unique consumer (there may be multiple edges from output
// to the single consumer).
CAFFE2_API bool hasUniqueConsumer(NNGraph::NodeRef nodeRef);

CAFFE2_API NNMatchPredicate matchExternalTensorNode();

} // namespace nn

} // namespace repr
} // namespace nom

#endif // NOM_REPRESENTATIONS_NEURALNET_H
