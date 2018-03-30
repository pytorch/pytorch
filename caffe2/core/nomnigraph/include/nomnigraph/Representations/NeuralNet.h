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

#include <string>
#include <type_traits>
#include <vector>

#include <assert.h>

namespace nom {
namespace repr {

class NeuralNetData;

/// \brief Annotations allow for generic manipulation of
/// neural network operations.  The base class contains
/// a saved void* pointer for external use.  Derived classes
/// add richer semantics to the annotation and it is encouraged
/// to use them.
class Annotation {
public:
  enum class AnnotationKind { Generic, Device };

  Annotation(AnnotationKind K) : Kind(K) {}
  Annotation() : Kind(AnnotationKind::Generic) {}

  AnnotationKind getKind() const { return Kind; }

  Annotation(const Annotation &) = delete;
  Annotation &operator=(Annotation &) = delete;

  void *getSaved() const { return Saved; }
  void setSaved(void *saved) { Saved = saved; }

private:
  const AnnotationKind Kind;
  void *Saved = nullptr;
};

class DeviceAnnotation : public Annotation {
public:
  DeviceAnnotation() : Annotation(AnnotationKind::Device) {}
  DeviceAnnotation(std::string device)
      : Annotation(AnnotationKind::Device), Device(device) {}
  void setDevice(std::string device) { Device = device; }
  const std::string getDevice() const { return Device; }

  static bool classof(const Annotation *A) {
    return A->getKind() == AnnotationKind::Device;
  }

private:
  std::string Device = 0;
};

class NeuralNetOperator : public Instruction {
public:
  /// Discriminator for LLVM-style RTTI (isa<>)
  enum class NNKind {
    Undefined,
    Conv,
    Relu,
    ConvRelu,
    DynamicInput,
    Send,
    Receive,
    While,
    NNPhi,
    GenericOperator
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

  NNKind getKind() const { return Kind; }

  void setLayout(NNLayout L) { Layout = L; }

  NNLayout getLayout() const { return Layout; }

  void setAnnotation(std::unique_ptr<Annotation> extraAnnotation) {
    ExtraAnnotation = std::move(extraAnnotation);
  }

  const Annotation *getAnnotation() const { return ExtraAnnotation.get(); }
  Annotation *getMutableAnnotation() { return ExtraAnnotation.get(); }

  const std::string getName() const;

  /// \brief Validate the inputs and outputs to this operator.
  ///
  /// \p inputs A vector of references to NeuralNetData types that
  /// represent the data being fed into the operator.
  /// \p outputs A vector of references to NeuralNetData types that
  /// represent the data being outputted by the operator.
  /// \return true if the inputs and outputs are compatible with the operator.
  bool checkInputsAndOutputs(std::vector<const NeuralNetData *> inputs,
                             std::vector<const NeuralNetData *> outputs) {
    return true;
  }

  virtual ~NeuralNetOperator() = 0;

  NeuralNetOperator(const NeuralNetOperator &) = delete;
  NeuralNetOperator &operator=(NeuralNetOperator &) = delete;

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

  NNDataKind getKind() const { return Kind; }

  virtual NeuralNetData *clone() = 0;

  const std::string getName() const;

  virtual ~NeuralNetData() = 0;

private:
  NNDataKind Kind;
  size_t Version = 0;
};

class Tensor : public NeuralNetData {
public:
  Tensor(std::string name) : NeuralNetData(NNDataKind::Tensor), name_(name) {}
  static bool classof(const NeuralNetData *D) {
    return D->getKind() == NNDataKind::Tensor;
  }

  NeuralNetData *clone() { return new Tensor(name_); }

  const std::string getName() const { return name_; }
  ~Tensor() {}

private:
  std::string name_;
};

class DynamicInput : public NeuralNetOperator {
public:
  DynamicInput() : NeuralNetOperator(NNKind::DynamicInput) {}
  ~DynamicInput() {}
};

#define NOMNIGRAPH_DEFINE_NN_RTTI(op)                                          \
  static bool classof(const NeuralNetOperator *N) {                            \
    return N->getKind() == NNKind::op;                                         \
  }

class Conv : public NeuralNetOperator {
public:
  Conv(std::vector<int> kernelShape, std::vector<int> dilations = {1, 1},
       int group = 1, std::vector<int> pads = {0, 0},
       std::vector<int> strides = {1, 1})
      : NeuralNetOperator(NNKind::Conv), KernelShape(kernelShape),
        Dilations(dilations), Group(group), Pads(pads), Strides(strides) {}

  NOMNIGRAPH_DEFINE_NN_RTTI(Conv);

  ~Conv() {}

  void setDilations(std::vector<int> &&dilations) {
    Dilations = std::move(dilations);
  }

  void setGroup(int &&group) { Group = std::move(group); }

  void setPads(std::vector<int> &&pads) { Pads = std::move(pads); }

  void setStrides(std::vector<int> &&strides) { Strides = std::move(strides); }

  std::vector<int> getDilations() { return Dilations; }

  int getGroup() { return Group; }

  std::vector<int> getPads() { return Pads; }

  std::vector<int> getStrides() { return Strides; }

  std::vector<int> getKernelShape() { return KernelShape; }

  bool checkInputsAndOutputs(std::vector<const NeuralNetData *> inputs,
                             std::vector<const NeuralNetData *> outputs) {
    assert(KernelShape.size() == Dilations.size());
    assert(KernelShape.size() == Pads.size());
    assert(KernelShape.size() == Strides.size());
    return true;
  }

protected:
  std::vector<int> KernelShape;
  std::vector<int> Dilations;
  int Group;
  std::vector<int> Pads;
  std::vector<int> Strides;
};

class ConvRelu : public NeuralNetOperator {
public:
  ConvRelu(std::vector<int> kernelShape, std::vector<int> dilations = {1, 1},
           int group = 1, std::vector<int> pads = {0, 0},
           std::vector<int> strides = {1, 1})
      : NeuralNetOperator(NNKind::ConvRelu),
        ConvPtr(util::make_unique<Conv>(kernelShape, dilations, group, pads,
                                        strides)) {}

  ConvRelu(Conv *conv)
      : NeuralNetOperator(NNKind::ConvRelu), ConvPtr(std::move(conv)) {}

  NOMNIGRAPH_DEFINE_NN_RTTI(ConvRelu);

  ~ConvRelu() {}

private:
  std::unique_ptr<Conv> ConvPtr = nullptr;
};

class Relu : public NeuralNetOperator {
public:
  Relu() : NeuralNetOperator(NNKind::Relu) {}
  NOMNIGRAPH_DEFINE_NN_RTTI(Relu);
  ~Relu() {}
};

class Send : public NeuralNetOperator {
public:
  Send() : NeuralNetOperator(NNKind::Send) {}
  NOMNIGRAPH_DEFINE_NN_RTTI(Send);
  ~Send() {}
};

class Receive : public NeuralNetOperator {
public:
  Receive() : NeuralNetOperator(NNKind::Receive) {}
  NOMNIGRAPH_DEFINE_NN_RTTI(Receive);
  ~Receive() {}
};

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
  std::string getName() const { return name_; }
  ~GenericOperator() {}

private:
  std::string name_;
};

using NNGraph = nom::Graph<std::unique_ptr<nom::repr::Value>, int>;
using NNCFGraph = nom::repr::ControlFlowGraph<NNGraph>;

struct NNModule {
  NNGraph dataFlow;
  NNCFGraph controlFlow;
  NNModule(const NNModule &) = delete;
  NNModule(NNModule &&) = default;
  NNModule() {}
};

// Although these seem generic, they make subtle assumptions
// about the structure of the graph that is 100% valid for NNModule graphs
// but not any graph (such as data being a unique_ptr).
namespace nn {

template< bool B, class T = void >
using enable_if_t = typename std::enable_if<B,T>::type;

template <typename T>
constexpr bool inheritedFromNeuralNetOperator() {
  return std::is_base_of<NeuralNetOperator, T>::value &&
    !std::is_same<NeuralNetOperator, T>::value;
}

template <typename T>
constexpr bool inheritedFromNeuralNetData() {
  return std::is_base_of<NeuralNetData, T>::value &&
    !std::is_same<NeuralNetData, T>::value;
}

// This is just a way to fix issues when the isa<> implementation
// can't automatically downcast.
template <typename T, typename N, typename = void> struct is_impl {
  inline static bool impl(N n) { return isa<T>(n->data()); }
};

template <typename T, typename N>
struct is_impl<T, N, enable_if_t<inheritedFromNeuralNetOperator<T>()>> {
  inline static bool impl(N n) {
    auto nno = dyn_cast<NeuralNetOperator>(n->data().get());
    return isa<T>(nno);
  }
};

template <typename T, typename N>
struct is_impl<T, N, enable_if_t<inheritedFromNeuralNetData<T>()>> {
  inline static bool impl(N n) {
    auto nno = dyn_cast<NeuralNetData>(n->data().get());
    return isa<T>(nno);
  }
};

template <typename T, typename N> inline bool is(N n) {
  return is_impl<T, N>::impl(n);
}

// This is just a way to fix issues when the dyn_cast<> implementation
// can't automatically downcast.
template <typename T, typename N, typename = void> struct get_impl {
  inline static T *impl(N n) { return dyn_cast<T>(n->data().get()); }
};

template <typename T, typename N>
struct get_impl<T, N, enable_if_t<inheritedFromNeuralNetOperator<T>()>> {
  inline static T *impl(N n) {
    auto nno = dyn_cast<NeuralNetOperator>(n->data().get());
    return dyn_cast<T>(nno);
  }
};

template <typename T, typename N>
struct get_impl<T, N, enable_if_t<inheritedFromNeuralNetData<T>()>> {
  inline static T *impl(N n) {
    auto nno = dyn_cast<NeuralNetData>(n->data().get());
    return dyn_cast<T>(nno);
  }
};

template <typename T, typename N> inline T *get(N n) {
  return get_impl<T, N>::impl(n);
}

template <typename T, typename G>
std::vector<std::pair<T *, typename G::NodeRef>> dataIterator(G &g) {
  std::vector<std::pair<T *, typename G::NodeRef>> out;
  for (auto node : g.getMutableNodes()) {
    if (!is<T>(node)) {
      continue;
    }
    auto d = get<T>(node);
    out.emplace_back(std::make_pair(d, node));
  }
  return out;
}

/// NeuralNetData specific helpers.
bool hasProducer(NNGraph::NodeRef n);
NNGraph::NodeRef getProducer(NNGraph::NodeRef n);
std::vector<NNGraph::NodeRef> getConsumers(NNGraph::NodeRef n);

bool hasInputs(NNGraph::NodeRef n);
std::vector<NNGraph::NodeRef> getInputs(NNGraph::NodeRef n);
std::vector<NNGraph::NodeRef> getOutputs(NNGraph::NodeRef n);

void coalesceInsertedDataDependencies(repr::NNModule* m);

template <NNGraph* G>
struct NodeHelper {
};

} // namespace nn

} // namespace repr
} // namespace nom

#endif // NOM_REPRESENTATIONS_NEURALNET_H
