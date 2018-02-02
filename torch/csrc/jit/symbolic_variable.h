#pragma once

#include "torch/csrc/jit/ir.h"

namespace torch { namespace jit {

struct SymbolicVariable {
  SymbolicVariable() : v(nullptr) {}
  /* implicit */ SymbolicVariable(Value * v) : v(v) {}
  static SymbolicVariable asNewInput(Graph & g, std::string name = "") {
    return g.addInput(name);
  }
  static SymbolicVariable asNewInput(Graph & g, TypePtr type) {
    return g.addInput()->setType(std::move(type));
  }
  void addAsOutput() {
    v->owningGraph()->registerOutput(v);
  }

  static std::vector<SymbolicVariable> create(Symbol kind, ArrayRef<SymbolicVariable> inputs,
                                 int num_outputs = 1,
                                 Node** created_node = nullptr,
                                 Graph * g = nullptr) {
      if(g == nullptr) {
        g = inputs.at(0).value()->owningGraph();
      }
      Node * n = g->appendNode(g->create(kind, num_outputs));
      for(auto i : inputs) {
        n->addInput(i.value());
      }
      if(created_node) {
        *created_node = n;
      }
      std::vector<SymbolicVariable> out;
      for(auto v : n->outputs()) {
        out.emplace_back(v);
      }
      return out;
  }
  static bool isConstInt(at::Scalar s, int32_t i) {
    // int32_t is safely convertible to both double and int64_t
    if(s.isFloatingPoint()) {
      return (double) i == s.toDouble();
    } else {
      return (int64_t) i == s.toLong();
    }
  }
  SymbolicVariable operator*(SymbolicVariable rhs) const {
    return create(kmul, {*this, rhs})[0].typeLike(*this);
  }
  SymbolicVariable operator*(at::Scalar rhs) const {
    if(isConstInt(rhs, 1))
      return *this;
    Node * n;
    auto r = create(kmul, {*this}, 1, &n)[0];
    n->t_(kother, rhs.toTensor());
    return r;
  }
  SymbolicVariable operator+(SymbolicVariable rhs) const {
    Node * n;
    auto r = create(kadd, {*this, rhs}, 1, &n)[0].typeLike(*this);
    n->t_(kalpha, at::Scalar(1).toTensor());
    return r;
  }
  SymbolicVariable operator-() const {
    return create(kneg, {*this})[0].typeLike(*this);
  }
  SymbolicVariable mm(SymbolicVariable rhs) const {
    // TODO: set types
    return create(s("mm"), {*this, rhs})[0];
  }
  SymbolicVariable sigmoid() const {
    return create(ksigmoid, {*this})[0].typeLike(*this);
  }
  SymbolicVariable tanh() const {
    return create(ktanh, {*this})[0].typeLike(*this);
  }
  std::vector<SymbolicVariable> chunk(int32_t chunks, uint32_t dim) const {
    Node * n;
    auto r = create(s("chunk"), { *this }, chunks, &n);
    n->i_(s("chunks"), chunks)
    ->i_(s("dim"), dim);
    return r;
  }
  static SymbolicVariable cat(ArrayRef<SymbolicVariable> inputs, int32_t dim) {
    Node* n;
    auto r = create(kcat, inputs, 1, &n)[0];
    n->i_(kdim, dim);
    return r;
  }
  Value * value() const {
    return v;
  }
private:
  SymbolicVariable typeLike(SymbolicVariable other) {
    if (auto other_type = other.v->typeOption())
      v->setType(other_type->expect<TensorType>()->contiguous());
    return *this;
  }
  static Symbol s(const char * s_) {
    return Symbol(s_);
  }
  Value * v;
};

}}
