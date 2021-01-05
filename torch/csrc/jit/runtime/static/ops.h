#pragma once

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/static/impl.h>

namespace torch {
namespace jit {

using SROperator = std::function<void(ProcessedNode*)>;
using SROpFunctor = std::function<SROperator(Node* n)>;
struct SROperatorFunctor {
  virtual SROperator Generate(Node*) {
    std::function<void(ProcessedNode*)> out;
    return out;
  }
  virtual bool CanReuseInput() {
    return false;
  }
  virtual bool CanReuseOutput() {
    return false;
  }
  virtual ~SROperatorFunctor() = default;
};

C10_DECLARE_REGISTRY(SROperatorRegistry, SROperatorFunctor);

// TODO: reuse_inp reuse_out can be deprecated with further analysis
// try to avoid this API.
#define REGISTER_OPERATOR_FUNCTOR(name, id, ...)             \
  struct SROperatorFunctor_##id : public SROperatorFunctor { \
    const SROpFunctor fn = __VA_ARGS__;                      \
    SROperator Generate(Node* n) override {                  \
      return fn(n);                                          \
    }                                                        \
  };                                                         \
  C10_REGISTER_CLASS(SROperatorRegistry, name, SROperatorFunctor_##id);

inline at::Tensor create_empty_from(const at::Tensor& t) {
  return at::empty({0}, t.options());
}

bool canRunOutOfPlace(Node* n);
bool mustRunOutOfPlace(Node* n);
bool canReuseInputs(Node* n);
bool canReuseOutputs(Node* n);

std::function<void(ProcessedNode*)> getOutOfPlaceOperation(Node* n);

bool canRunNatively(Node* n);
std::function<void(ProcessedNode*)> getNativeOperation(Node* n);

} // namespace jit
} // namespace torch
