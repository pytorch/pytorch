#pragma once

/*

Nothing particularly complex here, just a way to construct training graphs for
NNC.

Skips all layers above NNC on the stack which is useful for performance ablation
studies.

*/

#include <torch/csrc/jit/tensorexpr/tensor.h>

#include <functional>
#include <list>
#include <vector>

// Virtual "graph" for testing/benchmarking full training in NNC
struct VTensor; // Virtual tensors of symbolic shapes
struct VOp; // Virtual operators
struct VGraph; // Owner of bipartite VTensor/VOp graph

// VOps reference VMethods, or "virtual" methods that store
// 1) TensorExpr construction function (for lowering)
// 2) Grad construction function (for differentiating)
// 3) Shape functions (TODO this actually comes for free from TE)
struct VMethod;

// Utility for graph construction by op
std::vector<VTensor*> call(
    const std::string& name,
    const std::vector<VTensor*>& vs);
// Utility for graph construction by differentiation
VTensor* grad(VTensor* y, VTensor* x, VTensor* j);

std::string dot(const VGraph& g);
std::tuple<
    torch::jit::tensorexpr::Stmt*,
    std::map<const VTensor*, torch::jit::tensorexpr::Placeholder>,
    std::map<const VTensor*, torch::jit::tensorexpr::Tensor*>,
    std::map<std::string, torch::jit::tensorexpr::VarHandle>>
to_tensorexpr(const VGraph& graph, std::vector<VTensor*> outputs = {});

/* IMPL */

struct VMethod {
  using LowerFn = std::function<std::vector<torch::jit::tensorexpr::Tensor*>(
      const std::vector<torch::jit::tensorexpr::Tensor*>&,
      const std::vector<VTensor*>&,
      const std::map<std::string, torch::jit::tensorexpr::VarHandle>&)>;
  using GradFn = std::function<std::vector<VTensor*>(
      const std::vector<VTensor*>&,
      const std::vector<VTensor*>&)>;
  using ShapeFn = std::function<std::vector<std::vector<std::string>>(
      const std::vector<VTensor*>&)>;

  // Lookup from name
  static const VMethod& get(const std::string& name);

  LowerFn lower;
  GradFn grad;
  ShapeFn shape;
  std::string name;
  size_t num_outputs;
};

struct VTensor {
  VTensor(std::vector<std::string> shape_) : shape(shape_) {}
  std::vector<std::string> shape;
  VOp* op = nullptr;
  std::vector<VOp*> consumers;
  VGraph* graph;
};

struct VOp {
  VOp(const std::string& method_name,
      const std::vector<VTensor*>& inputs_,
      size_t num_outputs,
      VGraph* graph_);
  std::vector<VTensor*> inputs = {};
  std::vector<VTensor*> outputs = {};
  const VMethod* method;
  VGraph* graph;
};

struct VGraph {
  inline VTensor* create_tensor(std::vector<std::string> dims) {
    vtensors.emplace_back(dims);
    for (auto d : dims) {
    }
    auto* v = &vtensors.back();
    v->graph = this;
    return v;
  }

  inline VOp* create_op(
      std::string method,
      const std::vector<VTensor*>& inputs,
      size_t num_outputs) {
    vops.emplace_back(method, inputs, num_outputs, this);
    auto* o = &vops.back();
    o->graph = this;
    return o;
  }

  std::list<VTensor> vtensors;
  std::list<VOp> vops;
};

class RegMethod {
 public:
  RegMethod(
      std::string name,
      VMethod::LowerFn lower,
      VMethod::GradFn grad,
      VMethod::ShapeFn shape,
      size_t num_out = 1);
};

#define REGISTER_METHOD(name, ...) \
  static RegMethod _reg_method_##name(#name, __VA_ARGS__);
