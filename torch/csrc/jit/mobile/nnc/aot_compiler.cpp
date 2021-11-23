#include <torch/csrc/jit/mobile/nnc/aot_compiler.h>

#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/lower_tuples.h>
#include <torch/csrc/jit/passes/peephole.h>
#include <torch/csrc/jit/passes/remove_mutation.h>
#include <torch/csrc/jit/passes/shape_analysis.h>
#include <torch/csrc/jit/passes/symbolic_shape_analysis.h>
#include <torch/csrc/jit/runtime/jit_trace.h>
#include <torch/csrc/jit/tensorexpr/graph_opt.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_simplifier.h>
#include <torch/csrc/jit/tensorexpr/kernel.h>

using namespace torch::jit;
using namespace torch::jit::tensorexpr;

namespace torch {
namespace jit {
namespace mobile {
namespace nnc {

std::vector<int64_t> getConstSizes(const BufPtr b) {
  std::vector<int64_t> r;
  for (const auto& dim : b->dims()) {
    LongImmPtr imm_dim = to<LongImm>(dim);
    // TODO: assert it's actually immediate
    int64_t s = imm_dim->value();
    r.push_back(s);
  }
  return r;
}

std::vector<mobile::nnc::InputSpec> toInputSpecs(
    const std::vector<std::vector<int64_t>>& inputSizes) {
  std::vector<mobile::nnc::InputSpec> specs;
  for (const auto& sizes : inputSizes) {
    mobile::nnc::InputSpec spec;
    spec.sizes_ = sizes;
    // TODO: Use user specified input type. For now using Long for BI model
    spec.dtype_ = c10::ScalarType::Long;
    specs.emplace_back(std::move(spec));
  }
  return specs;
}

std::unique_ptr<Function> compileMethod(
    std::shared_ptr<tensorexpr::TensorExprKernel> kernel,
    const std::string& method_name,
    const std::vector<std::vector<int64_t>>& sizes) {
  auto func = std::make_unique<Function>();
  func->set_name(method_name);
  func->set_input_specs(toInputSpecs(sizes));

  auto params = c10::impl::GenericList(c10::AnyType::get());
  auto const_descriptors = kernel->getConstantDescriptors();
  for (const auto& cd : const_descriptors) {
    auto sizes = getConstSizes(cd.buf);
    if (cd.ptr) {
      // sizes.empty() needs to be handled as sizes can be empty for Scalar
      // Tensors
      at::Tensor const_tensor = !sizes.empty()
          ? at::from_blob(cd.ptr, sizes).clone()
          : at::native::wrapped_scalar_tensor(*static_cast<double*>(cd.ptr));
      params.push_back(const_tensor);
    } else {
      params.emplace_back(toIValue(cd.node->output()));
    }
  }
  func->set_parameters(params);

  MemoryPlan plan;
  plan.buffer_sizes_ = {}; // temp_sizes_;
  // TODO: implement prealloc optimization and fill in temp_sizes
  func->set_memory_plan(plan);

  int64_t n_inputs = kernel->graph()->inputs().size();
  int64_t n_outputs = kernel->graph()->outputs().size();
  std::vector<OutputSpec> out_spec;
  for (int64_t idx = n_inputs; idx < n_inputs + n_outputs; idx++) {
    const auto& ba = kernel->getBufferArgs()[idx];
    OutputSpec output;
    output.sizes_ = getConstSizes(ba.buf());
    // TODO: assert the output is a buffer and not a scalar
    output.dtype_ = ba.buf()->dtype().scalar_type();
    if (isQIntType(output.dtype_)) {
      // Supporting only static qscale/qzero
      output.qscale_ =
          to<DoubleImm>(torch::jit::tensorexpr::IRSimplifier::simplify(
                            ba.buf()->qscale()))
              ->value();
      output.qzero_ =
          to<LongImm>(
              torch::jit::tensorexpr::IRSimplifier::simplify(ba.buf()->qzero()))
              ->value();
    }
    out_spec.push_back(output);
  }
  func->set_output_specs(out_spec);

  return func;
}

std::pair<std::unique_ptr<Function>, const std::string> aotCompile(
    const std::string& method_name,
    std::shared_ptr<Graph>& g,
    const std::vector<std::vector<int64_t>>& sizes,
    const std::string& kernel_func_name) {
  GRAPH_DEBUG("Input sizes ", sizes);
  GRAPH_DEBUG("Method name ", method_name);

  std::vector<at::IValue> example_values;
  std::vector<c10::optional<at::Tensor>> example_inputs;
  for (const auto& size : sizes) {
    // TODO: Use user specified input type. For now using Long for BI model
    auto example_input = at::rand(size).to(at::dtype(at::kLong));
    example_values.emplace_back(example_input);
    example_inputs.emplace_back(example_input);
  }

  GRAPH_DUMP("graph before compiler passes ", g);
  tensorexpr::removeUnusedSelfArgument(g);
  g = TraceGraph(g, example_values);
  // TODO: Remove annotateInputShapes pass when TraceGraph can also capture
  // input shapes
  tensorexpr::annotateInputShapes(g, example_inputs);
  RemoveListMutation(g);
  RemoveTensorMutation(g);
  EliminateDeadCode(g);
  LowerAllTuples(g);
  GRAPH_DUMP("graph after compiler passes ", g);

  std::shared_ptr<tensorexpr::TensorExprKernel> kernel =
      std::make_shared<tensorexpr::TensorExprKernel>(
          TensorExprKernel(g, kernel_func_name));

  const std::string compiled_assembly = kernel->getCodeText();

  auto func = compileMethod(kernel, method_name, sizes);
  return std::make_pair(std::move(func), compiled_assembly);
}

} // namespace nnc
} // namespace mobile
} // namespace jit
} // namespace torch
