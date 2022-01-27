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

// Construct input-specs vector from the inputs of the original graph
std::vector<mobile::nnc::InputSpec> toInputSpecs(
    const std::shared_ptr<Graph>& g) {
  std::vector<mobile::nnc::InputSpec> specs;
  for (auto v : g->inputs()) {
    const auto& t = v->type();
    mobile::nnc::InputSpec spec;
    TORCH_CHECK(t->kind() == TypeKind::TensorType, "Unsupported input type");
    const auto& tt = t->cast<TensorType>();
    spec.sizes_ = {};
    auto sizes_vec = *tt->sizes().sizes();
    for (auto s : sizes_vec) {
      spec.sizes_.push_back(s ? *s : 0);
    }
    spec.dtype_ = *tt->scalarType();
    specs.emplace_back(std::move(spec));
  }
  return specs;
}

// Locate symbolic shapes in shapes of the inputs.
//
// For each symbolic shape we're trying to find the input from which it can be
// extracted and the dimension index in that input.
// For instance, if we have
// graph(%x : Float(SS(-1), 10), %y : Long(20, SS(-2), %ss_1 : int, %ss_2 : int)
// then we would need to find locations of two symbolic shapes: SS(-1) and
// SS(-2). The first one corresponds to the first dimension of the first input,
// the second one corresponds to the second dimension of the second input,
// so we will return {{0, 0}, {1, 1}}.
//
// If a symbolic shape cannot be found among dimensions of inputs, we
// will throw an error (this situation is possible when symbolic shape
// corresponds to the size of an intermediate - we don't support this
// case here yet).
//
// If a symbolic shape can be found in several different positions, we
// return the first one we find (TODO: maybe we should return all and
// verify that they all match at runtime).
std::vector<SymbolicShapePosition> findSymbolicShapePositions(
    std::shared_ptr<tensorexpr::TensorExprKernel> kernel) {
  std::vector<SymbolicShapePosition> res;
  for (int64_t sym_idx : kernel->getSymbolicShapeInputs()) {
    bool found = false;
    for (int64_t input_idx : c10::irange(kernel->graph()->inputs().size())) {
      auto input = kernel->graph()->inputs()[input_idx];

      if (!input->type()->cast<TensorType>()) {
        continue;
      }
      auto tt = input->type()->expect<TensorType>();
      if (!tt->symbolic_sizes().sizes()) {
        continue;
      }
      std::vector<at::ShapeSymbol> shape_vec = *tt->symbolic_sizes().sizes();
      for (int64_t dim_idx : c10::irange(shape_vec.size())) {
        if (shape_vec[dim_idx].value() == sym_idx) {
          res.push_back({input_idx, dim_idx});
          found = true;
          break;
        }
      }
      if (found) {
        break;
      }
    }
    TORCH_CHECK(
        found, "Could not locate a symbolic shape among input tensor shapes");
  }
  return res;
}

std::unique_ptr<Function> compileMethod(
    std::shared_ptr<tensorexpr::TensorExprKernel> kernel,
    const std::string& method_name,
    const std::vector<std::vector<int64_t>>& sizes,
    const std::vector<at::ScalarType>& types) {
  auto func = std::make_unique<Function>();
  func->set_name(method_name);
  func->set_input_specs(toInputSpecs(kernel->graph()));

  auto params = c10::impl::GenericList(c10::AnyType::get());
  auto const_descriptors = kernel->getConstantDescriptors();
  for (const auto& cd : const_descriptors) {
    auto sizes = getConstSizes(cd.buf);
    if (!cd.node) {
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
  func->set_sym_shape_positions(findSymbolicShapePositions(kernel));

  return func;
}

std::pair<std::unique_ptr<Function>, const std::string> aotCompile(
    const std::string& method_name,
    std::shared_ptr<Graph>& g,
    const std::vector<std::vector<int64_t>>& sizes,
    const std::vector<at::ScalarType>& types,
    const std::string& kernel_func_name) {
  GRAPH_DEBUG("Input sizes ", sizes);
  GRAPH_DEBUG("Input types ", types);
  GRAPH_DEBUG("Method name ", method_name);
  GRAPH_DEBUG("Kernel func name ", kernel_func_name);

  CAFFE_ENFORCE(
      sizes.size() == types.size(),
      "Number of input sizes and input types should be the same");

  std::vector<at::IValue> example_values;
  std::vector<c10::optional<at::Tensor>> example_inputs;
  for (int i = 0; i < sizes.size(); ++i) {
    auto example_input = at::rand(sizes[i]).to(at::dtype(types[i]));
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

  auto func = compileMethod(kernel, method_name, sizes, types);
  return std::make_pair(std::move(func), compiled_assembly);
}

} // namespace nnc
} // namespace mobile
} // namespace jit
} // namespace torch
