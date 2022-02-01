#include <torch/csrc/jit/mobile/nnc/aot_compiler.h>

#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#include <torch/csrc/jit/backends/backend.h>
#include <torch/csrc/jit/backends/backend_detail.h>
#include <torch/csrc/jit/backends/backend_preprocess.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/frozen_graph_optimizations.h>
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
#include <torch/csrc/jit/tensorexpr/llvm_codegen.h>
#include <fstream>

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

  std::shared_ptr<tensorexpr::TensorExprKernel> kernel =
      std::make_shared<tensorexpr::TensorExprKernel>(
          TensorExprKernel(g, kernel_func_name));

  const std::string compiled_assembly = kernel->getCodeText("asm");

  auto func = compileMethod(kernel, method_name, sizes, types);
  return std::make_pair(std::move(func), compiled_assembly);
}

void writeOutputLlvmAssembly(
    const std::string& asm_code,
    const std::string& output_llvm_file_name) {
  std::ofstream output(output_llvm_file_name);
  output << asm_code;
  GRAPH_DEBUG(
      "The compiled llvm assembly code was saved to ", output_llvm_file_name);
}

std::vector<std::string> split(
    char separator,
    const std::string& string,
    bool ignore_empty = true) {
  std::vector<std::string> pieces;
  std::stringstream ss(string);
  std::string item;
  while (getline(ss, item, separator)) {
    if (!ignore_empty || !item.empty()) {
      pieces.push_back(std::move(item));
    }
  }
  return pieces;
}

std::vector<std::vector<int64_t>> parseInputShapes(
    const std::string& input_dims_s) {
  std::vector<std::string> input_dims_list = split(';', input_dims_s);
  std::vector<std::vector<int64_t>> inputs;
  for (const auto& input_dims_item : input_dims_list) {
    auto input_dims_str = split(',', input_dims_item);
    std::vector<int64_t> input_dims;
    input_dims.reserve(input_dims_str.size());
    for (const auto& s : input_dims_str) {
      input_dims.push_back(c10::stoi(s));
    }
    inputs.push_back(input_dims);
  }
  return inputs;
}

std::vector<at::ScalarType> parseInputTypes(
    const std::string& input_types_str) {
  std::vector<std::string> inputTypes = split(';', input_types_str);
  std::vector<at::ScalarType> scalarTypes;
  for (const auto& inputType : inputTypes) {
    at::ScalarType scalarType;
    if (inputType == "float") {
      scalarType = at::ScalarType::Float;
    } else if (inputType == "uint8") {
      scalarType = at::ScalarType::Byte;
    } else if (inputType == "int64") {
      scalarType = at::ScalarType::Long;
    } else {
      CAFFE_THROW("Unsupported input type: ", inputType);
    }
    scalarTypes.push_back(scalarType);
  }
  return scalarTypes;
}

std::string getNncKernelId(
    const std::string& model_name,
    const std::string& model_version,
    const std::string& method_name) {
  // TODO: calculate the version_token.
  const std::string version_token = "VERTOKEN";
  return model_name + ":" + model_version + ":" + method_name + ":" +
      version_token;
}

std::string getNncKernelFuncName(
    const std::string& model_name,
    const std::string& model_version,
    const std::string& method_name) {
  return "nnc_" + model_name + "_" + model_version + "_" + method_name;
}

std::shared_ptr<Graph> preprocessGraphPasses(
    std::shared_ptr<Graph>& graph,
    const std::vector<c10::optional<at::Tensor>>& example_inputs) {
  GRAPH_DEBUG("Before preprocessing graph passes: ", *graph);
  torch::jit::RemoveTensorMutation(graph);
  torch::jit::EliminateDeadCode(graph->block());
  graph = torch::jit::tensorexpr::removeUnusedSelfArgument(graph);

  torch::jit::tensorexpr::annotateInputShapes(graph, example_inputs);
  torch::jit::OptimizeFrozenGraph(graph, true);
  torch::jit::PropagateShapesOnGraph(graph);
  torch::jit::PeepholeOptimize(graph, false);
  torch::jit::ConstantPropagation(graph);
  torch::jit::PropagateShapesOnGraph(graph);
  torch::jit::PeepholeOptimize(graph, false);
  torch::jit::ConstantPropagation(graph);

  tensorexpr::removeUnusedSelfArgument(graph);

  std::vector<at::IValue> example_values;
  example_values.reserve(example_inputs.size());
  for (auto example_input : example_inputs) {
    example_values.emplace_back(*example_input);
  }
  graph = TraceGraph(graph, example_values);
  // TODO: Remove annotateInputShapes pass when TraceGraph can also capture
  // input shapes
  tensorexpr::annotateInputShapes(graph, example_inputs);

  RemoveListMutation(graph);
  RemoveTensorMutation(graph);
  EliminateDeadCode(graph);
  LowerAllTuples(graph);
  GRAPH_DEBUG("After preprocessing graph passes: ", *graph);
  return graph;
}

std::vector<c10::optional<at::Tensor>> generateExampleInputs(
    const std::vector<std::vector<int64_t>>& inputShapes,
    const std::vector<at::ScalarType>& inputTypes) {
  std::vector<c10::optional<at::Tensor>> example_inputs;
  example_inputs.reserve(inputShapes.size());
  for (int i = 0; i < inputShapes.size(); ++i) {
    example_inputs.emplace_back(
        at::rand(inputShapes[i]).to(at::dtype(inputTypes[i])));
  }
  return example_inputs;
}

c10::IValue preprocess(
    const torch::jit::Module& mod,
    const c10::Dict<c10::IValue, c10::IValue>& compile_spec,
    const torch::jit::BackendDebugHandleGenerator& generate_debug_handles) {
  torch::jit::mobile::nnc::CompilationUnit cu;
  for (const auto& kv : compile_spec) {
    GRAPH_DEBUG("Key: ", kv.key());
    GRAPH_DEBUG("Value: ", kv.value());
    std::string method_name = *(kv.key().toString());
    GRAPH_DEBUG("Method name: ", method_name);
    auto method_spec = kv.value().toGenericDict();
    std::string model_name = *method_spec.at("model_name").toString();
    std::string model_version = *method_spec.at("model_version").toString();
    std::string asmfile_name = *method_spec.at("asmfile").toString();
    std::string arch = *method_spec.at("arch").toString();
    GRAPH_DEBUG("Model name: ", model_name);
    GRAPH_DEBUG("Model version: ", model_version);
    GRAPH_DEBUG("Asm file name: ", asmfile_name);
    GRAPH_DEBUG("Arch: ", arch);

    if (arch == "x86-64") {
      LLVMTargetTriple() = "x86_64-unknown-unknown";
      LLVMTargetAttrs() = "";
    } else if (arch == "aarch64") {
      LLVMTargetTriple() = "aarch64-unknown-unknown";
      LLVMTargetAttrs() = "+neon,+thumb2";
    } else if (arch == "arm") {
      LLVMTargetTriple() = "arm-unknown-unknown";
      LLVMTargetAttrs() = "+neon,+thumb2";
    } else if (arch == "") {
      // Do nothing
    } else {
      TORCH_CHECK(false, "Unknown architecture");
    }

    auto method = mod.get_method(method_name);
    auto graph = toGraphFunction(method.function()).graph()->copy();

    auto sizes = parseInputShapes(*method_spec.at("sizes").toString());
    auto types = parseInputTypes(*method_spec.at("types").toString());

    auto example_inputs = generateExampleInputs(sizes, types);
    graph = preprocessGraphPasses(graph, example_inputs);

    auto kernel_func_name =
        getNncKernelFuncName(model_name, model_version, method_name);
    auto compiled = torch::jit::mobile::nnc::aotCompile(
        method_name, graph, sizes, types, kernel_func_name);
    writeOutputLlvmAssembly(compiled.second, asmfile_name);
    auto func = std::move(compiled.first);
    func->set_nnc_kernel_id(
        getNncKernelId(model_name, model_version, method_name));
    cu.register_function(std::move(func));
  }
  return cu.serialize();
}

static auto reg = torch::jit::backend_preprocess_register("nnc", preprocess);

} // namespace nnc
} // namespace mobile
} // namespace jit
} // namespace torch
