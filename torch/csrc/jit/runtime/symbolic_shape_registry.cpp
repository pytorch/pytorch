#include <c10/util/Exception.h>
#include <torch/csrc/jit/frontend/ir_emitter.h>
#include <torch/csrc/jit/ir/ir_views.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/runtime/graph_iterator.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/csrc/jit/runtime/symbolic_shape_registry.h>
#include <torch/csrc/jit/runtime/symbolic_shape_registry_util.h>
#include <torch/csrc/jit/serialization/import_source.h>
#include <unordered_map>

namespace torch {
namespace jit {
namespace {
std::mutex lock;

// split here to satisfy MSVC++
// https://docs.microsoft.com/en-us/cpp/error-messages/compiler-errors-1/compiler-error-c2026?view=msvc-170
const std::string _shape_compute_functions =
#include <torch/csrc/jit/runtime/shape_functions.h>
    ;
const std::string _shape_compute_functions_1 =
#include <torch/csrc/jit/runtime/shape_functions_1.h>
    ;

const std::string _xnnpack_shape_compute_functions =
#ifdef USE_XNNPACK
    R"(
def prepacked_conv2d_clamp_run(input: List[int], conv2dOpContext: Any):
    assert isinstance(conv2dOpContext, __torch__.torch.classes.xnnpack.Conv2dOpContext)
    (weight, bias, stride, padding, dilation, groups) = unchecked_cast(
        Tuple[List[int], Optional[List[int]], List[int], List[int], List[int], int],
        ops.prepacked.unpack_prepacked_sizes_conv2d(conv2dOpContext),
    )
    return conv2d(input, weight, bias, stride, padding, dilation, groups)

def prepacked_linear_clamp_run(input: List[int], linearOpContext: Any):
    assert isinstance(linearOpContext, __torch__.torch.classes.xnnpack.LinearOpContext)
    (weight, bias) = unchecked_cast(
        Tuple[List[int], Optional[List[int]]],
        ops.prepacked.unpack_prepacked_sizes_linear(linearOpContext),
    )
    return linear(input, weight, bias)
    )"
#else
    ""
#endif
    ;

// mapping function schema to shape compute graphs allows multiple functions to
// share the same shape compute graph, which is memory efficient and also will
// help speed up shape analysis by caching the result of running consecutive ops
// for a particular set of inputs with the same graph, e.g. running a series
// of pointwise ops
// we need a map from schema to shape compute graph, because the aten schema
// is not recoverable from the shape compute graph, since the shape compute
// graph replaces Tensor inputs with List[int] and there are operators like Conv
// which natively have List[int] inputs
// TODO: consider storing shape compute graph directly on operator,
// and merge into native_functions.yaml

// wrapped in function so that operators get registered before map is
// initialized
static const OperatorMap<std::string>& get_schema_to_function_graph() {
  // clang-format off
  static const OperatorMap<std::string> schema_to_function_graph{
      {"aten::contiguous(Tensor(a) self, *, MemoryFormat memory_format=contiguous_format) -> Tensor(a)", "unary"},
      {"aten::rsub.Tensor(Tensor self, Scalar other, Scalar alpha=1) -> Tensor", "unary"},
      {"aten::dropout(Tensor input, float p, bool train) -> Tensor", "unary"},
      {"aten::adaptive_avg_pool2d(Tensor self, int[2] output_size) -> Tensor", "adaptive_avg_pool2d"},
      {"prim::NumToTensor.Scalar(Scalar a) -> Tensor", "zero_dim_tensor"},
      {"prim::NumToTensor.bool(bool a) -> Tensor", "zero_dim_tensor"},
      {"aten::zeros(int[] size, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None) -> (Tensor)", "unary"},
      {"aten::to.dtype(Tensor(a) self, int dtype, bool non_blocking=False, bool copy=False, int? memory_format=None) -> (Tensor(a))", "unary"},
      {"aten::arange(Scalar end, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None) -> (Tensor)", "arange_end"},
      {"aten::arange.start(Scalar start, Scalar end, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor", "arange_start"},
      {"aten::arange.start_step(Scalar start, Scalar end, Scalar step, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor", "arange_start_step"},
      {"aten::squeeze(Tensor(a) self) -> Tensor(a)", "squeeze_nodim"},
      {"aten::squeeze.dim(Tensor(a) self, int dim) -> Tensor(a)", "squeeze"},
      {"aten::unsqueeze(Tensor(a) self, int dim) -> Tensor(a)", "unsqueeze"},
      {"aten::slice.Tensor(Tensor(a) self, int dim=0, int? start=None, int? end=None, int step=1) -> Tensor(a)", "slice"},
      {"aten::select.int(Tensor(a) self, int dim, int index) -> Tensor(a)", "select"},
      {"aten::index_select(Tensor self, int dim, Tensor index) -> Tensor", "index_select"},
      {"aten::layer_norm(Tensor input, int[] normalized_shape, Tensor? weight=None, Tensor? bias=None, "
       "float eps=1e-05, bool cudnn_enable=True) -> Tensor", "unary"},
      {"aten::softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor", "unary"},
      {"aten::_no_grad_embedding_renorm_(Tensor weight, Tensor input, float max_norm, float norm_type) -> Tensor", "unary"},
      {"aten::embedding_renorm_(Tensor(a!) self, Tensor indices, float max_norm, float norm_type) -> Tensor(a!)", "unary"},
      {"aten::embedding(Tensor weight, Tensor indices, int padding_idx=-1, bool scale_grad_by_freq=False, bool sparse=False) -> Tensor", "embedding"},
      {"aten::mm(Tensor self, Tensor mat2) -> Tensor", "mm"},
      {"aten::dot(Tensor self, Tensor tensor) -> Tensor", "dot"},
      {"aten::mv(Tensor self, Tensor vec) -> Tensor", "mv"},
      {"aten::matmul(Tensor self, Tensor other) -> Tensor", "matmul"},
      {"aten::linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor", "linear"},
      {"aten::max_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> Tensor", "max_pool2d"},
      {"aten::max_pool2d_with_indices(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)", "max_pool2d_with_indices"},
      {"aten::t(Tensor(a) self) -> Tensor(a)", "t"},
      {"aten::transpose.int(Tensor(a) self, int dim0, int dim1) -> Tensor(a)", "transpose"},
      {"aten::conv1d(Tensor input, Tensor weight, Tensor? bias=None, int[1] stride=1, int[1] padding=0, int[1] dilation=1, int groups=1) -> Tensor", "conv1d"},
      {"aten::conv2d(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1, int groups=1) -> Tensor", "conv2d"},
      {"aten::batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled) -> Tensor", "batch_norm"},
      {"aten::conv3d(Tensor input, Tensor weight, Tensor? bias=None, int[3] stride=1, int[3] padding=0, int[3] dilation=1, int groups=1) -> Tensor", "conv3d"},
      {"aten::flatten.using_ints(Tensor(a) self, int start_dim=0, int end_dim=-1) -> Tensor(a)", "flatten"},
      {"aten::cat(Tensor[] tensors, int dim=0) -> Tensor", "cat"},
      {"aten::permute(Tensor(a) self, int[] dims) -> Tensor(a)", "permute"},
      {"aten::view(Tensor(a) self, int[] size) -> Tensor(a)", "view"},
      {"aten::expand_as(Tensor(a) self, Tensor other) -> Tensor(a)", "expand"},
      {"aten::expand(Tensor(a) self, int[] size, *, bool implicit=False) -> Tensor(a)", "expand_one_unused"},
      {"aten::mean.dim(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor", "mean_dim"},
      {"aten::sum.dim_IntList(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor", "mean_dim"},
      {"aten::max.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)", "max_dim"},
      {"aten::mean(Tensor self, *, ScalarType? dtype=None) -> Tensor", "zero_dim_tensor"},
      {"aten::sum(Tensor self, *, ScalarType? dtype=None) -> Tensor", "zero_dim_tensor"},
      {"aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor", "addmm"},
      {"aten::upsample_nearest2d.vec(Tensor input, int[]? output_size, float[]? scale_factors) -> (Tensor)", "upsample_nearest2d"},
      {"aten::quantize_per_tensor(Tensor self, float scale, int zero_point, ScalarType dtype) -> Tensor", "unary"},
      {"aten::quantize_per_tensor.tensor_qparams(Tensor self, Tensor scale, Tensor zero_point, ScalarType dtype) -> Tensor", "unary"},
      {"aten::dequantize(Tensor self) -> Tensor", "unary"},
      {"quantized::conv2d.new(Tensor qx, __torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weight, float output_scale, int output_zero_point) -> Tensor", "quantized_prepacked_conv2d"},
      {"quantized::conv2d_relu.new(Tensor qx, __torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weight, float output_scale, int output_zero_point) -> Tensor", "quantized_prepacked_conv2d"},
      {"quantized::add(Tensor qa, Tensor qb, float scale, int zero_point) -> Tensor qc", "broadcast"},
#ifdef USE_XNNPACK
      {"prepacked::conv2d_clamp_run(Tensor X, __torch__.torch.classes.xnnpack.Conv2dOpContext W_prepack) -> Tensor Y", "prepacked_conv2d_clamp_run"},
      {"prepacked::linear_clamp_run(Tensor X, __torch__.torch.classes.xnnpack.LinearOpContext W_prepack) -> Tensor Y", "prepacked_linear_clamp_run"},
#endif
  };
  // clang-format on
  return schema_to_function_graph;
}

std::unordered_map<const FunctionSchema*, std::shared_ptr<Graph>>
    cached_schema_to_graph;

// CompilationUnit that holds all these Functions and keeps them alive.
auto compilation_unit = std::make_shared<CompilationUnit>();

const at::optional<const FunctionSchema*> getInplaceVariant(
    const FunctionSchema& base_schema) {
  auto& inplace_variants =
      getAllOperatorsFor(c10::Symbol::fromQualString(base_schema.name() + "_"));

  for (const auto& variant : inplace_variants) {
    // Need to check that all args are the same except for the first, which
    // is almost the same except for the Alias info
    const FunctionSchema* schema = &variant->schema();
    if (!schema->isSubtypeOf(base_schema, false)) {
      continue;
    }

    Argument self_arg = schema->arguments()[0];
    if (!self_arg.alias_info()->isWrite()) {
      continue;
    }

    Argument ret_arg = schema->returns()[0];
    if (!ret_arg.alias_info()->isWrite()) {
      continue;
    }

    return schema;
  }
  return at::nullopt;
}

TypePtr mapTensorToListOfInts(TypePtr type) {
  if (type->cast<TensorType>()) {
    return ListType::ofInts();
  }
  at::ArrayRef<TypePtr> contained = type->containedTypes();
  if (contained.empty()) {
    return type;
  }
  return type->withContained(
      fmap(type->containedTypes(), mapTensorToListOfInts));
}

void checkForWhileLoop(
    const FunctionSchema* schema,
    std::shared_ptr<Graph> graph) {
  DepthFirstGraphNodeIterator graph_it(graph);
  for (auto* node = graph_it.next(); node != nullptr; node = graph_it.next()) {
    if (node->kind() != prim::Loop) {
      continue;
    }
    LoopView loop(node);
    if (loop.loopType() != LoopView::For) {
      TORCH_WARN(
          "While loops are not yet implemented in unrolling which may make this shape function difficult to partially evaluate: ",
          *node,
          " for schema ",
          *schema);
    }
  }
}

void checkInputReturnedAsOutput(
    const FunctionSchema* schema,
    const std::shared_ptr<Graph>& graph) {
  // Could use alias db here as well but would have to warn because it's
  // imprecise
  for (size_t i : c10::irange(graph->inputs().size())) {
    Value* input = graph->inputs().at(i);
    for (size_t j : c10::irange(graph->outputs().size())) {
      Value* output = graph->outputs().at(j);
      TORCH_CHECK(
          input != output,
          "For schema: ",
          *schema,
          " input index ",
          i,
          " is returned as output index ",
          j,
          ". Shape functions must return new unaliased lists");
    }
  }
}

void checkInputAndOutputTypes(
    const FunctionSchema* schema,
    const std::shared_ptr<Graph>& graph) {
  // allow extra unused arguments to map multiple functions to e.g. unary
  TORCH_CHECK(
      graph->inputs().size() <= schema->arguments().size(),
      "Shape function must have fewer arguments than schema. Got ",
      graph->inputs().size(),
      " graph arguments and ",
      schema->arguments().size(),
      " schema arguments of schema: ",
      *schema);

  for (auto i : c10::irange(graph->inputs().size())) {
    auto inp_type = schema->arguments().at(i).type();
    auto mapped_type = mapTensorToListOfInts(inp_type);
    auto graph_type = graph->inputs().at(i)->type();
    TORCH_INTERNAL_ASSERT(
        mapped_type->isSubtypeOf(graph->inputs().at(i)->type()),
        "For schema type: ",
        inp_type->str(),
        " Expected supertype of ",
        mapped_type->str(),
        " but got graph_type ",
        graph_type->str(),
        " at index ",
        i,
        " of schema: ",
        *schema);
  }

  TORCH_CHECK(
      graph->outputs().size() == schema->returns().size(),
      "Shape function equal number of outputs as schema. Got ",
      graph->outputs().size(),
      " graph outputs and ",
      schema->returns().size(),
      " schema returns of schema: ",
      *schema);

  for (auto i : c10::irange(schema->returns().size())) {
    auto out_type = schema->returns().at(i).type();
    auto mapped_type = mapTensorToListOfInts(out_type);
    auto graph_type = graph->outputs().at(i)->type();
    TORCH_INTERNAL_ASSERT(
        mapped_type->isSubtypeOf(graph->outputs().at(i)->type()),
        "For schema type: ",
        out_type->str(),
        " Expected supertype of ",
        mapped_type->str(),
        " but got graph_type ",
        graph_type->str(),
        " at output index ",
        i,
        " of schema: ",
        *schema);
  }
}

void checkShapeFunction(
    const FunctionSchema* schema,
    const std::shared_ptr<Graph>& graph) {
  checkInputAndOutputTypes(schema, graph);
  checkForWhileLoop(schema, graph);
  checkInputReturnedAsOutput(schema, graph);
  // TODO: other checks ? list ops which we don't symbolically optimize, etc ?
}

void transformShapeFunction(
    const FunctionSchema* schema_string,
    std::shared_ptr<Graph> graph) {
  Inline(*graph);

  // ATEN operators can return multiple unboxed values, this in contrast to
  // functions defined in TorchScript or User-Registered Operators
  // Which must use a Tuple
  // Here, modify the shape graph of aten operators with multiple outputs
  // so that they correspond to each other
  if (schema_string->returns().size() > 1) {
    TORCH_INTERNAL_ASSERT(
        graph->outputs().size() == 1 &&
        graph->outputs().at(0)->node()->kind() == prim::TupleConstruct);
    auto tuple_node = graph->outputs().at(0)->node();
    graph->eraseOutput(0);
    for (Value* v : tuple_node->inputs()) {
      graph->registerOutput(v);
    }
  }
}

void registerSchema(
    const FunctionSchema* schema_string,
    const std::string& shape_compute_function_name,
    std::unordered_map<std::string, std::shared_ptr<Graph>>& reused_functions,
    const CompilationUnit& module) {
  if (reused_functions.count(shape_compute_function_name)) {
    auto graph = reused_functions[shape_compute_function_name];

    // allow extra unused arguments to map multiple functions to e.g. unary
    TORCH_INTERNAL_ASSERT(
        graph->inputs().size() <= schema_string->arguments().size());

    cached_schema_to_graph[schema_string] = graph;
    return;
  }

  Function& shape_compute_function =
      module.get_function(shape_compute_function_name);
  std::shared_ptr<Graph> graph =
      toGraphFunction(shape_compute_function).graph();

  transformShapeFunction(schema_string, graph);
  checkShapeFunction(schema_string, graph);

  cached_schema_to_graph[schema_string] = graph;
  reused_functions[shape_compute_function_name] = graph;
}

void loadModule(const CompilationUnit& module) {
  std::unordered_map<std::string, std::shared_ptr<Graph>> reused_functions;

  std::vector<std::pair<std::shared_ptr<Operator>, std::string>>
      operator_pairs = get_schema_to_function_graph().getAllKeysAndValues();
  auto te_ops = get_tensorexpr_elementwise_set().getAllKeysAndValues();
  operator_pairs.insert(operator_pairs.end(), te_ops.begin(), te_ops.end());

  for (const auto& pair : operator_pairs) {
    const FunctionSchema* schema_string = &pair.first->schema();
    const std::string& shape_compute_function_name = pair.second;

    registerSchema(
        schema_string, shape_compute_function_name, reused_functions, module);

    // Register the inplace variant if any for functions with common shape forms
    if (shape_compute_function_name == "unary") {
      auto inplace_schema = getInplaceVariant(*schema_string);
      if (inplace_schema.has_value()) {
        registerSchema(
            inplace_schema.value(), "unary", reused_functions, module);
      }
    }
    if (shape_compute_function_name == "broadcast") {
      auto inplace_schema = getInplaceVariant(*schema_string);
      if (inplace_schema.has_value()) {
        registerSchema(
            inplace_schema.value(),
            "broadcast_inplace",
            reused_functions,
            module);
      }
    }
  }
}

void loadFunctions() {
  // these should be static casts but not possible until C++17
  // https://stackoverflow.com/a/43335753/9045206
  auto start = _shape_compute_functions.find(
      "####    SHAPE COMPUTE FUNCTIONS START   ###");
  auto end = _shape_compute_functions.find(
      "####    SHAPE COMPUTE FUNCTIONS END   ###");
  auto start_1 = _shape_compute_functions_1.find(
      "####    SHAPE COMPUTE FUNCTIONS START   ###");
  auto end_1 = _shape_compute_functions_1.find(
      "####    SHAPE COMPUTE FUNCTIONS END   ###");
  TORCH_INTERNAL_ASSERT(start != std::string::npos && end != std::string::npos);
  TORCH_INTERNAL_ASSERT(
      start_1 != std::string::npos && end_1 != std::string::npos);

  auto shape_compute_functions =
      _shape_compute_functions.substr(start, end - start) +
      _shape_compute_functions_1.substr(start_1, end_1 - start_1) +
      _xnnpack_shape_compute_functions;

  auto src = std::make_shared<Source>(shape_compute_functions);
  std::stringstream ss;
  std::vector<at::IValue> constantTable;
  auto resolver = std::make_shared<SourceImporterImpl>(
      compilation_unit,
      &constantTable,
      [&](const std::string& name) -> std::shared_ptr<Source> { return src; },
      1);
  compilation_unit->define(
      c10::nullopt, shape_compute_functions, resolver, nullptr);
  loadModule(*compilation_unit);
}
} // anonymous namespace

c10::optional<std::shared_ptr<Graph>> shapeComputeGraphForSchema(
    const FunctionSchema& schema) {
  std::lock_guard<std::mutex> guard(lock);
  if (cached_schema_to_graph.size() == 0) {
    loadFunctions();
  }

  GRAPH_DEBUG("Trying to find schema: ", schema);
  auto cache_it = cached_schema_to_graph.find(&schema);
  if (cache_it != cached_schema_to_graph.end()) {
    return cache_it->second;
  }
  GRAPH_DEBUG("Could not find schema: ", schema);

  return c10::nullopt;
}

void RegisterShapeComputeGraphForSchema(
    const FunctionSchema& schema,
    std::shared_ptr<Graph> g) {
  std::lock_guard<std::mutex> guard(lock);
  if (cached_schema_to_graph.size() == 0) {
    loadFunctions();
  }
  transformShapeFunction(&schema, g);
  checkShapeFunction(&schema, g);

  cached_schema_to_graph[&schema] = g;
}

} // namespace jit
} // namespace torch
