#include <torch/csrc/jit/frontend/ir_emitter.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/csrc/jit/runtime/symbolic_shape_registry.h>
#include <unordered_map>

namespace torch {
namespace jit {
namespace {
std::mutex lock;

const std::string shape_compute_functions =
    R"(
        ####     SHAPE COMPUTE FUNCTIONS    ###
        def broadcast(a: List[int], b: List[int]):
          dimsA = len(a)
          dimsB = len(b)
          ndim = max(dimsA, dimsB)
          expandedSizes : List[int] = []

          for i in range(ndim):
            offset = ndim - 1 - i
            dimA = dimsA - 1 - offset
            dimB = dimsB - 1 - offset
            sizeA = a[dimA] if (dimA >= 0) else 1
            sizeB = b[dimB] if (dimB >= 0) else 1

            if sizeA != sizeB and sizeA != 1 and sizeB != 1:
                # TODO: only assertion error is bound in C++ compilation right now
                raise AssertionError("The size of tensor a {} must match the size of tensor b ("
                                "{}) at non-singleton dimension {}".format(sizeA, sizeB, i))

            expandedSizes.append(sizeB if sizeA == 1 else sizeA)

          return expandedSizes

        def adaptive_avg_pool2d(self: List[int], out: List[int]):
          # TODO: return out directly, list len refiner would need to
          # annotate the List Type with len directly in IR
          assert len(out) == 2
          return [out[0], out[1]]

        # TODO: maybe make it customary that extra arguments are unused ?
        # TODO: return self directly
        def unary_two_unused_inputs(self: List[int], inp0: Any, inp1: Any):
          out: List[int] = []
          for elem in self:
            out.append(elem)
          return out

        def broadcast_one_unused_input(self: List[int], other: List[int], unused: Any):
          return broadcast(self, other)
    )";

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
      {"aten::mul.Tensor(Tensor self, Tensor other) -> Tensor", "broadcast"},
      {"aten::div.Tensor(Tensor self, Tensor other) -> Tensor", "broadcast"},
      {"aten::gt.Tensor(Tensor self, Tensor other) -> Tensor", "broadcast"},
      {"aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor", "broadcast_one_unused_input"},
      {"aten::hardtanh(Tensor self, Scalar min_val=-1, Scalar max_val=1) -> Tensor", "unary_two_unused_inputs"},
      {"aten::adaptive_avg_pool2d(Tensor self, int[2] output_size) -> Tensor", "adaptive_avg_pool2d"},
  };
  // clang-format on
  return schema_to_function_graph;
}

std::unordered_map<const FunctionSchema*, std::shared_ptr<Graph>>
    cached_schema_to_graph;

// CompilationUnit that holds all these Functions and keeps them alive.
CompilationUnit compilation_unit;

void loadModule(const CompilationUnit& module) {
  std::unordered_map<std::string, std::shared_ptr<Graph>> reused_functions;

  for (const auto& pair :
       get_schema_to_function_graph().getAllKeysAndValues()) {
    const FunctionSchema* schema_string = &pair.first->schema();
    const std::string& shape_compute_function_name = pair.second;

    if (reused_functions.count(shape_compute_function_name)) {
      cached_schema_to_graph[schema_string] =
          reused_functions[shape_compute_function_name];
      continue;
    }

    Function& shape_compute_function =
        module.get_function(shape_compute_function_name);
    std::shared_ptr<Graph> graph = shape_compute_function.graph();
    Inline(*graph);

    cached_schema_to_graph[schema_string] = graph;
    reused_functions[shape_compute_function_name] = graph;
  }
}

void loadFunctions() {
  compilation_unit.define(
      c10::nullopt, shape_compute_functions, nativeResolver(), nullptr);
  loadModule(compilation_unit);
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

} // namespace jit
} // namespace torch
