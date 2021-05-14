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
// TODO: we are matching on exact string here, which is brittle and can fail in
// non-obvious ways, e.g., you have to return (Tensor) instead of Tensor. It
// would be nice to use something like OperatorMap<T> like `OperatorSet` but for
// mapping Operators to T
// TODO: if a unary function has an equivalent inplace operation, then the shape
// function must be an identity. We could add those automatically.

// clang-format off
std::unordered_map<std::string,  std::string> schema_string_to_function_name_mappings = {
    {"aten::mul.Tensor(Tensor self, Tensor other) -> (Tensor)", "broadcast"},
    {"aten::div.Tensor(Tensor self, Tensor other) -> (Tensor)", "broadcast"},
    {"aten::adaptive_avg_pool2d(Tensor self, int[2] output_size) -> (Tensor)", "adaptive_avg_pool2d"},
};
// clang-format on

std::unordered_map<std::string, std::shared_ptr<Graph>>
    schema_string_to_function_graph;

std::unordered_map<const FunctionSchema*, std::shared_ptr<Graph>>
    cached_schema_to_graph;

// CompilationUnit that holds all these Functions and keeps them alive.
CompilationUnit compilation_unit;

void loadModule(const CompilationUnit& module) {
  std::unordered_map<std::string, std::shared_ptr<Graph>> reused_functions;

  for (const auto& pair : schema_string_to_function_name_mappings) {
    const std::string& schema_string = pair.first;
    const std::string& shape_compute_function_name = pair.second;

    if (reused_functions.count(shape_compute_function_name)) {
      schema_string_to_function_graph[schema_string] =
          reused_functions[shape_compute_function_name];
      continue;
    }

    Function& shape_compute_function =
        module.get_function(shape_compute_function_name);
    std::shared_ptr<Graph> graph = shape_compute_function.graph();
    Inline(*graph);

    schema_string_to_function_graph[schema_string] = graph;
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
  if (schema_string_to_function_graph.size() == 0) {
    loadFunctions();
  }

  auto cache_it = cached_schema_to_graph.find(&schema);
  if (cache_it != cached_schema_to_graph.end()) {
    return cache_it->second;
  } else {
    auto schema_str = toString(schema);
    GRAPH_DEBUG("Trying to find schema: " + schema_str);

    auto sym_shape_it = schema_string_to_function_graph.find(schema_str);

    if (sym_shape_it != schema_string_to_function_graph.end()) {
      cached_schema_to_graph[&schema] = sym_shape_it->second;
      return sym_shape_it->second;
    }
  }
  return c10::nullopt;
}

} // namespace jit
} // namespace torch
