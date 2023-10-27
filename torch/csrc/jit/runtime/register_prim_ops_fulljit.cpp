#include <torch/csrc/jit/codegen/fuser/interface.h>
#include <torch/csrc/jit/runtime/register_ops_utils.h>

#include <ATen/core/ivalue.h>
#include <c10/util/irange.h>
#include <torch/csrc/autograd/profiler.h>
#include <torch/csrc/jit/frontend/tracer.h>

#include <algorithm>
#include <bitset>
#include <cctype>
#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <ostream>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace torch::jit {

namespace {

RegisterOperators reg({
    Operator(
        prim::profile,
        [](const Node* node) -> Operation {
          return [](Stack& stack) {
            AT_ERROR(
                "Must be lowered to Interpreter's PROFILE instruction"); // NOLINT
          };
        },
        aliasAnalysisSpecialCase()),
    Operator(
        prim::profile_ivalue,
        [](const Node* node) -> Operation {
          return [](Stack& stack) {
            AT_ERROR(
                "Must be lowered to Interpreter's PROFILE instruction"); // NOLINT
          };
        },
        aliasAnalysisSpecialCase()),
    Operator(
        prim::FusionGroup,
        [](const Node* node) -> Operation {
          const auto key = registerFusion(node);
          return [key](Stack& stack) {
            RECORD_FUNCTION("FusionGroup", std::vector<c10::IValue>());
            runFusion(key, stack);
          };
        },
        aliasAnalysisSpecialCase()),
    Operator(
        prim::RequiresGradCheck /* (...)  -> (..., bool) */,
        [](const Node* node) -> Operation {
          std::vector<bool> rg_props =
              fmap(node->tys(attr::types), [](const TypePtr& t) {
                // if an rg property changes we assume a tensor does require
                // gradients which is set in `guardDifferentiableGraph`
                TORCH_INTERNAL_ASSERT(
                    t->castRaw<TensorType>()->requiresGrad().has_value());
                return *t->castRaw<TensorType>()->requiresGrad();
              });
          return [rg_props](Stack& stack) {
            auto num_inputs = rg_props.size();
            // Check every input's shape against profiled (expected) shape.
            for (const auto i : c10::irange(num_inputs)) {
              auto& input = peek(stack, i, num_inputs);
              const auto& t = input.toTensor();
              if (rg_props[i] != t.requires_grad()) {
                push(stack, false);
                return;
              }
            }

            push(stack, true);
          };
        },
        aliasAnalysisSpecialCase()),
    Operator(
        prim::ConstantChunk,
        [](const Node* node) -> Operation {
          int64_t chunks = node->i(attr::chunks);
          int64_t dim = node->i(attr::dim);
          auto outputs_used = fmap(node->outputs(), [](const Value* v) {
            return !v->uses().empty();
          });
          return [=](Stack& stack) {
            RECORD_FUNCTION("chunk", last(stack, 1));

            at::Tensor t;
            pop(stack, t);
            auto result = at::chunk(t, chunks, dim);
            stack.insert(
                stack.end(),
                std::make_move_iterator(result.begin()),
                std::make_move_iterator(result.end()));
            // NB: Chunk can sometimes return a smaller number of outputs.
            int64_t num_results = result.size();
            if (num_results != chunks) {
              if (num_results > chunks) {
                TORCH_CHECK(
                    num_results == chunks,
                    "Expected chunk to return ",
                    chunks,
                    " outputs, but got ",
                    num_results);
              }
              for (const auto i : c10::irange(num_results, chunks)) {
                TORCH_CHECK(
                    !outputs_used[i],
                    "Expected chunk to return at least ",
                    chunks,
                    " outputs, but got only ",
                    num_results);
                // We know that the output is unused, so it's ok to push
                // anything on the stack.
                stack.emplace_back();
              }
            }
          };
        },
        aliasAnalysisSpecialCase()),
    Operator(
        prim::ChunkSizes,
        [](const Node* node) -> Operation {
          int64_t raw_dim = node->i(attr::dim);
          int64_t chunks = node->i(attr::chunks);
          return [raw_dim, chunks](Stack& stack) {
            c10::List<int64_t> shape = pop(stack).toIntList();
            c10::List<int64_t> regular_shape = shape.copy();
            c10::List<int64_t> last_shape = shape.copy();
            int64_t dim = at::maybe_wrap_dim(raw_dim, shape.size());
            TORCH_CHECK(
                dim < (int64_t)regular_shape.size(),
                "Dimension out of range for chunk");
            int64_t split_size = (regular_shape[dim] + chunks - 1) / chunks;
            regular_shape[dim] = split_size;
            if (shape[dim] % chunks == 0) {
              last_shape[dim] = split_size;
            } else {
              int64_t num_splits = std::max<int64_t>(
                  (shape[dim] + split_size - 1) / split_size, 1);
              last_shape[dim] =
                  split_size - (split_size * num_splits - shape[dim]);
              AT_ASSERT(last_shape[dim] >= 0);
            }
            push(stack, std::move(regular_shape));
            push(stack, std::move(last_shape));
          };
        },
        aliasAnalysisSpecialCase()),
    Operator(
        "aten::_grad_sum_to_size(Tensor(a) self, int[]? size) -> Tensor(a)",
        [](Stack& stack) {
          RECORD_FUNCTION("_grad_sum_to_size", std::vector<c10::IValue>());
          IValue self, size;
          pop(stack, self, size);
          if (size.isNone()) {
            push(stack, std::move(self));
          } else {
            push(stack, at::sum_to(self.toTensor(), size.toDimVector()));
          }
        },
        aliasAnalysisFromSchema()),
    // This operator is generated inside the compiler for indexing into
    // ModuleDict without a statically determinable key. Accordingly,
    // self must be a ModuleType and the output must be an InterfaceType.
    OperatorGenerator(
        TORCH_SELECTIVE_SCHEMA(
            "prim::ModuleContainerIndex.dict(Any self, str ind) -> Any"),
        [](Stack& stack) {
          IValue ind = pop(stack);
          IValue module_dict = pop(stack);
          push(stack, module_dict.toModule().attr(ind.toStringRef()));
        },
        aliasAnalysisFromSchema()),
    Operator(
        prim::TypeCheck /* (...)  -> (..., bool) */,
        [](const Node* /* node */) -> Operation {
          return [](Stack& /* stack */) {
            AT_ERROR("prim::TypeCheck not yet implemented"); // NOLINT
          };
        },
        aliasAnalysisSpecialCase()),
    Operator(
        prim::FallbackGraph,
        [](const Node* node) -> Operation {
          return [](Stack& stack) {
            AT_ERROR(
                "Must be converted to prim::FunctionCall by replaceFallbackGraphWithFallbackFunction"); // NOLINT
          };
        },
        aliasAnalysisSpecialCase()),
    Operator(
        "prim::Guard(Tensor(a) t) -> Tensor(a)",
        [](Stack& stack) { AT_ERROR("Should be replaced by prim::BailOut"); },
        aliasAnalysisFromSchema()),
    Operator(
        "prim::BailOut(...) -> Tensor(a)",
        [](Stack& /* stack */) {
          AT_ERROR("prim::BailOut not yet implemented"); // NOLINT
        },
        aliasAnalysisFromSchema()),
    Operator(
        "prim::BailoutTemplate() -> int",
        [](Stack& stack) {
          // TODO: today, we put a single bailout template at the front to
          // carry the un-optimized graph for bailout nodes to use. Ideally
          // this should never run, but we haven't written the code to remove
          // it yet.
          // TORCH_INTERNAL_ASSERT(false);

          // Returns an int so that we have an easy way to do graph traversal
          push(stack, 1);
        },
        aliasAnalysisFromSchema()),
    Operator(
        "aten::grad(Tensor[] outputs, Tensor[] inputs, Tensor?[]? grad_outputs=None, bool? retain_graph=None, bool create_graph=False, bool allow_unused=False) -> Tensor?[]",
        [](Stack& stack) {
          bool allow_unused = pop(stack).toBool();
          bool create_graph = pop(stack).toBool();
          auto retain_graph = pop(stack).toOptional<bool>();
          auto grad_outputs = pop(stack);
          auto inputs = pop(stack).toTensorList();
          auto outputs = pop(stack).toTensorList();
          std::vector<torch::autograd::Variable> input_vars(
              inputs.begin(), inputs.end());
          std::vector<torch::autograd::Variable> output_vars(
              outputs.begin(), outputs.end());
          std::vector<torch::autograd::Variable> gradients;

          if (!grad_outputs.isNone()) {
            for (const IValue& v : grad_outputs.toListRef()) {
              gradients.emplace_back(v.isNone() ? at::Tensor() : v.toTensor());
            }
          }

          auto res = torch::autograd::grad(
              output_vars,
              input_vars,
              gradients,
              retain_graph,
              create_graph,
              allow_unused);

          c10::impl::GenericList res_list{OptionalType::ofTensor()};
          for (const at::Tensor& t : res) {
            res_list.emplace_back(t.defined() ? t : IValue());
          }
          push(stack, res_list);
        },
        aliasAnalysisFromSchema()),
    // NB: backward op might write to every input tensors in the graph and it's
    // much more expensive to analyze the leaves and sometimes it might retain
    // the whole gradients in every tensor of the Autograd graph with
    // create_graph=True so we use aliasAnalysisConservative for these two OPs
    Operator(
        "aten::backward.TensorList(Tensor[] tensors, Tensor?[]? grad_tensors=None, bool? retain_graph=None, bool create_graph=False) -> ()",
        [](Stack& stack) {
          bool create_graph = pop(stack).toBool();
          auto retain_graph = pop(stack).toOptional<bool>();
          auto grad_tensors = pop(stack);
          auto outputs = pop(stack).toTensorList();
          std::vector<torch::autograd::Variable> output_vars(
              outputs.begin(), outputs.end());
          std::vector<torch::autograd::Variable> gradients;

          if (!grad_tensors.isNone()) {
            for (const IValue& v : grad_tensors.toListRef()) {
              gradients.emplace_back(v.isNone() ? at::Tensor() : v.toTensor());
            }
          }

          torch::autograd::backward(
              output_vars, gradients, retain_graph, create_graph);
        },
        aliasAnalysisConservative()),
    Operator(
        "aten::save(t item, str filename) -> ()",
        [](Stack& stack) {
          auto filename = pop(stack).toStringRef();
          auto ivalue = pop(stack);

          // Pickle the tensor
          auto data = jit::pickle_save(ivalue);

          // Write file
          std::fstream output(filename, std::ios::out | std::ios::binary);
          output.write(data.data(), data.size());
        },
        aliasAnalysisFromSchema()),
    Operator(
        "prim::IgnoredPythonOp(...) -> None",
        [](Stack& stack) {
          throw JITException(
              "This Python function is annotated to be ignored"
              " and cannot be and has not been included in the exported"
              " binary, meaning that it cannot be executed now."
              " Make sure that ignored operations are never executed after"
              " import");
        },
        aliasAnalysisFromSchema()),
    Operator(
        "aten::wait(Future(t) self) -> t",
        [](Stack& stack) {
          TORCH_CHECK(false, "wait is implemented directly in the interpreter");
        },
        aliasAnalysisSpecialCase()),
    Operator(
        "prim::awaitable_wait(Await(t) self) -> t",
        [](Stack& stack) {
          auto aw = stack.back().toAwait();
          aw->wait();
          stack.pop_back();
          stack.emplace_back(aw->value());
        },
        aliasAnalysisSpecialCase()),
    Operator(
        "prim::awaitable_nowait(t self) -> Await(t)",
        [](Stack& stack) {
          auto aw =
              c10::make_intrusive<c10::ivalue::Await>(stack.back().type());
          aw->markCompleted(pop(stack));
          push(stack, std::move(aw));
        },
        aliasAnalysisSpecialCase()),
});

RegisterOperators logging_operators(
    {Operator(
         "prim::AddStatValue(str key, int val) -> ()",
         [](Stack& stack) {
           auto val = pop(stack).toInt();
           auto key = pop(stack).toString();

           auto schema =
               parseSchema("prim::AddStatValue(str key, int val) -> ()");
           // TODO: remove this custom tracing code once the custom op bugfix
           // lands
           if (jit::tracer::isTracing()) {
             const auto& graph = tracer::getTracingState()->graph;
             Node* node = graph->create(prim::AddStatValue, /*num_outputs=*/0);
             tracer::recordSourceLocation(node);
             node->addInput(insertConstant(*graph, key));
             tracer::addInputs(node, "val", val);
             graph->insertNode(node);
           }
           torch::jit::logging::getLogger()->addStatValue(*key, val);
         },
         aliasAnalysisFromSchema()),
     Operator(
         "prim::TimePoint() -> int",
         [](Stack& stack) {
           auto schema = parseSchema("prim::TimePoint() -> int");
           Node* node = nullptr;
           // TODO: remove this custom tracing code once the custom op bugfix
           // lands
           if (jit::tracer::isTracing()) {
             const auto& graph = tracer::getTracingState()->graph;
             Node* node = graph->create(prim::TimePoint, /*num_outputs=*/0);
             tracer::recordSourceLocation(node);
             graph->insertNode(node);
           }
           auto output =
               torch::profiler::impl::getTime(/*allow_monotonic=*/true);
           push(stack, output);
           if (jit::tracer::isTracing()) {
             jit::tracer::addOutput(node, output);
           }
         },
         aliasAnalysisFromSchema())});

C10_UNUSED void hashValue(Stack& stack) {
  auto value = pop(stack);
  push(stack, value.hash());
}

bool isSortableTupleType(
    const TupleTypePtr& tuple_type,
    std::stringstream& why_not) {
  for (const TypePtr& ele_type : tuple_type->containedTypes()) {
    switch (ele_type->kind()) {
      case TypeKind::IntType:
      case TypeKind::BoolType:
      case TypeKind::FloatType:
      case TypeKind::StringType:
      case TypeKind::TensorType:
        continue;
      case TypeKind::TupleType:
        if (!isSortableTupleType(ele_type->expect<TupleType>(), why_not)) {
          return false;
        }
        continue;
      case TypeKind::ClassType:
        if (!c10::checkObjectSortSchema(
                ele_type->expect<ClassType>(), why_not)) {
          return false;
        }
        continue;
      default:
        why_not << "Contained elements in " << *tuple_type
                << " are not sortable. Only Int, Bool, Float, String, Tensor, "
                << "a User Defined Class with __lt__ method defined or Tuples "
                << "of aforementionted types can be sorted.";
        return false;
    }
  }

  return true;
}

bool isSortableListOfObjectsOrTuples(
    c10::List<IValue>& ivalues,
    std::stringstream& why_not) {
  if (ivalues.empty()) {
    return true;
  }

  auto type = ivalues.get(0).type();
  // We assume lists have homogenous types, use first element to determine
  // best sorting methods. If in the future we need to support heterogenous
  // types inside list, then sorting needs to have runtime sortable checks.
  const size_t n = ivalues.size();
  for (const auto i : c10::irange(n)) {
    const IValue& v = ivalues.get(i);
    auto curr_type = v.type();
    if (*curr_type != *type) {
      why_not << "Only values of same type can be compared. "
              << "Found " << type->repr_str() << " and "
              << curr_type->repr_str();
      return false;
    }
  }

  if (auto tuple_type = type->cast<TupleType>()) {
    return isSortableTupleType(tuple_type, why_not);
  }

  if (auto class_type = type->cast<ClassType>()) {
    return c10::checkObjectSortSchema(class_type, why_not) != nullptr;
  }

  // Basic types like tensors/ints/floats/bools/strs are not checked in this
  // method because they should have been schema matched to specialized
  // aten::sort kernels using listSort<T>.
  why_not << "Only list of Tensors, ints, floats, bools, strs, "
          << "a User Defined Class that defines the __lt__ compare method "
          << "or Tuples of aforementioned types can be sorted, got list of "
          << type->repr_str() << "\n";
  return false;
}

template <bool has_reverse_arg, bool copy_return_list>
void sort_op(Stack& stack) {
  bool reverse = has_reverse_arg ? pop(stack).toBool() : false;
  auto g_list = pop(stack).toList();

  if (copy_return_list) {
    g_list = g_list.copy();
  }

  if (!g_list.empty()) {
    std::stringstream error_str;
    if (!isSortableListOfObjectsOrTuples(g_list, error_str)) {
      throw std::runtime_error(error_str.str());
    }

    c10::IValueComparator comparator;
    if (reverse) {
      comparator = c10::getGreaterThanComparator(g_list.get(0));
    } else {
      comparator = c10::getLessThanComparator(g_list.get(0));
    }
    std::sort(g_list.begin(), g_list.end(), comparator);
  }

  if (copy_return_list) {
    push(stack, g_list);
  }
}

// NB: this must be registered after the other aten::sort operators
RegisterOperators regSort({
    Operator(
        "aten::sorted.any(t[](a) self) -> (t[])",
        sort_op</*has_reverse_arg*/ false, /*copy_return_list*/ true>,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::sort.any(t[](a!) self, bool reverse=False) -> ()",
        sort_op</*has_reverse_arg*/ true, /*copy_return_list*/ false>,
        aliasAnalysisFromSchema()),
});

// reference: _output_size in torch/nn/functional.py
// size can be none, int or intlist
// scale_factors can be none, float, or floatlist
std::vector<int64_t> _output_size(
    const at::Tensor& input,
    size_t dim,
    const IValue& size,
    const IValue& scale_factors) {
  if (!size.isNone()) {
    if (size.isInt()) {
      std::vector<int64_t> repeated(dim, size.toInt());
      return repeated;
    } else {
      return size.toIntVector();
    }
  }
  std::vector<double> scale_repeated;
  if (scale_factors.isDouble()) {
    scale_repeated = std::vector<double>(dim, scale_factors.toDouble());
  } else {
    scale_repeated = scale_factors.toDoubleVector();
  }
  std::vector<int64_t> ret;
  for (const auto i : c10::irange(dim)) {
    ret.push_back(std::floor(input.size(i + 2) * scale_repeated[i]));
  }
  return ret;
}

// return true if v is a real float
// and false if it is an integer
bool _is_floating_value(double v) {
  return std::floor(v) != v;
}

// reference: interpolate in torch/nn/functional.py
// size can be none, int or intlist
// scale_factors can be none, float, or floatlist
at::Tensor interpolate(
    const at::Tensor& input,
    const IValue& size,
    const IValue& scale_factors,
    const std::string& mode,
    c10::optional<bool> align_corners,
    c10::optional<bool> recompute_scale_factor) {
  if ((mode == "nearest" || mode == "area")) {
    if (align_corners != c10::nullopt) {
      throw std::runtime_error(
          "align_corners option can only be set with the "
          "interpolating modes: linear | bilinear | bicubic | trilinear");
    }
  } else {
    if (align_corners == c10::nullopt) {
      TORCH_WARN(
          "Default upsampling behavior when mode=",
          mode,
          " is changed "
          "to align_corners=False since 0.4.0. Please specify align_corners=True "
          "if the old behavior is desired. See the documentation of nn.Upsample for details");
      align_corners = false;
    }
  }

  double scale_factors_1 = -1.0;
  double scale_factors_2 = -1.0;
  double scale_factors_3 = -1.0;

  if (!scale_factors.isNone() && recompute_scale_factor == c10::nullopt) {
    recompute_scale_factor = true;
    bool warn_recompute_scale_factor = false;

    if (scale_factors.isDouble()) {
      // only warn when the scales have floating values since
      // the result for ints is the same with/without recompute_scale_factor
      if (_is_floating_value(scale_factors.toDouble())) {
        warn_recompute_scale_factor = true;
      }
    } else if (scale_factors.isDoubleList()) {
      auto scale_factors_list = scale_factors.toDoubleList();

      for (const auto& scales : scale_factors_list) {
        // only warn when the scales have floating values since
        // the result for ints is the same with/without recompute_scale_factor
        if (_is_floating_value(scales)) {
          warn_recompute_scale_factor = true;
          break;
        }
      }
    }

    if (warn_recompute_scale_factor) {
      TORCH_WARN(
          "The default behavior for interpolate/upsample with float scale_factor will change "
          "in 1.5.0 to align with other frameworks/libraries, and use scale_factor directly, "
          "instead of relying on the computed output size. "
          "If you wish to keep the old behavior, please set recompute_scale_factor=True. "
          "See the documentation of nn.Upsample for details.");
    }
  }

  if (recompute_scale_factor == false) {
    if (scale_factors.isDouble()) {
      scale_factors_1 = scale_factors.toDouble();
      scale_factors_2 = scale_factors.toDouble();
      scale_factors_3 = scale_factors.toDouble();
    } else if (scale_factors.isDoubleList()) {
      auto scale_factors_list = scale_factors.toDoubleList();
      scale_factors_1 = scale_factors_list[0];
      if (scale_factors_list.size() >= 2) {
        scale_factors_2 = scale_factors_list[1];
        if (scale_factors_list.size() >= 3) {
          scale_factors_3 = scale_factors_list[2];
        }
      }
    }
  }

  const auto dim1d = 3;
  const auto dim2d = 4;
  const auto dim3d = 5;

  auto input_dim = input.dim();
  if (input_dim == dim1d && mode == "nearest")
    return at::upsample_nearest1d(
        input,
        _output_size(input, 1, size, scale_factors),
        c10::make_optional(scale_factors_1));
  if (input_dim == dim2d && mode == "nearest")
    return at::upsample_nearest2d(
        input,
        _output_size(input, 2, size, scale_factors),
        scale_factors_1,
        scale_factors_2);
  if (input_dim == dim3d && mode == "nearest")
    return at::upsample_nearest3d(
        input,
        _output_size(input, 3, size, scale_factors),
        scale_factors_1,
        scale_factors_2,
        scale_factors_3);
  if (input_dim == dim1d && mode == "area")
    return at::adaptive_avg_pool1d(
        input, _output_size(input, 1, size, scale_factors));
  if (input_dim == dim2d && mode == "area")
    return at::adaptive_avg_pool2d(
        input, _output_size(input, 2, size, scale_factors));
  if (input_dim == dim3d && mode == "area")
    return at::adaptive_avg_pool3d(
        input, _output_size(input, 3, size, scale_factors));
  if (input_dim == dim1d && mode == "linear")
    return at::upsample_linear1d(
        input,
        _output_size(input, 1, size, scale_factors),
        *align_corners,
        c10::make_optional(scale_factors_1));
  if (input_dim == dim1d && mode == "bilinear")
    throw std::runtime_error("Got 3D input, but bilinear mode needs 4D input");
  if (input_dim == dim1d && mode == "bicubic")
    throw std::runtime_error("Got 3D input, but bicubic mode needs 4D input");
  if (input_dim == dim1d && mode == "trilinear")
    throw std::runtime_error("Got 3D input, but trilinear mode needs 5D input");
  if (input_dim == dim2d && mode == "linear")
    throw std::runtime_error("Got 4D input, but linear mode needs 3D input");
  if (input_dim == dim2d && mode == "bilinear")
    return at::upsample_bilinear2d(
        input,
        _output_size(input, 2, size, scale_factors),
        *align_corners,
        scale_factors_1,
        scale_factors_2);
  if (input_dim == dim2d && mode == "bicubic")
    return at::upsample_bicubic2d(
        input,
        _output_size(input, 2, size, scale_factors),
        *align_corners,
        scale_factors_1,
        scale_factors_2);
  if (input_dim == dim2d && mode == "trilinear")
    throw std::runtime_error("Got 4D input, but trilinear mode needs 5D input");
  if (input_dim == dim3d && mode == "linear")
    throw std::runtime_error("Got 5D input, but linear mode needs 3D input");
  if (input_dim == dim3d && mode == "bilinear")
    throw std::runtime_error("Got 5D input, but bilinear mode needs 4D input");
  if (input_dim == dim3d && mode == "bicubic")
    throw std::runtime_error("Got 5D input, but bicubic mode needs 4D input");
  if (input_dim == dim3d && mode == "trilinear")
    return at::upsample_trilinear3d(
        input,
        _output_size(input, 3, size, scale_factors),
        *align_corners,
        scale_factors_1,
        scale_factors_2,
        scale_factors_3);

  AT_ERROR(
      "Input Error: Only 3D, 4D and 5D input Tensors supported",
      " (got ",
      input_dim,
      "D) for the modes: nearest | linear | bilinear | trilinear",
      " (got ",
      mode,
      ") ");
}

void interpolate_op(Stack& stack) {
  at::Tensor input;
  IValue size;
  IValue scale_factors;
  std::string mode;
  IValue align_corners;
  IValue recompute_scale_factor;
  bool antialias = false;
  pop(stack,
      input,
      size,
      scale_factors,
      mode,
      align_corners,
      recompute_scale_factor,
      antialias);
  if (antialias) {
    throw std::runtime_error("Antialias is not yet supported");
  }
  at::Tensor res = interpolate(
      input,
      size,
      scale_factors,
      mode,
      align_corners.toOptional<bool>(),
      recompute_scale_factor.toOptional<bool>());
  push(stack, std::move(res));
}

// interpolate takes in float & float[] for scale factor
// upsample takes in int & int[], so convert the ints to floats before
// passing on to the interpolate op
IValue convert_scale_factor_to_double(const IValue& int_ivalue) {
  IValue scale_factor_double;
  if (int_ivalue.isInt()) {
    scale_factor_double = static_cast<double>(int_ivalue.toInt());
  } else if (int_ivalue.isIntList()) {
    auto int_list = int_ivalue.toDimVector();
    std::vector<double> double_vec(int_list.begin(), int_list.end());
    scale_factor_double = double_vec;
  } else if (int_ivalue.isNone()) {
    return IValue();
  } else {
    std::stringstream ss;
    ss << "Expecting optional int or int list arg for scale factor, got"
       << int_ivalue;
    throw std::runtime_error(ss.str());
  }
  return scale_factor_double;
}

void upsample_nearest_op(Stack& stack) {
  at::Tensor input;
  IValue size;
  IValue scale_factor_int;
  pop(stack, input, size, scale_factor_int);
  IValue scale_factor_double = convert_scale_factor_to_double(scale_factor_int);
  at::Tensor res = interpolate(
      input, size, scale_factor_double, "nearest", c10::nullopt, c10::nullopt);
  push(stack, std::move(res));
}

void upsample_op(Stack& stack) {
  at::Tensor input;
  IValue size;
  IValue scale_factor_int;
  std::string mode;
  IValue align_corners;
  pop(stack, input, size, scale_factor_int, mode, align_corners);
  IValue scale_factor_double = convert_scale_factor_to_double(scale_factor_int);
  at::Tensor res = interpolate(
      input,
      size,
      scale_factor_double,
      mode,
      align_corners.toOptional<bool>(),
      c10::nullopt);
  push(stack, std::move(res));
}

void upsample_bilinear_op(Stack& stack) {
  at::Tensor input;
  IValue size;
  IValue scale_factor_int;
  pop(stack, input, size, scale_factor_int);
  IValue scale_factor_double = convert_scale_factor_to_double(scale_factor_int);
  at::Tensor res = interpolate(
      input, size, scale_factor_double, "bilinear", true, c10::nullopt);
  push(stack, std::move(res));
}

// These ops are no longer generated, but remain here for BC
RegisterOperators reg3({
    Operator(
        "aten::__interpolate.scale_list(Tensor input, int? size = None, float[]? scale_factor = None, str mode = 'nearest', bool? align_corners = None, bool? recompute_scale_factor = None, bool antialias = False) -> Tensor",
        interpolate_op,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::__interpolate.size_list_scale_list(Tensor input, int[]? size = None, float[]? scale_factor = None, str mode = 'nearest', bool? align_corners = None, bool? recompute_scale_factor = None, bool antialias = False) -> Tensor",
        interpolate_op,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::__interpolate(Tensor input, int? size = None, float? scale_factor = None, str mode = 'nearest', bool? align_corners = None, bool? recompute_scale_factor = None, bool antialias = False) -> Tensor",
        interpolate_op,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::__interpolate.size_list(Tensor input, int[]? size = None, float? scale_factor = None, str mode = 'nearest', bool? align_corners = None, bool? recompute_scale_factor = None, bool antialias = False) -> Tensor",
        interpolate_op,
        aliasAnalysisFromSchema()),

    Operator(
        "aten::__upsample_nearest(Tensor input, int? size = None, int? scale_factor = None) -> Tensor",
        upsample_nearest_op,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::__upsample_nearest.size_list(Tensor input, int[]? size = None, int? scale_factor = None) -> Tensor",
        upsample_nearest_op,
        aliasAnalysisFromSchema()),

    Operator(
        "aten::__upsample(Tensor input, int? size = None, int? scale_factor = None, str mode = 'nearest', bool? align_corners = None) -> Tensor",
        upsample_op,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::__upsample.size_list(Tensor input, int[]? size = None, int? scale_factor = None, str mode = 'nearest', bool? align_corners = None) -> Tensor",
        upsample_op,
        aliasAnalysisFromSchema()),

    Operator(
        "aten::__upsample_bilinear(Tensor input, int? size = None, int? scale_factor = None) -> Tensor",
        upsample_bilinear_op,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::__upsample_bilinear.size_list(Tensor input, int[]? size = None, int? scale_factor = None) -> Tensor",
        upsample_bilinear_op,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::__upsample_bilinear.scale_list(Tensor input, int? size = None, int[]? scale_factor = None) -> Tensor",
        upsample_bilinear_op,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::__upsample_bilinear.size_list_scale_list(Tensor input, int[]? size = None, int[]? scale_factor = None) -> Tensor",
        upsample_bilinear_op,
        aliasAnalysisFromSchema()),

});

at::Tensor leaky_relu(const at::Tensor& tensor, double scalar) {
  return at::leaky_relu(tensor, scalar);
}
at::Tensor cat(const c10::List<at::Tensor>& tensors) {
  return at::cat(tensors.vec());
}

std::string get_first(const c10::List<c10::List<std::string>>& strings) {
  return strings.get(0).get(0);
}

static auto reg4 =
    torch::RegisterOperators()
        .op("_test::leaky_relu(Tensor self, float v=0.01) -> Tensor",
            &leaky_relu)
        .op("_test::cat(Tensor[] inputs) -> Tensor", &cat)
        .op("_test::get_first", &get_first);

} // namespace
} // namespace torch::jit
