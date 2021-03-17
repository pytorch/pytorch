#include <torch/csrc/jit/runtime/register_ops_utils.h>

#include <ATen/core/ivalue.h>
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

namespace torch {
namespace jit {

namespace {

RegisterOperators reg(
    {Operator(
         prim::profile,
         [](const Node* node) -> Operation {
           return [](Stack* stack) {
             AT_ERROR(
                 "Must be lowered to Interpreter's PROFILE instruction"); // NOLINT
           };
         },
         aliasAnalysisSpecialCase()),
     Operator(
         prim::profile_ivalue,
         [](const Node* node) -> Operation {
           return [](Stack* stack) {
             AT_ERROR(
                 "Must be lowered to Interpreter's PROFILE instruction"); // NOLINT
           };
         },
         aliasAnalysisSpecialCase()),
     Operator(
         prim::FusionGroup,
         [](const Node* node) -> Operation {
           const auto key = registerFusion(node);
           return [key](Stack* stack) {
             RECORD_FUNCTION("FusionGroup", std::vector<c10::IValue>());
             runFusion(key, *stack);
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
           return [rg_props](Stack* stack) {
             auto num_inputs = rg_props.size();
             // Check every input's shape against profiled (expected) shape.
             for (size_t i = 0; i < num_inputs; i++) {
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
         prim::TypeCheck /* (...)  -> (..., bool) */,
         [](const Node* /* node */) -> Operation {
           return [](Stack* /* stack */) {
             AT_ERROR("prim::TypeCheck not yet implemented"); // NOLINT
           };
         },
         aliasAnalysisSpecialCase()),
     Operator(
         prim::FallbackGraph,
         [](const Node* node) -> Operation {
           return [](Stack* stack) {
             AT_ERROR(
                 "Must be converted to prim::FunctionCall by replaceFallbackGraphWithFallbackFunction"); // NOLINT
           };
         },
         aliasAnalysisSpecialCase()),
     Operator(
         "prim::Guard(Tensor(a) t) -> Tensor(a)",
         [](Stack* stack) { AT_ERROR("Should be replaced by prim::BailOut"); },
         aliasAnalysisFromSchema()),
     Operator(
         "prim::BailOut(...) -> Tensor(a)",
         [](Stack* /* stack */) {
           AT_ERROR("prim::BailOut not yet implemented"); // NOLINT
         },
         aliasAnalysisFromSchema()),
     Operator(
         "prim::BailoutTemplate() -> int",
         [](Stack* stack) {
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
         [](Stack* stack) {
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
     // much more expensive to analayze the leaves and sometimes it might retain
     // the whole gradients in every tensor of the Autograd graph with
     // create_graph=True so we use aliasAnalysisConservative for these two OPs
     Operator(
         "aten::backward.TensorList(Tensor[] tensors, Tensor?[]? grad_tensors=None, bool? retain_graph=None, bool create_graph=False) -> ()",
         [](Stack* stack) {
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
         [](Stack* stack) {
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
         [](Stack* stack) {
           throw JITException(
               "This Python function is annotated to be ignored"
               " and cannot be and has not been included in the exported"
               " binary, meaning that it cannot be executed now."
               " Make sure that ignored operations are never executed after"
               " import");
         },
         aliasAnalysisFromSchema()),
     Operator(
         "prim::rangelist(int n) -> int[]",
         [](Stack* stack) {
           int64_t n;
           pop(stack, n);
           c10::List<int64_t> elems;
           elems.reserve(n);
           for (int i = 0; i < n; i++) {
             elems.push_back(i);
           }
           push(stack, std::move(elems));
         },
         aliasAnalysisFromSchema()),
     // note: this op needs to share a name with the Scalar -> Tensor conversion
     // because all _to_tensor conversion have to have the same operator namet
     Operator(
         "prim::NumToTensor.bool(bool a) -> Tensor",
         [](Stack* stack) {
           bool b;
           pop(stack, b);
           push(stack, at::scalar_to_tensor(b));
         },
         aliasAnalysisFromSchema()),
     Operator(
         "aten::device(str a) -> Device",
         [](Stack* stack) {
           push(stack, c10::Device(pop(stack).toStringRef()));
         },
         aliasAnalysisFromSchema()),
     Operator(
         "aten::percentFormat(str self, ...) -> str",
         [](Stack* stack) {
           size_t num_inputs = pop(stack).toInt();
           percentFormat(*stack, num_inputs);
         },
         aliasAnalysisFromSchema()),
     Operator(
         "aten::to.prim_other(Tensor(a) self, bool non_blocking=False, bool copy=False) -> Tensor(a|b)",
         [](Stack* stack) {
           at::Tensor self;
           bool non_blocking;
           bool copy;
           pop(stack, self, non_blocking, copy);
           c10::optional<c10::Device> device = c10::nullopt;
           c10::optional<at::ScalarType> scalarType = c10::nullopt;
           push(
               stack,
               to_dispatch(self, device, scalarType, non_blocking, copy));
         },
         aliasAnalysisFromSchema()),
     Operator(
         "prim::requires_grad(Tensor a) -> bool",
         [](Stack* stack) {
           at::Tensor a;
           pop(stack, a);
           push(stack, a.requires_grad());
         },
         aliasAnalysisFromSchema()),
     Operator(
         "prim::grad(Tensor a) -> Tensor(*)",
         [](Stack* stack) {
           at::Tensor a;
           pop(stack, a);
           push(stack, a.grad());
         },
         aliasAnalysisFromSchema()),
     Operator(
         "prim::is_sparse(Tensor a) -> bool",
         [](Stack* stack) {
           at::Tensor a;
           pop(stack, a);
           push(stack, a.is_sparse());
         },
         aliasAnalysisFromSchema()),
     Operator(
         "prim::is_mkldnn(Tensor a) -> bool",
         [](Stack* stack) {
           at::Tensor a;
           pop(stack, a);
           push(stack, a.is_mkldnn());
         },
         aliasAnalysisFromSchema()),
     Operator(
         "prim::is_mlc(Tensor a) -> bool",
         [](Stack* stack) {
           at::Tensor a;
           pop(stack, a);
           push(stack, a.is_mlc());
         },
         aliasAnalysisFromSchema()),
     Operator(
         "prim::is_vulkan(Tensor a) -> bool",
         [](Stack* stack) {
           at::Tensor a;
           pop(stack, a);
           push(stack, a.is_vulkan());
         },
         aliasAnalysisFromSchema()),
     Operator(
         "prim::is_quantized(Tensor a) -> bool",
         [](Stack* stack) {
           at::Tensor a;
           pop(stack, a);
           push(stack, a.is_quantized());
         },
         aliasAnalysisFromSchema()),
     Operator(
         "prim::is_meta(Tensor a) -> bool",
         [](Stack* stack) {
           at::Tensor a;
           pop(stack, a);
           push(stack, a.is_meta());
         },
         aliasAnalysisFromSchema()),
     Operator(
         "prim::name(Tensor a) -> str?",
         [](Stack* stack) {
           at::Tensor a;
           pop(stack, a);
           if (a.name() == "") {
             push(stack, IValue());
           } else {
             push(stack, a.name());
           }
         },
         aliasAnalysisFromSchema()),
     Operator(
         "prim::index(Device self) -> int?",
         [](Stack* stack) {
           auto d = pop(stack).toDevice();
           if (d.has_index()) {
             push(stack, d.index());
           } else {
             push(stack, IValue());
           }
         },
         aliasAnalysisFromSchema()),
     Operator(
         // TODO return generator object when torchscript supports RNG
         // first-class
         "aten::manual_seed(int seed) -> ()",
         [](Stack* stack) { at::manual_seed(pop(stack).toInt()); },
         aliasAnalysisFromSchema()),
     Operator(
         "aten::cuda(Tensor(a) self) -> Tensor(a|b)",
         [](Stack* stack) {
           at::Tensor a;
           pop(stack, a);
           push(stack, a.cuda());
         },
         aliasAnalysisFromSchema()),
     Operator(
         "prim::AutogradZero() -> Tensor",
         [](Stack* stack) { stack->emplace_back(at::Tensor()); },
         aliasAnalysisSpecialCase()),
     Operator(
         "prim::ReductionSizes(int[] size, int[] red_axes, bool keepdim = False) -> int[]",
         [](Stack* stack) {
           bool keepdim = pop(stack).toBool();
           c10::List<int64_t> axes = pop(stack).toIntList();
           c10::List<int64_t> size = pop(stack).toIntList();
           if (keepdim) {
             for (const auto& axis : axes) {
               size.set(axis, 1);
             }
           } else {
             int64_t index = 0;
             auto iter = size.begin();
             std::sort(axes.begin(), axes.end());
             for (const auto& axis : axes) {
               // move iter to the next axis
               iter += axis - index;

               // input iter points to axis and is updated to axis + 1
               iter = size.erase(iter);

               // update current index for iter
               index = axis + 1;
             }
           }
           push(stack, IValue(std::move(size)));
         },
         aliasAnalysisFromSchema()),
     Operator(
         "prim::BroadcastSizes(...) -> int[]",
         [](Stack* stack) {
           auto num_inputs = pop(stack).toInt();
           std::vector<int64_t> size;
           size.reserve(8);
           for (auto i = 0; i < num_inputs; ++i) {
             size =
                 at::infer_size(size, peek(stack, i, num_inputs).toIntVector());
           }
           drop(stack, num_inputs);
           push(stack, IValue(size));
         },
         aliasAnalysisSpecialCase()),
     Operator(
         prim::ChunkSizes,
         [](const Node* node) -> Operation {
           int64_t raw_dim = node->i(attr::dim);
           int64_t chunks = node->i(attr::chunks);
           return [raw_dim, chunks](Stack* stack) {
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
         "aten::warn(str message, int stacklevel=2) -> ()",
         [](Stack* stack) {
           TORCH_CHECK(
               false, "warn is implemented directly in the interpreter");
         },
         aliasAnalysisFromSchema()),

     Operator(
         "onnx::Reshape(Tensor input, Tensor shape) -> Tensor",
         [](Stack* stack) {
           at::Tensor input, shape;
           pop(stack, input, shape);
           shape = shape.contiguous();
           AT_ASSERT(shape.ndimension() == 1);
           at::IntArrayRef shape_list(shape.data_ptr<int64_t>(), shape.size(0));
           push(stack, input.reshape(shape_list));
         },
         aliasAnalysisSpecialCase()),
     Operator(
         "onnx::Shape(Tensor t) -> Tensor",
         [](Stack* stack) {
           auto t = pop(stack).toTensor();
           at::IntArrayRef sizes = t.sizes();
           auto sizes_tensor = torch::empty(
               {static_cast<int64_t>(sizes.size())}, at::dtype(at::kLong));
           auto accessor = sizes_tensor.accessor<int64_t, 1>();
           for (size_t i = 0; i < sizes.size(); ++i) {
             accessor[i] = sizes[i];
           }
           stack->emplace_back(sizes_tensor);
         },
         aliasAnalysisSpecialCase()),
     Operator(
         "prim::AutogradAnyNonZero(...) -> bool",
         [](Stack* stack) {
           auto num_inputs = pop(stack).toInt();
           bool result = false;
           for (const IValue& v : last(stack, num_inputs)) {
             if (v.isTensor()) {
               if (v.toTensor().defined()) {
                 result = true;
                 break;
               }
             } else if (v.isTensorList()) {
               for (const at::Tensor& t : v.toTensorVector()) {
                 if (t.defined()) {
                   result = true;
                 }
               }
               if (result) {
                 break;
               }
             } else {
               TORCH_INTERNAL_ASSERT(false);
             }
           }
           drop(stack, num_inputs);
           stack->emplace_back(result);
         },
         aliasAnalysisFromSchema()),
     Operator(
         "prim::AutogradAllZero(...) -> bool",
         [](Stack* stack) {
           auto num_inputs = pop(stack).toInt();
           bool result = true;
           for (const IValue& v : last(stack, num_inputs)) {
             TORCH_INTERNAL_ASSERT(v.isTensor());
             if (v.toTensor().defined()) {
               result = false;
               break;
             }
           }
           drop(stack, num_inputs);
           stack->emplace_back(result);
         },
         aliasAnalysisFromSchema()),
     Operator(
         "prim::AutogradAllNonZero(...) -> bool",
         [](Stack* stack) {
           auto num_inputs = pop(stack).toInt();
           bool result = true;
           for (const IValue& v : last(stack, num_inputs)) {
             TORCH_INTERNAL_ASSERT(v.isTensor());
             if (!v.toTensor().defined()) {
               result = false;
               break;
             }
           }
           drop(stack, num_inputs);
           stack->emplace_back(result);
         },
         aliasAnalysisFromSchema()),
     Operator(
         "prim::AutogradAdd(Any a, Any b) -> Any",
         [](Stack* stack) {
           at::Tensor a, b;
           pop(stack, a, b);
           if (!a.defined() && !b.defined()) {
             // undef + undef == undef
             stack->emplace_back(a);
           } else if (!a.defined()) {
             stack->emplace_back(b);
           } else if (!b.defined()) {
             stack->emplace_back(a);
           } else {
             stack->emplace_back(a + b);
           }
         },
         aliasAnalysisSpecialCase()),
     Operator(
         "aten::_grad_sum_to_size(Tensor(a) self, int[]? size) -> Tensor(a)",
         [](Stack* stack) {
           RECORD_FUNCTION("_grad_sum_to_size", std::vector<c10::IValue>());
           IValue self, size;
           pop(stack, self, size);
           if (size.isNone()) {
             push(stack, std::move(self));
           } else {
             push(stack, at::sum_to(self.toTensor(), size.toIntVector()));
           }
         },
         aliasAnalysisFromSchema()),
     Operator(
         "aten::_size_if_not_equal(int[] self_size, int[] other_size) -> int[]?",
         [](Stack* stack) {
           IValue self_size, other_size;
           pop(stack, self_size, other_size);
           auto s = self_size.toIntVector();
           auto o = other_size.toIntVector();
           if (s == o) {
             push(stack, IValue());
           } else {
             push(stack, s);
           }
         },
         aliasAnalysisFromSchema()),
     Operator(
         prim::ConstantChunk,
         [](const Node* node) -> Operation {
           int64_t chunks = node->i(attr::chunks);
           int64_t dim = node->i(attr::dim);
           auto outputs_used = fmap(node->outputs(), [](const Value* v) {
             return v->uses().size() > 0;
           });
           return [=](Stack* stack) {
             RECORD_FUNCTION("chunk", last(stack, 1));

             at::Tensor t;
             pop(stack, t);
             auto result = at::chunk(t, chunks, dim);
             stack->insert(
                 stack->end(),
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
               for (int64_t i = num_results; i < chunks; ++i) {
                 TORCH_CHECK(
                     !outputs_used[i],
                     "Expected chunk to return at least ",
                     chunks,
                     " outputs, but got only ",
                     num_results);
                 // We know that the output is unused, so it's ok to push
                 // anything on the stack.
                 stack->emplace_back();
               }
             }
           };
         },
         aliasAnalysisSpecialCase()),
     // This operator is generated inside the compiler for indexing into
     // ModuleList without a statically determinable key. Accordingly,
     // self must be a ModuleType and the output must be an InterfaceType.
     OperatorGenerator(
         TORCH_SELECTIVE_SCHEMA(
             "prim::ModuleContainerIndex.list(Any self, int ind) -> Any"),
         [](Stack* stack) {
           IValue ind = pop(stack);
           IValue module_dict = pop(stack);
           std::stringstream ss;
           ss << ind.toInt();
           push(stack, module_dict.toModule().attr(ss.str()));
         },
         aliasAnalysisFromSchema()),
     // This operator is generated inside the compiler for indexing into
     // ModuleDict without a statically determinable key. Accordingly,
     // self must be a ModuleType and the output must be an InterfaceType.
     OperatorGenerator(
         TORCH_SELECTIVE_SCHEMA(
             "prim::ModuleContainerIndex.dict(Any self, str ind) -> Any"),
         [](Stack* stack) {
           IValue ind = pop(stack);
           IValue module_dict = pop(stack);
           push(stack, module_dict.toModule().attr(ind.toStringRef()));
         },
         aliasAnalysisFromSchema()),
     Operator(
         "aten::_unwrap_optional(t(a)? optional) -> t(a)",
         [](Stack* stack) {
           auto val = pop(stack);
           TORCH_CHECK(!val.isNone(), "Unwrapping null optional");
           push(stack, std::move(val));
         },
         aliasAnalysisFromSchema()),
     Operator(
         "aten::wait(Future(t) self) -> t",
         [](Stack* stack) {
           TORCH_CHECK(
               false, "wait is implemented directly in the interpreter");
         },
         aliasAnalysisSpecialCase())});

RegisterOperators logging_operators(
    {Operator(
         "prim::AddStatValue(str key, int val) -> ()",
         [](Stack* stack) {
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
         [](Stack* stack) {
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
           auto output = autograd::profiler::getTime();
           push(stack, output);
           if (jit::tracer::isTracing()) {
             jit::tracer::addOutput(node, output);
           }
         },
         aliasAnalysisFromSchema())});

void hashValue(Stack* stack) {
  auto value = pop(stack);
  push(stack, value.hash());
}

RegisterOperators reg2({
    // registered as Any[] so that heterogenous tuples can be called with len()
    Operator(
        "aten::len.any(Any[] a) -> int",
        listLen,
        aliasAnalysisFromSchema()),

// these ops have a specialized implementation for the list element type
#define CREATE_SPECIALIZED_LIST_OPS(decl_type, value_type) \
  Operator(                                                \
      "aten::remove." decl_type "(" decl_type              \
      "[](a!) self,                                                           \
        " decl_type " el) -> ()",                          \
      listRemove<value_type>,                              \
      aliasAnalysisFromSchema()),                          \
      Operator(                                            \
          "aten::index.list_" decl_type "(" decl_type      \
          "[] self,                                                               \
        " decl_type " el) -> int",                         \
          listIndex<value_type>,                           \
          aliasAnalysisFromSchema()),                      \
      Operator(                                            \
          "aten::count." decl_type "(" decl_type           \
          "[] self,                                                               \
        " decl_type " el) -> int",                         \
          listCount<value_type>,                           \
          aliasAnalysisFromSchema()),

    CREATE_SPECIALIZED_LIST_OPS("int", int64_t)
        CREATE_SPECIALIZED_LIST_OPS("float", double)
            CREATE_SPECIALIZED_LIST_OPS("bool", bool)
                CREATE_SPECIALIZED_LIST_OPS("Tensor", at::Tensor)
                    CREATE_SPECIALIZED_LIST_OPS("str", std::string)

#undef CREATE_GENERIC_LIST_OPS
#undef CREATE_SPECIALIZED_LIST_OPS

    // `listContains<T>` is not implemented for non-primitive types
    // TODO: Add List[bool] once .to<c10::List<bool>> doesn't throw an error
    Operator(
        "aten::__contains__.float_list(float[] l, float item) -> bool",
        listContains<double>,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::sort.int(int[](a!) self, bool reverse=False) -> ()",
        listSort<int64_t>,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::sort.float(float[](a!) self, bool reverse=False) -> ()",
        listSort<double>,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::sort.Tensor(Tensor[](a!) self, bool reverse=False) -> ()",
        listSort<at::Tensor>,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::sort.bool(bool[](a!) self, bool reverse=False) -> ()",
        listSort<bool>,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::sort.str(str[](a!) self, bool reverse=False) -> ()",
        listSort<std::string>,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::sorted.int(int[](a) input) -> (int[])",
        listCopyAndSort<int64_t>,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::sorted.float(float[](a) input) -> (float[])",
        listCopyAndSort<double>,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::sorted.Tensor(Tensor[](a) input) -> (Tensor[])",
        listCopyAndSort<at::Tensor>,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::sorted.bool(bool[](a) input) -> (bool[])",
        listCopyAndSort<bool>,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::sorted.str(str[](a) input) -> (str[])",
        listCopyAndSort<std::string>,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::eq.float_list(float[] a, float[] b) -> bool",
        listEq<double>,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::eq.Tensor_list(Tensor[] a, Tensor[] b) -> bool",
        listEq<at::Tensor>,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::eq.bool_list(bool[] a, bool[] b) -> bool",
        listEq<bool>,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::eq.str_list(str[] a, str[] b) -> bool",
        listEq<std::string>,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::ne.float_list(float[] a, float[] b) -> bool",
        listNe<double>,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::ne.Tensor_list(Tensor[] a, Tensor[] b) -> bool",
        listNe<at::Tensor>,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::ne.bool_list(bool[] a, bool[] b) -> bool",
        listNe<bool>,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::ne.str_list(str[] a, str[] b) -> bool",
        listNe<std::string>,
        aliasAnalysisFromSchema()),

#define DEFINE_CONVERT_BASE_OP(op_name, prefix, char_op) \
  Operator(                                              \
      #op_name "(int i) -> str",                         \
      [](Stack* stack) {                                 \
        auto i = pop(stack).toInt();                     \
        std::stringstream ss;                            \
        if (i < 0) {                                     \
          ss << "-";                                     \
          i = -i;                                        \
        }                                                \
        ss << "0" << prefix << char_op << i;             \
        push(stack, ss.str());                           \
      },                                                 \
      aliasAnalysisFromSchema())

    DEFINE_CONVERT_BASE_OP(aten::hex, "x", std::hex),
    DEFINE_CONVERT_BASE_OP(aten::oct, "o", std::oct),

    Operator(
        "aten::bin(int i) -> str",
        [](Stack* stack) {
          auto i = pop(stack).toInt();
          std::stringstream ss;
          if (i == 0) {
            push(stack, "0b0");
          } else {
            if (i < 0) {
              ss << "-";
              i = -i;
            }
            std::string str = std::bitset<8 * sizeof(i)>(i).to_string();
            str.erase(0, std::min(str.find_first_not_of('0'), str.size() - 1));
            ss << "0b" << str;
            push(stack, ss.str());
          }
        },
        aliasAnalysisFromSchema()),
    // TODO: deprecate this in favor of aten::getelem
    Operator(
        "prim::StringIndex(str string, int index) -> str",
        [](Stack* stack) {
          auto index = pop(stack).toInt();
          auto string = pop(stack).toStringRef();
          auto norm_index = normalizeIndex(index, string.size());
          char c = string.at(norm_index);
          push(stack, std::string(&c, 1));
        },
        aliasAnalysisFromSchema()),
    Operator(
        "aten::chr(int i) -> str",
        [](Stack* stack) {
          auto i = pop(stack).toInt();
          std::stringstream ss;
          TORCH_CHECK(
              i >= 0 && i < 1114111,
              "chr() arg not in range(0x110000), found ",
              i);
          char c = i;
          ss << c;
          push(stack, ss.str());
        },
        aliasAnalysisFromSchema()),

    // only used in loop unrolling, not exposed to end users
    DEFINE_INT_OP(aten::__round_to_zero_floordiv, a / b),

    Operator(
        "aten::modf(float a) -> (float, float)",
        [](Stack* stack) {
          double a;
          pop(stack, a);
          double b, c;
          b = modf(a, &c);
          push(stack, b, c);
        },
        aliasAnalysisFromSchema()),
    Operator(
        "aten::frexp(float a) -> (float, int)",
        [](Stack* stack) {
          double a;
          pop(stack, a);
          double m;
          int e;
          m = std::frexp(a, &e);
          push(stack, m, e);
        },
        aliasAnalysisFromSchema()),
    Operator(
        "aten::ldexp(float x, int i) -> float",
        [](Stack* stack) {
          double a;
          int64_t b;
          pop(stack, a, b);
          push(stack, std::ldexp(a, b));
        },
        aliasAnalysisFromSchema()),
    DEFINE_BINARY_FLOAT_OP(aten::mathremainder, std::remainder(a, b)),

    DEFINE_INT_OP(aten::__and__, a& b),
    DEFINE_INT_OP(aten::__or__, a | b),
    DEFINE_INT_OP(aten::__xor__, a ^ b),
    DEFINE_INT_OP(aten::__lshift__, a << b),
    DEFINE_INT_OP(aten::__rshift__, a >> b),

    DEFINE_GENERIC_BINARY_OP(aten::log, std::log(a) / std::log(b), float),
    DEFINE_INT_FLOAT_OP(aten::log, std::log(a) / std::log(b), float),
    DEFINE_SCALAR_SCALAR_BINARY_OP(
        aten::log,
        std::log(a) / std::log(b),
        std::log(a) / std::log(b),
        float),
    DEFINE_UNARY_OP(aten::log1p, std::log1p(a), float, float),
    DEFINE_UNARY_OP(aten::log10, std::log10(a), float, float),
    DEFINE_UNARY_OP(aten::sqrt, std::sqrt(a), float, float),
    DEFINE_UNARY_OP(aten::acos, std::acos(a), float, float),
    DEFINE_UNARY_OP(aten::asin, std::asin(a), float, float),
    DEFINE_UNARY_OP(aten::atan, std::atan(a), float, float),
    DEFINE_GENERIC_OP(
        aten::atan2,
        std::atan2(a, b),
        std::atan2(a, b),
        float,
        float),
    DEFINE_INT_FLOAT_OP(aten::atan2, std::atan2(a, b), float),
    DEFINE_SCALAR_SCALAR_BINARY_OP(
        aten::atan2,
        std::atan2(a, b),
        std::atan2(a, b),
        float),
    DEFINE_UNARY_OP(aten::cos, std::cos(a), float, float),
    DEFINE_UNARY_OP(aten::sin, std::sin(a), float, float),
    DEFINE_UNARY_OP(aten::tan, std::tan(a), float, float),
    DEFINE_UNARY_OP(aten::asinh, std::asinh(a), float, float),
    DEFINE_UNARY_OP(aten::atanh, std::atanh(a), float, float),
    DEFINE_UNARY_OP(aten::acosh, std::acosh(a), float, float),
    DEFINE_UNARY_OP(aten::sinh, std::sinh(a), float, float),
    DEFINE_UNARY_OP(aten::cosh, std::cosh(a), float, float),
    DEFINE_UNARY_OP(aten::tanh, std::tanh(a), float, float),
    DEFINE_UNARY_OP(aten::degrees, degrees(a), float, float),
    DEFINE_UNARY_OP(aten::radians, radians(a), float, float),
    DEFINE_BINARY_FLOAT_OP(aten::fmod, std::fmod(a, b)),
    DEFINE_UNARY_INT_OP(aten::factorial, factorial(a), int),
    DEFINE_UNARY_FLOAT_OP(aten::isnan, std::isnan(a), bool),
    DEFINE_UNARY_FLOAT_OP(aten::isfinite, std::isfinite(a), bool),
    DEFINE_UNARY_FLOAT_OP(aten::isinf, std::isinf(a), bool),
    DEFINE_UNARY_OP(aten::gamma, std::tgamma(a), float, float),
    DEFINE_UNARY_OP(aten::erf, std::erf(a), float, float),
    DEFINE_UNARY_OP(aten::erfc, std::erfc(a), float, float),
    DEFINE_UNARY_OP(aten::expm1, std::expm1(a), float, float),
    DEFINE_UNARY_OP(aten::fabs, std::fabs(a), float, float),
    DEFINE_UNARY_OP(aten::lgamma, std::lgamma(a), float, float),

    // TODO: move abs to aten namespace because it's schematized!
    DEFINE_UNARY_OP(prim::abs, std::abs(a), int, float),
    Operator(
        "prim::abs(Tensor x) -> Tensor",
        [](Stack* stack) {
          at::Tensor x;
          pop(stack, x);
          push(stack, x.abs());
        },
        aliasAnalysisFromSchema()),

    DEFINE_INT_OP(aten::gcd, gcd(a, b)),

    DEFINE_GENERIC_OP(
        aten::copysign,
        std::copysign(a, b),
        std::copysign(a, b),
        float,
        float),
    DEFINE_INT_FLOAT_OP(aten::copysign, std::copysign(a, b), float),
    DEFINE_SCALAR_BINARY_OP(
        aten::copysign,
        std::copysign(a, b),
        std::copysign(a, b),
        float),
    Operator(
        "aten::_tensor_to_list(Tensor self) -> int[]",
        [](Stack* stack) {
          at::Tensor t;
          pop(stack, t);
          c10::List<int64_t> elems;
          elems.reserve(t.size(0));
          for (int i = 0; i < t.size(0); i++) {
            elems.push_back(*t[i].data_ptr<int32_t>());
          }
          push(stack, std::move(elems));
        },
        aliasAnalysisFromSchema()),
    Operator(
        "aten::_list_to_tensor(int[] self) -> Tensor",
        [](Stack* stack) {
          c10::List<int64_t> l = pop(stack).toIntList();
          auto t = torch::empty(
              {static_cast<int64_t>(l.size())}, at::dtype(at::kInt));
          for (size_t i = 0; i < l.size(); i++) {
            t[i] = l.get(i);
          }
          push(stack, std::move(t));
        },
        aliasAnalysisFromSchema()),
    Operator(
        "aten::sum.int(int[] self) -> int",
        [](Stack* stack) {
          c10::List<int64_t> l = pop(stack).toIntList();
          auto sum = 0;
          for (const auto& elem : l) {
            sum += elem;
          }
          push(stack, sum);
        },
        aliasAnalysisFromSchema()),
    Operator(
        "aten::sum.float(float[] self) -> float",
        [](Stack* stack) {
          c10::List<double> l = pop(stack).toDoubleList();
          auto sum = 0.0;
          for (const auto& elem : l) {
            sum += elem;
          }
          push(stack, sum);
        },
        aliasAnalysisFromSchema()),
    Operator(
        "aten::sum.bool(bool[] self) -> int",
        [](Stack* stack) {
          c10::List<bool> l = pop(stack).toBoolList();
          auto sum = 0;
          for (const auto& elem : l) {
            if (elem) {
              sum += 1;
            }
          }
          push(stack, sum);
        },
        aliasAnalysisFromSchema()),
    Operator(
        "aten::any.str(str[] self) -> bool",
        [](Stack* stack) {
          auto l = pop(stack).toList();
          for (const auto& elem : l) {
            if (elem != "") {
              push(stack, true);
              return;
            }
          }
          push(stack, false);
        },
        aliasAnalysisFromSchema()),
    Operator(
        "aten::any.int(int[] self) -> bool",
        [](Stack* stack) {
          c10::List<int64_t> l = pop(stack).toIntList();
          for (const auto& elem : l) {
            if (elem) {
              push(stack, true);
              return;
            }
          }
          push(stack, false);
        },
        aliasAnalysisFromSchema()),
    Operator(
        "aten::any.float(float[] self) -> bool",
        [](Stack* stack) {
          c10::List<double> l = pop(stack).toDoubleList();
          for (const auto& elem : l) {
            if (elem) {
              push(stack, true);
              return;
            }
          }
          push(stack, false);
        },
        aliasAnalysisFromSchema()),
    Operator(
        "aten::any.bool(bool[] self) -> bool",
        [](Stack* stack) {
          c10::List<bool> l = pop(stack).toBoolList();
          for (const auto& elem : l) {
            if (elem) {
              push(stack, true);
              return;
            }
          }
          push(stack, false);
        },
        aliasAnalysisFromSchema()),
    Operator(
        "aten::all.int(int[] self) -> bool",
        [](Stack* stack) {
          c10::List<int64_t> l = pop(stack).toIntList();
          for (const auto& elem : l) {
            if (!elem) {
              push(stack, false);
              return;
            }
          }
          push(stack, true);
        },
        aliasAnalysisFromSchema()),
    Operator(
        "aten::all.float(float[] self) -> bool",
        [](Stack* stack) {
          c10::List<double> l = pop(stack).toDoubleList();
          for (const auto& elem : l) {
            if (!elem) {
              push(stack, false);
              return;
            }
          }
          push(stack, true);
        },
        aliasAnalysisFromSchema()),
    Operator(
        "aten::all.bool(bool[] self) -> bool",
        [](Stack* stack) {
          c10::List<bool> l = pop(stack).toBoolList();
          for (const auto& elem : l) {
            if (!elem) {
              push(stack, false);
              return;
            }
          }
          push(stack, true);
        },
        aliasAnalysisFromSchema()),
    Operator(
        "aten::divmod.int(int x, int y) -> (int, int)",
        [](Stack* stack) {
          int64_t a, b;
          lldiv_t divresult = {};
          pop(stack, a, b);
          if (b == 0) {
            throw std::runtime_error(
                "ZeroDivisionError: integer division or modulo by zero");
          }
          divresult = lldiv(a, b);
          if (divresult.rem && (a < 0) != (b < 0)) {
            divresult.quot -= 1;
            divresult.rem += b;
          }
          push(
              stack,
              static_cast<int64_t>(divresult.quot),
              static_cast<int64_t>(divresult.rem));
        },
        aliasAnalysisFromSchema()),
    Operator(
        "aten::divmod.float(float x, float y) -> (float, float)",
        [](Stack* stack) {
          double a, b;
          pop(stack, a, b);
          if (b == 0) {
            throw std::runtime_error("ZeroDivisionError: float divmod()");
          }
          double rem = fmod(a, b);
          if (rem && (a < 0) != (b < 0)) {
            rem += b;
          }
          push(stack, (a - rem) / b, rem);
        },
        aliasAnalysisFromSchema()),
    Operator(
        "prim::id(AnyClassType? x) -> int",
        [](Stack* stack) {
          IValue a;
          pop(stack, a);
          if (a.isNone()) {
            push(stack, 0);
          } else {
            push(stack, reinterpret_cast<int64_t>(a.internalToPointer()));
          }
        },
        aliasAnalysisFromSchema()),

#define DEFINE_DIVMOD_MIXED_OP(type_a, type_b)                           \
  Operator(                                                              \
      "aten::divmod." #type_a "_" #type_b "(" #type_a " x," #type_b      \
      " y) -> (float, float)",                                           \
      [](Stack* stack) {                                                 \
        type_a a;                                                        \
        type_b b;                                                        \
        pop(stack, a, b);                                                \
        if (b == 0) {                                                    \
          throw std::runtime_error("ZeroDivisionError: float divmod()"); \
        }                                                                \
        double quot = floor(a / b);                                      \
        double rem = a - (quot * b);                                     \
        push(stack, quot, rem);                                          \
      },                                                                 \
      aliasAnalysisFromSchema())

    DEFINE_DIVMOD_MIXED_OP(int, float),
    DEFINE_DIVMOD_MIXED_OP(float, int),

#undef DEFINE_DIVMOD_MIXED_OP
    Operator(
        "aten::hash.generic(t value) -> int",
        hashValue,
        aliasAnalysisFromSchema()),

#define DEFINE_COMPLEX_OP(type_a, type_b, actual_type_a, actual_type_b) \
  Operator(                                                             \
      "aten::complex." #type_a "_" #type_b "(" #type_a " x," #type_b    \
      " y) -> complex",                                                 \
      [](Stack* stack) {                                                \
        actual_type_a a;                                                \
        actual_type_b b;                                                \
        pop(stack, a, b);                                               \
        auto comp = c10::complex<double>(a, b);                         \
        push(stack, comp);                                              \
      },                                                                \
      aliasAnalysisFromSchema())

    DEFINE_COMPLEX_OP(int, bool, int, bool),
    DEFINE_COMPLEX_OP(bool, int, bool, int),
    DEFINE_COMPLEX_OP(float, bool, double, bool),
    DEFINE_COMPLEX_OP(bool, float, bool, double),
    DEFINE_COMPLEX_OP(float, int, double, int),
    DEFINE_COMPLEX_OP(int, float, int, double),
    DEFINE_COMPLEX_OP(int, int, int, int),
    DEFINE_COMPLEX_OP(bool, bool, bool, bool),
    DEFINE_COMPLEX_OP(float, float, double, double),
});

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
  for (size_t i = 0; i < n; ++i) {
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
void sort_op(Stack* stack) {
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
  for (size_t i = 0; i < dim; ++i) {
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

void interpolate_op(Stack* stack) {
  at::Tensor input;
  IValue size;
  IValue scale_factors;
  std::string mode;
  IValue align_corners;
  IValue recompute_scale_factor;
  pop(stack,
      input,
      size,
      scale_factors,
      mode,
      align_corners,
      recompute_scale_factor);
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
    auto int_list = int_ivalue.toIntVector();
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

void upsample_nearest_op(Stack* stack) {
  at::Tensor input;
  IValue size;
  IValue scale_factor_int;
  pop(stack, input, size, scale_factor_int);
  IValue scale_factor_double = convert_scale_factor_to_double(scale_factor_int);
  at::Tensor res = interpolate(
      input, size, scale_factor_double, "nearest", c10::nullopt, c10::nullopt);
  push(stack, std::move(res));
}

void upsample_op(Stack* stack) {
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

void upsample_bilinear_op(Stack* stack) {
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
        "aten::__interpolate.scale_list(Tensor input, int? size = None, float[]? scale_factor = None, str mode = 'nearest', bool? align_corners = None, bool? recompute_scale_factor = None) -> Tensor",
        interpolate_op,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::__interpolate.size_list_scale_list(Tensor input, int[]? size = None, float[]? scale_factor = None, str mode = 'nearest', bool? align_corners = None, bool? recompute_scale_factor = None) -> Tensor",
        interpolate_op,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::__interpolate(Tensor input, int? size = None, float? scale_factor = None, str mode = 'nearest', bool? align_corners = None, bool? recompute_scale_factor = None) -> Tensor",
        interpolate_op,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::__interpolate.size_list(Tensor input, int[]? size = None, float? scale_factor = None, str mode = 'nearest', bool? align_corners = None, bool? recompute_scale_factor = None) -> Tensor",
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
} // namespace jit
} // namespace torch
