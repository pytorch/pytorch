#include <torch/csrc/jit/runtime/register_ops_utils.h>

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
           auto callback = node->cast<ProfileOp>()->getCallback();
           return [](Stack& stack) {
             AT_ERROR(
                 "Must be lowered to Interpreter's PROFILE instruction"); // NOLINT
             return 0;
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
             return 0;
           };
         },
         aliasAnalysisSpecialCase()),
     Operator(
         "prim::Guard(Tensor(a) t) -> Tensor(a)",
         [](Stack& stack) {
           AT_ERROR("Should be replaced by prim::BailOut");
           return 0;
         },
         aliasAnalysisFromSchema()),
     Operator(
         "prim::BailOut(...) -> Tensor(a)",
         [](Stack& /* stack */) {
           AT_ERROR("prim::BailOut not yet implemented"); // NOLINT
           return 0;
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
           return 0;
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
           return 0;
         },
         aliasAnalysisFromSchema()),
     // NB: backward op might write to every input tensors in the graph and it's
     // much more expensive to analayze the leaves and sometimes it might retain
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
           return 0;
         },
         aliasAnalysisConservative()),
     Operator(
         "aten::backward(Tensor self, Tensor? gradient=None, bool? retain_graph=None, bool create_graph=False) -> ()",
         [](Stack& stack) {
           bool create_graph = pop(stack).toBool();
           auto retain_graph = pop(stack).toOptional<bool>();
           IValue gradient_ivalue = pop(stack);
           at::Tensor gradient = gradient_ivalue.isNone()
               ? at::Tensor()
               : gradient_ivalue.toTensor();
           at::Tensor self = pop(stack).toTensor();
           bool keep_graph = retain_graph ? retain_graph.value() : create_graph;
           self.backward(gradient, keep_graph, create_graph);
           return 0;
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
           return 0;
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
           return 0;
         },
         aliasAnalysisFromSchema()),
     Operator(
         "prim::rangelist(int n) -> int[]",
         [](Stack& stack) {
           int64_t n;
           pop(stack, n);
           c10::List<int64_t> elems;
           elems.reserve(n);
           for (int i = 0; i < n; i++) {
             elems.push_back(i);
           }
           push(stack, std::move(elems));
           return 0;
         },
         aliasAnalysisFromSchema()),
     // note: this op needs to share a name with the Scalar -> Tensor conversion
     // because all _to_tensor conversion have to have the same operator namet
     Operator(
         "prim::NumToTensor.bool(bool a) -> Tensor",
         [](Stack& stack) {
           bool b;
           pop(stack, b);
           push(stack, at::scalar_to_tensor(b));
           return 0;
         },
         aliasAnalysisFromSchema()),
     Operator(
         "aten::str(t elem) -> str",
         [](Stack& stack) {
           std::stringstream ss;
           ss << pop(stack);
           push(stack, ss.str());
           return 0;
         },
         aliasAnalysisFromSchema()),
     Operator(
         "aten::device(str a) -> Device",
         [](Stack& stack) {
           push(stack, c10::Device(pop(stack).toStringRef()));
           return 0;
         },
         aliasAnalysisFromSchema()),
     Operator(
         "aten::to.prim_other(Tensor(a) self, bool non_blocking=False, bool copy=False) -> Tensor(a|b)",
         [](Stack& stack) {
           at::Tensor self;
           bool non_blocking;
           bool copy;
           pop(stack, self, non_blocking, copy);
           c10::optional<c10::Device> device = c10::nullopt;
           c10::optional<at::ScalarType> scalarType = c10::nullopt;
           push(
               stack,
               to_dispatch(self, device, scalarType, non_blocking, copy));
           return 0;
         },
         aliasAnalysisFromSchema()),
     Operator(
         "aten::eq(Device a, Device b) -> bool",
         [](Stack& stack) {
           auto a = pop(stack).toDevice();
           auto b = pop(stack).toDevice();
           push(stack, a == b);
           return 0;
         },
         aliasAnalysisFromSchema()),
     Operator(
         "prim::requires_grad(Tensor a) -> bool",
         [](Stack& stack) {
           at::Tensor a;
           pop(stack, a);
           push(stack, a.requires_grad());
           return 0;
         },
         aliasAnalysisFromSchema()),
     Operator(
         "prim::grad(Tensor a) -> Tensor(*)",
         [](Stack& stack) {
           at::Tensor a;
           pop(stack, a);
           push(stack, a.grad());
           return 0;
         },
         aliasAnalysisFromSchema()),
     Operator(
         "prim::is_sparse(Tensor a) -> bool",
         [](Stack& stack) {
           at::Tensor a;
           pop(stack, a);
           push(stack, a.is_sparse());
           return 0;
         },
         aliasAnalysisFromSchema()),
     Operator(
         "prim::is_mkldnn(Tensor a) -> bool",
         [](Stack& stack) {
           at::Tensor a;
           pop(stack, a);
           push(stack, a.is_mkldnn());
           return 0;
         },
         aliasAnalysisFromSchema()),
     Operator(
         "prim::is_quantized(Tensor a) -> bool",
         [](Stack& stack) {
           at::Tensor a;
           pop(stack, a);
           push(stack, a.is_quantized());
           return 0;
         },
         aliasAnalysisFromSchema()),
     Operator(
         "prim::is_meta(Tensor a) -> bool",
         [](Stack& stack) {
           at::Tensor a;
           pop(stack, a);
           push(stack, a.is_meta());
           return 0;
         },
         aliasAnalysisFromSchema()),
     Operator(
         "prim::name(Tensor a) -> str?",
         [](Stack& stack) {
           at::Tensor a;
           pop(stack, a);
           if (a.name() == "") {
             push(stack, IValue());
           } else {
             push(stack, a.name());
           }
           return 0;
         },
         aliasAnalysisFromSchema()),
     Operator(
         "prim::layout(Tensor a) -> int",
         [](Stack& stack) {
           at::Tensor a;
           pop(stack, a);
           push(stack, a.layout());
           return 0;
         },
         aliasAnalysisFromSchema()),
     Operator(
         "aten::cpu(Tensor(a) self) -> Tensor(a|b)",
         [](Stack& stack) {
           at::Tensor a;
           pop(stack, a);
           push(stack, a.cpu());
           return 0;
         },
         aliasAnalysisFromSchema()),

     Operator(
         "prim::index(Device self) -> int?",
         [](Stack& stack) {
           auto d = pop(stack).toDevice();
           if (d.has_index()) {
             push(stack, d.index());
           } else {
             push(stack, IValue());
           }
           return 0;
         },
         aliasAnalysisFromSchema()),
     Operator(
         // TODO return generator object when torchscript supports RNG
         // first-class
         "aten::manual_seed(int seed) -> ()",
         [](Stack& stack) {
           at::manual_seed(pop(stack).toInt());
           return 0;
         },
         aliasAnalysisFromSchema()),
     Operator(
         "aten::cuda(Tensor(a) self) -> Tensor(a|b)",
         [](Stack& stack) {
           at::Tensor a;
           pop(stack, a);
           push(stack, a.cuda());
           return 0;
         },
         aliasAnalysisFromSchema()),
     Operator(
         "aten::requires_grad_(Tensor(a!) self, bool requires_grad=True) -> Tensor(a!)",
         [](Stack& stack) {
           bool _requires_grad = pop(stack).toBool();
           at::Tensor self = pop(stack).toTensor();
           self.requires_grad_(_requires_grad);
           return 0;
         },
         aliasAnalysisConservative()),
     Operator(
         "prim::AutogradZero() -> Tensor",
         [](Stack& stack) {
           stack.emplace_back(at::Tensor());
           return 0;
         },
         aliasAnalysisSpecialCase()),
     Operator(
         "prim::BroadcastSizes(...) -> int[]",
         [](Stack& stack) {
           auto num_inputs = pop(stack).toInt();
           std::vector<int64_t> size;
           size.reserve(8);
           for (auto i = 0; i < num_inputs; ++i) {
             size =
                 at::infer_size(size, peek(stack, i, num_inputs).toIntVector());
           }
           drop(stack, num_inputs);
           push(stack, IValue(std::move(size)));
           return 0;
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
             return 0;
           };
         },
         aliasAnalysisSpecialCase()),
     Operator(
         "aten::warn(str message, int stacklevel=2) -> ()",
         [](Stack& stack) {
           TORCH_CHECK(
               false, "warn is implemented directly in the interpreter");
           return 0;
         },
         aliasAnalysisFromSchema()),

     Operator(
         "onnx::Reshape(Tensor input, Tensor shape) -> Tensor",
         [](Stack& stack) {
           at::Tensor input, shape;
           pop(stack, input, shape);
           shape = shape.contiguous();
           AT_ASSERT(shape.ndimension() == 1);
           at::IntArrayRef shape_list(shape.data_ptr<int64_t>(), shape.size(0));
           push(stack, input.reshape(shape_list));
           return 0;
         },
         aliasAnalysisSpecialCase()),
     Operator(
         "onnx::Shape(Tensor t) -> Tensor",
         [](Stack& stack) {
           auto t = pop(stack).toTensor();
           at::IntArrayRef sizes = t.sizes();
           auto sizes_tensor = torch::empty(
               {static_cast<int64_t>(sizes.size())}, at::dtype(at::kLong));
           auto accessor = sizes_tensor.accessor<int64_t, 1>();
           for (size_t i = 0; i < sizes.size(); ++i) {
             accessor[i] = sizes[i];
           }
           stack.emplace_back(sizes_tensor);
           return 0;
         },
         aliasAnalysisSpecialCase()),
     Operator(
         "prim::AutogradAnyNonZero(...) -> bool",
         [](Stack& stack) {
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
           stack.emplace_back(result);
           return 0;
         },
         aliasAnalysisFromSchema()),
     Operator(
         "prim::AutogradAdd(Any a, Any b) -> Any",
         [](Stack& stack) {
           at::Tensor a, b;
           pop(stack, a, b);
           if (!a.defined() && !b.defined()) {
             // undef + undef == undef
             stack.emplace_back(a);
           } else if (!a.defined()) {
             stack.emplace_back(b);
           } else if (!b.defined()) {
             stack.emplace_back(a);
           } else {
             stack.emplace_back(a + b);
           }
           return 0;
         },
         aliasAnalysisSpecialCase()),
     Operator(
         "aten::_grad_sum_to_size(Tensor(a) self, int[]? size) -> Tensor(a)",
         [](Stack& stack) {
           IValue self, size;
           pop(stack, self, size);
           if (size.isNone()) {
             push(stack, std::move(self));
           } else {
             push(stack, at::sum_to(self.toTensor(), size.toIntVector()));
           }
           return 0;
         },
         aliasAnalysisFromSchema()),
     Operator(
         "aten::_size_if_not_equal(int[] self_size, int[] other_size) -> int[]?",
         [](Stack& stack) {
           IValue self_size, other_size;
           pop(stack, self_size, other_size);
           auto s = self_size.toIntVector();
           auto o = other_size.toIntVector();
           if (s == o) {
             push(stack, IValue());
           } else {
             push(stack, s);
           }
           return 0;
         },
         aliasAnalysisFromSchema()),
     Operator(
         prim::tolist,
         // This operator has to be unschematized because the return type
         // depends on the type hint and input. The implementation of this
         // operator below is intended to be as close to the Python
         // implementation in torch/csrc/utils/tensor_list.cpp as possible.
         [](const Node* node) -> Operation {
           return [](Stack& stack) {
             int elem_ty_val;
             int dim_val;
             at::Tensor t;

             pop(stack, elem_ty_val);
             pop(stack, dim_val);
             pop(stack, t);

             // If the Tensor is not on the CPU, transfer it.
             if (!t.device().is_cpu()) {
               t = t.cpu();
             }

             // Rebuild the output type using elem_ty_val and dim_val. Start
             // with the element type corresponding to elem_ty_val.
             TypePtr out_ty;
             if (elem_ty_val == 0) {
               out_ty = IntType::get();
             } else if (elem_ty_val == 1) {
               out_ty = FloatType::get();
             } else if (elem_ty_val == 2) {
               out_ty = BoolType::get();
             } else {
               TORCH_CHECK(
                   false,
                   "Unsupported element type for tolist; only int, float and bool are supported");
             }

             // Check that type of the Tensor matches that of the annotation.
             // Make an exception for the case in which the annotated type is
             // float and the Tensor data type is also float; the elements will
             // be casted to double later.
             auto scalarTypeForJitType = tryScalarTypeFromJitType(out_ty);
             if (scalarTypeForJitType != at::ScalarType::Double ||
                 t.scalar_type() != at::ScalarType::Float) {
               TORCH_CHECK(
                   tryScalarTypeFromJitType(out_ty) == t.scalar_type(),
                   "Output annotation element type and runtime tensor element type must match for tolist()")
             }

             // Check that the dimension of the Tensor matches that of the
             // annotation.
             TORCH_CHECK(
                 dim_val == t.dim(),
                 "Output annotation list dimension and runtime tensor dimension must match for tolist()")

             // Wrap out_ty in a ListType dim times.
             for (int i = 0; i < dim_val; ++i) {
               out_ty = ListType::create(out_ty);
             }

             int64_t dim = t.dim();
             auto sizes = t.sizes();
             auto strides = t.strides();
             size_t element_size = t.element_size();
             char* data = static_cast<char*>(t.data_ptr());
             auto result = tensorToListRecursive(
                 data,
                 0,
                 dim,
                 out_ty,
                 t.scalar_type(),
                 sizes,
                 strides,
                 element_size);
             push(stack, std::move(result));
             return 0;
           };
         },
         aliasAnalysisSpecialCase()),
     Operator(
         prim::ConstantChunk,
         [](const Node* node) -> Operation {
           int64_t chunks = node->i(attr::chunks);
           int64_t dim = node->i(attr::dim);
           auto outputs_used = fmap(node->outputs(), [](const Value* v) {
             return v->uses().size() > 0;
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
               for (int64_t i = num_results; i < chunks; ++i) {
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
             return 0;
           };
         },
         aliasAnalysisSpecialCase()),
     Operator(
         "aten::dict() -> Dict(str, Tensor)",
         [](Stack& stack) {
           auto dict =
               c10::impl::GenericDict(StringType::get(), TensorType::get());
           push(stack, dict);
           return 0;
         },
         aliasAnalysisFromSchema()),
     Operator(
         "aten::_unwrap_optional(t(a)? optional) -> t(a)",
         [](Stack& stack) {
           auto val = pop(stack);
           TORCH_CHECK(!val.isNone(), "Unwrapping null optional");
           push(stack, std::move(val));
           return 0;
         },
         aliasAnalysisFromSchema()),
     // This op is no longer generated, but old models use it instead of
     // unchecked_cast, so we keep it here so it gets handled correctly.
     Operator(
         "prim::unchecked_cast(t x) -> t",
         noop,
         aliasAnalysisSpecialCase()),
     Operator(
         "aten::wait(Future(t) self) -> t",
         [](Stack& stack) {
           TORCH_CHECK(
               false, "wait is implemented directly in the interpreter");
           return 0;
         },
         aliasAnalysisSpecialCase())});

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
           return 0;
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
           auto output = autograd::profiler::getTime();
           push(stack, output);
           if (jit::tracer::isTracing()) {
             jit::tracer::addOutput(node, output);
           }
           return 0;
         },
         aliasAnalysisFromSchema())});

template <typename T>
int hashValue(Stack& stack) {
  auto value = pop(stack);
  auto hash = std::hash<T>()(value.to<T>());
  push(stack, int64_t(hash));
  return 0;
}

RegisterOperators reg2({

#define DEFINE_STRING_OP(op_name, string_op, result) \
  Operator(                                          \
      #op_name "(str a, str b) ->" #result,          \
      [](Stack& stack) {                             \
        auto b = pop(stack).toStringRef();           \
        auto a = pop(stack).toStringRef();           \
        push(stack, string_op);                      \
        return 0;                                    \
      },                                             \
      aliasAnalysisFromSchema())

    DEFINE_STRING_OP(aten::eq, a == b, bool),
    DEFINE_STRING_OP(aten::ne, a != b, bool),
    DEFINE_STRING_OP(aten::add, a + b, str),
#undef DEFINE_STRING_OP
    Operator(
        "aten::len.str(str s) -> int",
        [](Stack& stack) {
          auto string = pop(stack).toStringRef();
          push(stack, static_cast<int64_t>(string.size()));
          return 0;
        },
        aliasAnalysisFromSchema()),
    Operator(
        "aten::__getitem__.str(str s, int index) -> str",
        [](Stack& stack) {
          auto index = pop(stack).toInt();
          auto string = pop(stack).toStringRef();
          auto norm_index = normalizeIndex(index, string.size());
          char c = string.at(norm_index);
          push(stack, std::string(&c, 1));
          return 0;
        },
        aliasAnalysisFromSchema()),
    Operator(
        "aten::list(str t) -> str[]",
        [](Stack& stack) {
          auto str = pop(stack).toStringRef();
          c10::List<std::string> chars;
          chars.reserve(str.size());
          for (auto c : str) {
            chars.push_back(std::string(1, c));
          }
          push(stack, std::move(chars));
          return 0;
        },
        aliasAnalysisFromSchema()),

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
          "aten::index." decl_type "(" decl_type           \
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

#undef CREATE_GENERIC_LIST_OPS
#undef CREATE_SPECIALIZED_LIST_OPS

    // `listContains<T>` is not implemented for non-primitive types
    // TODO: Add List[bool] once .to<c10::List<bool>> doesn't throw an error
    Operator(
        "aten::__contains__.int(int[] l, int item) -> bool",
        listContains<int64_t>,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::__contains__.float(float[] l, float item) -> bool",
        listContains<double>,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::__contains__.str(str[] l, str item) -> bool",
        listContains<std::string>,
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
        "aten::eq.int_list(int[] a, int[] b) -> bool",
        listEq<int64_t>,
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

#define DEFINE_CONVERT_BASE_OP(op_name, prefix, char_op) \
  Operator(                                              \
      #op_name "(int i) -> str",                         \
      [](Stack& stack) {                                 \
        auto i = pop(stack).toInt();                     \
        std::stringstream ss;                            \
        if (i < 0) {                                     \
          ss << "-";                                     \
          i = -i;                                        \
        }                                                \
        ss << "0" << prefix << char_op << i;             \
        push(stack, ss.str());                           \
        return 0;                                        \
      },                                                 \
      aliasAnalysisFromSchema())

    DEFINE_CONVERT_BASE_OP(aten::hex, "x", std::hex),
    DEFINE_CONVERT_BASE_OP(aten::oct, "o", std::oct),

    Operator(
        "aten::bin(int i) -> str",
        [](Stack& stack) {
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
          return 0;
        },
        aliasAnalysisFromSchema()),
    // TODO: deprecate this in favor of aten::getelem
    Operator(
        "prim::StringIndex(str string, int index) -> str",
        [](Stack& stack) {
          auto index = pop(stack).toInt();
          auto string = pop(stack).toStringRef();
          auto norm_index = normalizeIndex(index, string.size());
          char c = string.at(norm_index);
          push(stack, std::string(&c, 1));
          return 0;
        },
        aliasAnalysisFromSchema()),
    Operator(
        "aten::ord(str string) -> int",
        [](Stack& stack) {
          auto string = pop(stack).toStringRef();
          TORCH_CHECK(
              string.size() == 1,
              "String for ord() must be 1 character, found ",
              string.size());
          uint8_t ord = string.at(0);
          push(stack, int64_t(ord));
          return 0;
        },
        aliasAnalysisFromSchema()),
    Operator(
        "aten::chr(int i) -> str",
        [](Stack& stack) {
          auto i = pop(stack).toInt();
          std::stringstream ss;
          TORCH_CHECK(
              i >= 0 && i < 1114111,
              "chr() arg not in range(0x110000), found ",
              i);
          char c = i;
          ss << c;
          push(stack, ss.str());
          return 0;
        },
        aliasAnalysisFromSchema()),
#define CREATE_COPY_OP(other_type, c_type)                                 \
  Operator(                                                                \
      "aten::copy_(Tensor(a!) self, " #other_type " other) -> Tensor(a!)", \
      [](Stack& stack) {                                                   \
        at::Tensor t;                                                      \
        c_type other;                                                      \
        pop(stack, t, other);                                              \
        std::move(t) = other; /* NOLINT(bugprone-use-after-move) */        \
        push(stack, std::move(t)); /* NOLINT(bugprone-use-after-move) */   \
        return 0;                                                          \
      },                                                                   \
      aliasAnalysisFromSchema())

    CREATE_COPY_OP(Tensor, at::Tensor),
    CREATE_COPY_OP(int, int64_t),
    CREATE_COPY_OP(float, double),
#undef CREATE_COPY_OP

    // only used in loop unrolling, not exposed to end users
    DEFINE_INT_OP(aten::__round_to_zero_floordiv, a / b),

    // only used internally in range() translation
    Operator(
        "aten::__range_length(int lo, int hi, int step) -> int",
        [](Stack& stack) {
          int64_t lo, hi, step;
          pop(stack, lo, hi, step);
          // error handling when step_val = 0 during runtime
          if (step == 0) {
            throw std::runtime_error("range() arg 3 must not be zero");
          }
          if (step > 0 && lo < hi)
            push(stack, 1 + (hi - 1 - lo) / step);
          else if (step < 0 && lo > hi)
            push(stack, 1 + (lo - 1 - hi) / (0 - step));
          else
            push(stack, 0);
          return 0;
        },
        aliasAnalysisFromSchema()),
    Operator(
        "aten::__derive_index(int index, int start, int step) -> int",
        [](Stack& stack) {
          int64_t index, start, step;
          pop(stack, index, start, step);
          push(stack, start + index * step);
          return 0;
        },
        aliasAnalysisFromSchema()),
    Operator(
        "aten::modf(float a) -> (float, float)",
        [](Stack& stack) {
          double a;
          pop(stack, a);
          double b, c;
          b = modf(a, &c);
          push(stack, b, c);
          return 0;
        },
        aliasAnalysisFromSchema()),
    Operator(
        "aten::frexp(float a) -> (float, int)",
        [](Stack& stack) {
          double a;
          pop(stack, a);
          double m;
          int e;
          m = std::frexp(a, &e);
          push(stack, m, e);
          return 0;
        },
        aliasAnalysisFromSchema()),
    Operator(
        "aten::ldexp(float x, int i) -> float",
        [](Stack& stack) {
          double a;
          int64_t b;
          pop(stack, a, b);
          push(stack, std::ldexp(a, b));
          return 0;
        },
        aliasAnalysisFromSchema()),
    DEFINE_BINARY_FLOAT_OP(aten::mathremainder, std::remainder(a, b)),

    DEFINE_INT_OP(aten::__and__, a& b),
    DEFINE_INT_OP(aten::__or__, a | b),
    DEFINE_INT_OP(aten::__xor__, a ^ b),
    DEFINE_INT_OP(aten::__lshift__, a << b),
    DEFINE_INT_OP(aten::__rshift__, a >> b),

    DEFINE_UNARY_OP(aten::round, std::round(a), float, float),
    DEFINE_UNARY_OP(aten::log, std::log(a), float, float),
    DEFINE_GENERIC_BINARY_OP(aten::log, std::log(a) / std::log(b), float),
    DEFINE_INT_FLOAT_OP(aten::log, std::log(a) / std::log(b), float),
    DEFINE_SCALAR_BINARY_OP(
        aten::log,
        std::log(a) / std::log(b),
        std::log(a) / std::log(b),
        float),
    DEFINE_UNARY_OP(aten::log1p, std::log1p(a), float, float),
    DEFINE_UNARY_OP(aten::log10, std::log10(a), float, float),
    DEFINE_UNARY_OP(aten::exp, std::exp(a), float, float),
    DEFINE_UNARY_OP(aten::sqrt, std::sqrt(a), float, float),
    DEFINE_UNARY_OP(aten::acos, std::acos(a), float, float),
    DEFINE_UNARY_OP(aten::asin, std::asin(a), float, float),
    DEFINE_UNARY_OP(aten::atan, std::atan(a), float, float),
    DEFINE_BINARY_FLOAT_OP(aten::atan2, std::atan2(a, b)),
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
        [](Stack& stack) {
          at::Tensor x;
          pop(stack, x);
          push(stack, x.abs());
          return 0;
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
        "aten::isnan(float a) -> bool",
        [](Stack& stack) {
          double a;
          pop(stack, a);
          push(stack, std::isnan(a));
          return 0;
        },
        aliasAnalysisFromSchema()),

    Operator(
        "aten::_tensor_to_list(Tensor self) -> int[]",
        [](Stack& stack) {
          at::Tensor t;
          pop(stack, t);
          c10::List<int64_t> elems;
          elems.reserve(t.size(0));
          for (int i = 0; i < t.size(0); i++) {
            elems.push_back(*t[i].data_ptr<int32_t>());
          }
          push(stack, std::move(elems));
          return 0;
        },
        aliasAnalysisFromSchema()),
    Operator(
        "aten::_list_to_tensor(int[] self) -> Tensor",
        [](Stack& stack) {
          c10::List<int64_t> l = pop(stack).toIntList();
          auto t = torch::empty(
              {static_cast<int64_t>(l.size())}, at::dtype(at::kInt));
          for (size_t i = 0; i < l.size(); i++) {
            t[i] = l.get(i);
          }
          push(stack, std::move(t));
          return 0;
        },
        aliasAnalysisFromSchema()),
    Operator(
        "aten::all.int(int[] self) -> bool",
        [](Stack& stack) {
          c10::List<int64_t> l = pop(stack).toIntList();
          for (const auto& elem : l) {
            if (!elem) {
              push(stack, false);
              return 0;
            }
          }
          push(stack, true);
          return 0;
        },
        aliasAnalysisFromSchema()),
    Operator(
        "aten::all.float(float[] self) -> bool",
        [](Stack& stack) {
          c10::List<double> l = pop(stack).toDoubleList();
          for (const auto& elem : l) {
            if (!elem) {
              push(stack, false);
              return 0;
            }
          }
          push(stack, true);
          return 0;
        },
        aliasAnalysisFromSchema()),
    Operator(
        "aten::all.bool(bool[] self) -> bool",
        [](Stack& stack) {
          c10::List<bool> l = pop(stack).toBoolList();
          for (const auto& elem : l) {
            if (!elem) {
              push(stack, false);
              return 0;
            }
          }
          push(stack, true);
          return 0;
        },
        aliasAnalysisFromSchema()),
    Operator(
        "aten::divmod.int(int x, int y) -> (int, int)",
        [](Stack& stack) {
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
          return 0;
        },
        aliasAnalysisFromSchema()),
    Operator(
        "aten::divmod.float(float x, float y) -> (float, float)",
        [](Stack& stack) {
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
          return 0;
        },
        aliasAnalysisFromSchema()),
    Operator(
        "prim::id(AnyClassType? x) -> int",
        [](Stack& stack) {
          IValue a;
          pop(stack, a);
          if (a.isNone()) {
            push(stack, 0);
          } else {
            push(stack, reinterpret_cast<int64_t>(a.internalToPointer()));
          }
          return 0;
        },
        aliasAnalysisFromSchema()),

#define DEFINE_DIVMOD_MIXED_OP(type_a, type_b)                           \
  Operator(                                                              \
      "aten::divmod(" #type_a " x," #type_b " y) -> (float, float)",     \
      [](Stack& stack) {                                                 \
        type_a a;                                                        \
        type_b b;                                                        \
        pop(stack, a, b);                                                \
        if (b == 0) {                                                    \
          throw std::runtime_error("ZeroDivisionError: float divmod()"); \
        }                                                                \
        double quot = floor(a / b);                                      \
        double rem = a - (quot * b);                                     \
        push(stack, quot, rem);                                          \
        return 0;                                                        \
      },                                                                 \
      aliasAnalysisFromSchema())

    DEFINE_DIVMOD_MIXED_OP(int, float),
    DEFINE_DIVMOD_MIXED_OP(float, int),

#undef DEFINE_DIVMOD_MIXED_OP
    Operator(
        "aten::hash(str t) -> int",
        hashValue<std::string>,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::hash(int t) -> int",
        hashValue<int>,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::hash(float t) -> int",
        hashValue<double>,
        aliasAnalysisFromSchema()),
});

bool simpleClassTypeArg(const Argument& arg, const ClassTypePtr& type) {
  return arg.type() == type && !arg.kwarg_only() && !arg.default_value();
}

Function* checkSortSchema(const c10::TypePtr& list_element_type) {
  std::stringstream error_str;
  if (auto class_type = list_element_type->cast<ClassType>()) {
    if (auto method = class_type->findMethod("__lt__")) {
      const auto& lt_schema = method->getSchema();
      const auto& schema_args = lt_schema.arguments();
      bool error =
          (schema_args.size() != 2 ||
           !simpleClassTypeArg(schema_args[0], class_type) ||
           !simpleClassTypeArg(schema_args[1], class_type) ||
           lt_schema.returns().size() != 1 ||
           lt_schema.returns()[0].type() != BoolType::get());
      if (!error) {
        return method;
      }
    }
    error_str << "To sort a list of " << class_type->repr_str()
              << " it must define a "
              << "__lt__ method with two inputs of type "
              << class_type->repr_str() << " that "
              << "returns a bool";
  } else {
    error_str << "To sort a list of " << list_element_type->repr_str()
              << " must be of Tensors, ints, floats, bools or "
              << "a User Defined Class that defines the __lt__ compare method"
              << ", got list of " << list_element_type->repr_str() << "\n";
  }
  throw std::runtime_error(error_str.str());
}

template <bool has_reverse_arg, bool copy_return_list>
int sort_op(Stack& stack) {
  bool reverse = has_reverse_arg ? pop(stack).toBool() : false;
  auto g_list = pop(stack).toList();
  if (copy_return_list) {
    g_list = g_list.copy();
  }
  Stack sort_stack;
  Function* lt_func = nullptr;
  std::sort(
      g_list.begin(),
      g_list.end(),
      [reverse, &sort_stack, &lt_func](IValue a, IValue b) -> bool {
        // "strict weak ordering" issue - see other sort
        if (a.is(b)) {
          return false;
        }
        if (!lt_func) {
          lt_func = checkSortSchema(a.type());
        }
        sort_stack.push_back(a);
        sort_stack.push_back(b);
        lt_func->run(sort_stack);
        return pop(sort_stack).toBool() != reverse;
      });
  if (copy_return_list) {
    push(stack, g_list);
  }
  return 0;
}

// NB: this must be registered after the other aten::sort operators
RegisterOperators regSort({
    Operator(
        "aten::sorted(t[](a) self) -> (t[])",
        sort_op</*has_reverse_arg*/ false, /*copy_return_list*/ true>,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::sort(t[](a!) self, bool reverse=False) -> ()",
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
        input, _output_size(input, 1, size, scale_factors), scale_factors_1);
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
        scale_factors_1);
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

int interpolate_op(Stack& stack) {
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
  return 0;
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

int upsample_nearest_op(Stack& stack) {
  at::Tensor input;
  IValue size;
  IValue scale_factor_int;
  pop(stack, input, size, scale_factor_int);
  IValue scale_factor_double = convert_scale_factor_to_double(scale_factor_int);
  at::Tensor res = interpolate(
      input, size, scale_factor_double, "nearest", c10::nullopt, c10::nullopt);
  push(stack, std::move(res));
  return 0;
}

int upsample_op(Stack& stack) {
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
  return 0;
}

int upsample_bilinear_op(Stack& stack) {
  at::Tensor input;
  IValue size;
  IValue scale_factor_int;
  pop(stack, input, size, scale_factor_int);
  IValue scale_factor_double = convert_scale_factor_to_double(scale_factor_int);
  at::Tensor res = interpolate(
      input, size, scale_factor_double, "bilinear", true, c10::nullopt);
  push(stack, std::move(res));
  return 0;
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
        "aten::__upsample_bilinear(Tensor input, int? size = None, int[]? scale_factor = None) -> Tensor",
        upsample_bilinear_op,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::__upsample_bilinear.size_list(Tensor input, int[]? size = None, int[]? scale_factor = None) -> Tensor",
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
