#include <aten/src/ATen/Context.h>
#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/autograd/edge.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/autograd/profiler.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/jit/custom_operator.h>
#include <torch/csrc/jit/fuser/interface.h>
#include <torch/csrc/jit/graph_executor.h>
#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/operator.h>
#include <torch/csrc/jit/pickle.h>
#include <torch/csrc/jit/print_handler.h>
#include <torch/csrc/jit/profiling_record.h>
#include <torch/csrc/jit/script/compilation_unit.h>
#include <torch/csrc/jit/script/error_report.h>
#include <torch/csrc/jit/script/jit_exception.h>
#include <torch/csrc/jit/script/logging.h>

#include <ATen/ExpandUtils.h>
#include <ATen/Parallel.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/core/Dict.h>
#include <ATen/core/ivalue.h>
#include <c10/core/thread_pool.h>
#include <c10/util/SmallVector.h>

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

template <class T>
c10::List<T> make_result_list(const TypePtr& elemType) {
  return c10::List<T>();
}
template <>
c10::impl::GenericList make_result_list<IValue>(const TypePtr& elemType) {
  return c10::impl::GenericList(elemType);
}

Operation noop(const Node* n) {
  return [](Stack& stack) { return 0; };
}

c10::OperatorOptions aliasAnalysisFromSchema() {
  c10::OperatorOptions result;
  result.setAliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA);
  return result;
}

c10::OperatorOptions aliasAnalysisConservative() {
  c10::OperatorOptions result;
  result.setAliasAnalysis(c10::AliasAnalysisKind::CONSERVATIVE);
  return result;
}

c10::OperatorOptions aliasAnalysisSpecialCase() {
  c10::OperatorOptions result;
  result.setAliasAnalysis(c10::AliasAnalysisKind::INTERNAL_SPECIAL_CASE);
  return result;
}

// using the rules from python_arg_parser FunctionParameter::check
// tensor cannot have grad set, tensor must be 0 dim,
// and if the dest is an int the source must be integral type
void checkImplicitTensorToNum(at::Tensor t, bool toInt) {
  if (autograd::as_variable_ref(t).requires_grad()) {
    throw std::runtime_error(
        "Cannot input a tensor that requires grad as a scalar argument");
  }
  if (t.sizes().size() != 0) {
    throw std::runtime_error(
        "Cannot input a tensor of dimension other than 0 as a scalar argument");
  }
  if (toInt && !isIntegralType(t.scalar_type(), /*includeBool=*/false)) {
    std::stringstream ss;
    ss << "Cannot input a tensor of type " << t.scalar_type()
       << " as an integral argument";
    throw std::runtime_error(ss.str());
  }
}

template <typename dtype> // int64_t, bool, double
Operation listConstruct(int64_t num_inputs) {
  return [=](Stack& stack) {
    auto inputs = peekSlice(stack, 0, num_inputs, num_inputs);
    c10::List<dtype> vals =
        c10::impl::toList(fmap(inputs, [](const IValue& v) { return v.to<dtype>(); }));
    drop(stack, num_inputs);
    push(stack, std::move(vals));
    return 0;
  };
}

static int64_t floordiv(int64_t a, int64_t b) {
  if (b == 0) {
    throw std::runtime_error("division by 0");
  }
  if ((a > 0) == (b > 0)) {
    // simple case, both have same sign
    return a / b;
  } else {
    // in python division rounds down, it doesnt not truncate like in c++
    auto r = lldiv(a, b);
    return (r.rem) ? r.quot - 1 : r.quot;
  }
}
void checkDoubleInRange(double a) {
  if (std::isnan(a) || std::isinf(a) ||
      a > double(std::numeric_limits<int64_t>::max()) ||
      a < double(std::numeric_limits<int64_t>::min())) {
    throw c10::Error(
        "Cannot convert float " + std::to_string(a) + " to integer", "");
    return;
  }
}
static int64_t floor(double a) {
  checkDoubleInRange(a);
  return std::floor(a);
}
static int64_t ceil(double a) {
  checkDoubleInRange(a);
  return std::ceil(a);
}

static int64_t gcd(int64_t a, int64_t b) {
  while (b != 0) {
    int64_t r = a % b;
    a = b;
    b = r;
  }
  // in python gcd returns non-negative values
  return std::abs(a);
}

int64_t partProduct(int n, int m) {
  if (m <= (n + 1))
    return (int64_t)n;
  if (m == (n + 2))
    return (int64_t)n * m;
  int k = (n + m) / 2;
  if ((k & 1) != 1)
    k = k - 1;
  return partProduct(n, k) * partProduct(k + 2, m);
}

void loop(int n, int64_t& p, int64_t& r) {
  if (n <= 2)
    return;
  loop(n / 2, p, r);
  p = p * partProduct(n / 2 + 1 + ((n / 2) & 1), n - 1 + (n & 1));
  r = r * p;
}

int nminussumofbits(int v) {
  long w = (long)v;
  w -= (0xaaaaaaaa & w) >> 1;
  w = (w & 0x33333333) + ((w >> 2) & 0x33333333);
  w = (w + (w >> 4)) & 0x0f0f0f0f;
  w += w >> 8;
  w += w >> 16;
  return v - (int)(w & 0xff);
}

int64_t factorial(int n) {
  if (n < 0) {
    throw std::runtime_error("factorial() not defined for negative values");
  }
  int64_t p = 1, r = 1;
  loop(n, p, r);
  return r << nminussumofbits(n);
}
static const double degToRad = std::acos(-1.0) / 180.0;
static const double radToDeg = 180.0 / std::acos(-1.0);
double degrees(double x) {
  return x * radToDeg;
}
double radians(double x) {
  return x * degToRad;
}

// reference function THPVariable_to in python_variable_methods.cpp
static at::Tensor to_dispatch(
    at::Tensor self,
    c10::optional<at::Device> device,
    c10::optional<at::ScalarType> scalarType,
    bool non_blocking,
    bool copy) {
  if (device && device->is_cuda()) {
    at::globalContext().lazyInitCUDA();
  }
  if (!device && !scalarType && !copy) {
    return self;
  } else if (!device) {
    return self.to(*scalarType, non_blocking, copy);
  } else if (!scalarType) {
    return self.to(*device, non_blocking, copy);
  } else {
    return self.to(*device, *scalarType, non_blocking, copy);
  }
}

// Convert an python index (which may be negative) into an index usable for a
// C++ container
int64_t normalizeIndex(int64_t idx, int64_t list_size) {
  if (idx < 0) {
    // Handle negative indexing
    idx = list_size + idx;
  }
  return idx;
}

RegisterOperators reg(
    {Operator(
         prim::profile,
         [](const Node* node) -> Operation {
           auto callback = node->cast<ProfileOp>()->getCallback();
           return [callback](Stack& stack) {
             callback(stack);
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
         [](const Node* node) -> Operation {
           return [](Stack& stack) {
             AT_ERROR("Should be replaced by prim::BailOut");
             return 0;
           };
         },
         aliasAnalysisFromSchema()),
     Operator(
         "prim::BailOut(...) -> Tensor(a)",
         [](const Node* /* node */) -> Operation {
           return [](Stack& /* stack */) {
             AT_ERROR("prim::BailOut not yet implemented"); // NOLINT
             return 0;
           };
         },
         aliasAnalysisFromSchema()),
     Operator(
         "prim::BailoutTemplate() -> int",
         [](const Node* /* node */) -> Operation {
           return [](Stack& stack) {
             // TODO: today, we put a single bailout template at the front to
             // carry the un-optimized graph for bailout nodes to use. Ideally
             // this should never run, but we haven't written the code to remove
             // it yet.
             // TORCH_INTERNAL_ASSERT(false);

             // Returns an int so that we have an easy way to do graph traversal
             push(stack, 1);
             return 0;
           };
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
     Operator(
         "prim::ImplicitTensorToNum(Tensor a) -> Scalar",
         [](const Node* node) -> Operation {
           if (node->output()->type() == IntType::get()) {
             return [](Stack& stack) {
               at::Tensor a;
               pop(stack, a);
               checkImplicitTensorToNum(a, /*to int*/ true);
               push(stack, a.item<int64_t>());
               return 0;
             };
           } else {
             return [](Stack& stack) {
               at::Tensor a;
               pop(stack, a);
               checkImplicitTensorToNum(a, /*to int*/ false);
               push(stack, a.item<double>());
               return 0;
             };
           }
         },
         aliasAnalysisFromSchema()),
     Operator(
         "prim::NumToTensor(Scalar a) -> Tensor",
         [](Stack& stack) {
           at::Scalar s;
           pop(stack, s);
           push(stack, autograd::make_variable(at::scalar_to_tensor(s)));
           return 0;
         },
         aliasAnalysisFromSchema()),
     // note: this op needs to share a name with the Scalar -> Tensor conversion
     // because all _to_tensor conversion have to have the same operator namet
     Operator(
         "prim::NumToTensor(bool a) -> Tensor",
         [](Stack& stack) {
           bool b;
           pop(stack, b);
           push(stack, autograd::make_variable(at::scalar_to_tensor(b)));
           return 0;
         },
         aliasAnalysisFromSchema()),
     Operator(
         "aten::Bool(Tensor a) -> bool",
         [](Stack& stack) {
           at::Tensor a;
           pop(stack, a);
           push(stack, a.is_nonzero());
           return 0;
         },
         aliasAnalysisFromSchema()),
     Operator(
         "aten::Bool(int a) -> bool",
         [](Stack& stack) {
           int64_t i;
           pop(stack, i);
           push(stack, (bool)i);
           return 0;
         },
         aliasAnalysisFromSchema()),
     Operator(
         "aten::Bool(float a) -> bool",
         [](Stack& stack) {
           double d;
           pop(stack, d);
           push(stack, (bool)d);
           return 0;
         },
         aliasAnalysisFromSchema()),
     Operator(
         "aten::Int(Tensor a) -> int",
         [](Stack& stack) {
           at::Tensor a;
           pop(stack, a);
           push(stack, a.item<int64_t>());
           return 0;
         },
         aliasAnalysisFromSchema()),
     Operator(
         "aten::Int(bool a) -> int",
         [](Stack& stack) {
           bool b;
           pop(stack, b);
           push(stack, (int)b);
           return 0;
         },
         aliasAnalysisFromSchema()),
     Operator(
         "aten::Int(float a) -> int",
         [](Stack& stack) {
           double d;
           pop(stack, d);
           push(stack, (int64_t)d);
           return 0;
         },
         aliasAnalysisFromSchema()),
     Operator(
         "aten::Int(Scalar a) -> int",
         [](Stack& stack) {
           IValue scalar;
           pop(stack, scalar);
           if (scalar.isInt()) {
             push(stack, std::move(scalar));
           } else {
             push(stack, static_cast<int64_t>(scalar.toDouble()));
           }
           return 0;
         },
         aliasAnalysisFromSchema()),
     Operator(
         "aten::Float(Tensor a) -> float",
         [](Stack& stack) {
           at::Tensor a;
           pop(stack, a);
           push(stack, a.item<double>());
           return 0;
         },
         aliasAnalysisFromSchema()),
     Operator(
         "aten::Float(Scalar a) -> float",
         [](Stack& stack) {
           IValue scalar;
           pop(stack, scalar);
           if (scalar.isDouble()) {
             push(stack, std::move(scalar));
           } else {
             push(stack, static_cast<double>(scalar.toInt()));
           }
           return 0;
         },
         aliasAnalysisFromSchema()),
     Operator(
         "aten::Float(int a) -> float",
         [](Stack& stack) {
           int64_t i;
           pop(stack, i);
           push(stack, (float)i);
           return 0;
         },
         aliasAnalysisFromSchema()),
     Operator(
         "aten::Float(bool a) -> float",
         [](Stack& stack) {
           bool b;
           pop(stack, b);
           push(stack, (float)b);
           return 0;
         },
         aliasAnalysisFromSchema()),
     Operator(
         "aten::Float(str a) -> float",
         [](Stack& stack) {
           auto s = pop(stack).toString();
           if (s->string() == "inf")
             push(stack, std::numeric_limits<double>::infinity());
           else if (s->string() == "-inf")
             push(stack, -std::numeric_limits<double>::infinity());
           else
             AT_ERROR(
                 "Only 'inf' or '-inf' can be cast to a float, but got '",
                 s->string(),
                 "'");
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
     // reference function parse_to_conversion in python_arg_parsing.h
     Operator(
         "aten::to(Tensor(a) self, Device? device, int? dtype=None, bool non_blocking=False, bool copy=False) -> Tensor(a|b)",
         [](Stack& stack) {
           bool non_blocking;
           bool copy;
           pop(stack, non_blocking, copy);
           c10::optional<at::ScalarType> scalarType =
               pop(stack).toOptional<at::ScalarType>();
           c10::optional<c10::Device> device =
               pop(stack).toOptional<c10::Device>();
           at::Tensor self = pop(stack).toTensor();
           push(
               stack,
               to_dispatch(self, device, scalarType, non_blocking, copy));
           return 0;
         },
         aliasAnalysisFromSchema()),
     Operator(
         "aten::to(Tensor(a) self, int? dtype=None, bool non_blocking=False, bool copy=False) -> Tensor(a|b)",
         [](Stack& stack) {
           bool non_blocking;
           bool copy;
           pop(stack, non_blocking, copy);
           c10::optional<at::ScalarType> scalarType =
               pop(stack).toOptional<at::ScalarType>();
           c10::optional<c10::Device> device = c10::nullopt;
           at::Tensor self = pop(stack).toTensor();
           push(
               stack,
               to_dispatch(self, device, scalarType, non_blocking, copy));
           return 0;
         },
         aliasAnalysisFromSchema()),
     Operator(
         "aten::to(Tensor(a) self, bool non_blocking=False, bool copy=False) -> Tensor(a|b)",
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
         "prim::device(Tensor a) -> Device",
         [](Stack& stack) {
           push(stack, pop(stack).toTensor().device());
           return 0;
         },
         aliasAnalysisFromSchema()),
     Operator(
         "prim::dtype(Tensor a) -> int",
         [](Stack& stack) {
           at::Tensor a;
           pop(stack, a);
           push(stack, static_cast<int64_t>(a.scalar_type()));
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
         "prim::shape(Tensor a) -> int[]",
         [](Stack& stack) {
           at::Tensor a;
           pop(stack, a);
           push(stack, a.sizes());
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
         "prim::data(Tensor(a) a) -> Tensor(a)",
         [](Stack& stack) {
           at::Tensor a;
           pop(stack, a);
           push(stack, autograd::Variable(a).variable_data());
           return 0;
         },
         aliasAnalysisFromSchema()),
     Operator(
         "prim::is_cuda(Tensor a) -> bool",
         [](Stack& stack) {
           at::Tensor a;
           pop(stack, a);
           push(stack, a.is_cuda());
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
         "aten::cpu(Tensor(a) self) -> Tensor(a|b)",
         [](Stack& stack) {
           at::Tensor a;
           pop(stack, a);
           push(stack, a.cpu());
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
         "aten::grad(Tensor[] outputs, Tensor[] inputs, Tensor?[]? grad_outputs=None, bool? keep_graph=None, bool create_graph=False, bool allow_unused=False) -> Tensor[]",
         [](Stack& stack) {
           bool allow_unused = pop(stack).toBool();
           bool create_graph = pop(stack).toBool();
           auto keep_graph = pop(stack).toOptional<bool>();
           auto grad_outputs = pop(stack);
           auto inputs = pop(stack).toTensorList();
           auto outputs = pop(stack).toTensorList();
           std::vector<torch::autograd::Variable> input_vars(
               inputs.begin(), inputs.end());
           std::vector<torch::autograd::Variable> output_vars(outputs.begin(), outputs.end());
           std::vector<torch::autograd::Variable> gradients;

           if (!grad_outputs.isNone()) {
             for (const IValue& v : grad_outputs.toGenericListRef()) {
               gradients.emplace_back(v.isNone() ? at::Tensor() : v.toTensor());
             }
           }

           auto res = torch::autograd::grad(
               output_vars,
               input_vars,
               gradients,
               keep_graph,
               create_graph,
               allow_unused);

           std::vector<at::Tensor> res_tensors(res.begin(), res.end());
           push(stack, c10::impl::toList<at::Tensor>(std::move(res_tensors)));
           return 0;
         },
         aliasAnalysisFromSchema()),
     // NB: backward op might write to every input tensors in the graph and it's
     // much more expensive to analayze the leaves and sometimes it might retain
     // the whole gradients in every tensor of the Autograd graph with
     // create_graph=True so we use aliasAnalysisConservative for these two OPs
     Operator(
         "aten::backward(Tensor[](a!) tensors, Tensor?[]? grad_tensors=None, bool? retain_graph=None, bool create_graph=False) -> ()",
         [](Stack& stack) {
           bool create_graph = pop(stack).toBool();
           auto retain_graph = pop(stack).toOptional<bool>();
           auto grad_tensors = pop(stack);
           auto tensors = pop(stack).toTensorList();
           std::vector<torch::autograd::Variable> output_vars(
               tensors.begin(), tensors.end());
           std::vector<torch::autograd::Variable> gradients;

           if (!grad_tensors.isNone()) {
             for (const IValue& v : grad_tensors.toGenericListRef()) {
               gradients.emplace_back(v.isNone() ? at::Tensor() : v.toTensor());
             }
           }

           torch::autograd::backward(
               output_vars, gradients, retain_graph, create_graph);
           return 0;
         },
         aliasAnalysisConservative()),
     Operator(
         "aten::backward(Tensor(a!) self, Tensor? gradient=None, bool? retain_graph=None, bool create_graph=False) -> ()",
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
         "prim::AutogradZero() -> Tensor",
         [](const Node* node) -> Operation {
           return [](Stack& stack) {
             stack.emplace_back(at::Tensor());
             return 0;
           };
         },
         aliasAnalysisSpecialCase()),
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
         prim::Print,
         [](const Node* node) -> Operation {
           size_t num_inputs = node->inputs().size();
           return [num_inputs](Stack& stack) {
             std::stringstream ss;
             bool first = true;
             for (const IValue& i : last(stack, num_inputs)) {
               if (!first)
                 ss << " ";
               first = false;
               ss << i;
             }
             drop(stack, num_inputs);
             ss << std::endl;
             auto* handler = getPrintHandler();
             TORCH_INTERNAL_ASSERT(handler);
             handler(ss.str());
             return 0;
           };
         },
         aliasAnalysisSpecialCase()),
     Operator(
         prim::BroadcastSizes,
         [](const Node* node) -> Operation {
           size_t num_inputs = node->inputs().size();
           return [num_inputs](Stack& stack) {
             std::vector<int64_t> size;
             size.reserve(8);
             for (size_t i = 0; i < num_inputs; ++i) {
               size = at::infer_size(
                   size, peek(stack, i, num_inputs).toIntListRef());
             }
             drop(stack, num_inputs);
             push(stack, c10::impl::toList(std::move(size)));
             return 0;
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
             regular_shape[dim] =split_size;
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
         FunctionSchema(
             "aten::warn",
             "",
             {Argument("message", StringType::get()),
              Argument("stacklevel", IntType::get(), c10::nullopt, 2, true)},
             {}),
         [](const Node* node) -> Operation {
           auto range = node->sourceRange().source();
           if (range->filename()) {
             auto line = range->starting_line_no() +
                 range->lineno_for_offset(node->sourceRange().start());
             return [=](Stack& stack) {
               drop(stack, 1);
               c10::SourceLocation location{
                   "", range->filename()->c_str(), uint32_t(line)};
               c10::Warning::warn(location, pop(stack).toStringRef());
               return 0;
             };
           }

           return [=](Stack& stack) {
             drop(stack, 1);
             AT_WARN(pop(stack).toStringRef());
             return 0;
           };
         },
         aliasAnalysisFromSchema()),
     Operator(
         "prim::RaiseException(str msg) -> ()",
         [](Stack& stack) {
           throw JITException(pop(stack).toStringRef());
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

     // Load x, y
     // loads values from registers onto the stack, the actual callback does
     // nothing since the stack manipulation is already encoded in inst.inputs
     // and inst.outputs
     Operator(prim::Load, noop, aliasAnalysisSpecialCase()),
     // x, y = Store
     // stores vales from stack into registers, the actual callback does
     // nothing since the stack manipulation is already encoded in inst.inputs
     // and inst.outputs
     Operator(prim::Store, noop, aliasAnalysisSpecialCase()),
     Operator(
         prim::Drop,
         [](const Node* node) -> Operation {
           auto N = node->inputs().size();
           return [=](Stack& stack) {
             drop(stack, N);
             return 0;
           };
         },
         aliasAnalysisSpecialCase()),
     Operator(
         c10::onnx::Reshape,
         [](const Node* node) -> Operation {
           return [=](Stack& stack) {
             at::Tensor input, shape;
             pop(stack, input, shape);
             shape = shape.contiguous();
             AT_ASSERT(shape.ndimension() == 1);
             at::IntArrayRef shape_list(shape.data_ptr<int64_t>(), shape.size(0));
             push(stack, input.reshape(shape_list));
             return 0;
           };
         },
         aliasAnalysisSpecialCase()),
     Operator(
         c10::onnx::Shape,
         [](const Node* node) -> Operation {
           return [=](Stack& stack) {
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
           };
         },
         aliasAnalysisSpecialCase()),
     Operator(
         prim::AutogradAnyNonZero,
         [](const Node* node) -> Operation {
           size_t num_inputs = node->inputs().size();
           return [=](Stack& stack) {
             bool result = false;
             for (const IValue& t : last(stack, num_inputs)) {
               if (t.toTensor().defined()) {
                 result = true;
                 break;
               }
             }
             drop(stack, num_inputs);
             stack.emplace_back(result);
             return 0;
           };
         },
         aliasAnalysisSpecialCase()),
     Operator(
         prim::AutogradAdd,
         [](const Node* node) -> Operation {
           return [=](Stack& stack) {
             at::Tensor a, b;
             pop(stack, a, b);
             if (!a.defined())
               stack.emplace_back(b);
             else if (!b.defined())
               stack.emplace_back(a);
             else
               stack.emplace_back(a + b);
             return 0;
           };
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
             push(
                 stack,
                 at::sum_to(self.toTensor(), size.toIntListRef()));
           }
           return 0;
         },
         aliasAnalysisFromSchema()),
     Operator(
         "aten::_size_if_not_equal(int[] self_size, int[] other_size) -> int[]?",
         [](Stack& stack) {
           IValue self_size, other_size;
           pop(stack, self_size, other_size);
           auto s = self_size.toIntListRef();
           if (s.equals(other_size.toIntListRef())) {
             push(stack, IValue());
           } else {
             push(stack, s);
           }
           return 0;
         },
         aliasAnalysisFromSchema()),
     Operator(
         prim::TupleUnpack,
         [](const Node* node) -> Operation {
           size_t num_elems = node->outputs().size();
           return [=](Stack& stack) {
             auto tuple = pop(stack).toTuple();
             if (tuple->elements().size() != num_elems) {
               AT_ERROR(
                   "Expected a tuple of ",
                   num_elems,
                   " elements, but got ",
                   tuple->elements().size());
             }
             stack.insert(
                 stack.end(),
                 tuple->elements().begin(),
                 tuple->elements().end());
             return 0;
           };
         },
         aliasAnalysisSpecialCase()),
     Operator(
         prim::TupleSlice,
         [](const Node* node) -> Operation {
           int64_t beg_ind = node->i(attr::beg);
           int64_t end_ind = node->i(attr::end);
           return [=](Stack& stack) {
             auto tuple = pop(stack).toTuple();
             std::vector<IValue> output_elems;
             output_elems.reserve(end_ind - beg_ind);
             for (int64_t i = beg_ind; i < end_ind; ++i) {
               output_elems.emplace_back(tuple->elements()[i]);
             }
             push(stack, c10::ivalue::Tuple::create(std::move(output_elems)));
             return 0;
           };
         },
         aliasAnalysisSpecialCase()),
     Operator(
         prim::TupleIndex,
         [](const Node* node) -> Operation {
           return [](Stack& stack) {
             int64_t index = pop(stack).toInt();
             auto tuple = pop(stack).toTuple();
             auto norm_index = normalizeIndex(index, tuple->elements().size());
             if (norm_index < 0 ||
                 norm_index > static_cast<int64_t>(tuple->elements().size())) {
               throw std::out_of_range("Tuple list index out of range");
             }
             stack.emplace_back(tuple->elements()[norm_index]);
             return 0;
           };
         },
         aliasAnalysisSpecialCase()),
     Operator(
         prim::TupleConstruct,
         [](const Node* node) -> Operation {
           size_t num_inputs = node->inputs().size();
           auto type = node->output()->type()->expect<TupleType>();
           bool named = type->name().has_value();
           return [=](Stack& stack) {
             std::vector<IValue> elems{
                 std::make_move_iterator(stack.end() - num_inputs),
                 std::make_move_iterator(stack.end())};
             drop(stack, num_inputs);
             push(
                 stack,
                 named ? c10::ivalue::Tuple::createNamed(std::move(elems), type)
                       : c10::ivalue::Tuple::create(std::move(elems)));
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
         prim::ListUnpack,
         [](const Node* node) -> Operation {
           const auto num_outputs = node->outputs().size();
           ListTypePtr lt = node->input()->type()->expect<ListType>();
           if (lt->getElementType() == IntType::get()) {
             return [=](Stack& stack) {
               auto list = pop(stack).toIntList();
               TORCH_CHECK(
                   list.size() == num_outputs,
                   "Expected ",
                   num_outputs,
                   " elements in a list but found ",
                   list.size());
               push_list_elements(stack, list);
               return 0;
             };
           } else if (lt->getElementType() == FloatType::get()) {
             return [=](Stack& stack) {
               auto list = pop(stack).toDoubleList();
               TORCH_CHECK(
                   list.size() == num_outputs,
                   "Expected ",
                   num_outputs,
                   " elements in a list but found ",
                   list.size());
               push_list_elements(stack, list);
               return 0;
             };
           } else if (lt->getElementType() == TensorType::get()) {
             return [=](Stack& stack) {
               auto list = pop(stack).toTensorList();
               TORCH_CHECK(
                   list.size() == num_outputs,
                   "Expected ",
                   num_outputs,
                   " elements in a list but found ",
                   list.size());
               push_list_elements(stack, list);
               return 0;
             };
           } else {
             return [=](Stack& stack) {
               auto list = pop(stack).toGenericList();
               TORCH_CHECK(
                   list.size() == num_outputs,
                   "Expected ",
                   num_outputs,
                   " elements in a list but found ",
                   list.size());
               push_list_elements(stack, list);
               return 0;
             };
           }
         },
         aliasAnalysisSpecialCase()),
     Operator(
         prim::ListConstruct,
         [](const Node* node) -> Operation {
           const auto num_inputs = node->inputs().size();
           ListTypePtr lt = node->output()->type()->expect<ListType>();
           if (IntType::get() == lt->getElementType()) {
             return listConstruct<int64_t>(num_inputs);
           } else if (FloatType::get() == lt->getElementType()) {
             return listConstruct<double>(num_inputs);
           } else if (lt->getElementType() == BoolType::get()) {
             return listConstruct<bool>(num_inputs);
           } else if (lt->getElementType()->isSubtypeOf(TensorType::get())) {
             return [=](Stack& stack) {
               const size_t stack_size = stack.size();
               c10::List<at::Tensor> vals;
               vals.reserve(num_inputs);
               for (size_t i = stack_size - num_inputs; i < stack_size; ++i) {
                 vals.emplace_back(std::move(stack[i]).toTensor());
               }
               drop(stack, num_inputs);
               push(stack, std::move(vals));
               return 0;
             };
           } else {
             TypePtr elementType = lt->getElementType();
             return [=](Stack& stack) {
               const size_t stack_size = stack.size();
               auto vals = c10::impl::GenericList(elementType);
               vals.reserve(num_inputs);
               for (size_t i = stack_size - num_inputs; i < stack_size; ++i) {
                 vals.emplace_back(std::move(stack[i]));
               }
               drop(stack, num_inputs);
               push(stack, std::move(vals));
               return 0;
             };
           }
         },
         aliasAnalysisSpecialCase()),
     Operator(
         prim::DictConstruct,
         [](const Node* node) -> Operation {
           const auto num_inputs = node->inputs().size();
           if (num_inputs % 2 != 0) {
             throw std::runtime_error(
                 "DictConstruct must have an even number of inputs");
           }
           TORCH_INTERNAL_ASSERT(
               node->outputs().size() == 1,
               "DictConstruct must have exactly one output");
           TypePtr output_type = node->outputs()[0]->type();
           auto dt = output_type->expect<DictType>();
           TypePtr key_type = dt->getKeyType();
           TypePtr value_type = dt->getValueType();
           return [=](Stack& stack) {
             auto vals = c10::impl::GenericDict(key_type, value_type);
             vals.reserve(num_inputs / 2);
             for (size_t i = 0; i < num_inputs; i += 2) {
               auto val = pop(stack);
               auto key = pop(stack);
               vals.insert_or_assign(std::move(key), std::move(val));
             }
             push(stack, std::move(vals));
             return 0;
           };
         },
         aliasAnalysisSpecialCase()),
     Operator(
         "aten::dict() -> Dict(str, Tensor)",
         [](const Node* node) -> Operation {
           return [](Stack& stack) {
             auto dict =
                 c10::impl::GenericDict(StringType::get(), TensorType::get());
             push(stack, dict);
             return 0;
           };
         }),
     Operator(
         "aten::_unwrap_optional(t(a)? optional) -> t(a)",
         [](Stack& stack) {
           auto val = pop(stack);
           TORCH_CHECK(!val.isNone(), "Unwrapping null optional");
           push(stack, std::move(val));
           return 0;
         },
         aliasAnalysisFromSchema()),
     // This op can be removed in preprocessing before being run in the
     // interpreter (but is currently not removed), even when it is removed it
     // needs to remain a registered op so that constant prop can run.
     Operator(
         "prim::unchecked_unwrap_optional(t(a)? optional) -> t(a)",
         noop,
         aliasAnalysisFromSchema()),
     Operator(
         prim::fork,
         [](const Node* node) -> Operation {
           Code code(node->g(attr::Subgraph));
           int n_inputs = node->inputs().size();
           AT_ASSERT(node->blocks().size() == 0);
           AT_ASSERT(node->hasAttribute(attr::Subgraph));
           return [=](Stack& stack) {
             // Move inputs to a separate stack
             InterpreterState forked_interprester(code);
             InterpreterContinuation continuation(
                 forked_interprester,
                 Stack(stack.end() - n_inputs, stack.end()),
                 autograd::GradMode::is_enabled());
             drop(stack, n_inputs);

             push(stack, forked_interprester.getFuture());

             at::launch(std::move(continuation));
             return 0;
           };
         },
         aliasAnalysisSpecialCase()),
     Operator(
         "aten::wait(Future(t) self) -> t",
         [](Stack& stack) {
           TORCH_CHECK(
               false, "wait is implemented directly in the interpreter");
           return 0;
         },
         aliasAnalysisSpecialCase()),
     Operator(
         prim::Uninitialized,
         [](const Node* node) -> Operation {
           return [](Stack& stack) {
             push(stack, IValue::uninitialized());
             return 0;
           };
         },
         aliasAnalysisSpecialCase()),
     Operator(
         prim::CreateObject,
         [](const Node* node) -> Operation {
           const auto type = node->output()->type()->expect<ClassType>();
           const size_t numAttrs = type->numAttributes();
           auto cu = type->compilation_unit();
           return [cu, type, numAttrs](Stack& stack) {
             auto userObj = c10::ivalue::Object::create(
                 c10::StrongTypePtr(cu, type), numAttrs);
             push(stack, std::move(userObj));
             return 0;
           };
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

// define implementations for primitive number ops
#define DEFINE_GENERIC_OP(aten_op, int_op, float_op, int_result, float_result) \
  Operator(                                                                    \
      #aten_op "(int a, int b) -> " #int_result,                               \
      [](Stack& stack) {                                                       \
        int64_t a, b;                                                          \
        pop(stack, a, b);                                                      \
        push(stack, int_op);                                                   \
        return 0;                                                              \
      },                                                                       \
      aliasAnalysisFromSchema()),                                              \
      Operator(                                                                \
          #aten_op "(float a, float b) -> " #float_result,                     \
          [](Stack& stack) {                                                   \
            double a, b;                                                       \
            pop(stack, a, b);                                                  \
            push(stack, float_op);                                             \
            return 0;                                                          \
          },                                                                   \
          aliasAnalysisFromSchema())

#define DEFINE_INT_FLOAT_OP(aten_op, op, result)   \
  Operator(                                        \
      #aten_op "(int a, float b) -> " #result,     \
      [](Stack& stack) {                           \
        int64_t a;                                 \
        double b;                                  \
        pop(stack, a, b);                          \
        push(stack, op);                           \
        return 0;                                  \
      },                                           \
      aliasAnalysisFromSchema()),                  \
      Operator(                                    \
          #aten_op "(float a, int b) -> " #result, \
          [](Stack& stack) {                       \
            double a;                              \
            int64_t b;                             \
            pop(stack, a, b);                      \
            push(stack, op);                       \
            return 0;                              \
          },                                       \
          aliasAnalysisFromSchema())

#define DEFINE_INT_OP(aten_op, op)                          \
  Operator(                                                 \
      #aten_op "(int a, int b) -> int",                     \
      [](Stack& stack) {                                    \
        int64_t a, b;                                       \
        pop(stack, a, b);                                   \
        push(stack, op); /* NOLINT(hicpp-signed-bitwise) */ \
        return 0;                                           \
      },                                                    \
      aliasAnalysisFromSchema())

#define DEFINE_STR_CMP_OP(aten_op, op)     \
  Operator(                                \
      #aten_op "(str a, str b) -> bool",   \
      [](Stack& stack) {                   \
        auto b = pop(stack).toStringRef(); \
        auto a = pop(stack).toStringRef(); \
        push(stack, op);                   \
        return 0;                          \
      },                                   \
      aliasAnalysisFromSchema())

// define a primitive op over Scalar operands.
// it's necessary to register this overload following
// int/float variations to avoid trapping Scalar args
// in unintended implicit conversions
#define DEFINE_SCALAR_BINARY_OP(aten_op, int_op, float_op, result) \
  Operator( \
    #aten_op "(Scalar a, Scalar b) -> " #result, \
    [](Stack& stack) { \
      IValue x, y; \
      pop(stack, x, y); \
      if (x.isDouble()) { \
        if (y.isDouble()) { \
          double a = x.toDouble(); \
          double b = y.toDouble(); \
          push(stack, float_op); \
        } else { \
          double a = x.toDouble(); \
          int64_t b = y.toInt(); \
          push(stack, float_op); \
        } \
      } else { \
        if (y.isDouble()) { \
          int64_t a = x.toInt(); \
          double b = y.toDouble(); \
          push(stack, float_op); \
        } else { \
          int64_t a = x.toInt(); \
          int64_t b = y.toInt(); \
          push(stack, int_op); \
        } \
      } \
      return 0; \
    }, \
    aliasAnalysisFromSchema() \
  )

#define DEFINE_BINARY_OP(aten_op, op)               \
  DEFINE_GENERIC_OP(aten_op, op, op, int, float),   \
  DEFINE_INT_FLOAT_OP(aten_op, op, float),          \
  DEFINE_SCALAR_BINARY_OP(aten_op, op, op, Scalar)

#define DEFINE_BINARY_FLOAT_OP(aten_op, op)         \
  DEFINE_GENERIC_OP(aten_op, op, op, float, float), \
  DEFINE_INT_FLOAT_OP(aten_op, op, float),          \
  DEFINE_SCALAR_BINARY_OP(aten_op, op, op, float)

#define DEFINE_COMPARISON_OP(aten_op, op)         \
  DEFINE_GENERIC_OP(aten_op, op, op, bool, bool), \
  DEFINE_INT_FLOAT_OP(aten_op, op, bool),         \
  DEFINE_SCALAR_BINARY_OP(aten_op, op, op, bool), \
  DEFINE_STR_CMP_OP(aten_op, op)

#define DEFINE_UNARY_INT_OP(aten_op, op, result) \
  Operator(                                      \
      #aten_op "(int a) -> " #result,            \
      [](Stack& stack) {                         \
        int64_t a;                               \
        pop(stack, a);                           \
        push(stack, op);                         \
        return 0;                                \
      },                                         \
      aliasAnalysisFromSchema())

#define DEFINE_UNARY_FLOAT_OP(aten_op, op, result) \
  Operator(                                        \
      #aten_op "(float a) -> " #result,            \
      [](Stack& stack) {                           \
        double a;                                  \
        pop(stack, a);                             \
        push(stack, op);                           \
        return 0;                                  \
      },                                           \
      aliasAnalysisFromSchema())

#define DEFINE_UNARY_OP(aten_op, op, int_result, float_result) \
  DEFINE_UNARY_INT_OP(aten_op, op, int_result),                \
  DEFINE_UNARY_FLOAT_OP(aten_op, op, float_result),            \
  Operator( \
  #aten_op "(Scalar a) -> Scalar", \
  [](Stack& stack) { \
    IValue x; \
    pop(stack, x); \
    if (x.isDouble()) { \
      double a = x.toDouble(); \
      push(stack, static_cast<float_result>(op)); \
    } else { \
      int64_t a = x.toInt(); \
      push(stack, static_cast<int_result>(op)); \
    } \
    return 0; \
  }, \
  aliasAnalysisFromSchema() \
)
#define DEFINE_BOOL_OP(aten_op, op)        \
  Operator(                                \
      #aten_op "(bool a, bool b) -> bool", \
      [](Stack& stack) {                   \
        bool a, b;                         \
        pop(stack, a, b);                  \
        push(stack, op);                   \
        return 0;                          \
      },                                   \
      aliasAnalysisFromSchema())

// Equivalent to list.at(idx)
template <typename T>
T getItem(const c10::List<T>& list, int64_t idx) {
  const int64_t list_size = list.size();
  const int64_t normalized_idx = normalizeIndex(idx, list_size);
  if (normalized_idx < 0 || normalized_idx >= list_size) {
    throw std::out_of_range("list index out of range");
  }
  return list.get(normalized_idx);
}

template <typename T>
void setItem(const c10::List<T>& list, int64_t idx, T&& value) {
  const int64_t list_size = list.size();
  const int64_t normalized_idx = normalizeIndex(idx, list_size);
  if (normalized_idx < 0 || normalized_idx >= list_size) {
    throw std::out_of_range("list index out of range");
  }
  list.set(normalized_idx, std::move(value));
}

template <typename T>
int listAppend(Stack& stack) {
  T el = pop(stack).to<T>();
  c10::List<T> list = pop(stack).to<c10::List<T>>();

  list.push_back(std::move(el));
  push(stack, std::move(list));

  return 0;
}

template <typename T>
int listReverse(Stack& stack) {
  c10::List<T> list = pop(stack).to<c10::List<T>>();

  std::reverse(list.begin(), list.end());

  return 0;
}

template <typename T> int minList(Stack &stack) {
  c10::List<T> a = pop(stack).to<c10::List<T>>();
  c10::List<T> b = pop(stack).to<c10::List<T>>();

  size_t min_size = std::min(a.size(), b.size());
  for (size_t i = 0; i < min_size; i++) {
    if (a[i] == b[i]) {
      continue;
    }

    push(stack, a[i] < b[i] ? a : b);
    return 0;
  }

  push(stack, b.size() < a.size() ? b : a);
  return 0;
}

template <typename T> int maxList(Stack &stack) {
  c10::List<T> a = pop(stack).to<c10::List<T>>();
  c10::List<T> b = pop(stack).to<c10::List<T>>();

  size_t min_size = std::min(a.size(), b.size());
  for (size_t i = 0; i < min_size; i++) {
    if (a[i] == b[i]) {
      continue;
    }

    push(stack, a[i] > b[i] ? a : b);
    return 0;
  }

  push(stack, b.size() > a.size() ? b : a);
  return 0;
}

template <typename T>
int listPop(Stack& stack) {
  int64_t idx = pop(stack).to<int64_t>();
  c10::List<T> list = pop(stack).to<c10::List<T>>();

  const int64_t list_size = list.size();
  const int64_t normalized_idx = normalizeIndex(idx, list_size);

  if (list_size == 0) {
    AT_ERROR("pop from empty list");
  }

  push(stack, getItem(list, idx));
  list.erase(list.begin() + normalized_idx);

  return 0;
}

template <typename T>
int listClear(Stack& stack) {
  c10::List<T> list = pop(stack).to<c10::List<T>>();

  list.clear();
  return 0;
}

template <typename T>
int listInsert(Stack& stack) {
  T elem = pop(stack).to<T>();
  int64_t idx = pop(stack).to<int64_t>();
  c10::List<T> list = pop(stack).to<c10::List<T>>();

  const int64_t list_size = list.size();
  const int64_t normalized_idx = normalizeIndex(idx, list_size);

  if (normalized_idx < 0 || normalized_idx >= list_size) {
    if (normalized_idx < 0) {
      list.insert(list.begin(), elem);
    } else {
      list.push_back(elem);
    }
  } else {
    list.insert(list.begin() + normalized_idx, elem);
  }

  return 0;
}

template <typename T>
int listRemove(Stack& stack) {
  T elem = pop(stack).to<T>();
  c10::List<T> list = pop(stack).to<c10::List<T>>();

  auto pos = std::find(list.begin(), list.end(), elem);

  if (pos != list.end()) {
    list.erase(pos);
  } else {
    AT_ERROR("list.remove(x): x not in list");
  }

  return 0;
}

template <typename T>
int listMin(Stack& stack) {
  c10::List<T> list = pop(stack).to<c10::List<T>>();
  size_t list_size = list.size();
  if (list_size == 0) {
    throw std::runtime_error("min() arg is an empty sequence");
  }

  T min_elem = list[0];
  for (size_t i = 1; i < list_size; ++i) {
    T elem = list[i];
    min_elem = elem < min_elem ? elem : min_elem;
  }

  stack.push_back(min_elem);
  return 0;
}

template <typename T>
int listMax(Stack& stack) {
  c10::List<T> list = pop(stack).to<c10::List<T>>();
  size_t list_size = list.size();
  if (list_size == 0) {
    throw std::runtime_error("max() arg is an empty sequence");
  }

  T max_elem = list[0];
  for (size_t i = 1; i < list_size; ++i) {
    T elem = list[i];
    max_elem = elem > max_elem ? elem : max_elem;
  }

  stack.push_back(max_elem);
  return 0;
}

template <>
int listRemove<at::Tensor>(Stack& stack) {
  at::Tensor elem = pop(stack).to<at::Tensor>();
  c10::List<at::Tensor> list = pop(stack).to<c10::List<at::Tensor>>();

  auto pos = std::find_if(
      list.begin(), list.end(), [&](const at::Tensor& b) {
        const auto cmp_result = elem.eq(b);
        return cmp_result.is_nonzero();
      });

  if (pos != list.end()) {
    list.erase(pos);
  } else {
    AT_ERROR("list.remove(x): x not in list");
  }

  return 0;
}

template <typename T>
int listIndex(Stack& stack) {
  T elem = pop(stack).to<T>();
  c10::List<T> list = pop(stack).to<c10::List<T>>();

  auto pos = std::find(list.begin(), list.end(), elem);

  if (pos != list.end()) {
    push(stack, static_cast<int64_t>(std::distance(list.begin(), pos)));
  } else {
    AT_ERROR("'", elem, "' is not in list");
  }

  return 0;
}

template <>
int listIndex<at::Tensor>(Stack& stack) {
  at::Tensor elem = pop(stack).to<at::Tensor>();
  c10::List<at::Tensor> list = pop(stack).to<c10::List<at::Tensor>>();

  auto pos = std::find_if(
      list.begin(), list.end(), [elem](const at::Tensor& b) {
        const auto cmp_result = elem.eq(b);
        return cmp_result.is_nonzero();
      });

  if (pos != list.end()) {
    push(stack, static_cast<int64_t>(std::distance(list.begin(), pos)));
  } else {
    AT_ERROR("'", elem, "' is not in list");
  }

  return 0;
}

template <typename T>
int listCount(Stack& stack) {
  T elem = pop(stack).to<T>();
  c10::List<T> list = pop(stack).to<c10::List<T>>();

  const int64_t count = std::count(list.begin(), list.end(), elem);
  push(stack, count);

  return 0;
}

template <>
int listCount<at::Tensor>(Stack& stack) {
  at::Tensor elem = pop(stack).to<at::Tensor>();
  c10::List<at::Tensor> list = pop(stack).to<c10::List<at::Tensor>>();

  const int64_t count = std::count_if(
      list.begin(), list.end(), [&](const at::Tensor& b) {
        const auto cmp_result = elem.eq(b);
        return cmp_result.is_nonzero();
      });
  push(stack, count);

  return 0;
}

template <typename T>
Operation listExtend(const Node* node) {
  return [](Stack& stack) {
    c10::List<T> b = pop(stack).to<c10::List<T>>();
    c10::List<T> a = pop(stack).to<c10::List<T>>();

    a.reserve(a.size() + b.size());
    for (size_t i = 0; i < b.size(); ++i) {
      a.push_back(b.get(i));
    }
    return 0;
  };
}

template <typename T>
Operation listCopy(const Node* node) {
  return [](Stack& stack) {
    c10::List<T> list = pop(stack).to<c10::List<T>>();
    push(stack, list.copy());
    return 0;
  };
}

template <typename T>
int listSelect(Stack& stack) {
  int64_t idx = pop(stack).to<int64_t>();
  c10::List<T> list = pop(stack).to<c10::List<T>>();

  auto element = getItem(list, idx);
  push(stack, std::move(element));
  return 0;
}

template <typename T>
int listLen(Stack& stack) {
  c10::List<T> a = pop(stack).to<c10::List<T>>();

  const int64_t size = a.size();
  push(stack, size);
  return 0;
}

template <typename T>
int listEq(Stack& stack) {
  c10::List<T> b = pop(stack).to<c10::List<T>>();
  c10::List<T> a = pop(stack).to<c10::List<T>>();
  push(stack, list_is_equal(a, b));
  return 0;
}

template <typename T>
int listNe(Stack& stack) {
  c10::List<T> b = pop(stack).to<c10::List<T>>();
  c10::List<T> a = pop(stack).to<c10::List<T>>();
  push(stack, !list_is_equal(a, b));
  return 0;
}

inline bool tensor_list_equal(const c10::List<at::Tensor>& a, const c10::List<at::Tensor>& b) {
  if (a.size() != b.size()) {
    return false;
  }

  for (size_t i = 0; i < a.size(); ++i) {
    at::Tensor a_element = a[i];
    at::Tensor b_element = b[i];
    // This preserves Python's semantics, which uses eq() to compare two
    // elements, then passes the result to bool().
    // see: https://docs.python.org/3.4/reference/datamodel.html#object.__ge__
    const auto cmp_result = a_element.eq(b_element);
    if (!cmp_result.is_nonzero()) {
      return false;
    }
  }

  return true;
}

// Specialization for at::Tensor, since it doesn't define operator==
template <>
int listEq<at::Tensor>(Stack& stack) {
  c10::List<at::Tensor> b = pop(stack).to<c10::List<at::Tensor>>();
  c10::List<at::Tensor> a = pop(stack).to<c10::List<at::Tensor>>();
  push(stack, tensor_list_equal(a, b));
  return 0;
}

// Specialization for at::Tensor, since it doesn't define operator==
template <>
int listNe<at::Tensor>(Stack& stack) {
  c10::List<at::Tensor> b = pop(stack).to<c10::List<at::Tensor>>();
  c10::List<at::Tensor> a = pop(stack).to<c10::List<at::Tensor>>();
  push(stack, !tensor_list_equal(a, b));
  return 0;
}

template <typename T>
int listList(Stack& stack) {
  c10::List<T> a = pop(stack).to<c10::List<T>>();
  push(stack, a.copy());
  return 0;
}

template <typename T>
int listContains(Stack& stack) {
  auto key = pop(stack).to<T>();
  auto list = pop(stack).to<c10::List<T>>();
  for (const T& item : list) {
    if (item == key) {
      push(stack, true);
      return 0;
    }
  }
  push(stack, false);
  return 0;
}

template <class T>
int listAdd(Stack& stack) {
  c10::List<T> b = pop(stack).to<c10::List<T>>();
  c10::List<T> a = pop(stack).to<c10::List<T>>();

  c10::List<T> ret = make_result_list<T>(a.elementType());

  if (a.use_count() == 1) {
    ret = std::move(a);
  } else {
    ret = a.copy();
  }

  ret.append(std::move(b));

  push(stack, std::move(ret));
  return 0;
}

template <class T>
int listInplaceAdd(Stack& stack) {
  c10::List<T> b = pop(stack).to<List<T>>();
  c10::List<T> a = pop(stack).to<List<T>>();
  a.append(std::move(b));
  push(stack, std::move(a));
  return 0;
}

template <class T>
int listMulIntLeft(Stack& stack) {
  int64_t n = pop(stack).to<int64_t>();
  c10::List<T> list = pop(stack).to<c10::List<T>>();

  c10::List<T> ret = make_result_list<T>(list.elementType());
  const auto size = list.size() * n;
  ret.reserve(size);

  for (auto i = 0; i < n; i++) {
    for (T e : list) {
      ret.push_back(std::move(e));
    }
  }

  push(stack, std::move(ret));
  return 0;
}

template <class T>
int listMulIntRight(Stack& stack) {
  c10::List<T> list = pop(stack).to<c10::List<T>>();
  int64_t n = pop(stack).to<int64_t>();

  c10::List<T> ret = make_result_list<T>(list.elementType());
  const auto size = list.size() * n;
  ret.reserve(size);

  for (auto i = 0; i < n; i++) {
    for (T e : list) {
      ret.push_back(std::move(e));
    }
  }

  push(stack, std::move(ret));
  return 0;
}

template <typename T>
int listSlice(Stack& stack) {
  int64_t step = pop(stack).to<int64_t>();
  int64_t end = pop(stack).to<int64_t>();
  int64_t start = pop(stack).to<int64_t>();
  c10::List<T> list = pop(stack).to<c10::List<T>>();

  const int64_t list_size = list.size();

  // clamp start and end to the bounds of the list
  const auto normalized_start =
      std::max((int64_t)0, normalizeIndex(start, list_size));
  const auto normalized_end =
      std::min(list_size, normalizeIndex(end, list_size));

  c10::List<T> sliced_list = make_result_list<T>(list.elementType());
  if (normalized_end <= normalized_start) {
    // early exit if the slice is trivially empty
    push(stack, std::move(sliced_list));
    return 0;
  }

  sliced_list.reserve(normalized_end - normalized_start);

  for (auto i = normalized_start; i < normalized_end;) {
    sliced_list.push_back(list.get(i));
    i += step;
  }

  push(stack, std::move(sliced_list));
  return 0;
}

template <typename T>
int listSort(Stack& stack) {
  bool reverse = pop(stack).toBool();
  c10::List<T> list = pop(stack).to<c10::List<T>>();
  std::sort(list.begin(), list.end(), [reverse](const T& a, const T& b) {
    // FBCode errors without this check - "strict weak ordering"
    // TODO: remove when possible, since it just slows down
    // sorting and doesn't do anything useful
    if (a == b) {
      return false;
    }
    return (a < b) != reverse;
  });
  return 0;
}

// Specialization for at::Tensor
template <>
int listSort<at::Tensor>(Stack& stack) {
  bool reverse = pop(stack).toBool();
  c10::List<at::Tensor> list = pop(stack).toTensorList();
  std::sort(
      list.begin(),
      list.end(),
      [reverse](const at::Tensor& a, const at::Tensor& b) -> bool {
        // "strict weak ordering" issue - see other sort
        if (a.getIntrusivePtr() == b.getIntrusivePtr()) {
          return false;
        }
        return (a.lt(b).is_nonzero()) ^ reverse;
      });
  return 0;
}

template <typename T>
int listCopyAndSort(Stack& stack) {
  c10::List<T> list = pop(stack).to<c10::List<T>>();
  auto list_copied = list.copy();
  std::sort(list_copied.begin(), list_copied.end(), [](const T& a, const T& b) {
    // "strict weak ordering" issue - see other sort
    if (a == b) {
      return false;
    }
    return a < b;
  });
  push(stack, list_copied);
  return 0;
}

// Specialization for at::Tensor
template <>
int listCopyAndSort<at::Tensor>(Stack& stack) {
  c10::List<at::Tensor> list = pop(stack).toTensorList();
  std::sort(
      list.begin(),
      list.end(),
      [](const at::Tensor& a, const at::Tensor& b) {
        return a.lt(b).is_nonzero();
      });
  push(stack, list);
  return 0;
}

template <typename T>
int listSetItem(Stack& stack) {
  T value = pop(stack).to<T>();
  int64_t idx = pop(stack).to<int64_t>();
  c10::List<T> list = pop(stack).to<c10::List<T>>();

  setItem(list, idx, std::move(value));

  push(stack, std::move(list));
  return 0;
}

int dictSetItem(Stack& stack) {
  auto value = pop(stack);
  auto idx = pop(stack);
  auto dict = pop(stack).toGenericDict();
  dict.insert_or_assign(std::move(idx), std::move(value));
  return 0;
}

int dictLen(Stack& stack) {
  auto dict = pop(stack).toGenericDict();
  push(stack, int64_t(dict.size()));
  return 0;
}

template <unsigned int Index, typename Elem>
c10::List<Elem> makeListForDictKeysOrValues(
    const std::pair<c10::optional<TypePtr>, c10::optional<TypePtr>>& types,
    const std::vector<std::pair<IValue, IValue>>& order) {
  TORCH_INTERNAL_ASSERT(
      (!std::get<Index>(types).has_value())
      || (*std::get<Index>(types) == getTypePtr<Elem>()),
      "Type mismatch when trying to get a List of keys/values from Dict. ",
      "Type in Dict is ", toString(*std::get<Index>(types)),
      ". Type in List is ", toString(getTypePtr<Elem>()),
      ". Index is ", c10::guts::to_string(Index));
  c10::List<Elem> values;
  values.reserve(order.size());
  for (const auto& item : order) {
    values.push_back(std::get<Index>(item).template to<Elem>());
  }
  return values;
}

template <unsigned int Index>
c10::impl::GenericList makeGenericListForDictKeysOrValues(
    const std::pair<TypePtr, TypePtr>& types,
    const std::vector<std::pair<IValue, IValue>>& order) {
  auto type = std::get<Index>(types);
  auto values = c10::impl::GenericList(type);
  values.reserve(order.size());
  for (const auto& item : order) {
    values.push_back(std::get<Index>(item));
  }
  return values;
}

template <unsigned int Index>
Operation dictKeysOrValues(const Node* n) {
  auto outputType = n->output()->type()->expect<ListType>();
  return [=](Stack& stack) -> int {
    auto dict = pop(stack).toGenericDict();
    const auto& order = iterationOrder(dict);
    const auto types = std::make_pair(dict.keyType(), dict.valueType());
    if (outputType->getElementType()->isSubtypeOf(TensorType::get())) {
      push(stack, makeListForDictKeysOrValues<Index, at::Tensor>(types, order));
    } else if (outputType->getElementType() == IntType::get()) {
      push(stack, makeListForDictKeysOrValues<Index, int64_t>(types, order));
    } else if (outputType->getElementType() == FloatType::get()) {
      push(stack, makeListForDictKeysOrValues<Index, double>(types, order));
    } else if (outputType->getElementType() == BoolType::get()) {
      push(stack, makeListForDictKeysOrValues<Index, bool>(types, order));
    } else {
      push(stack, makeGenericListForDictKeysOrValues<Index>(types, order));
    }
    return 0;
  };
}

Operation dictKeys(const Node* n) {
  // getting first dict pair
  return dictKeysOrValues<0>(n);
}

Operation dictValues(const Node* n) {
  // getting second of dict pair
  return dictKeysOrValues<1>(n);
}

int dictIndex(Stack& stack) {
  auto key = pop(stack);
  auto dict = pop(stack).toGenericDict();
  auto value = dict.find(key);
  if (value == dict.end()) {
    AT_ERROR("KeyError: ", key);
  }
  push(stack, value->value());
  return 0;
}

template<bool has_default>
int dictGet(Stack& stack) {
  IValue default_value;
  if (has_default) {
    default_value = pop(stack);
  }
  auto key = pop(stack);
  auto dict = pop(stack).toGenericDict();
  auto value = dict.find(key);
  if (value == dict.end()) {
    push(stack, std::move(default_value));
  } else {
    push(stack, value->value());
  }
  return 0;
}

// If the key is in the dict, return it. Else set it to the default value and
// return that.
int dictSetDefault(Stack& stack) {
  auto default_value = pop(stack);
  auto key = pop(stack);
  auto dict = pop(stack).toGenericDict();
  auto value = dict.find(key);
  if (value == dict.end()) {
    dict.insert(key, default_value);
    push(stack, std::move(default_value));
  } else {
    push(stack, value->value());
  }
  return 0;
}

template<bool has_default>
int dictPop(Stack& stack) {
  IValue default_value;
  if (has_default) {
    default_value = pop(stack);
  }
  auto key = pop(stack);
  auto dict = pop(stack).toGenericDict();
  auto iter = dict.find(key);
  if (iter == dict.end()) {
    if (has_default) {
      push(stack, default_value);
    } else {
      AT_ERROR("KeyError: ", key);
    }
  } else {
    // note: before erase
    push(stack, iter->value());
    auto erase_count = dict.erase(key);
    TORCH_CHECK(
        erase_count == 1, "Expected to erase 1 item, found ", erase_count);
  }
  return 0;
}

int dictPopItem(Stack& stack) {
  auto dict = pop(stack).toGenericDict();
  if (dict.size() == 0) {
    AT_ERROR("popitem(): dictionary is empty");
  }
  auto item = iterationOrder(dict).at(0);
  auto erase_count = dict.erase(item.first);
  TORCH_CHECK(
      erase_count == 1, "Expected to erase 1 item, found ", erase_count);

  IValue tuple = c10::ivalue::Tuple::create({item.first, item.second});
  push(stack, tuple);
  return 0;
}

int dictContains(Stack& stack) {
  auto key = pop(stack);
  auto dict = pop(stack).toGenericDict();
  push(stack, dict.contains(key));
  return 0;
}

int dictClear(Stack& stack) {
  auto dict = pop(stack).toGenericDict();
  dict.clear();
  return 0;
}

int dictUpdate(Stack& stack) {
  auto to_add = pop(stack).toGenericDict();
  auto dict = pop(stack).toGenericDict();

  for (const auto& item : to_add) {
    dict.insert(item.key(), item.value());
  }
  return 0;
}

int dictItems(Stack& stack) {
  auto dict = pop(stack).toGenericDict();
  auto key_type = dict.keyType();
  auto value_type = dict.valueType();
  auto items =
      c10::impl::GenericList(TupleType::create({key_type, value_type}));
  items.reserve(dict.size());
  for (const auto& item : iterationOrder(dict)) {
    items.emplace_back(c10::ivalue::Tuple::create({item.first, item.second}));
  }
  push(stack, std::move(items));
  return 0;
}

int dictCopy(Stack& stack) {
  push(stack, pop(stack).toGenericDict().copy());
  return 0;
}

Operation dictConstructFromList(const Node* node) {
  TypePtr output_type = node->outputs()[0]->type();
  TypePtr key_type =
      static_cast<const DictType*>(output_type.get())->getKeyType();
  TypePtr value_type =
      static_cast<const DictType*>(output_type.get())->getValueType();
  return [key_type, value_type](Stack& stack) {
    auto input_list = pop(stack);
    auto list_ref = input_list.toGenericListRef();
    auto dict = c10::impl::GenericDict(key_type, value_type);
    dict.reserve(list_ref.size());
    for (const auto& input : list_ref) {
      const auto tup = input.toTuple()->elements();
      dict.insert_or_assign(tup[0], tup[1]);
    }
    push(stack, dict);
    return 0;
  };
}

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
        "aten::len(str s) -> int",
        [](Stack& stack) {
          auto string = pop(stack).toStringRef();
          push(stack, static_cast<int64_t>(string.size()));
          return 0;
        },
        aliasAnalysisFromSchema()),
    // tensor length op (size of 1st dimension)
    Operator(
        "aten::len(Tensor t) -> int",
        [](Stack& stack) {
          at::Tensor t = pop(stack).toTensor();
          if (t.dim() == 0) {
            AT_ERROR("len() of a 0-d tensor");
          }
          push(stack, t.sizes()[0]);
          return 0;
        },
        aliasAnalysisFromSchema()),
    Operator(
        "aten::__getitem__(str s, int index) -> str",
        [](Stack& stack) {
          auto index = pop(stack).toInt();
          auto string = pop(stack).toStringRef();
          char c = string.at(index);
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
// Mutable ops for lists containing mutable types.
#define CREATE_MUTABLE_LIST_OPS(decl_type, value_type)                        \
  Operator(                                                                   \
      "aten::select(" decl_type "[](a) list, int idx) -> " decl_type "(*)",   \
      listSelect<value_type>,                                                 \
      aliasAnalysisFromSchema()),                                             \
      Operator(                                                               \
          "aten::__getitem__(" decl_type "[](a) list, int idx) -> " decl_type \
          "(*)",                                                              \
          listSelect<value_type>,                                             \
          aliasAnalysisFromSchema()),                                         \
      Operator(                                                               \
          "aten::append( " decl_type "[](a!) self, " decl_type                \
          "(c -> *) el) -> " decl_type "[](a!)",                              \
          listAppend<value_type>,                                             \
          aliasAnalysisFromSchema()),                                         \
      Operator(                                                               \
          "aten::reverse( " decl_type "[](a!) self) -> ()",                   \
          listReverse<value_type>,                                            \
          aliasAnalysisFromSchema()),                                         \
      Operator(                                                               \
          "aten::extend(" decl_type "[](a!) self, " decl_type                 \
          " [] other) -> ()",                                                 \
          listExtend<value_type>,                                             \
          aliasAnalysisFromSchema()),                                         \
      Operator(                                                               \
          "aten::copy(" decl_type                                             \
          "[](a) self)"                                                       \
          " -> " decl_type "[]",                                              \
          listCopy<value_type>,                                               \
          aliasAnalysisFromSchema()),                                         \
      Operator(                                                               \
          "aten::_set_item(" decl_type "[](a!) l, int idx, " decl_type        \
          "(b -> *) el) -> " decl_type "[](a!)",                              \
          listSetItem<value_type>,                                            \
          aliasAnalysisFromSchema()),                                         \
      Operator(                                                               \
          "aten::clear( " decl_type "[](a!) self) -> ()",                     \
          listClear<value_type>,                                              \
          aliasAnalysisFromSchema()),                                         \
      Operator(                                                               \
          "aten::insert( " decl_type                                          \
          "[](a!) self, int idx,                                                    \
          " decl_type "(b -> *) el) -> ()",                                   \
          listInsert<value_type>,                                             \
          aliasAnalysisFromSchema()),                                         \
      Operator(                                                               \
          "aten::pop(" decl_type                                              \
          "[](a!) self, int idx=-1)                                                 \
        -> " decl_type "(*)",                                                 \
          listPop<value_type>,                                                \
          aliasAnalysisFromSchema())

    CREATE_MUTABLE_LIST_OPS("Tensor", at::Tensor),

    Operator(
        "aten::remove(Tensor[](a!) self, Tensor el) -> ()",
        listRemove<at::Tensor>,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::index(Tensor[] self, Tensor el) -> int",
        listIndex<at::Tensor>,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::count(Tensor[] self, Tensor el) -> int",
        listCount<at::Tensor>,
        aliasAnalysisFromSchema()),

// Mutable ops for lists containing immutable types.
#define CREATE_IMMUTABLE_LIST_OPS(decl_type, value_type)                       \
  Operator(                                                                    \
      "aten::select(" decl_type "[] a, int b) -> " decl_type,                  \
      listSelect<value_type>,                                                  \
      aliasAnalysisFromSchema()),                                              \
      Operator(                                                                \
          "aten::__getitem__(" decl_type "[](a) list, int idx) -> " decl_type, \
          listSelect<value_type>,                                              \
          aliasAnalysisFromSchema()),                                          \
      Operator(                                                                \
          "prim::min(" decl_type "[] l, " decl_type "[] r) -> " decl_type "[]",\
          minList<value_type>,                                                 \
          aliasAnalysisFromSchema()),                                          \
      Operator(                                                                \
          "prim::max(" decl_type "[] l, " decl_type "[] r) -> " decl_type "[]",\
          maxList<value_type>,                                                 \
          aliasAnalysisFromSchema()),                                          \
      Operator(                                                                \
          "aten::append(" decl_type "[](a!) self, " decl_type                  \
          " el) -> " decl_type "[](a!)",                                       \
          listAppend<value_type>,                                              \
          aliasAnalysisFromSchema()),                                          \
      Operator(                                                                \
          "aten::reverse(" decl_type "[](a!) self) -> ()",                     \
          listReverse<value_type>,                                             \
          aliasAnalysisFromSchema()),                                          \
      Operator(                                                                \
          "prim::min(" decl_type "[] self) -> " decl_type,                     \
          listMin<value_type>,                                                 \
          aliasAnalysisFromSchema()),                                          \
      Operator(                                                                \
          "prim::max(" decl_type "[] self) -> " decl_type,                     \
          listMax<value_type>,                                                 \
          aliasAnalysisFromSchema()),                                          \
      Operator(                                                                \
          "aten::extend(" decl_type "[](a!) self, " decl_type                  \
          " [] other) -> ()",                                                  \
          listExtend<value_type>,                                              \
          aliasAnalysisFromSchema()),                                          \
      Operator(                                                                \
          "aten::copy(" decl_type                                              \
          "[](a) self)"                                                        \
          " -> " decl_type "[]",                                               \
          listCopy<value_type>,                                                \
          aliasAnalysisFromSchema()),                                          \
      Operator(                                                                \
          "aten::_set_item(" decl_type "[](a!) l, int idx, " decl_type         \
          " el) -> " decl_type "[](a!)",                                       \
          listSetItem<value_type>,                                             \
          aliasAnalysisFromSchema()),                                          \
      Operator(                                                                \
          "aten::clear( " decl_type "[](a!) self) -> ()",                      \
          listClear<value_type>,                                               \
          aliasAnalysisFromSchema()),                                          \
      Operator(                                                                \
          "aten::insert( " decl_type                                           \
          "[](a!) self, int idx,                                                  \
          " decl_type " el) -> ()",                                            \
          listInsert<value_type>,                                              \
          aliasAnalysisFromSchema()),                                          \
      Operator(                                                                \
          "aten::remove(" decl_type                                            \
          "[](a!) self,                                                           \
          " decl_type " el) -> ()",                                            \
          listRemove<value_type>,                                              \
          aliasAnalysisFromSchema()),                                          \
      Operator(                                                                \
          "aten::index(" decl_type                                             \
          "[] self,                                                               \
          " decl_type " el) -> int",                                           \
          listIndex<value_type>,                                               \
          aliasAnalysisFromSchema()),                                          \
      Operator(                                                                \
          "aten::count(" decl_type                                             \
          "[] self,                                                               \
          " decl_type " el) -> int",                                           \
          listCount<value_type>,                                               \
          aliasAnalysisFromSchema()),                                          \
      Operator(                                                                \
          "aten::pop(" decl_type                                               \
          "[](a!) self, int idx=-1)                                               \
          -> " decl_type,                                                      \
          listPop<value_type>,                                                 \
          aliasAnalysisFromSchema())

    CREATE_IMMUTABLE_LIST_OPS("int", int64_t),
    CREATE_IMMUTABLE_LIST_OPS("float", double),
    CREATE_IMMUTABLE_LIST_OPS("bool", bool),

    // NOTE: this must be after the other list specializations so that operator
    // resolution doesn't pick this up first
    CREATE_MUTABLE_LIST_OPS("t", IValue),

    // TODO: remove once tests that rely on
    // https://github.com/pytorch/pytorch/issues/24856
    // behavior have been fixed
    Operator(
      "aten::append(str[](a!) self, str? el) -> str[](a!)",
      listAppend<std::string>,
      aliasAnalysisFromSchema()),

#undef CREATE_IMMUTABLE_LIST_OPS
#undef CREATE_MUTABLE_LIST_OPS

#define CREATE_LIST_OPS(decl_type, c_type)                                          \
  Operator(                                                                         \
      "aten::len(" decl_type "[] a) -> int",                                        \
      listLen<c_type::value_type>,                                                  \
      aliasAnalysisFromSchema()),                                                   \
      Operator(                                                                     \
          "aten::add(" decl_type "[] a, " decl_type "[] b) -> " decl_type           \
          "[]",                                                                     \
          listAdd<c_type::value_type>,                                              \
          aliasAnalysisFromSchema()),                                               \
      Operator(                                                                     \
          "aten::add_(" decl_type "[](a!) self, " decl_type                         \
          "[] b) -> " decl_type "[]",                                               \
          listInplaceAdd<c_type::value_type>,                                       \
          aliasAnalysisFromSchema()),                                               \
      Operator(                                                                     \
          "aten::slice(" decl_type                                                  \
          "[] l, int start, int end=9223372036854775807, int step=1) -> " decl_type \
          "[]",                                                                     \
          listSlice<c_type::value_type>,                                            \
          aliasAnalysisFromSchema()),                                               \
      Operator(                                                                     \
          "aten::list(" decl_type "[] l) -> " decl_type "[]",                       \
          listList<c_type::value_type>,                                             \
          aliasAnalysisFromSchema()),                                               \
      Operator(                                                                     \
          "aten::mul(" decl_type "[] l, int n) -> " decl_type "[]",                 \
          listMulIntLeft<c_type::value_type>,                                       \
          aliasAnalysisFromSchema()),                                               \
      Operator(                                                                     \
          "aten::mul(int n, " decl_type "[] l) -> " decl_type "[]",                 \
          listMulIntRight<c_type::value_type>,                                      \
          aliasAnalysisFromSchema())

    CREATE_LIST_OPS("int", c10::List<int64_t>),
    CREATE_LIST_OPS("float", c10::List<double>),
    CREATE_LIST_OPS("bool", c10::List<bool>),
    CREATE_LIST_OPS("Tensor", c10::List<at::Tensor>),
    CREATE_LIST_OPS("t", c10::List<IValue>),

    // `listContains<T>` is not implemented for non-primitive types
    // TODO: Add List[bool] once .to<c10::List<bool>> doesn't throw an error
    Operator(
        "aten::__contains__(int[] l, int item) -> bool",
        listContains<int64_t>,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::__contains__(float[] l, float item) -> bool",
        listContains<double>,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::__contains__(str[] l, str item) -> bool",
        listContains<std::string>,
        aliasAnalysisFromSchema()),
#undef CREATE_LIST_OPS
    Operator(
        "aten::sort(int[](a!) self, bool reverse=False) -> ()",
        listSort<int64_t>,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::sort(float[](a!) self, bool reverse=False) -> ()",
        listSort<double>,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::sort(Tensor[](a!) self, bool reverse=False) -> ()",
        listSort<at::Tensor>,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::sort(bool[](a!) self, bool reverse=False) -> ()",
        listSort<bool>,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::sorted(int[](a) input) -> (int[])",
        listCopyAndSort<int64_t>,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::sorted(float[](a) input) -> (float[])",
        listCopyAndSort<double>,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::sorted(Tensor[](a) input) -> (Tensor[])",
        listCopyAndSort<at::Tensor>,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::sorted(bool[](a) input) -> (bool[])",
        listCopyAndSort<bool>,
        aliasAnalysisFromSchema()),

    Operator(
        "aten::eq(int[] a, int[] b) -> bool",
        listEq<int64_t>,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::eq(float[] a, float[] b) -> bool",
        listEq<double>,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::eq(Tensor[] a, Tensor[] b) -> bool",
        listEq<at::Tensor>,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::eq(bool[] a, bool[] b) -> bool",
        listEq<bool>,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::ne(int[] a, int[] b) -> bool",
        listNe<int64_t>,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::ne(float[] a, float[] b) -> bool",
        listNe<double>,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::ne(Tensor[] a, Tensor[] b) -> bool",
        listNe<at::Tensor>,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::ne(bool[] a, bool[] b) -> bool",
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
          char c = string.at(index);
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

    DEFINE_BINARY_OP(aten::add, a + b),
    DEFINE_BINARY_OP(aten::sub, a - b),
    DEFINE_BINARY_OP(aten::mul, a * b),
    DEFINE_BINARY_OP(aten::pow, pow(a, b)),
    // min and max are in prim:: because there is a difference between
    // the python builtin 'min' and 'torch.min'
    DEFINE_BINARY_OP(prim::min, a < b ? a : b),
    DEFINE_BINARY_OP(prim::max, a > b ? a : b),

    // Pass in two ops for handling int and float separately as % in C++ only
    // works for int The modulus calculation is different between C++ and Python
    // (on negative), we preserve the python behavior as it's more common and
    // match python syntax, hence the conversion.
    DEFINE_GENERIC_OP(
        aten::remainder,
        (b + (a % b)) % b,
        fmod((b + fmod(a, b)), b),
        int,
        float),
    DEFINE_INT_FLOAT_OP(aten::remainder, fmod((b + fmod(a, b)), b), float),
    DEFINE_SCALAR_BINARY_OP(
        aten::remainder,
        (b + (a % b)) % b,
        fmod((b + fmod(a, b)), b),
        Scalar),

    DEFINE_GENERIC_OP(
        aten::floordiv,
        floordiv(a, b),
        std::floor(a / b),
        int,
        float),
    DEFINE_INT_FLOAT_OP(aten::floordiv, std::floor(a / b), float),
    DEFINE_SCALAR_BINARY_OP(
        aten::floordiv,
        floordiv(a, b),
        std::floor(a / b),
        Scalar),

    // NB: This is the python truediv operation
    DEFINE_GENERIC_OP(
        aten::div,
        static_cast<double>(a) / static_cast<double>(b),
        a / b,
        float,
        float),
    DEFINE_SCALAR_BINARY_OP(
        aten::div,
        static_cast<double>(a) / static_cast<double>(b),
        a / b,
        float),

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

    DEFINE_INT_OP(aten::__and__, a& b),
    DEFINE_INT_OP(aten::__or__, a | b),
    DEFINE_INT_OP(aten::__xor__, a ^ b),

    DEFINE_UNARY_OP(aten::floor, floor(a), int, int),
    DEFINE_UNARY_OP(aten::ceil, ceil(a), int, int),
    DEFINE_UNARY_OP(aten::round, std::round(a), float, float),
    DEFINE_UNARY_OP(aten::log, std::log(a), float, float),
    DEFINE_BINARY_FLOAT_OP(aten::log, std::log(a) / std::log(b)),
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

    DEFINE_UNARY_OP(aten::gamma, std::tgamma(a), float, float),
    DEFINE_UNARY_OP(aten::erf, std::erf(a), float, float),
    DEFINE_UNARY_OP(aten::erfc, std::erfc(a), float, float),
    DEFINE_UNARY_OP(aten::expm1, std::expm1(a), float, float),
    DEFINE_UNARY_OP(aten::fabs, std::fabs(a), float, float),
    DEFINE_UNARY_OP(aten::lgamma, std::lgamma(a), float, float),
    DEFINE_UNARY_OP(aten::asinh, std::asinh(a), float, float),
    DEFINE_UNARY_OP(aten::atanh, std::atanh(a), float, float),
    DEFINE_UNARY_OP(aten::cosh, std::cosh(a), float, float),
    DEFINE_UNARY_OP(aten::sinh, std::sinh(a), float, float),
    DEFINE_UNARY_OP(aten::tanh, std::tanh(a), float, float),

    Operator(
        "aten::isnan(float a) -> bool",
        [](Stack& stack) {
          double a;
          pop(stack, a);
          push(stack, std::isnan(a));
          return 0;
        },
        aliasAnalysisFromSchema()),

    DEFINE_COMPARISON_OP(aten::ne, a != b),
    DEFINE_COMPARISON_OP(aten::eq, a == b),
    DEFINE_COMPARISON_OP(aten::lt, a < b),
    DEFINE_COMPARISON_OP(aten::gt, a > b),
    DEFINE_COMPARISON_OP(aten::le, a <= b),
    DEFINE_COMPARISON_OP(aten::ge, a >= b),
    DEFINE_BOOL_OP(aten::__and__, a && b),
    DEFINE_BOOL_OP(aten::__or__, a || b),
    DEFINE_BOOL_OP(aten::__xor__, a != b),

    DEFINE_UNARY_OP(aten::neg, -a, int, float),
    Operator(
        "aten::__not__(bool self) -> bool",
        [](Stack& stack) {
          push(stack, !pop(stack).toBool());
          return 0;
        },
        aliasAnalysisFromSchema()),
    Operator(
        "aten::__is__(t1 self, t2 obj) -> bool",
        [](Stack& stack) {
          IValue self, obj;
          pop(stack, self, obj);
          push(stack, self.isSameIdentity(obj));
          return 0;
        },
        aliasAnalysisFromSchema()),
    Operator(
        "aten::__isnot__(t1 self, t2 obj) -> bool",
        [](Stack& stack) {
          IValue self, obj;
          pop(stack, self, obj);
          push(stack, !self.isSameIdentity(obj));
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
#define CREATE_DICT_OPS(key_type)                                             \
  Operator(                                                                   \
      "aten::len(Dict(" key_type ", t) self) -> int",                         \
      dictLen,                                                                \
      aliasAnalysisFromSchema()),                                             \
      Operator(                                                               \
          "aten::keys(Dict(" key_type ", t) self) -> " key_type "[](*)",      \
          dictKeys,                                                           \
          aliasAnalysisFromSchema()),                                         \
      Operator(                                                               \
          "aten::values(Dict(" key_type ", t) self) -> t[](*)",               \
          dictValues,                                                         \
          aliasAnalysisFromSchema()),                                         \
      Operator(                                                               \
          "aten::__getitem__(Dict(" key_type ", t) self, " key_type           \
          " key) -> t(*)",                                                    \
          dictIndex,                                                          \
          aliasAnalysisFromSchema()),                                         \
      Operator(                                                               \
          "aten::get(Dict(" key_type ", t) self, " key_type " key) -> t(*)?", \
          dictGet<false>,                                                     \
          aliasAnalysisFromSchema()),                                         \
      Operator(                                                               \
          "aten::get(Dict(" key_type ", t) self, " key_type                   \
          " key, t default_value) -> t(*)",                                   \
          dictGet<true>,                                                      \
          aliasAnalysisFromSchema()),                                         \
      Operator(                                                               \
          "aten::setdefault(Dict(" key_type ", t)(a!) self, " key_type        \
          " key, t default_value) -> t(*)",                                   \
          dictSetDefault,                                                     \
          aliasAnalysisFromSchema()),                                         \
      Operator(                                                               \
          "aten::pop(Dict(" key_type ", t)(a!) self, " key_type               \
          " key) -> t(*)",                                                    \
          dictPop<false>,                                                     \
          aliasAnalysisFromSchema()),                                         \
      Operator(                                                               \
          "aten::pop(Dict(" key_type ", t)(a!) self, " key_type               \
          " key, t default_value) -> t(*)",                                   \
          dictPop<true>,                                                      \
          aliasAnalysisFromSchema()),                                         \
      Operator(                                                               \
          "aten::popitem(Dict(" key_type ", t)(a!) self) -> ((" key_type      \
          ", t))",                                                            \
          dictPopItem,                                                        \
          aliasAnalysisFromSchema()),                                         \
      Operator(                                                               \
          "aten::clear(Dict(" key_type ", t)(a!) self) -> ()",                \
          dictClear,                                                          \
          aliasAnalysisFromSchema()),                                         \
      Operator(                                                               \
          "aten::update(Dict(" key_type ", t)(a!) self, Dict(" key_type       \
          ", t)(a!) to_add) -> ()",                                           \
          dictUpdate,                                                         \
          aliasAnalysisFromSchema()),                                         \
      Operator(                                                               \
          "aten::items(Dict(" key_type ", t) self) -> ((" key_type ", t)[])", \
          dictItems,                                                          \
          aliasAnalysisFromSchema()),                                         \
      Operator(                                                               \
          "aten::copy(Dict(" key_type ", t)(a) self) -> Dict(" key_type       \
          ", t)",                                                             \
          dictCopy,                                                           \
          aliasAnalysisFromSchema()),                                         \
      Operator(                                                               \
          "aten::__contains__(Dict(" key_type ", t) dict, " key_type          \
          " key) -> bool",                                                    \
          dictContains,                                                       \
          aliasAnalysisFromSchema()),                                         \
      Operator(                                                               \
          "aten::_set_item(Dict(" key_type ", t)(a!) l, " key_type            \
          " idx, t(b -> *) v) -> ()",                                         \
          dictSetItem,                                                        \
          aliasAnalysisFromSchema()),                                         \
      Operator(                                                               \
          "aten::dict((" key_type ", tVal)[] inputs) -> Dict(" key_type       \
          ", tVal)",                                                          \
          dictConstructFromList,                                              \
          aliasAnalysisFromSchema())

    CREATE_DICT_OPS("str"),
    CREATE_DICT_OPS("int"),
    CREATE_DICT_OPS("float"),
    CREATE_DICT_OPS("Tensor"),
#undef CREATE_DICT_OPS

    Operator(
        "aten::divmod(int x, int y) -> (int, int)",
        [](Stack& stack) {
          int64_t a, b;
          lldiv_t divresult = {};
          pop(stack, a, b);
          if (b == 0) {
            throw std::runtime_error("ZeroDivisionError: integer division or modulo by zero");
          }
          divresult = lldiv(a, b);
          if (divresult.rem && (a < 0) != (b < 0)) {
            divresult.quot -= 1;
            divresult.rem  += b;
          }
          push(stack, static_cast<int64_t>(divresult.quot), \
            static_cast<int64_t>(divresult.rem));
          return 0;
        },
        aliasAnalysisFromSchema()),
    Operator(
        "aten::divmod(float x, float y) -> (float, float)",
        [](Stack& stack) {
          double a, b;
          pop(stack, a, b);
          if (b == 0) {
            throw std::runtime_error("ZeroDivisionError: float divmod()");
          }
          double rem = fmod(a, b);
          if (rem && (a < 0) != (b < 0)) {
            rem  += b;
          }
          push(stack, (a - rem)/b, rem);
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

void checkSortSchema(const Node* node, const c10::TypePtr& list_element_type) {
  std::stringstream error_str;
  if (auto class_type = list_element_type->cast<ClassType>()) {
    if (auto method = class_type->getMethod("__lt__")) {
      const auto& lt_schema = method->getSchema();
      const auto& schema_args = lt_schema.arguments();
      bool error =
          (schema_args.size() != 2 ||
           !simpleClassTypeArg(schema_args[0], class_type) ||
           !simpleClassTypeArg(schema_args[1], class_type) ||
           lt_schema.returns().size() != 1 ||
           lt_schema.returns()[0].type() != BoolType::get());
      if (!error) {
        return;
      }
    }
    error_str << "To sort a list of " << class_type->python_str()
              << " it must define a "
              << "__lt__ method with two inputs of type "
              << class_type->python_str() << " that "
              << "returns a bool";
  } else {
    error_str << "Input to " << node->kind().toUnqualString()
              << "must be of Tensors, ints, floats, bools or "
              << "a User Defined Class that defines the __lt__ compare method"
              << ", got list of " << list_element_type->python_str() << "\n";
  }

  auto error_msg = script::ErrorReport(node->sourceRange());
  error_msg << error_str.str();
  throw error_msg;
}

Operation sort_op(
    Function* lt_func,
    bool has_reverse_arg,
    bool copy_return_list) {
  return [lt_func, has_reverse_arg, copy_return_list](Stack& stack) {
    bool reverse = has_reverse_arg ? pop(stack).toBool() : false;
    auto g_list = pop(stack).toGenericList();
    if (copy_return_list) {
      g_list = g_list.copy();
    }
    Stack sort_stack;
    std::sort(
        g_list.begin(),
        g_list.end(),
        [lt_func, reverse, &sort_stack](IValue a, IValue b) -> bool {
          // "strict weak ordering" issue - see other sort
          if (a.isSameIdentity(b)) {
            return false;
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
  };
}

Function* getLtFuncFromListOfClassTypes(const Node* node) {
  const auto list_type = node->inputs().at(0)->type()->expect<ListType>();
  checkSortSchema(node, list_type->getElementType());
  const auto elem = list_type->getElementType()->expect<ClassType>();
  return elem->getMethod("__lt__");
}

// NB: this must be registered after the other aten::sort operators
RegisterOperators regSort({
    Operator(
        "aten::sorted(t[](a) self) -> (t[])",
        [](const Node* node) -> Operation {
          return sort_op(
              getLtFuncFromListOfClassTypes(node),
              /*has_reverse_arg*/ false,
              /*copy_return_list*/ true);
        },
        aliasAnalysisFromSchema()),
    Operator(
        "aten::sort(t[](a!) self, bool reverse=False) -> ()",
        [](const Node* node) -> Operation {
          return sort_op(
              getLtFuncFromListOfClassTypes(node),
              /*has_reverse_arg*/ true,
              /*copy_return_list*/ false);
        },
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
      return size.toIntListRef().vec();
    }
  }
  std::vector<double> scale_repeated;
  if (scale_factors.isDouble()) {
    scale_repeated = std::vector<double>(dim, scale_factors.toDouble());
  } else {
    scale_repeated = scale_factors.toDoubleListRef().vec();
  }
  std::vector<int64_t> ret;
  for (size_t i = 0; i < dim; ++i) {
    ret.push_back(std::floor(input.size(i + 2) * scale_repeated[i]));
  }
  return ret;
}

// reference: interpolate in torch/nn/functional.py
// size can be none, int or intlist
// scale_factors can be none, float, or floatlist
at::Tensor interpolate(
    const at::Tensor& input,
    const IValue& size,
    const IValue& scale_factors,
    const std::string& mode,
    c10::optional<bool> align_corners) {
  if ((mode == "nearest" || mode == "area")) {
    if (align_corners != c10::nullopt) {
      throw std::runtime_error(
          "align_corners option can only be set with the "
          "interpolating modes: linear | bilinear | bicubic | trilinear");
    }
  } else {
    if (align_corners == c10::nullopt) {
      AT_WARN(
          "Default upsampling behavior when mode=",
          mode,
          " is changed "
          "to align_corners=False since 0.4.0. Please specify align_corners=True "
          "if the old behavior is desired. See the documentation of nn.Upsample for details");
      align_corners = false;
    }
  }

  auto input_dim = input.dim();
  if (input_dim == 3 && mode == "nearest")
    return at::upsample_nearest1d(
        input, _output_size(input, 1, size, scale_factors));
  if (input_dim == 4 && mode == "nearest")
    return at::upsample_nearest2d(
        input, _output_size(input, 2, size, scale_factors));
  if (input_dim == 5 && mode == "nearest")
    return at::upsample_nearest3d(
        input, _output_size(input, 3, size, scale_factors));
  if (input_dim == 3 && mode == "area")
    return at::adaptive_avg_pool1d(
        input, _output_size(input, 1, size, scale_factors));
  if (input_dim == 4 && mode == "area")
    return at::adaptive_avg_pool2d(
        input, _output_size(input, 2, size, scale_factors));
  if (input_dim == 5 && mode == "area")
    return at::adaptive_avg_pool3d(
        input, _output_size(input, 3, size, scale_factors));
  if (input_dim == 3 && mode == "linear")
    return at::upsample_linear1d(
        input, _output_size(input, 1, size, scale_factors), *align_corners);
  if (input_dim == 3 && mode == "bilinear")
    throw std::runtime_error("Got 3D input, but bilinear mode needs 4D input");
  if (input_dim == 3 && mode == "bicubic")
    throw std::runtime_error("Got 3D input, but bicubic mode needs 4D input");
  if (input_dim == 3 && mode == "trilinear")
    throw std::runtime_error("Got 3D input, but trilinear mode needs 5D input");
  if (input_dim == 4 && mode == "linear")
    throw std::runtime_error("Got 4D input, but linear mode needs 3D input");
  if (input_dim == 4 && mode == "bilinear")
    return at::upsample_bilinear2d(
        input, _output_size(input, 2, size, scale_factors), *align_corners);
  if (input_dim == 4 && mode == "bicubic")
    return at::upsample_bicubic2d(
        input, _output_size(input, 2, size, scale_factors), *align_corners);
  if (input_dim == 4 && mode == "trilinear")
    throw std::runtime_error("Got 4D input, but trilinear mode needs 5D input");
  if (input_dim == 5 && mode == "linear")
    throw std::runtime_error("Got 5D input, but linear mode needs 3D input");
  if (input_dim == 5 && mode == "bilinear")
    throw std::runtime_error("Got 5D input, but bilinear mode needs 4D input");
  if (input_dim == 5 && mode == "bicubic")
    throw std::runtime_error("Got 5D input, but bicubic mode needs 4D input");
  if (input_dim == 5 && mode == "trilinear")
    return at::upsample_trilinear3d(
        input, _output_size(input, 3, size, scale_factors), *align_corners);

  AT_ERROR(
      "Input Error: Only 3D, 4D and 5D input Tensors supported",
      " (got ",
      input_dim,
      "D) for the modes: nearest | linear | bilinear | trilinear",
      " (got ",
      mode,
      ") ");
}

Operation interpolate_op(const Node* n) {
  return [](Stack& stack) {
    at::Tensor input;
    IValue size;
    IValue scale_factors;
    std::string mode;
    IValue align_corners;
    pop(stack, input, size, scale_factors, mode, align_corners);
    at::Tensor res = interpolate(
        input, size, scale_factors, mode, align_corners.toOptional<bool>());
    push(stack, std::move(res));
    return 0;
  };
}

// interpolate takes in float & float[] for scale factor
// upsample takes in int & int[], so convert the ints to floats before
// passing on to the interpolate op
IValue convert_scale_factor_to_double(const IValue& int_ivalue) {
  IValue scale_factor_double;
  if (int_ivalue.isInt()) {
    scale_factor_double = static_cast<double>(int_ivalue.toInt());
  } else if (int_ivalue.isIntList()) {
    auto int_list = int_ivalue.toIntListRef();
    std::vector<double> double_vec(int_list.begin(), int_list.end());
    scale_factor_double = c10::impl::toList(double_vec);
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

Operation upsample_nearest_op(const Node* n) {
  return [](Stack& stack) {
    at::Tensor input;
    IValue size;
    IValue scale_factor_int;
    pop(stack, input, size, scale_factor_int);
    IValue scale_factor_double =
        convert_scale_factor_to_double(scale_factor_int);
    at::Tensor res =
        interpolate(input, size, scale_factor_double, "nearest", c10::nullopt);
    push(stack, std::move(res));
    return 0;
  };
}

Operation upsample_op(const Node* n) {
  return [](Stack& stack) {
    at::Tensor input;
    IValue size;
    IValue scale_factor_int;
    std::string mode;
    IValue align_corners;
    pop(stack, input, size, scale_factor_int, mode, align_corners);
    IValue scale_factor_double =
        convert_scale_factor_to_double(scale_factor_int);
    at::Tensor res = interpolate(
        input,
        size,
        scale_factor_double,
        mode,
        align_corners.toOptional<bool>());
    push(stack, std::move(res));
    return 0;
  };
}

Operation upsample_bilinear_op(const Node* n) {
  return [](Stack& stack) {
    at::Tensor input;
    IValue size;
    IValue scale_factor_int;
    pop(stack, input, size, scale_factor_int);
    IValue scale_factor_double =
        convert_scale_factor_to_double(scale_factor_int);
    at::Tensor res =
        interpolate(input, size, scale_factor_double, "bilinear", true);
    push(stack, std::move(res));
    return 0;
  };
}

RegisterOperators reg3({
    Operator(
        "aten::__interpolate(Tensor input, int? size = None, float[]? scale_factor = None, str mode = 'nearest', bool? align_corners = None) -> Tensor",
        interpolate_op,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::__interpolate(Tensor input, int[]? size = None, float[]? scale_factor = None, str mode = 'nearest', bool? align_corners = None) -> Tensor",
        interpolate_op,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::__interpolate(Tensor input, int? size = None, float? scale_factor = None, str mode = 'nearest', bool? align_corners = None) -> Tensor",
        interpolate_op,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::__interpolate(Tensor input, int[]? size = None, float? scale_factor = None, str mode = 'nearest', bool? align_corners = None) -> Tensor",
        interpolate_op,
        aliasAnalysisFromSchema()),

    Operator(
        "aten::__upsample_nearest(Tensor input, int? size = None, int? scale_factor = None) -> Tensor",
        upsample_nearest_op,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::__upsample_nearest(Tensor input, int[]? size = None, int? scale_factor = None) -> Tensor",
        upsample_nearest_op,
        aliasAnalysisFromSchema()),

    Operator(
        "aten::__upsample(Tensor input, int? size = None, int? scale_factor = None, str mode = 'nearest', bool? align_corners = None) -> Tensor",
        upsample_op,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::__upsample(Tensor input, int[]? size = None, int? scale_factor = None, str mode = 'nearest', bool? align_corners = None) -> Tensor",
        upsample_op,
        aliasAnalysisFromSchema()),

    Operator(
        "aten::__upsample_bilinear(Tensor input, int? size = None, int? scale_factor = None) -> Tensor",
        upsample_bilinear_op,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::__upsample_bilinear(Tensor input, int[]? size = None, int? scale_factor = None) -> Tensor",
        upsample_bilinear_op,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::__upsample_bilinear(Tensor input, int? size = None, int[]? scale_factor = None) -> Tensor",
        upsample_bilinear_op,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::__upsample_bilinear(Tensor input, int[]? size = None, int[]? scale_factor = None) -> Tensor",
        upsample_bilinear_op,
        aliasAnalysisFromSchema()),

});

at::Tensor leaky_relu(const at::Tensor& tensor, double scalar) {
  return at::leaky_relu(tensor, scalar);
}
at::Tensor cat(const c10::List<at::Tensor>& tensors) {
  return at::cat(c10::impl::toVector(tensors));
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
