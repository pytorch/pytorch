#include <aten/src/ATen/Context.h>

#include <ATen/core/jit_type.h>
#include <aten/src/ATen/ExpandUtils.h>
#include <torch/csrc/api/include/torch/utils.h>
#include <torch/csrc/autograd/profiler.h>
#include <torch/csrc/jit/custom_operator.h>
#include <torch/csrc/jit/operator.h>
#include <torch/csrc/jit/ir.h>

#include <aten/src/ATen/InitialTensorOptions.h>
#include <c10/core/ScalarType.h>
#include <torch/csrc/jit/script/error_report.h>

#include <regex>
#include <sstream>

namespace torch {
namespace jit {

namespace {

c10::OperatorOptions aliasAnalysisFromSchema() {
  c10::OperatorOptions result;
  result.setAliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA);
  return result;
}

void checkListInputType(const c10::TypePtr& elem_type, const Node* node) {
  std::string op_name = node->kind().toUnqualString();
  if (!elem_type->isSubtypeOf(NumberType::get()) &&
      elem_type != BoolType::get()) {
    auto error = script::ErrorReport(node->sourceRange());
    error << "Input list to torch." << op_name
          << " must be of ints, floats, or bools, "
          << "got " << elem_type->python_str();
    // special case empty list torch.tensor([])
    if (elem_type->isSubtypeOf(TensorType::get())) {
      auto input = node->inputs().at(0);
      if (input->node()->kind() == prim::ListConstruct &&
          input->node()->inputs().size() == 0) {
        error << "\nEmpty lists default to List[Tensor]. Add a variable "
                 "annotation to the assignment to create an empty list "
                 "of another type (torch.jit.annotate(List[T, []]) where T "
                 "is the type of elements in the list for Python 2)";
      }
    }
    throw error;
  }
}

at::Tensor castTensorTo(
    at::Tensor self,
    const IValue& dtype,
    const IValue& device) {
  at::ScalarType scalar_type =
      dtype.isNone() ? self.scalar_type() : dtype.toScalarType();
  c10::Device dev =
      device.isNone() ? self.device() : device.toDevice();
  if (scalar_type != self.scalar_type() || dev != self.device()) {
    self = self.to(dev, scalar_type);
  }
  return self;
}

int64_t list_size(const IValue& list) {
  if (list.isGenericList()) {
    return list.toGenericListRef().size();
  } else if (list.isIntList()) {
    return list.toIntListRef().size();
  } else if (list.isDoubleList()) {
    return list.toDoubleListRef().size();
  } else if (list.isBoolList()) {
    return list.toBoolList().size();
  }
  AT_ASSERTM(0, "Unexpected list type", list);
}

std::vector<int64_t> compute_sizes(const IValue& seq) {
  std::vector<int64_t> sizes;
  // because bool, int, and float lists are specialized, inner array will
  // will not be generic list
  auto seq_recur = seq;
  while (seq_recur.isGenericList()) {
    auto seq_list = seq_recur.toGenericListRef();
    auto length = seq_list.size();
    AT_ASSERT(length != 0);
    sizes.push_back(length);
    seq_recur = seq_list[0];
  }
  sizes.push_back(list_size(seq_recur));
  return sizes;
}

void checkSequenceSize(int64_t n, int64_t dim, int64_t seq_size) {
  if (seq_size != n) {
    AT_ERROR(
        "Expected sequence of length ",
        n,
        " at dim ",
        dim,
        " (got ",
        seq_size,
        ")");
  }
}

template <typename DTYPE, typename List>
void storeLastDimension(
    char* data,
    const std::vector<int64_t>& sizes,
    const c10::ArrayRef<int64_t>& strides,
    int64_t dim,
    int elementSize,
    List obj) {
  auto n = sizes[dim];
  auto seq_size = obj.size();
  checkSequenceSize(n, dim, seq_size);
  for (int64_t i = 0; i < n; i++) {
    *(DTYPE*)data = obj[i];
    data += strides[dim] * elementSize;
  }
}

template <>
void storeLastDimension<bool>(
    char* data,
    const std::vector<int64_t>& sizes,
    const c10::ArrayRef<int64_t>& strides,
    int64_t dim,
    int elementSize,
    const c10::List<bool>& obj) {
  auto n = sizes[dim];
  auto seq_size = obj.size();
  checkSequenceSize(n, dim, seq_size);
  for (int64_t i = 0; i < n; i++) {
    *(bool*)data = static_cast<bool>(obj[i]);
    data += strides[dim] * elementSize;
  }
}

// refernce python implementation recursive_store in tensor_new.cpp

void recursiveStore(
    char* data,
    const std::vector<int64_t>& sizes,
    const c10::ArrayRef<int64_t>& strides,
    int64_t dim,
    int elementSize,
    const IValue& obj) {
  auto ndim = sizes.size();
  auto n = sizes[dim];
  auto seq_size = list_size(obj);
  checkSequenceSize(n, dim, seq_size);
  if (dim + 1 < static_cast<long>(ndim)) {
    auto items = obj.toGenericListRef();
    for (int64_t i = 0; i < n; i++) {
      recursiveStore(data, sizes, strides, dim + 1, elementSize, items[i]);
      data += strides[dim] * elementSize;
    }
  } else {
    AT_ASSERT(obj.isIntList() || obj.isDoubleList() || obj.isBoolList());
    if (obj.isIntList()) {
      storeLastDimension<int64_t>(
          data, sizes, strides, dim, elementSize, obj.toIntListRef());
    } else if (obj.isDoubleList()) {
      storeLastDimension<double>(
          data, sizes, strides, dim, elementSize, obj.toDoubleListRef());
    } else {
      storeLastDimension<bool>(
          data, sizes, strides, dim, elementSize, obj.toBoolList());
    }
  }
}

template<bool if_set_requires_grad>
Operation createTensorFromList(const Node* node) {
  // torch.tensor has a fourth requires_grad arg but torch.as_tensor not, so
  // we use the template arg to distinguish between these two cases
  auto input = node->inputs().at(0);
  auto elem_type = input->type();
  while (auto list_type = elem_type->cast<ListType>()) {
    elem_type = list_type->getElementType();
  }
  checkListInputType(elem_type, node);
  at::ScalarType initial_scalar_type = scalarTypeFromJitType(elem_type);
  return [initial_scalar_type, elem_type](Stack& stack) {
    bool requires_grad;
    IValue data;
    IValue dtype;
    IValue device;
    if (if_set_requires_grad) {
      pop(stack, data, dtype, device, requires_grad);
    } else {
      pop(stack, data, dtype, device);
    }
    auto sizes = compute_sizes(data);
    auto tensor = autograd::make_variable(at::empty(
        sizes, at::initialTensorOptions().dtype(initial_scalar_type)));

    recursiveStore(
        (char*)tensor.data_ptr(),
        sizes,
        tensor.strides(),
        0,
        tensor.element_size(),
        data);

    tensor = castTensorTo(tensor, dtype, device);
    auto default_type = at::typeMetaToScalarType(at::get_default_dtype());

    if (dtype.isNone() && tensor.scalar_type() != default_type &&
        tensor.numel() == 0) {
      AT_WARN(
          "Creating a tensor from an empty ",
          elem_type->python_str(),
          "list will create a tensor of default floating point type  (currently ",
          default_type,
          ") in python but a tensor of type ",
          elem_type->python_str(),
          " in torchscript.\n",
          "Pass in a dtype argument to ensure consistent behavior");
    }
    if (if_set_requires_grad) {
      tensor.set_requires_grad(requires_grad);
    }
    push(stack, std::move(tensor));
    return 0;
  };
}

RegisterOperators reg({
    Operator(
        "aten::split(Tensor self, int[] split_sizes, int dim=0) -> Tensor[]",
        [](Stack& stack) {
          RECORD_FUNCTION("split_with_sizes", last(stack, 3));

          auto result = at::split_with_sizes(
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toIntListRef(),
              (std::move(peek(stack, 2, 3))).toInt());
          drop(stack, 3);
          pack(stack, c10::impl::toList(std::move(result)));
          return 0;
        },
        aliasAnalysisFromSchema()),
    Operator(
        "aten::Size(int[] sizes) -> int[]",
        [](Stack& stack) { return 0; },
        aliasAnalysisFromSchema()),
    Operator(
        "aten::size(Tensor self) -> int[]",
        [](Stack& stack) {
          RECORD_FUNCTION("size", last(stack, 1));

          auto t = std::move(pop(stack)).toTensor();
          pack(stack, t.sizes().vec());
          return 0;
        },
        aliasAnalysisFromSchema()),
    Operator(
        "aten::list_with_default(int[] list, int[] defaults) -> int[]",
        [](Stack& stack) {
          RECORD_FUNCTION("sizes", last(stack, 2));

          auto list = peek(stack, 0, 2).toIntList().copy();
          auto defaults = peek(stack, 1, 2).toIntListRef();
          drop(stack, 2);

          AT_ASSERT(defaults.size() > list.size());

          // TODO: allow list of optionals to be filled in with defaults
          // i.e. list_with_default([1, 2, None], [1, 2, 3]) -> [1, 2, 3]

          push(stack, std::move(list));
          return 0;
        },
        aliasAnalysisFromSchema()),
    Operator(
        "aten::_infer_size(int[] a, int[] b) -> int[]",
        [](Stack& stack) {
          auto a = pop(stack);
          auto b = pop(stack);
          push(stack, at::infer_size(a.toIntListRef(), b.toIntListRef()));
          return 0;
        },
        aliasAnalysisFromSchema()),
    Operator(
        "aten::_no_grad_embedding_renorm_(Tensor weight, Tensor input, float max_norm, float norm_type) -> Tensor",
        [](Stack& stack) {
          at::Tensor weight;
          at::Tensor input;
          double max_norm;
          double norm_type;
          pop(stack, weight, input, max_norm, norm_type);

          // TODO: remove when script supports setting grad mode
          torch::NoGradGuard no_grad;

          at::Tensor result =
              at::embedding_renorm_(weight, input, max_norm, norm_type);
          push(stack, std::move(result));

          return 0;
        },
        aliasAnalysisFromSchema()),
    Operator(
        "aten::format(str self, ...) -> str",
        [](const Node* node) -> Operation {
          size_t num_inputs = node->inputs().size();
          std::regex unsupported_options(R"(\{([^\}]+)\})");
          return [num_inputs, unsupported_options](Stack& stack) {
            auto format = peek(stack, 0, num_inputs).toStringRef();

            if (std::regex_search(format, unsupported_options)) {
              AT_WARN("Format options are not supported.");
            }

            auto args = last(stack, num_inputs - 1);
            std::stringstream ss;
            for (size_t begin = 0, used_args = 0; true; ++used_args) {
              size_t loc = format.find("{}", begin);
              if (loc == std::string::npos) {
                ss << format.substr(begin);
                break;
              }
              ss << format.substr(begin, loc - begin);
              if (used_args >= args.size()) {
                AT_ERROR("Too few arguments for format string: ", format);
              }
              ss << args[used_args];
              begin = loc + 2;
            }

            drop(stack, num_inputs);
            push(stack, ss.str());
            return 0;
          };
        },
        aliasAnalysisFromSchema()),

#define DEFINE_TORCH_TENSOR_OP(operator_type, c_type, tensor_creation_op)  \
  Operator(                                                                \
      "aten::tensor(" #operator_type                                       \
      " t, *, ScalarType? dtype=None, Device? device=None"                 \
      ", bool requires_grad=False) -> Tensor",                             \
      [](Stack& stack) {                                                   \
        c_type scalar_val;                                                 \
        IValue dtype;                                                      \
        IValue device;                                                     \
        bool requires_grad;                                                \
        pop(stack, scalar_val, dtype, device, requires_grad);              \
        auto tensor = autograd::make_variable(tensor_creation_op);         \
        tensor = castTensorTo(tensor, dtype, device);                      \
        tensor.set_requires_grad(requires_grad);                           \
        push(stack, std::move(tensor));                                    \
        return 0;                                                          \
      },                                                                   \
      aliasAnalysisFromSchema()),                                          \
      Operator(                                                            \
          "aten::as_tensor(" #operator_type                                \
          " t, *, ScalarType? dtype=None, Device? device=None) -> Tensor", \
          [](Stack& stack) {                                               \
            c_type scalar_val;                                             \
            IValue dtype;                                                  \
            IValue device;                                                 \
            pop(stack, scalar_val, dtype, device);                         \
            auto tensor = autograd::make_variable(tensor_creation_op);     \
            tensor = castTensorTo(tensor, dtype, device);                  \
            push(stack, std::move(tensor));                                \
            return 0;                                                      \
          },                                                               \
          aliasAnalysisFromSchema()),

    DEFINE_TORCH_TENSOR_OP(float, double, at::scalar_to_tensor(scalar_val))
        DEFINE_TORCH_TENSOR_OP(int, int64_t, at::scalar_to_tensor(scalar_val))
            DEFINE_TORCH_TENSOR_OP(
                bool,
                bool,
                at::empty({}, at::CPU(at::kBool).options()).fill_(scalar_val))

    // reference python implementation: internal_new_from_data in
    // tensor_new.cpp
    Operator(
        "aten::_infer_size(int[] a, int[] b) -> int[]",
        [](Stack& stack) {
          auto a = pop(stack);
          auto b = pop(stack);
          push(stack, at::infer_size(a.toIntListRef(), b.toIntListRef()));
          return 0;
        },
        aliasAnalysisFromSchema()),
    Operator(
        "aten::_no_grad_embedding_renorm_(Tensor weight, Tensor input, float max_norm, float norm_type) -> Tensor",
        [](Stack& stack) {
          at::Tensor weight;
          at::Tensor input;
          double max_norm;
          double norm_type;
          pop(stack, weight, input, max_norm, norm_type);

          // TODO: remove when script supports setting grad mode
          torch::NoGradGuard no_grad;

          at::Tensor result =
              at::embedding_renorm_(weight, input, max_norm, norm_type);
          push(stack, std::move(result));

          return 0;
        },
        aliasAnalysisFromSchema()),
    Operator(
        "aten::tensor(t[] data, *, ScalarType? dtype=None, Device? device=None, bool requires_grad=False) -> Tensor",
        createTensorFromList<true>,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::as_tensor(Tensor(a) data, *, ScalarType? dtype=None, Device? device=None) -> Tensor(a|b)",
        [](Stack& stack) {
          auto device = pop(stack).toOptional<c10::Device>();
          auto dtype = pop(stack).toOptional<at::ScalarType>();
          at::Tensor data = pop(stack).toTensor();
          at::ScalarType scalar_type =
              dtype ? dtype.value() : data.scalar_type();
          c10::Device dev = device ? device.value() : data.device();

          if (scalar_type != data.scalar_type() || dev != data.device()) {
            data = data.to(
                dev, scalar_type, /*non_blocking=*/false, /*copy=*/false);
          }
          push(stack, std::move(data));
          return 0;
        },
        aliasAnalysisFromSchema()),
    Operator(
        "aten::as_tensor(t[] data, *, ScalarType? dtype=None, Device? device=None) -> Tensor",
        createTensorFromList<false>,
        aliasAnalysisFromSchema()),
    Operator(
        "aten::_assert_int_or_pair(int[] vals, str name, str message) -> Tensor",
        [](Stack& stack) {
          // Everything is a list at the point this is used, so don't do
          // anything
          drop(stack, 3);
          return 0;
        },
        aliasAnalysisFromSchema()),
    Operator(
        "aten::_pack_sequence(Tensor output, Tensor batch_sizes, Tensor? sorted_indices, "
        "Tensor? unsorted_indices) -> (Tensor, Tensor, Tensor?, Tensor?)",
        [](Stack& stack) { return 0; },
        aliasAnalysisFromSchema()),
    Operator(
        "aten::_get_tracing_state() -> bool",
        [](Stack& stack) {
          push(stack, false);
          return 0;
        },
        aliasAnalysisFromSchema()),
    Operator(
        "aten::is_scripting() -> bool",
        [](Stack& stack) {
          push(stack, true);
          return 0;
        },
        aliasAnalysisFromSchema()),
    Operator(
        "aten::_no_grad_uniform_(Tensor(a!) tensor, float a, float b) -> Tensor(a!)",
        [](Stack& stack) {
          // TODO: remove when script supports setting grad mode
          torch::NoGradGuard no_grad;

          at::Tensor tensor;
          double a;
          double b;
          pop(stack, tensor, a, b);
          push(stack, tensor.uniform_(a, b));
          return 0;
        },
        aliasAnalysisFromSchema()),
    Operator(
        "aten::_no_grad_normal_(Tensor(a!) tensor, float mean, float std) -> Tensor(a!)",
        [](Stack& stack) {
          // TODO: remove when script supports setting grad mode
          torch::NoGradGuard no_grad;

          at::Tensor tensor;
          double mean;
          double std;
          pop(stack, tensor, mean, std);
          push(stack, tensor.normal_(mean, std));
          return 0;
        },
        aliasAnalysisFromSchema()),
    Operator(
        "aten::_no_grad_fill_(Tensor(a!) tensor, float val) -> Tensor(a!)",
        [](Stack& stack) {
          // TODO: remove when script supports setting grad mode
          torch::NoGradGuard no_grad;

          at::Tensor tensor;
          double val;
          pop(stack, tensor, val);
          push(stack, at::fill_(tensor, val));
          return 0;
        },
        aliasAnalysisFromSchema()),
    Operator(
        "aten::_no_grad_zero_(Tensor(a!) tensor) -> Tensor(a!)",
        [](Stack& stack) {
          // TODO: remove when script supports setting grad mode
          torch::NoGradGuard no_grad;

          at::Tensor tensor;
          pop(stack, tensor);
          push(stack, at::zero_(tensor));
          return 0;
        },
        aliasAnalysisFromSchema()),

});
} // namespace
} // namespace jit
} // namespace torch
