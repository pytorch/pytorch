#include <torch/csrc/autograd/profiler.h>
#include <torch/csrc/jit/custom_operator.h>
#include <torch/csrc/jit/operator.h>
#include <torch/csrc/api/include/torch/utils.h>
#include <aten/src/ATen/ExpandUtils.h>

#include <c10/core/ScalarType.h>
#include <aten/src/ATen/InitialTensorOptions.h>
#include <torch/csrc/jit/script/error_report.h>

#include <regex>
#include <sstream>

namespace torch {
namespace jit {

namespace {


void checkListInputType(const c10::TypePtr& elem_type, const Node* node) {
  if (!elem_type->isSubtypeOf(NumberType::get()) && elem_type != BoolType::get()) {
    auto error = script::ErrorReport(node->getSourceLocation());
    error << "Input list to torch.tensor must be of ints, floats, or bools, " <<
      "got " << elem_type->str();
    // special case empty list torch.tensor([])
    if (elem_type->isSubtypeOf(TensorType::get())) {
      auto input = node->inputs().at(0);
      if (input->node()->kind() == prim::ListConstruct && input->node()->inputs().size() == 0) {
        error << "\n(Note: empty lists are constructed as Tensor[]; \n"
               << "if you want an empty list of a different type, \n"
               << "use `torch.jit.annotate(List[T], [])`, \n"
               << "where `T` is the type of elements in the list)";
      }
    }
    throw error;
  }
}

at::ScalarType scalarTypeFromJitType(const c10::TypePtr& type) {
  if (type == FloatType::get()) {
    return at::ScalarType::Double;
  } else if (type == IntType::get()) {
    return at::ScalarType::Long;
  } else if (type == BoolType::get()) {
    return at::ScalarType::Byte;
  }
  AT_ASSERTM(0, "Add new condition, expected Float, Int, or Bool but got",
      type->str());
}


int64_t list_size(const IValue& list) {
  if (list.isGenericList()) {
    return list.toGenericListRef().size();
  } else if (list.isIntList()) {
    return list.toIntListRef().size();
  } else if (list.isDoubleList()){
    return list.toDoubleListRef().size();
  } else if (list.isBoolList()) {
    return list.toBoolListRef().size();
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
    AT_ERROR("Expected sequence of length ", n, " at dim ", dim, " (got ", seq_size, ")");
  }
}

template <typename DTYPE>
void storeLastDimension(char* data, const std::vector<int64_t>& sizes, const c10::ArrayRef<int64_t>& strides, int64_t dim,
    int elementSize, const std::vector<DTYPE>& obj) {
  auto n = sizes[dim];
  auto seq_size = obj.size();
  checkSequenceSize(n, dim, seq_size);
  for (int64_t i = 0; i < n; i++) {
    *(DTYPE*)data = obj[i];
    data += strides[dim] * elementSize;
  }
}

// bool vector needs to be cast to uint8_t
template<>
void storeLastDimension<bool>(char* data, const std::vector<int64_t>& sizes, const c10::ArrayRef<int64_t>& strides, int64_t dim,
    int elementSize, const std::vector<bool>& obj) {
  auto n = sizes[dim];
  auto seq_size = obj.size();
  checkSequenceSize(n, dim, seq_size);
  for (int64_t i = 0; i < n; i++) {
    *(uint8_t*)data = static_cast<uint8_t>(obj[i]);
    data += strides[dim] * elementSize;
  }
}

// refernce python implementation recursive_store in tensor_new.cpp

void recursiveStore(char* data, const std::vector<int64_t>& sizes, const c10::ArrayRef<int64_t>& strides, int64_t dim,
   int elementSize, const IValue& obj) {

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
      storeLastDimension<int64_t>(data, sizes, strides, dim, elementSize, obj.toIntListRef());
    } else if (obj.isDoubleList()){
      storeLastDimension<double>(data, sizes, strides, dim, elementSize, obj.toDoubleListRef());
    } else {
      storeLastDimension<bool>(data, sizes, strides, dim, elementSize, obj.toBoolListRef());
    }
  }
}

RegisterOperators reg({
    Operator(
        "aten::split(Tensor self, int[] split_sizes, int dim=0) -> Tensor[]",
        [](Stack& stack) {
          autograd::profiler::RecordFunction record("split_with_sizes");
          auto result = at::split_with_sizes(
              (std::move(peek(stack, 0, 3))).toTensor(),
              (std::move(peek(stack, 1, 3))).toIntList()->elements(),
              (std::move(peek(stack, 2, 3))).toInt());
          drop(stack, 3);
          pack(stack, std::move(result));
          return 0;
        }),
    Operator(
        "aten::Size(int[] sizes) -> int[]",
        [](Stack& stack) { return 0; }),
    Operator(
        "aten::size(Tensor self) -> int[]",
        [](Stack& stack) {
          autograd::profiler::RecordFunction record("sizes");
          auto t = std::move(pop(stack)).toTensor();
          pack(stack, t.sizes().vec());
          return 0;
        }),
    Operator(
        "aten::list_with_default(int[] list, int[] defaults) -> int[]",
        [](Stack& stack) {
          autograd::profiler::RecordFunction record("sizes");
          auto list = peek(stack, 0, 2).toIntListRef();
          auto defaults = peek(stack, 1, 2).toIntListRef();
          drop(stack, 2);

          AT_ASSERT(defaults.size() > list.size());

          // TODO: allow list of optionals to be filled in with defaults
          // i.e. list_with_default([1, 2, None], [1, 2, 3]) -> [1, 2, 3]

          push(stack, list);
          return 0;
        }),
    Operator(
        "aten::_infer_size(int[] a, int[] b) -> int[]",
        [](const Node* node) {
          return [](Stack& stack) {
            auto a = pop(stack).toIntList()->elements();
            auto b = pop(stack).toIntList()->elements();
            push(stack, at::infer_size(a, b));
            return 0;
          };
        }),
    Operator(
      "aten::_no_grad_embedding_renorm_(Tensor weight, Tensor input, float max_norm, float norm_type) -> Tensor",
      [](const Node* node) {
        return [](Stack& stack) {
          at::Tensor weight;
          at::Tensor input;
          double max_norm;
          double norm_type;
          pop(stack, weight, input, max_norm, norm_type);

          // TODO: remove when script supports setting grad mode
          torch::NoGradGuard no_grad;

          at::Tensor result = at::embedding_renorm_(weight, input, max_norm, norm_type);
          push(stack, result);

          return 0;
        };
      }),
    Operator(
        "aten::format(str self, ...) -> str",
        [](const Node* node) {
          size_t num_inputs = node->inputs().size();
          std::regex unsupported_options("\\{(.*)\\}");
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
        }),

#define DEFINE_TORCH_TENSOR_OP(operator_type, c_type, tensor_creation_op)             \
Operator(                                                                             \
  "aten::tensor(" #operator_type " t, *, ScalarType? dtype=None, Device? device=None"\
      ") -> Tensor",                                                                  \
  [](const Node* node) {                                                              \
    auto initial_scalar_type = scalarTypeFromJitType(node->inputs().at(0)->type());   \
    return [initial_scalar_type](Stack& stack) {                                      \
      c_type scalar_val;                                                              \
      IValue dtype;                                                                   \
      IValue device;                                                                  \
      pop(stack, scalar_val, dtype, device);                                          \
      auto tensor = autograd::make_variable(tensor_creation_op);                      \
      at::ScalarType scalar_type = dtype.isNone() ?                                   \
        tensor.scalar_type() : dtype.toScalarType();                                  \
      c10::Device dev = device.isNone() ? tensor.device() : device.toDevice();        \
      if (scalar_type != initial_scalar_type || dev != tensor.device()) {             \
        tensor = tensor.to(dev, scalar_type);                                         \
      }                                                                               \
      push(stack, tensor);                                                            \
      return 0;                                                                       \
    };                                                                                \
  }),

DEFINE_TORCH_TENSOR_OP(float, double, at::scalar_to_tensor(scalar_val))
DEFINE_TORCH_TENSOR_OP(int, int64_t, at::scalar_to_tensor(scalar_val))
DEFINE_TORCH_TENSOR_OP(bool, bool, at::empty({}, at::CPU(at::kByte).options()).fill_(scalar_val))


    // reference python implementation: internal_new_from_data in tensor_new.cpp
    Operator(
        "aten::_infer_size(int[] a, int[] b) -> int[]",
        [](const Node* node) {
          return [](Stack& stack) {
            auto a = pop(stack).toIntList()->elements();
            auto b = pop(stack).toIntList()->elements();
            push(stack, at::infer_size(a, b));
            return 0;
          };
        }),
    Operator(
        "aten::_no_grad_embedding_renorm_(Tensor weight, Tensor input, float max_norm, float norm_type) -> Tensor",
        [](const Node* node) {
          return [](Stack& stack) {
            at::Tensor weight;
            at::Tensor input;
            double max_norm;
            double norm_type;
            pop(stack, weight, input, max_norm, norm_type);

            // TODO: remove when script supports setting grad mode
            torch::NoGradGuard no_grad;

            at::Tensor result =
                at::embedding_renorm_(weight, input, max_norm, norm_type);
            push(stack, result);

            return 0;
          };
        }),
    Operator(
      "aten::tensor(t[] data, *, ScalarType? dtype=None, Device? device=None) -> Tensor",
      [](const Node* node) {
        auto input = node->inputs().at(0);
        auto elem_type = input->type();
        while (auto list_type = elem_type->cast<ListType>()) {
          elem_type = list_type->getElementType();
        }
        checkListInputType(elem_type, node);
        at::ScalarType initial_scalar_type = scalarTypeFromJitType(elem_type);
        return [initial_scalar_type, elem_type](Stack& stack) {
          IValue data;
          IValue dtype;
          IValue device;
          pop(stack, data, dtype, device);
          auto sizes = compute_sizes(data);
          auto tensor = autograd::make_variable(
            at::empty(sizes, at::initialTensorOptions().dtype(initial_scalar_type)));

          recursiveStore((char*)tensor.data_ptr(), sizes, tensor.strides(), 0,
              tensor.type().elementSizeInBytes(), data);

          at::ScalarType scalar_type = dtype.isNone() ? tensor.scalar_type() : dtype.toScalarType();
          c10::Device dev = device.isNone() ? tensor.device() : device.toDevice();
          if (scalar_type != initial_scalar_type || dev != tensor.device()) {
            tensor = tensor.to(dev, scalar_type);
          }

          auto default_type = at::typeMetaToScalarType(at::get_default_dtype());

          if (dtype.isNone() && tensor.scalar_type() != default_type &&
              tensor.numel() == 0) {
            AT_WARN("Creating a tensor from an empty ", elem_type->str(),
              "list will create a tensor of default floating point type  (currently ", default_type,
              ") in python but a tensor of type ", elem_type->str(), " in torchscript.\n",
              "Pass in a dtype argument to ensure consistent behavior");
          }

          push(stack, tensor);
          return 0;
        };
      }),
    Operator(
        "aten::_assert_int_or_pair(int[] vals, str name, str message) -> Tensor",
        [](const Node* node) {
          return [](Stack& stack) {
            // Everything is a list at the point this is used, so don't do
            // anything
            drop(stack, 3);
            return 0;
          };
        }),

});
}
} // namespace jit
} // namespace torch
