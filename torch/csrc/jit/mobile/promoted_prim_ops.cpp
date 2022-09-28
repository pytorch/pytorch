#include <ATen/ScalarOps.h>
#include <torch/csrc/jit/mobile/promoted_prim_ops.h>
namespace torch {
namespace jit {

void tupleIndex(Stack& stack) {
  int64_t index = pop(stack).toInt();
  auto tuple = pop(stack).toTuple();
  auto norm_index = normalizeIndex(index, tuple->elements().size());
  if (norm_index < 0 ||
      norm_index >= static_cast<int64_t>(tuple->elements().size())) {
    throw std::out_of_range("Tuple list index out of range");
  }
  stack.emplace_back(tuple->elements()[norm_index]);
}

void raiseException(Stack& stack) {
  // this kernel supports RaiseException with only one argument: the error
  // DEPRECATED from bytecode_version 8;
  // Please do not make any changes to this to support BC
  throw JITException(pop(stack).toStringRef());
}

void raiseExceptionWithMessage(Stack& stack) {
  // this kernel supports RaiseException with only two arguments: the error and
  // the message Please make changes only to this kernel
  c10::optional<std::string> qualified_class_name =
      pop(stack).toOptional<std::string>();
  std::string message;
  pop(stack, message);

  throw JITException(message, qualified_class_name);
}

void is(Stack& stack) {
  IValue self, obj;
  pop(stack, self, obj);
  push(stack, self.is(obj));
}

void unInitialized(Stack& stack) {
  push(stack, IValue::uninitialized());
}

void isNot(Stack& stack) {
  IValue self, obj;
  pop(stack, self, obj);
  push(stack, !self.is(obj));
}

void aten_format(Stack& stack) {
  size_t num_inputs = pop(stack).toInt();
  format(stack, num_inputs);
}

void size(Stack& stack) {
  auto t = std::move(pop(stack)).toTensor();
  pack(stack, t.sizes().vec());
}

void sym_size(Stack& stack) {
  auto t = std::move(pop(stack)).toTensor();
  pack(stack, t.sym_sizes().vec());
}
void sym_size_int(Stack& stack) {
  auto dim = pop(stack).toInt();
  auto t = pop(stack).toTensor();
  push(stack, t.sym_sizes()[dim]);
}
void sym_stride_int(Stack& stack) {
  auto dim = pop(stack).toInt();
  auto t = pop(stack).toTensor();
  push(stack, t.sym_strides()[dim]);
}

void sym_numel(Stack& stack) {
  auto t = std::move(pop(stack)).toTensor();
  push(stack, t.sym_numel());
}

void sym_storage_offset(Stack& stack) {
  auto t = std::move(pop(stack)).toTensor();
  push(stack, t.sym_storage_offset());
}

void sym_stride(Stack& stack) {
  auto t = std::move(pop(stack)).toTensor();
  pack(stack, t.sym_strides().vec());
}

void device(Stack& stack) {
  push(stack, pop(stack).toTensor().device());
}

void dtype(Stack& stack) {
  at::Tensor a;
  pop(stack, a);
  push(stack, static_cast<int64_t>(a.scalar_type()));
}

void layout(Stack& stack) {
  push(stack, pop(stack).toTensor().layout());
}

void toPrimDType(Stack& stack) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  bool non_blocking;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  bool copy;
  pop(stack, non_blocking, copy);
  c10::optional<at::ScalarType> scalarType =
      pop(stack).toOptional<at::ScalarType>();
  c10::optional<c10::Device> device = c10::nullopt;
  at::Tensor self = pop(stack).toTensor();
  push(stack, to_dispatch(self, device, scalarType, non_blocking, copy));
}

void dim(Stack& stack) {
  at::Tensor arg = pop(stack).toTensor();
  push(stack, arg.dim());
}

void _not(Stack& stack) {
  push(stack, !pop(stack).toBool());
}

void boolTensor(Stack& stack) {
  at::Tensor a;
  pop(stack, a);
  push(stack, at::native::is_nonzero(a));
}

void toList(Stack& stack) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int elem_ty_val;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
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
  at::TypePtr out_ty;
  if (elem_ty_val == 0) {
    out_ty = at::IntType::get();
  } else if (elem_ty_val == 1) {
    out_ty = at::FloatType::get();
  } else if (elem_ty_val == 2) {
    out_ty = at::BoolType::get();
  } else if (elem_ty_val == 3) {
    out_ty = at::ComplexType::get();
  } else {
    TORCH_CHECK(
        false,
        "Unsupported element type for tolist; only int, float, complex and bool are supported");
  }

  // Check that type of the Tensor matches that of the annotation.
  // Make an exception for the case in which the annotated type is
  // float/complex and the Tensor data type is also float/complex;
  // the elements will be casted to double/c10::complex<double>
  // later.
  TORCH_CHECK(
      (out_ty == at::FloatType::get() && t.is_floating_point()) ||
          (out_ty == at::ComplexType::get() && t.is_complex()) ||
          tryScalarTypeFromJitType(*out_ty) == t.scalar_type(),
      "Output annotation element type and runtime tensor element type must match for tolist()");

  // Check that the dimension of the Tensor matches that of the
  // annotation.
  TORCH_CHECK(
      dim_val == t.dim(),
      "Output annotation list dimension and runtime tensor dimension must match for tolist()");

  // Wrap out_ty in a ListType dim times.
  for (const auto i : c10::irange(dim_val)) {
    (void)i; // Suppress unused variable warning
    out_ty = at::ListType::create(out_ty);
  }

  int64_t dim = t.dim();
  auto sizes = t.sizes();
  auto strides = t.strides();
  size_t element_size = t.element_size();
  char* data = static_cast<char*>(t.data_ptr());
  auto result = tensorToListRecursive(
      data, 0, dim, out_ty, t.scalar_type(), sizes, strides, element_size);
  push(stack, std::move(result));
}

void numToTensorScalar(Stack& stack) {
  at::Scalar s;
  pop(stack, s);
  push(stack, c10::scalar_to_tensor(s));
}

void isCuda(Stack& stack) {
  at::Tensor a;
  pop(stack, a);
  push(stack, a.is_cuda());
}

void numToTensorBool(Stack& stack) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  bool b;
  pop(stack, b);
  push(stack, c10::scalar_to_tensor(b));
}

void dictIndex(Stack& stack) {
  auto key = pop(stack);
  auto dict = pop(stack).toGenericDict();
  auto value = dict.find(key);
  if (value == dict.end()) {
    AT_ERROR("KeyError: ", key);
  }
  push(stack, value->value());
}

static const C10_UNUSED std::array<mobile::prim_op_fn_register, 16> op_reg = {
    mobile::prim_op_fn_register("prim::TupleIndex", tupleIndex),
    mobile::prim_op_fn_register("aten::Bool.Tensor", boolTensor),
    mobile::prim_op_fn_register("aten::format", aten_format),
    mobile::prim_op_fn_register("prim::NumToTensor.Scalar", numToTensorScalar),
    mobile::prim_op_fn_register(
        "prim::RaiseException",
        raiseExceptionWithMessage),
    mobile::prim_op_fn_register("prim::device", device),
    mobile::prim_op_fn_register("prim::dtype", dtype),
    mobile::prim_op_fn_register("prim::layout", layout),
    mobile::prim_op_fn_register("aten::__not__", _not),
    mobile::prim_op_fn_register("aten::__is__", is),
    mobile::prim_op_fn_register("aten::__isnot__", isNot),
    mobile::prim_op_fn_register("aten::dim", dim),
    mobile::prim_op_fn_register("prim::Uninitialized", unInitialized),
    mobile::prim_op_fn_register("prim::is_cuda", isCuda),
    mobile::prim_op_fn_register("aten::__getitem__.Dict_str", dictIndex),
    mobile::prim_op_fn_register("prim::unchecked_cast", noop),
    // TODO: (@pavithran) size is overloaded with int[] and Tensor
    // so this throws error expecting int not Tensor
    // mobile::prim_op_fn_register("aten::size", size)
};

} // namespace jit
} // namespace torch
