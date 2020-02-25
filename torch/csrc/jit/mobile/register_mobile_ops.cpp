#include <ATen/core/op_registration/op_registration.h>
#include <ATen/ATen.h>
#include <ATen/core/stack.h>

using Stack = std::vector<c10::IValue>;
using torch::jit::peek;
using torch::jit::drop;
using torch::jit::pack;
using torch::jit::push;
using torch::jit::pop;
using torch::jit::last;
using at::Tensor;
using at::Scalar;

namespace {
at::Tensor toOptionalTensor(const c10::IValue& v) {
  if (v.isNone()) {
    return at::Tensor();
  }
  return v.toTensor();
}

at::Tensor optional_to_tensor(c10::optional<at::Tensor> v) {
  return v.has_value() ? *v : at::Tensor();
}

void _convolution_kernel(const c10::OperatorHandle& op, Stack* stack) {
#ifdef USE_STATIC_DISPATCH
  at::AutoNonVariableTypeMode non_var_type_mode(true);
#endif
  auto result_ = at::_convolution(
      (std::move(peek(*stack, 0, 12))).toTensor(),
      (std::move(peek(*stack, 1, 12))).toTensor(),
      toOptionalTensor((std::move(peek(*stack, 2, 12)))),
      (std::move(peek(*stack, 3, 12))).toIntVector(),
      (std::move(peek(*stack, 4, 12))).toIntVector(),
      (std::move(peek(*stack, 5, 12))).toIntVector(),
      (std::move(peek(*stack, 6, 12))).toBool(),
      (std::move(peek(*stack, 7, 12))).toIntVector(),
      (std::move(peek(*stack, 8, 12))).toInt(),
      (std::move(peek(*stack, 9, 12))).toBool(),
      (std::move(peek(*stack, 10, 12))).toBool(),
      (std::move(peek(*stack, 11, 12))).toBool()
      );
  drop(*stack, 12);
  pack(*stack, std::move(result_));
}

void conv2d_kernel(const c10::OperatorHandle& op, Stack* stack) {
#ifdef USE_STATIC_DISPATCH
    at::AutoNonVariableTypeMode non_var_type_mode(true);
#endif
    auto result_ = at::conv2d(
        (std::move(peek(*stack, 0, 7))).toTensor(),
        (std::move(peek(*stack, 1, 7))).toTensor(),
        toOptionalTensor((std::move(peek(*stack, 2, 7)))),
        (std::move(peek(*stack, 3, 7))).toIntVector(),
        (std::move(peek(*stack, 4, 7))).toIntVector(),
        (std::move(peek(*stack, 5, 7))).toIntVector(),
        (std::move(peek(*stack, 6, 7))).toInt()
        );
    drop(*stack, 7);
    pack(*stack, std::move(result_));
}

void view_kernel(const c10::OperatorHandle& op, Stack* stack) {
#ifdef USE_STATIC_DISPATCH
  at::AutoNonVariableTypeMode non_var_type_mode(true);
#endif
  auto result_ = ((std::move(peek(*stack, 0, 2))).toTensor()).view(
      (std::move(peek(*stack, 1, 2))).toIntVector()
      );
  drop(*stack, 2);
  pack(*stack, std::move(result_));
}

void permute_kernel(const c10::OperatorHandle& op, Stack* stack) {
  auto result_ = ((std::move(peek(*stack, 0, 2))).toTensor()).permute(
      (std::move(peek(*stack, 1, 2))).toIntVector()
  );
  drop(*stack, 2);
  pack(*stack, std::move(result_));
}

void cat_kernel(const c10::OperatorHandle& op, Stack* stack) {
  auto result_ = at::cat(
      (std::move(peek(*stack, 0, 2))).toTensorVector(),
      (std::move(peek(*stack, 1, 2))).toInt()
  );
  drop(*stack, 2);
  pack(*stack, std::move(result_));
}

void __is__kernel(const c10::OperatorHandle& op, Stack* stack) {
  c10::IValue self, obj;
  pop(*stack, self, obj);
  push(*stack, self.isSameIdentity(obj));
}

void log_softmax_kernel(const c10::OperatorHandle& op, Stack* stack) {
  auto result_ = at::log_softmax(
    (std::move(peek(*stack, 0, 3))).toTensor(),
    (std::move(peek(*stack, 1, 3))).toInt(),
    (std::move(peek(*stack, 2, 3))).toOptional<c10::ScalarType>()
  );
  drop(*stack, 3);
  pack(*stack, std::move(result_));
}

void softmax_kernel(const c10::OperatorHandle& op, Stack* stack) {
  auto result_ = at::softmax(
    (std::move(peek(*stack, 0, 3))).toTensor(),
    (std::move(peek(*stack, 1, 3))).toInt(),
    (std::move(peek(*stack, 2, 3))).toOptional<c10::ScalarType>()
  );
  drop(*stack, 3);
  pack(*stack, std::move(result_));
}

void upsample_nearest2d_kernel(const c10::OperatorHandle& op, Stack* stack) {
  auto result_ = at::upsample_nearest2d(
    (std::move(peek(*stack, 0, 4))).toTensor(),
    (std::move(peek(*stack, 1, 4))).toIntVector(),
    (std::move(peek(*stack, 2, 4))).toOptional<double>(),
    (std::move(peek(*stack, 3, 4))).toOptional<double>()
  );
  drop(*stack, 4);
  pack(*stack, std::move(result_));
}

void warn_kernel(const c10::OperatorHandle& op, Stack* stack) {
  drop(*stack, 1);
  pop(*stack);
}

void tupleunpack_kernel(const c10::OperatorHandle& op, Stack* stack) {
  auto tuple = pop(*stack).toTuple();
  stack->insert(
     stack->end(),
     tuple->elements().begin(),
     tuple->elements().end());
}

void format_kernel(const c10::OperatorHandle& op, Stack* stack) {
  size_t num_inputs = pop(*stack).toInt();
  auto format = peek(*stack, 0, num_inputs).toStringRef();
  auto args = last(*stack, num_inputs - 1);
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

  drop(*stack, num_inputs);
  push(*stack, ss.str());
}

void to_dtype_kernal(const c10::OperatorHandle& op, Stack* stack) {
  auto result_ = ((std::move(peek(*stack, 0, 5))).toTensor()).to(
     (std::move(peek(*stack, 1, 5))).toScalarType(),
     (std::move(peek(*stack, 2, 5))).toBool(),
     (std::move(peek(*stack, 3, 5))).toBool(),
     (std::move(peek(*stack, 4, 5))).toOptional<c10::MemoryFormat>()
  );
  drop(*stack, 5);
  pack(*stack, std::move(result_));
}

int64_t normalizeIndex(int64_t idx, int64_t list_size) {
  if (idx < 0) {
    // Handle negative indexing
    idx = list_size + idx;
  }
  return idx;
}

void TupleIndex_kernel(const c10::OperatorHandle& op, Stack* stack) {
   int64_t index = pop(*stack).toInt();
   auto tuple = pop(*stack).toTuple();
   auto norm_index = normalizeIndex(index, tuple->elements().size());
   if (norm_index < 0 ||
       norm_index > static_cast<int64_t>(tuple->elements().size())) {
     throw std::out_of_range("Tuple list index out of range");
   }
   stack->emplace_back(tuple->elements()[norm_index]);
}

void pop_kernal(const c10::OperatorHandle& op, Stack* stack) {
  pop(*stack);
}

template <typename T>
void listAppend(const c10::OperatorHandle& op, Stack* stack) {
  T el = pop(*stack).to<T>();
  c10::List<T> list = pop(*stack).to<c10::List<T>>();

  list.push_back(std::move(el));
  push(*stack, std::move(list));
}

static auto registry = torch::RegisterOperators().op(
  "_aten::add.Tensor",
  torch::RegisterOperators::options().kernel(c10::DispatchKey::CPUTensorId,
  [](at::Tensor a, at::Tensor b, at::Scalar c) -> at::Tensor {
    return at::add(a, b, c);
  })
).op(
  "_aten::sub.Tensor",
  torch::RegisterOperators::options().kernel(c10::DispatchKey::CPUTensorId,
  [](at::Tensor a, at::Tensor b, at::Scalar c) -> at::Tensor {
    return at::sub(a, b, c);
  })
).op(
  "_aten::add.Scalar",
  torch::RegisterOperators::options().kernel(c10::DispatchKey::CPUTensorId,
  [](at::Tensor a, at::Scalar b, at::Scalar c) -> at::Tensor {
    return at::add(a, b, c);
  })
).op(
  "_aten::add_.Tensor",
  torch::RegisterOperators::options().kernel(c10::DispatchKey::CPUTensorId,
                                           [](at::Tensor a, at::Tensor b, at::Scalar c) -> at::Tensor {
                                             return at::add(a, b, c);
                                           })
).op(
  "_aten::adaptive_avg_pool2d",
  torch::RegisterOperators::options().kernel(c10::DispatchKey::CPUTensorId,
  [](at::Tensor a, c10::List<int64_t> b) -> at::Tensor {
  #ifdef USE_STATIC_DISPATCH
   at::AutoNonVariableTypeMode non_var_type_mode(true);
  #endif
   return at::adaptive_avg_pool2d(a, b.vec());
  })
).op(
  "_aten::mm",
  torch::RegisterOperators::options().kernel(c10::DispatchKey::CPUTensorId,
  [](at::Tensor a, at::Tensor b) -> at::Tensor {
    return at::mm(a, b);
  })
).op(
  "_aten::_convolution(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups, bool benchmark, bool deterministic, bool cudnn_enabled) -> Tensor",
  torch::RegisterOperators::options().kernel<&_convolution_kernel>(c10::DispatchKey::CPUTensorId)
).op(
  "_aten::conv2d(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1, int groups=1) -> Tensor",
  torch::RegisterOperators::options().kernel<&conv2d_kernel>(c10::DispatchKey::CPUTensorId)
).op(
  "_aten::batch_norm",
  torch::RegisterOperators::options().kernel(c10::DispatchKey::CPUTensorId,
  [] (at::Tensor input, c10::optional<at::Tensor> weight, c10::optional<at::Tensor> bias,
    c10::optional<at::Tensor> running_mean, c10::optional<at::Tensor> running_var,
    bool training, double momentum, double eps, bool cudnn_enabled) {
   return at::batch_norm(input, optional_to_tensor(weight), optional_to_tensor(bias),
                         optional_to_tensor(running_mean), optional_to_tensor(running_var),
                         training, momentum, eps, cudnn_enabled);
  })
).op(
  "_aten::max_pool2d_with_indices(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)",
  torch::RegisterOperators::options().kernel(c10::DispatchKey::CPUTensorId,
  [](const Tensor & self, c10::List<int64_t> kernel_size, c10::List<int64_t> stride,
      c10::List<int64_t> padding, c10::List<int64_t> dilation, bool ceil_mode) {
  #ifdef USE_STATIC_DISPATCH
     at::AutoNonVariableTypeMode non_var_type_mode(true);
  #endif
     return at::max_pool2d_with_indices(self, kernel_size.vec(), stride.vec(),
      padding.vec(), dilation.vec(), ceil_mode);
  })
).op(
  "_aten::max_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> Tensor",
  torch::RegisterOperators::options().kernel(c10::DispatchKey::CPUTensorId,
  [](const Tensor & self, c10::List<int64_t> kernel_size, c10::List<int64_t> stride, c10::List<int64_t> padding, c10::List<int64_t> dilation, bool ceil_mode=false) {
#ifdef USE_STATIC_DISPATCH
   at::AutoNonVariableTypeMode non_var_type_mode(true);
#endif
   return at::max_pool2d(self, kernel_size.vec(), stride.vec(), padding.vec(), dilation.vec(), ceil_mode);
  })
).op(
  "_aten::threshold",
  torch::RegisterOperators::options().kernel(c10::DispatchKey::CPUTensorId,
  [](at::Tensor self, at::Scalar threshold, at::Scalar value) {
   return at::threshold_(self, threshold, value);
  })
).op(
  "_aten::relu(Tensor self) -> Tensor",
  torch::RegisterOperators::options().kernel(c10::DispatchKey::CPUTensorId,
  [](const Tensor & self) {

  #ifdef USE_STATIC_DISPATCH
  at::AutoNonVariableTypeMode non_var_type_mode(true);
  #endif
  return at::relu(self);
})
).op(
  "_aten::relu_",
  torch::RegisterOperators::options().kernel(c10::DispatchKey::CPUTensorId,
  [](at::Tensor a) -> at::Tensor {
    return at::relu_(a);
  })
).op(
  "_aten::t(Tensor(a) self) -> Tensor(a)",
  torch::RegisterOperators::options().kernel(c10::DispatchKey::CPUTensorId,
  [](const Tensor & self) {

  #ifdef USE_STATIC_DISPATCH
  at::AutoNonVariableTypeMode non_var_type_mode(true);
  #endif
  return at::t(self);
  }).aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA)
).op(
  "_aten::size.int",
  torch::RegisterOperators::options().kernel(c10::DispatchKey::CPUTensorId,
  [](at::Tensor a, int64_t dim) -> int64_t {
   return at::size(a, dim);
  })
).op(
  "_aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor",
  torch::RegisterOperators::options().kernel(c10::DispatchKey::CPUTensorId,
  [](const Tensor & self, const Tensor & mat1, const Tensor & mat2,
      Scalar beta, Scalar alpha) {

  #ifdef USE_STATIC_DISPATCH
  at::AutoNonVariableTypeMode non_var_type_mode(true);
  #endif
  return at::addmm(self, mat1, mat2, beta, alpha);
  })
).op(
  "_aten::view(Tensor(a) self, int[] size) -> Tensor(a)",
  torch::RegisterOperators::options()
    .kernel<&view_kernel>(c10::DispatchKey::CPUTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA)
).op(
  "_aten::dim",
  torch::RegisterOperators::options().kernel(c10::DispatchKey::CPUTensorId,
  [](at::Tensor a) -> int64_t {
   return a.dim();
  })
).op(
  "_aten::eq",
  torch::RegisterOperators::options().catchAllKernel(
    [](int64_t a, int64_t b) -> bool {
      return a == b;
    })
).op(
  "_aten::log_softmax",
  torch::RegisterOperators::options().kernel(c10::DispatchKey::CPUTensorId,
  [](at::Tensor a, int64_t b, c10::optional<int64_t> c) -> at::Tensor {
    if (c.has_value()) {
     return at::log_softmax(a, b, static_cast<c10::ScalarType>(c.value()));
    } else {
     return at::log_softmax(a, b);
    }
  })
).op(
  "_aten::flatten.using_ints(Tensor self, int start_dim=0, int end_dim=-1) -> Tensor",
  torch::RegisterOperators::options().kernel(c10::DispatchKey::CPUTensorId,
  [](const Tensor & self, int64_t start_dim, int64_t end_dim) {
  #ifdef USE_STATIC_DISPATCH
     at::AutoNonVariableTypeMode non_var_type_mode(true);
  #endif
     return at::flatten(self, start_dim, end_dim);
  })
).op(
  "_prim::NumToTensor",
  torch::RegisterOperators::options().catchAllKernel(
  [](at::Scalar s) -> at::Tensor {
      return at::scalar_to_tensor(s);
  })
).op(
  // Dummy operator that does nothing. Used to reserve a location of an operator table.
  "_prim::ListConstruct.int",
  torch::RegisterOperators::options().catchAllKernel(
  []() {
  })
).op(
  "_prim::ListConstruct.float",
  torch::RegisterOperators::options().catchAllKernel(
  []() {
  })
).op(
  "_prim::ListConstruct.bool",
  torch::RegisterOperators::options().catchAllKernel(
  []() {
  })
).op(
  "_prim::ListConstruct.Tensor",
  torch::RegisterOperators::options().catchAllKernel(
  []() {
  })
).op(
  "_prim::ListConstruct.generic",
  torch::RegisterOperators::options().catchAllKernel(
  []() {
  })

// Pytext operators
).op(
  "_aten::embedding(Tensor weight, Tensor indices, int padding_idx=-1, bool scale_grad_by_freq=False, bool sparse=False) -> Tensor",
  torch::RegisterOperators::options().kernel(c10::DispatchKey::CPUTensorId,
  [](const Tensor & weight, const Tensor & indices, int64_t padding_idx, bool scale_grad_by_freq, bool sparse) {
     return at::embedding(weight, indices, padding_idx, scale_grad_by_freq, sparse);
  })
).op(
  "_aten::dropout(Tensor input, float p, bool train) -> Tensor",
  torch::RegisterOperators::options().kernel(c10::DispatchKey::CPUTensorId,
  [](const Tensor & input, double p, bool train) {
     return at::dropout(input, p, train);
  })
).op(
  "_aten::permute(Tensor(a) self, int[] dims) -> Tensor(a)",
  torch::RegisterOperators::options()
    .kernel<&permute_kernel>(c10::DispatchKey::CPUTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA)
).op(
  "_aten::matmul(Tensor self, Tensor other) -> Tensor",
  torch::RegisterOperators::options().kernel(c10::DispatchKey::CPUTensorId,
  [](const Tensor & self, const Tensor & other) {
     return at::matmul(self, other);
  })
).op(
  "_aten::mul.Tensor(Tensor self, Tensor other) -> Tensor",
  torch::RegisterOperators::options().kernel(c10::DispatchKey::CPUTensorId,
  [](const Tensor & self, const Tensor & other) {
     return at::mul(self, other);
  })
).op(
  "_aten::upsample_nearest2d(Tensor self, int[2] output_size, float? scales_h=None, float? scales_w=None) -> Tensor",
  torch::RegisterOperators::options().kernel<&upsample_nearest2d_kernel>(c10::DispatchKey::CPUTensorId)
).op(
  "_aten::tanh(Tensor self) -> Tensor",
  torch::RegisterOperators::options().kernel(c10::DispatchKey::CPUTensorId,
  [](const Tensor & self) {
     return at::tanh(self);
  })
).op(
  "_aten::max.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)",
  torch::RegisterOperators::options().kernel(c10::DispatchKey::CPUTensorId,
  [](const Tensor & self, int64_t dim, bool keepdim) {
     return at::max(self, dim, keepdim);
  })
).op(
  "_aten::cat(Tensor[] tensors, int dim=0) -> Tensor",
  torch::RegisterOperators::options().kernel<&cat_kernel>(c10::DispatchKey::CPUTensorId)
).op(
  "_aten::__is__(t1 self, t2 obj) -> bool",
  torch::RegisterOperators::options().catchAllKernel<&__is__kernel>()
).op(
  "_aten::log_softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor",
  torch::RegisterOperators::options().kernel<&log_softmax_kernel>(c10::DispatchKey::CPUTensorId)
).op(
  "_aten::softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor",
  torch::RegisterOperators::options().kernel<&softmax_kernel>(c10::DispatchKey::CPUTensorId)
).op(
  "_aten::softplus(Tensor self, Scalar beta=1, Scalar threshold=20) -> Tensor",
  torch::RegisterOperators::options().kernel(c10::DispatchKey::CPUTensorId,
  [](const Tensor& self, Scalar beta, Scalar threshold) {
    return at::softplus(self, beta, threshold);
  })
).op(
  "_aten::warn() -> void",
  torch::RegisterOperators::options().catchAllKernel<&warn_kernel>()
).op(
  "_prim::unchecked_cast",
  torch::RegisterOperators::options().catchAllKernel(
  []() {
  })
).op(
  "_prim::TupleConstruct",
  torch::RegisterOperators::options().catchAllKernel(
  []() {
  })
).op(
  "_prim::TupleUnpack(any self) -> any",
  torch::RegisterOperators::options().catchAllKernel<&tupleunpack_kernel>()
).op(
  "_aten::format(str self, ...) -> str",
  torch::RegisterOperators::options().catchAllKernel<&format_kernel>()
      .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA)
).op(
  "_aten::append.Tensor(Tensor self) -> void",
  torch::RegisterOperators::options().kernel<&listAppend<at::Tensor>>(c10::DispatchKey::CPUTensorId)
).op(
  "_aten::append.int(int self) -> void",
  torch::RegisterOperators::options().catchAllKernel<&listAppend<int64_t>>()
  // Segmentation
).op(
  "_aten::sigmoid(Tensor self) -> Tensor",
  torch::RegisterOperators::options().kernel(c10::DispatchKey::CPUTensorId,
  [](const Tensor & self) {
     return at::sigmoid(self);
  })
).op(torch::RegisterOperators::options()
    .schema("_aten::to.dtype(Tensor self, ScalarType dtype, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor")
    .kernel<&to_dtype_kernal>(c10::DispatchKey::CPUTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
.op(torch::RegisterOperators::options()
    .schema("_prim::TupleIndex(any self, int index) -> any")
    .catchAllKernel<&TupleIndex_kernel>())
.op(torch::RegisterOperators::options()
    .schema("_prim::RaiseException(str msg) -> ()")
    .kernel<&pop_kernal>(c10::DispatchKey::CPUTensorId)
    .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
    ;
}
