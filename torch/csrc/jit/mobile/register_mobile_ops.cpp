#include <ATen/core/op_registration/op_registration.h>
#include <ATen/ATen.h>
#include <ATen/core/stack.h>

using Stack = std::vector<c10::IValue>;
using torch::jit::peek;
using torch::jit::drop;
using torch::jit::pack;
using torch::jit::push;
using torch::jit::pop;

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

template <typename T>
void listAppend(Stack& stack) {
  T el = pop(stack).to<T>();
  c10::List<T> list = pop(stack).to<c10::List<T>>();

  list.push_back(std::move(el));
  push(stack, std::move(list));
}
}

static auto registry0 = torch::RegisterOperators().op(
  "_aten::add.Tensor",
  torch::RegisterOperators::options().kernel(c10::TensorTypeId::CPUTensorId,
  [](at::Tensor a, at::Tensor b, at::Scalar c) -> at::Tensor {
    return at::add(a, b, c);
  })
).op(
  "_aten::add.Scalar",
  torch::RegisterOperators::options().kernel(c10::TensorTypeId::CPUTensorId,
  [](at::Tensor a, at::Scalar b, at::Scalar c) -> at::Tensor {
    return at::add(a, b, c);
  })
).op(
  "_aten::add_.Tensor",
  torch::RegisterOperators::options().kernel(c10::TensorTypeId::CPUTensorId,
                                           [](at::Tensor a, at::Tensor b, at::Scalar c) -> at::Tensor {
                                             return at::add(a, b, c);
                                           })
).op(
  "_aten::adaptive_avg_pool2d(Tensor self, int[2] output_size) -> Tensor",
  torch::RegisterOperators::options().kernel(c10::TensorTypeId::CPUTensorId,
  [](c10::OperatorKernel* kernel, Stack* stack) {
  #ifdef USE_STATIC_DISPATCH
   at::AutoNonVariableTypeMode non_var_type_mode(true);
  #endif
   auto result_ = at::adaptive_avg_pool2d(
       (std::move(peek(*stack, 0, 2))).toTensor(),
       (std::move(peek(*stack, 1, 2))).toIntListRef()
       );
   drop(*stack, 2);
   pack(*stack, std::move(result_));
  })
).op(
  "_aten::mm",
  torch::RegisterOperators::options().kernel(c10::TensorTypeId::CPUTensorId,
  [](at::Tensor a, at::Tensor b) -> at::Tensor {
    return at::mm(a, b);
  })
).op(
  "_aten::_convolution(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups, bool benchmark, bool deterministic, bool cudnn_enabled) -> Tensor",
  torch::RegisterOperators::options().kernel(c10::TensorTypeId::CPUTensorId,
  [](c10::OperatorKernel* kernel, Stack* stack) {
#ifdef USE_STATIC_DISPATCH
     at::AutoNonVariableTypeMode non_var_type_mode(true);
#endif
     auto result_ = at::_convolution(
         (std::move(peek(*stack, 0, 12))).toTensor(),
         (std::move(peek(*stack, 1, 12))).toTensor(),
         toOptionalTensor((std::move(peek(*stack, 2, 12)))),
         (std::move(peek(*stack, 3, 12))).toIntListRef(),
         (std::move(peek(*stack, 4, 12))).toIntListRef(),
         (std::move(peek(*stack, 5, 12))).toIntListRef(),
         (std::move(peek(*stack, 6, 12))).toBool(),
         (std::move(peek(*stack, 7, 12))).toIntListRef(),
         (std::move(peek(*stack, 8, 12))).toInt(),
         (std::move(peek(*stack, 9, 12))).toBool(),
         (std::move(peek(*stack, 10, 12))).toBool(),
         (std::move(peek(*stack, 11, 12))).toBool()
         );
     drop(*stack, 12);
     pack(*stack, std::move(result_));
   })
).op(
  "_aten::conv2d(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1, int groups=1) -> Tensor",
  torch::RegisterOperators::options().kernel(c10::TensorTypeId::CPUTensorId,
  [](c10::OperatorKernel* kernel, Stack* stack) {
#ifdef USE_STATIC_DISPATCH
     at::AutoNonVariableTypeMode non_var_type_mode(true);
#endif
     auto result_ = at::conv2d(
         (std::move(peek(*stack, 0, 7))).toTensor(),
         (std::move(peek(*stack, 1, 7))).toTensor(),
         toOptionalTensor((std::move(peek(*stack, 2, 7)))),
         (std::move(peek(*stack, 3, 7))).toIntListRef(),
         (std::move(peek(*stack, 4, 7))).toIntListRef(),
         (std::move(peek(*stack, 5, 7))).toIntListRef(),
         (std::move(peek(*stack, 6, 7))).toInt()
         );
     drop(*stack, 7);
     pack(*stack, std::move(result_));
  })
).op(
  "_aten::batch_norm",
  torch::RegisterOperators::options().kernel(c10::TensorTypeId::CPUTensorId,
  [] (at::Tensor input, c10::optional<at::Tensor> weight, c10::optional<at::Tensor> bias,
    c10::optional<at::Tensor> running_mean, c10::optional<at::Tensor> running_var,
    bool training, double momentum, double eps, bool cudnn_enabled) {
   return at::batch_norm(input, optional_to_tensor(weight), optional_to_tensor(bias),
                         optional_to_tensor(running_mean), optional_to_tensor(running_var),
                         training, momentum, eps, cudnn_enabled);
  })
).op(
  "_aten::max_pool2d_with_indices(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)",
  torch::RegisterOperators::options().kernel(c10::TensorTypeId::CPUTensorId,
  [](c10::OperatorKernel* kernel, Stack* stack) {
  #ifdef USE_STATIC_DISPATCH
     at::AutoNonVariableTypeMode non_var_type_mode(true);
  #endif
     auto result_ = at::max_pool2d_with_indices(
         (std::move(peek(*stack, 0, 6))).toTensor(),
         (std::move(peek(*stack, 1, 6))).toIntListRef(),
         (std::move(peek(*stack, 2, 6))).toIntListRef(),
         (std::move(peek(*stack, 3, 6))).toIntListRef(),
         (std::move(peek(*stack, 4, 6))).toIntListRef(),
         (std::move(peek(*stack, 5, 6))).toBool()
         );
     drop(*stack, 6);
     pack(*stack, std::move(result_));
  })
).op(
  "_aten::max_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> Tensor",
  torch::RegisterOperators::options().kernel(c10::TensorTypeId::CPUTensorId,
  [](c10::OperatorKernel* kernel, Stack* stack) {
#ifdef USE_STATIC_DISPATCH
   at::AutoNonVariableTypeMode non_var_type_mode(true);
#endif
   auto result_ = at::max_pool2d(
       (std::move(peek(*stack, 0, 6))).toTensor(),
       (std::move(peek(*stack, 1, 6))).toIntListRef(),
       (std::move(peek(*stack, 2, 6))).toIntListRef(),
       (std::move(peek(*stack, 3, 6))).toIntListRef(),
       (std::move(peek(*stack, 4, 6))).toIntListRef(),
       (std::move(peek(*stack, 5, 6))).toBool()
       );
   drop(*stack, 6);
   pack(*stack, std::move(result_));
  })
).op(
  "_aten::threshold",
  torch::RegisterOperators::options().kernel(c10::TensorTypeId::CPUTensorId,
  [](at::Tensor self, at::Scalar threshold, at::Scalar value) {
   return at::threshold_(self, threshold, value);
  })
).op(
  "_aten::relu(Tensor self) -> Tensor",
  torch::RegisterOperators::options().kernel(c10::TensorTypeId::CPUTensorId,
  [](c10::OperatorKernel* kernel, Stack* stack) {

  #ifdef USE_STATIC_DISPATCH
  at::AutoNonVariableTypeMode non_var_type_mode(true);
  #endif
  auto result_ = at::relu(
  (std::move(peek(*stack, 0, 1))).toTensor()
  );
  drop(*stack, 1);
  pack(*stack, std::move(result_));
})
).op(
  "_aten::relu_",
  torch::RegisterOperators::options().kernel(c10::TensorTypeId::CPUTensorId,
  [](at::Tensor a) -> at::Tensor {
    return at::relu_(a);
  })
).op(
  "_aten::t(Tensor(a) self) -> Tensor(a)",
  torch::RegisterOperators::options().kernel(c10::TensorTypeId::CPUTensorId,
  [](c10::OperatorKernel* kernel, Stack* stack) {

  #ifdef USE_STATIC_DISPATCH
  at::AutoNonVariableTypeMode non_var_type_mode(true);
  #endif
  auto result_ = at::t(
  (std::move(peek(*stack, 0, 1))).toTensor()
  );
  drop(*stack, 1);
  pack(*stack, std::move(result_));
  }).aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA)
).op(
  "_aten::size.int",
  torch::RegisterOperators::options().kernel(c10::TensorTypeId::CPUTensorId,
  [](at::Tensor a, int64_t dim) -> int64_t {
   return at::size(a, dim);
  })
).op(
  "_aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor",
  torch::RegisterOperators::options().kernel(c10::TensorTypeId::CPUTensorId,
  [](c10::OperatorKernel* kernel, Stack* stack) {

  #ifdef USE_STATIC_DISPATCH
  at::AutoNonVariableTypeMode non_var_type_mode(true);
  #endif
  constexpr size_t nvar = 5;
  auto result_ = at::addmm(
  (std::move(peek(*stack, 0, nvar))).toTensor(),
  (std::move(peek(*stack, 1, nvar))).toTensor(),
  (std::move(peek(*stack, 2, nvar))).toTensor(),
  (std::move(peek(*stack, 3, nvar))).toScalar(),
  (std::move(peek(*stack, 4, nvar))).toScalar()
  );
  drop(*stack, 5);
  pack(*stack, std::move(result_));
  })
).op(
  "_aten::view(Tensor(a) self, int[] size) -> Tensor(a)",
  torch::RegisterOperators::options().kernel(c10::TensorTypeId::CPUTensorId,
  [](c10::OperatorKernel* kernel, Stack* stack) {
#ifdef USE_STATIC_DISPATCH
   at::AutoNonVariableTypeMode non_var_type_mode(true);
#endif
   auto result_ = ((std::move(peek(*stack, 0, 2))).toTensor()).view(
       (std::move(peek(*stack, 1, 2))).toIntListRef()
       );
   drop(*stack, 2);
   pack(*stack, std::move(result_));
  }).aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA)
).op(
  "_aten::dim",
  torch::RegisterOperators::options().kernel(c10::TensorTypeId::CPUTensorId,
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
  torch::RegisterOperators::options().kernel(c10::TensorTypeId::CPUTensorId,
  [](at::Tensor a, int64_t b, c10::optional<int64_t> c) -> at::Tensor {
    if (c.has_value()) {
     return at::log_softmax(a, b, static_cast<c10::ScalarType>(c.value()));
    } else {
     return at::log_softmax(a, b);
    }
  })
).op(
  "_aten::flatten.using_ints(Tensor self, int start_dim=0, int end_dim=-1) -> Tensor",
  torch::RegisterOperators::options().kernel(c10::TensorTypeId::CPUTensorId,
  [](c10::OperatorKernel* kernel, Stack* stack) {
  #ifdef USE_STATIC_DISPATCH
     at::AutoNonVariableTypeMode non_var_type_mode(true);
  #endif
     auto result_ = at::flatten(
         (std::move(peek(*stack, 0, 3))).toTensor(),
         (std::move(peek(*stack, 1, 3))).toInt(),
         (std::move(peek(*stack, 2, 3))).toInt()
     );
     drop(*stack, 3);
     pack(*stack, std::move(result_));
  })
).op(
  "_aten::Int",
  torch::RegisterOperators::options().kernel(c10::TensorTypeId::CPUTensorId,
                                           [](at::Tensor a) -> int64_t {
                                             return a.item<int64_t>();
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
  torch::RegisterOperators::options().kernel(c10::TensorTypeId::CPUTensorId,
  [](c10::OperatorKernel* kernel, Stack* stack) {
     constexpr int N = 5;
     auto result_ = at::embedding(
         (std::move(peek(*stack, 0, N))).toTensor(),
         (std::move(peek(*stack, 1, N))).toTensor(),
         (std::move(peek(*stack, 2, N))).toInt(),
         (std::move(peek(*stack, 3, N))).toBool(),
         (std::move(peek(*stack, 4, N))).toBool()
     );
     drop(*stack, N);
     pack(*stack, std::move(result_));
  })
).op(
  "_aten::dropout(Tensor input, float p, bool train) -> Tensor",
  torch::RegisterOperators::options().kernel(c10::TensorTypeId::CPUTensorId,
  [](c10::OperatorKernel* kernel, Stack* stack) {
     auto result_ = at::dropout(
         (std::move(peek(*stack, 0, 3))).toTensor(),
         (std::move(peek(*stack, 1, 3))).toDouble(),
         (std::move(peek(*stack, 2, 3))).toBool()
     );
     drop(*stack, 3);
     pack(*stack, std::move(result_));
  })
).op(
  "_aten::permute(Tensor(a) self, int[] dims) -> Tensor(a)",
  torch::RegisterOperators::options().kernel(c10::TensorTypeId::CPUTensorId,
  [](c10::OperatorKernel* kernel, Stack* stack) {
     auto result_ = ((std::move(peek(*stack, 0, 2))).toTensor()).permute(
         (std::move(peek(*stack, 1, 2))).toIntListRef()
     );
     drop(*stack, 2);
     pack(*stack, std::move(result_));
  }).aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA)
).op(
  "_aten::matmul(Tensor self, Tensor other) -> Tensor",
  torch::RegisterOperators::options().kernel(c10::TensorTypeId::CPUTensorId,
  [](c10::OperatorKernel* kernel, Stack* stack) {
     auto result_ = at::matmul(
         (std::move(peek(*stack, 0, 2))).toTensor(),
         (std::move(peek(*stack, 1, 2))).toTensor()
     );
     drop(*stack, 2);
     pack(*stack, std::move(result_));
  })
).op(
  "_aten::mul.Tensor(Tensor self, Tensor other) -> Tensor",
  torch::RegisterOperators::options().kernel(c10::TensorTypeId::CPUTensorId,
  [](c10::OperatorKernel* kernel, Stack* stack) {
     auto result_ = at::mul(
         (std::move(peek(*stack, 0, 2))).toTensor(),
         (std::move(peek(*stack, 1, 2))).toTensor()
     );
     drop(*stack, 2);
     pack(*stack, std::move(result_));
  })
).op(
  "_aten::tanh(Tensor self) -> Tensor",
  torch::RegisterOperators::options().kernel(c10::TensorTypeId::CPUTensorId,
  [](c10::OperatorKernel* kernel, Stack* stack) {
     auto result_ = at::tanh(
         (std::move(peek(*stack, 0, 1))).toTensor()
     );
     drop(*stack, 1);
     pack(*stack, std::move(result_));
  })
).op(
  "_aten::max.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)",
  torch::RegisterOperators::options().kernel(c10::TensorTypeId::CPUTensorId,
  [](c10::OperatorKernel* kernel, Stack* stack) {
     auto result_ = at::max(
         (std::move(peek(*stack, 0, 3))).toTensor(),
         (std::move(peek(*stack, 1, 3))).toInt(),
         (std::move(peek(*stack, 2, 3))).toBool()
     );
     drop(*stack, 3);
     pack(*stack, std::move(result_));
  })
).op(
  "_aten::cat(Tensor[] tensors, int dim=0) -> Tensor",
  torch::RegisterOperators::options().kernel(c10::TensorTypeId::CPUTensorId,
  [](c10::OperatorKernel* kernel, Stack* stack) {
     auto result_ = at::cat(
         (std::move(peek(*stack, 0, 2))).toTensorListRef(),
         (std::move(peek(*stack, 1, 2))).toInt()
     );
     drop(*stack, 2);
     pack(*stack, std::move(result_));
  })
).op(
  "_aten::__is__(t1 self, t2 obj) -> bool",
  torch::RegisterOperators::options().catchAllKernel(
  [](c10::OperatorKernel* kernel, Stack* stack) {
      c10::IValue self, obj;
      pop(*stack, self, obj);
      push(*stack, self.isSameIdentity(obj));
  })
).op(
  "_aten::log_softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor",
  torch::RegisterOperators::options().kernel(c10::TensorTypeId::CPUTensorId,
  [](c10::OperatorKernel* kernel, Stack* stack) {
     auto result_ = at::log_softmax(
       (std::move(peek(*stack, 0, 3))).toTensor(),
       (std::move(peek(*stack, 1, 3))).toInt(),
       (std::move(peek(*stack, 2, 3))).toOptional<c10::ScalarType>()
     );
     drop(*stack, 3);
     pack(*stack, std::move(result_));
  })
).op(
  "_aten::softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor",
  torch::RegisterOperators::options().kernel(c10::TensorTypeId::CPUTensorId,
  [](c10::OperatorKernel* kernel, Stack* stack) {
     auto result_ = at::softmax(
       (std::move(peek(*stack, 0, 3))).toTensor(),
       (std::move(peek(*stack, 1, 3))).toInt(),
       (std::move(peek(*stack, 2, 3))).toOptional<c10::ScalarType>()
     );
     drop(*stack, 3);
     pack(*stack, std::move(result_));
  })
).op(
  "_aten::warn() -> void",
  torch::RegisterOperators::options().catchAllKernel(
  [](c10::OperatorKernel* kernel, Stack* stack) {
    drop(*stack, 1);
    pop(*stack);
  })
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
  "_prim::TupleUnpack",
  torch::RegisterOperators::options().catchAllKernel(
  []() {
  })
).op(
  "_aten::format",
  torch::RegisterOperators::options().catchAllKernel(
  []() {
  })
).op(
  "_aten::append.Tensor(Tensor self) -> void",
  torch::RegisterOperators::options().kernel(c10::TensorTypeId::CPUTensorId,
  [](c10::OperatorKernel* kernel, Stack* stack) {
    listAppend<at::Tensor>(*stack);
  })
).op(
  "_aten::append.int(int self) -> void",
  torch::RegisterOperators::options().catchAllKernel(
  [](c10::OperatorKernel* kernel, Stack* stack) {
    listAppend<int64_t>(*stack);
  })
);

