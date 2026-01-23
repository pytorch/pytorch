#include <ATen/record_function.h>

#include <ATen/CPUFunctions.h>
#include <c10/util/irange.h>

#include <c10/util/Enumerate.h>
#include <torch/nativert/kernels/PrimKernelRegistry.h>

namespace torch::nativert {

C10_DEFINE_REGISTRY(PrimKernelRegistry, OpKernel, const Node*)

namespace {

class OpKernel_prim_listpack : public OpKernel {
 public:
  explicit OpKernel_prim_listpack(const Node* node)
      : OpKernel(node, OpKernelKind::kPrimKernel) {
    auto listType = node->outputs()[0]->type();
    switch (listType.kind()) {
      case Type::Kind::TensorList:
        type_ = c10::TensorType::get();
        break;
      case Type::Kind::SymIntList:
        type_ = c10::IntType::get();
        break;
      case Type::Kind::OptionalTensorList:
        type_ = c10::OptionalType::create(c10::TensorType::get());
        break;
      default:
        TORCH_CHECK(false, "Unsupported list type: ", listType);
    }
  }

  void computeInternal(ExecutionFrame& executionFrame) const final {
    RECORD_USER_SCOPE("nativert::OpKernel_prim_listpack");
    c10::List<c10::IValue> list(type_);
    list.reserve(numInputs());
    for (size_t i = 0; i < numInputs(); ++i) {
      if (KernelInput(i).isNone()) {
        list.emplace_back();
      } else {
        list.push_back(KernelInput(i));
      }
    }
    KernelOutput(0) = std::move(list);
  }

 private:
  c10::TypePtr type_;
};

} // namespace

C10_REGISTER_TYPED_CLASS(
    PrimKernelRegistry,
    "prim.ListPack",
    OpKernel_prim_listpack)

REGISTER_PRIM_KERNEL("prim.ListUnpack", prim_listunpack, {
  RECORD_USER_SCOPE("nativert::OpKernel_prim_listunpack");
  auto inputListRef = KernelInput(0).toListRef();
  for (size_t i = 0; i < inputListRef.size(); ++i) {
    KernelOutput(i) = inputListRef[i];
  }
})

// Noop for input and output
REGISTER_PRIM_KERNEL("prim.Input", prim_input, {})
REGISTER_PRIM_KERNEL("prim.Output", prim_output, {})

namespace {

class OpKernel_variadic_concat : public OpKernel {
 public:
  explicit OpKernel_variadic_concat(const Node* node)
      : OpKernel(node, OpKernelKind::kPrimKernel) {
    dim_ = !node_->attributes().empty()
        ? constantToIValue(node_->getAttribute("dim").value).toInt()
        : 0;
  }
  void computeInternal(ExecutionFrame& executionFrame) const final {
    {
      const size_t numNodeInps = numInputs();
      auto numCatInps = numNodeInps;
      auto dim = dim_;
      if (KernelInput(numCatInps - 1).isInt()) {
        dim = KernelInput(numCatInps - 1).toInt();
        numCatInps--;
      }
      std::vector<at::Tensor> inputs(numCatInps);
      for (const auto i : c10::irange(numCatInps)) {
        inputs[i] = KernelInput(i).toTensor();
      }

      if (KernelOutput(0).isNone()) {
        KernelOutput(0) = at::cpu::cat(inputs, dim);
        return;
      }
      auto& out_t = KernelOutput(0).toTensor();
      at::cpu::cat_outf(inputs, dim, out_t);
    }
  }

 private:
  int dim_;
};

} // namespace

C10_REGISTER_TYPED_CLASS(
    PrimKernelRegistry,
    "prim.VarConcat",
    OpKernel_variadic_concat)

namespace {

class OpKernel_variadic_stack : public OpKernel {
 public:
  explicit OpKernel_variadic_stack(const Node* node)
      : OpKernel(node, OpKernelKind::kPrimKernel) {
    dim_ = !node_->attributes().empty()
        ? constantToIValue(node_->getAttribute("dim").value).toInt()
        : 0;
  }
  void computeInternal(ExecutionFrame& executionFrame) const final {
    {
      const size_t numNodeInps = numInputs();
      auto numStackInps = numNodeInps;
      auto dim = dim_;
      if (KernelInput(numStackInps - 1).isInt()) {
        dim = KernelInput(numStackInps - 1).toInt();
        numStackInps--;
      }
      std::vector<at::Tensor> inputs(numStackInps);
      for (const auto i : c10::irange(numStackInps)) {
        inputs[i] = KernelInput(i).toTensor();
      }
      auto& out = KernelOutput(0);
      if (out.isNone()) {
        out = at::native::_stack_cpu(inputs, dim);
        return;
      }
      auto& out_t = out.toTensor();
      at::native::_stack_out_cpu(inputs, dim, out_t);
    }
  }

 private:
  int64_t dim_;
};
} // namespace

C10_REGISTER_TYPED_CLASS(
    PrimKernelRegistry,
    "prim.VarStack",
    OpKernel_variadic_stack)

} // namespace torch::nativert
