#pragma once

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/interpreter.h>
#include <torch/csrc/jit/tensorexpr/codegen.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>

namespace torch {
namespace jit {
namespace tensorexpr {

template <typename T>
inline std::vector<int64_t> bufferSizes(const T& t) {
  std::vector<int64_t> sizes;
  for (int i = 0; i < t->function()->ndim(); i++) {
    sizes.push_back(
        dynamic_cast<const IntImm*>(t->function()->dim(i))->value());
  }
  return sizes;
}

template <typename T>
inline std::vector<ExprHandle> computeIndicesToBroadcast(
    const std::vector<T>& output_axes,
    const std::vector<ExprHandle>& input_sizes) {
  TORCH_CHECK(
      output_axes.size() >= input_sizes.size(),
      "Cannot broadcast to a lower rank tensor");
  std::vector<ExprHandle> bcast;
  auto axis_it = output_axes.rbegin();
  auto size_it = input_sizes.rbegin();
  while (size_it != input_sizes.rend()) {
    auto const& size = size_it->AsNode<IntImm>();
    if (size && size->value() == 1) {
      bcast.push_back(0);
    } else {
      bcast.push_back(*axis_it);
    }
    ++axis_it;
    ++size_it;
  }
  std::reverse(bcast.begin(), bcast.end());
  return bcast;
}

class TensorExprKernel {
 public:
  explicit TensorExprKernel(const std::shared_ptr<Graph>& subgraph);

  void run(Stack& stack);

  void fallback(Stack& stack) {
    InterpreterState(code_).run(stack);
  }

 private:
  enum BackendType {
    kUninitialized,
    kSimpleIREval,
    kLLVMCodeGen,
    kCudaCodeGen,
  };

  void compile();

  void runKernel(Stack& stack);

  ExprHandle constant(const torch::jit::Value* v);

  template <typename T, typename T1>
  ExprHandle broadcast(const T& t, const std::vector<T1>& axes) {
    return t->call(computeIndicesToBroadcast(
        axes, ExprVectorToExprHandleVector(t->function()->dims())));
  }

  template <typename T, typename T1>
  ExprHandle chunk(
      const T& t,
      size_t chunk_idx,
      size_t dim,
      size_t chunks,
      const std::vector<T1>& axes) {
    auto sizes = bufferSizes(t);
    size_t step = sizes[dim] / chunks;

    std::vector<ExprHandle> indices;
    for (size_t i = 0; i < axes.size(); ++i) {
      if (i == dim) {
        indices.push_back(axes[i] + IntImm::make(chunk_idx * step));
      } else {
        indices.push_back(axes[i]);
      }
    }

    return t->call(indices);
  }

  std::vector<ExprHandle> valueShape(const torch::jit::Value* v);

  void promoteInputs(std::vector<ExprHandle>& inputs);

  ExprHandle demoteOutput(const ExprHandle& e, const torch::jit::Value* v);

  template <typename T>
  ExprHandle tensorOrConstant(
      const torch::jit::Value* v,
      const std::vector<T>& axes) {
    auto ti = tensors_.find(v->unique());
    if (ti != tensors_.end()) {
      return broadcast(ti->second, axes);
    }
    return constant(v);
  }

  Tensor* ComputeOneOperand(
      const std::string& name,
      const torch::jit::Value* v,
      const std::function<ExprHandle(const ExprHandle&)>& inner_expr);

  Tensor* ComputeTwoOperand(
      const std::string& name,
      const torch::jit::Value* v,
      const std::function<ExprHandle(const ExprHandle&, const ExprHandle&)>&
          inner_expr);

  Tensor* ComputeTwoOperandWithAlpha(
      const std::string& name,
      const torch::jit::Value* v,
      const std::function<ExprHandle(const ExprHandle&, const ExprHandle&)>&
          inner_expr);

  Tensor* ComputeThreeOperand(
      const std::string& name,
      const torch::jit::Value* v,
      const std::function<
          ExprHandle(const ExprHandle&, const ExprHandle&, const ExprHandle&)>&
          inner_expr);

  Tensor* ComputeConditionWithTwoOperand(
      const std::string& name,
      const torch::jit::Value* v,
      const std::function<
          ExprHandle(const ExprHandle&, const ExprHandle&, const ExprHandle&)>&
          inner_expr);

  Tensor* ComputeFourOperand(
      const std::string& name,
      const torch::jit::Value* v,
      const std::function<ExprHandle(
          const ExprHandle&,
          const ExprHandle&,
          const ExprHandle&,
          const ExprHandle&)>& inner_expr);

  Tensor* ComputeValue(const torch::jit::Value* v);

  void LowerToBackend(BackendType backend_type);

  void PickAndCheckBackendType(const at::ArrayRef<IValue>& inputs);

  void CodeGenRun(const std::vector<CodeGen::CallArg>& run_args);

  void bindInput(const torch::jit::Value* input);

  ExprHandle createInputIndexExpr(
      const Buffer& buffer,
      const std::vector<VarHandle>& axes,
      const c10::VaryingShape& sizes,
      const c10::VaryingStrides& strides,
      const c10::VaryingStrides& contiguity,
      const std::unordered_map<int64_t, VarHandle>& sizeVars);

 private:
  struct ShapeArg {
    size_t idx;
    VarHandle var;

    ShapeArg(size_t i, VarHandle v) : idx(i), var(v) {}
  };

  struct KernelArg {
    template <typename B>
    KernelArg(B&& b) : bufferArg_(std::forward<B>(b)) {}

    template <typename B, typename T>
    KernelArg(B&& b, T&& sizes, T&& strides)
        : bufferArg_(b),
          sizeArgs_(std::forward<T>(sizes)),
          strideArgs_(std::forward<T>(strides)) {}

    const CodeGen::BufferArg& buffer() const {
      return bufferArg_;
    }

    const std::vector<ShapeArg>& sizes() const {
      return sizeArgs_;
    }

    const std::vector<ShapeArg>& strides() const {
      return strideArgs_;
    }

    CodeGen::BufferArg bufferArg_;
    std::vector<ShapeArg> sizeArgs_;
    std::vector<ShapeArg> strideArgs_;
  };

  int64_t n_inputs_ = 0;
  std::vector<KernelArg> kernelArgs_;
  std::vector<Tensor*> tensor_outputs_;
  std::unordered_map<int64_t, Tensor*> tensors_;
  std::unordered_map<int64_t, VarHandle> scalars_;
  std::unique_ptr<CodeGen> codegen_;
  KernelArena kernel_arena_;
  BackendType backend_type_ = BackendType::kUninitialized;
  at::Device device_ = at::kCPU;
  std::vector<TypePtr> input_types_;
  std::shared_ptr<Graph> graph_;
  Code code_;
  bool fallback_{false};
  bool hasRandom_{false};
  bool hasBroadcast_{false};
};

TORCH_API int& GetTECudaPointwiseLoopLevels();
TORCH_API int& GetTECudaPointwiseBlockCount();
TORCH_API int& GetTECudaPointwiseBlockSize();

} // namespace tensorexpr
} // namespace jit
} // namespace torch
