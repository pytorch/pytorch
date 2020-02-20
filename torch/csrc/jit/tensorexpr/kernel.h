#pragma once

#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/tensorexpr/codegen.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>

namespace torch {
namespace jit {
namespace tensorexpr {

template <typename T>
inline std::vector<int64_t> bufferSizes(const T& t) {
  std::vector<int64_t> sizes;
  for (int i = 0; i < t.ndim(); i++) {
    sizes.push_back(t.dim(i).template AsNode<IntImm>()->value());
  }
  return sizes;
}

template <typename T>
inline std::vector<Expr> computeIndicesToBroadcast(
    const std::vector<T>& output_axes,
    const std::vector<int64_t>& input_sizes) {
  TORCH_CHECK(
      output_axes.size() >= input_sizes.size(),
      "Cannot broadcast to a lower rank tensor");
  std::vector<Expr> bcast;
  auto axis_it = output_axes.rbegin();
  auto size_it = input_sizes.rbegin();
  while (size_it != input_sizes.rend()) {
    if (*size_it == 1) {
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
  explicit TensorExprKernel(const Graph& subgraph);

  void run(Stack& stack);

 private:
  enum BackendType {
    kUninitialized,
    kSimpleIREval,
    kLLVMCodeGen,
    kCudaCodeGen,
  };

  Expr constant(const torch::jit::Value* v);

  template <typename T, typename T1>
  Expr broadcast(const T& t, const std::vector<T1>& axes) {
    return t.call(computeIndicesToBroadcast(axes, bufferSizes(t)));
  }

  template <typename T, typename T1>
  Expr chunk(
      const T& t,
      size_t chunk_idx,
      size_t dim,
      size_t chunks,
      const std::vector<T1>& axes) {
    auto sizes = bufferSizes(t);
    size_t step = sizes[dim] / chunks;

    std::vector<Expr> indices;
    for (size_t i = 0; i < axes.size(); ++i) {
      if (i == dim) {
        indices.push_back(axes[i] + IntImm::make(chunk_idx * step));
      } else {
        indices.push_back(axes[i]);
      }
    }

    return t.call(indices);
  }

  std::vector<Expr> valueShape(const torch::jit::Value* v);

  void promoteInputs(std::vector<Expr>& inputs);

  Expr demoteOutput(const Expr& e, const torch::jit::Value* v);

  template <typename T>
  Expr tensorOrConstant(
      const torch::jit::Value* v,
      const std::vector<T>& axes) {
    auto ti = tensors_.find(v->unique());
    if (ti != tensors_.end()) {
      return broadcast(ti->second, axes);
    }
    return constant(v);
  }

  Tensor ComputeOneOperand(
      const std::string& name,
      const torch::jit::Value* v,
      std::function<Expr(const Expr&)> inner_expr);

  Tensor ComputeTwoOperand(
      const std::string& name,
      const torch::jit::Value* v,
      std::function<Expr(const Expr&, const Expr&)> inner_expr);

  Tensor ComputeTwoOperandWithAlpha(
      const std::string& name,
      const torch::jit::Value* v,
      std::function<Expr(const Expr&, const Expr&)> inner_expr);

  Tensor ComputeThreeOperand(
      const std::string& name,
      const torch::jit::Value* v,
      std::function<Expr(const Expr&, const Expr&, const Expr&)> inner_expr);

  Tensor ComputeFourOperand(
      const std::string& name,
      const torch::jit::Value* v,
      std::function<Expr(const Expr&, const Expr&, const Expr&, const Expr&)>
          inner_expr);

  Tensor ComputeValue(const torch::jit::Value* v);

  void LowerToBackend(BackendType backend_type);

  void PickAndCheckBackendType(const at::ArrayRef<IValue>& inputs);

  void CodeGenRun(const std::vector<CodeGen::CallArg>& run_args);

  void bindInput(const torch::jit::Value* input);

 private:
  int64_t n_inputs_ = 0;
  std::vector<CodeGen::BufferArg> buffer_args_;
  std::vector<Tensor> tensor_outputs_;
  std::unordered_map<int64_t, Tensor> tensors_;
  std::unordered_map<int64_t, Var> scalars_;
  std::unique_ptr<CodeGen> codegen_;
  KernelArena kernel_arena_;
  BackendType backend_type_ = BackendType::kUninitialized;
  at::Device device_ = at::kCPU;
};

TORCH_API int& GetTECudaPointwiseLoopLevels();
TORCH_API int& GetTECudaPointwiseBlockCount();
TORCH_API int& GetTECudaPointwiseBlockSize();

} // namespace tensorexpr
} // namespace jit
} // namespace torch
