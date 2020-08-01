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
  for (size_t i = 0; i < t->buf()->ndim(); i++) {
    sizes.push_back(dynamic_cast<const IntImm*>(t->buf()->dim(i))->value());
  }
  return sizes;
}

template <typename T>
inline std::vector<ExprHandle> computeIndicesToBroadcast(
    const std::vector<T>& outputAxes,
    const std::vector<ExprHandle>& inputSizes) {
  if (outputAxes.size() < inputSizes.size()) {
    throw malformed_input("Cannot broadcast to a lower rank tensor");
  }
  std::vector<ExprHandle> bcast;
  auto axisIt = outputAxes.rbegin();
  auto sizeIt = inputSizes.rbegin();
  while (sizeIt != inputSizes.rend()) {
    auto const& size = sizeIt->AsNode<IntImm>();
    if (size && size->value() == 1) {
      bcast.push_back(0);
    } else {
      bcast.push_back(*axisIt);
    }
    ++axisIt;
    ++sizeIt;
  }
  std::reverse(bcast.begin(), bcast.end());
  return bcast;
}

class TORCH_API TensorExprKernel {
 public:
  explicit TensorExprKernel(const std::shared_ptr<Graph>& subgraph);

  void run(Stack& stack);

  void fallback(Stack& stack) {
    InterpreterState(code_).run(stack);
  }

  Stmt* getCodeGenStmt();

 private:
  enum BackendType {
    kUninitialized,
    kSimpleIREval,
    kLLVMCodeGen,
    kCudaCodeGen,
  };

  void compile();

  void runKernel(Stack& stack);

  std::vector<DimArg> dimsFromSizes(const std::vector<ExprHandle>& sizes);
  std::vector<ExprHandle> sizesForValue(const torch::jit::Value* v);
  std::vector<ExprHandle> inferSizesForValue(const torch::jit::Value* v);
  std::vector<ExprHandle> sizesFromVaryingShape(
      const c10::VaryingShape<int64_t>& shape);

  std::vector<ExprHandle> broadcastShapes(
      const std::vector<ExprHandle>& a,
      const std::vector<ExprHandle>& b);
  std::vector<ExprHandle> broadcastShapes(
      std::vector<std::vector<ExprHandle>> shapes);

  ExprHandle constant(const torch::jit::Value* v);

  template <typename T, typename T1>
  ExprHandle broadcast(const T& t, const std::vector<T1>& axes) {
    return t->call(computeIndicesToBroadcast(
        axes, ExprVectorToExprHandleVector(t->buf()->dims())));
  }

  template <typename T, typename T1>
  ExprHandle chunk(
      const T& t,
      size_t chunkIdx,
      size_t dim,
      size_t chunks,
      const std::vector<T1>& axes) {
    auto sizes = bufferSizes(t);
    size_t step = sizes[dim] / chunks;

    std::vector<ExprHandle> indices;
    for (size_t i = 0; i < axes.size(); ++i) {
      if (i == dim) {
        indices.push_back(axes[i] + IntImm::make(chunkIdx * step));
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

  Tensor* computeOneOperand(
      const std::string& name,
      const torch::jit::Value* v,
      const std::function<ExprHandle(const ExprHandle&)>& innerExpr);

  Tensor* computeTwoOperand(
      const std::string& name,
      const torch::jit::Value* v,
      const std::function<ExprHandle(const ExprHandle&, const ExprHandle&)>&
          innerExpr);

  Tensor* computeTwoOperandWithAlpha(
      const std::string& name,
      const torch::jit::Value* v,
      const std::function<ExprHandle(const ExprHandle&, const ExprHandle&)>&
          innerExpr);

  Tensor* computeThreeOperand(
      const std::string& name,
      const torch::jit::Value* v,
      const std::function<
          ExprHandle(const ExprHandle&, const ExprHandle&, const ExprHandle&)>&
          innerExpr);

  Tensor* computeConditionWithTwoOperand(
      const std::string& name,
      const torch::jit::Value* v,
      const std::function<
          ExprHandle(const ExprHandle&, const ExprHandle&, const ExprHandle&)>&
          innerExpr);

  Tensor* computeFourOperand(
      const std::string& name,
      const torch::jit::Value* v,
      const std::function<ExprHandle(
          const ExprHandle&,
          const ExprHandle&,
          const ExprHandle&,
          const ExprHandle&)>& innerExpr);

  Tensor* computeValue(const torch::jit::Value* v);

  void flattenTensors(BackendType backendType);
  Stmt* generateStmt(BackendType backendType);
  std::vector<CodeGen::BufferArg> prepareBufferArgs();

  std::string getCodeGenName(BackendType backendType);

  std::vector<CodeGen::CallArg> prepareRunArgs(
      const at::ArrayRef<IValue>& inputs,
      std::vector<at::Tensor>& outputs);
  BackendType inferBackendTypeFromDevice(at::Device device);
  at::Device pickDeviceType(const at::ArrayRef<torch::jit::Value*>& inputs);

  void bindInput(const torch::jit::Value* input);

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

  int64_t nInputs_ = 0;
  std::vector<KernelArg> kernelArgs_;
  std::vector<Tensor*> tensorOutputs_;
  std::vector<Tensor*> flatTensorOutputs_;
  std::unordered_map<int64_t, Tensor*> tensors_;
  std::unordered_map<int64_t, VarHandle> scalars_;
  std::unique_ptr<CodeGen> codegen_;
  at::Device device_ = at::kCPU;
  KernelArena kernelArena_;
  std::vector<TypePtr> inputTypes_;
  std::shared_ptr<Graph> graph_;
  Code code_;
  bool fallback_{false};
  bool hasRandom_{false};
  bool hasBroadcast_{false};
  std::unordered_map<const torch::jit::Value*, std::vector<ExprHandle>>
      known_sizes_;
};

TORCH_API int& getTECudaPointwiseLoopLevels();
TORCH_API int& getTECudaPointwiseBlockCount();
TORCH_API int& getTECudaPointwiseBlockSize();
TORCH_API bool fallbackAllowed();
TORCH_API bool setFallbackAllowed(bool value);

} // namespace tensorexpr
} // namespace jit
} // namespace torch
