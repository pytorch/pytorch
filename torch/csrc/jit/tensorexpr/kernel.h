#pragma once

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/interpreter.h>
#include <torch/csrc/jit/tensorexpr/analysis.h>
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

class TORCH_API TensorExprKernel {
 public:
  explicit TensorExprKernel(const std::shared_ptr<Graph>& subgraph);

  void run(Stack& stack);

  void fallback(Stack& stack) {
    InterpreterState(code_).run(stack);
  }

  Stmt* getCodeGenStmt();

  std::string getCodeText() {
    return codegen_->getCodeText();
  }

 private:
  enum BackendType {
    kUninitialized,
    kSimpleIREval,
    kLLVMCodeGen,
    kCudaCodeGen,
    kBlockCodeGen,
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
  ExprHandle broadcast(Tensor* t, const std::vector<ExprHandle>& axes);
  ExprHandle chunk(
      Tensor* t,
      size_t chunkIdx,
      size_t dim,
      size_t chunks,
      const std::vector<ExprHandle>& axes);

  std::vector<ExprHandle> valueShape(const torch::jit::Value* v);

  void promoteInputs(std::vector<ExprHandle>& inputs);

  ExprHandle demoteOutput(const ExprHandle& e, const torch::jit::Value* v);

  ExprHandle tensorOrConstant(
      const torch::jit::Value* v,
      const std::vector<ExprHandle>& axes);

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

  Tensor* computeSum(const torch::jit::Value* v);

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

  // Captures the information for reduction operation nodes.
  struct ReductionInfo {
    std::vector<DimArg> reductionDims;
    std::vector<DimArg> outputDims;
    std::vector<size_t> axes;
    bool keepdim;
    c10::optional<Dtype> dtype;
  };

  // Get the reduction info for the given node, based on properties and inputs.
  ReductionInfo getReductionInfo(const torch::jit::Node* node);

  // Get the reduction axes for the given node, based on properties and inputs.
  std::vector<int64_t> getReductionAxes(const torch::jit::Node* node);

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
TORCH_API bool& getTEGenerateBlockCode();
TORCH_API bool fallbackAllowed();
TORCH_API bool setFallbackAllowed(bool value);

} // namespace tensorexpr
} // namespace jit
} // namespace torch
