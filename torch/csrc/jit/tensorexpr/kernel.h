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

  std::string getCodeText(const std::string& attr = "") {
    return codegen_->getCodeText(attr);
  }

  const std::shared_ptr<Graph> graph() {
    return graph_;
  }

 private:
  enum ElementType {
    kAllTypes = 0,
    kIntegralTypes = 1 << 0,
    kFloatingPointTypes = 1 << 1,
    kBoolType = 1 << 2,
    kComplexTypes = 1 << 3,
    kQintTypes = 1 << 4,
    kNonComplexOrQintTypes = kIntegralTypes | kBoolType | kFloatingPointTypes,
  };

  enum BackendType {
    kUninitialized,
    kSimpleIREval,
    kLLVMCodeGen,
    kCudaCodeGen,
    kBlockCodeGen,
  };

  void compile();
  void genInputDebugNames();

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
      int64_t dim,
      int64_t chunks,
      const std::vector<ExprHandle>& axes);

  std::vector<ExprHandle> valueShape(const torch::jit::Value* v);

  bool checkTypes(const ScalarType highType, const int typeConstraints);

  void promoteInputs(
      std::vector<ExprHandle>& inputs,
      int typeConstraints = kAllTypes);

  ExprHandle demoteOutput(const ExprHandle& e, const torch::jit::Value* v);

  ExprHandle tensorOrConstant(
      const torch::jit::Value* v,
      const std::vector<ExprHandle>& axes);

  Tensor* computeOneOperand(
      const std::string& name,
      const torch::jit::Value* v,
      const std::function<ExprHandle(const ExprHandle&)>& innerExpr,
      const int checkParamTypes = kAllTypes);

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
          innerExpr,
      bool promote_inputs = true);

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

  Tensor* computeSoftmax(const torch::jit::Value* v, bool log_softmax);

  Tensor* computeCatWoConditionals(const torch::jit::Value* v);

  Tensor* computeConv2d(const torch::jit::Value* v);

  Tensor* computeValue(const torch::jit::Value* v);

  Stmt* transformLoops(BackendType backendType, Stmt* st);

  std::string getCodeGenName(BackendType backendType);

  std::vector<CodeGen::CallArg> prepareRunArgs(
      const at::ArrayRef<IValue>& inputs,
      std::vector<at::Tensor>& outputs);
  BackendType inferBackendTypeFromDevice(at::Device device);

  void bindInput(const torch::jit::Value* input);

  Tensor* convertOutputToCorrectStrides(torch::jit::Value* v);

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
  struct UnpackedTensorOptions {
    c10::optional<c10::ScalarType> dtype;
    c10::optional<c10::Layout> layout;
    c10::optional<c10::Device> device;
    c10::optional<bool> pinned_memory;

    UnpackedTensorOptions(const c10::TensorOptions& opts)
        : dtype(optTypeMetaToScalarType(opts.dtype_opt())),
          layout(opts.layout_opt()),
          device(opts.device_opt()),
          pinned_memory(opts.pinned_memory_opt()) {}
  };

  int64_t nInputs_ = 0;
  std::vector<CodeGen::BufferArg> bufferArgs_;
  std::vector<std::vector<int64_t>> tensorOutputSizes_;
  std::vector<std::vector<int64_t>> tensorOutputStrides_;
  std::vector<UnpackedTensorOptions> tensorOutputTensorOptions_;
  std::unordered_set<const Buf*> bufOutputs_;
  std::unordered_map<int64_t, Tensor*> tensors_;
  std::unordered_map<int64_t, VarHandle> scalars_;
  std::unordered_map<const torch::jit::Value*, std::string> input_name_map_;
  std::unique_ptr<CodeGen> codegen_;
  at::Device device_ = at::kCPU;
  KernelArena kernelArena_;
  std::vector<TypePtr> inputTypes_;
  std::shared_ptr<Graph> graph_;
  Code code_;
  bool allow_fallback_{false};
  bool use_fallback_{false};
  bool hasRandom_{false};
  bool hasBroadcast_{false};
  std::unordered_map<const torch::jit::Value*, std::vector<ExprHandle>>
      known_sizes_;
};

TORCH_API int& getTECudaPointwiseLoopLevels();
TORCH_API int& getTECudaPointwiseBlockCount();
TORCH_API int& getTECudaPointwiseBlockSize();
TORCH_API bool& getTEGenerateBlockCode();
TORCH_API bool& getTEMustUseLLVMOnCPU();
TORCH_API bool fallbackAllowed();
TORCH_API bool setFallbackAllowed(bool value);
TORCH_API bool& getCatWoConditionals();

TORCH_API c10::optional<at::Device> pickDeviceType(
    const at::ArrayRef<torch::jit::Value*>& inputs);

} // namespace tensorexpr
} // namespace jit
} // namespace torch
