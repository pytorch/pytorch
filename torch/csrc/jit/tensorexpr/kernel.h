#pragma once

#include <c10/util/variant.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/interpreter.h>
#include <torch/csrc/jit/tensorexpr/analysis.h>
#include <torch/csrc/jit/tensorexpr/codegen.h>
#include <torch/csrc/jit/tensorexpr/tensor.h>

namespace torch {
namespace jit {
namespace tensorexpr {

// Returns true if the TE fuser supports this conv2d.
bool conv2dIsSupportedJit(const Node* node);
// Returns true if the TE fuser supports this matmul.
bool matmulIsSupported(const Node* node);
template <typename T>
inline std::vector<int64_t> bufferSizes(const T& t) {
  std::vector<int64_t> sizes;
  for (size_t i = 0; i < t->ndim(); i++) {
    sizes.push_back(*intValue(t->dim(i)));
  }
  return sizes;
}

enum ElementType {
  kAllTypes = 0,
  kIntegralTypes = 1 << 0,
  kFloatingPointTypes = 1 << 1,
  kBoolType = 1 << 2,
  kComplexTypes = 1 << 3,
  kQintTypes = 1 << 4,
  kNonComplexOrQintTypes = kIntegralTypes | kBoolType | kFloatingPointTypes,
};

using ArgNone = c10::monostate;
using BufList = std::vector<tensorexpr::BufHandle>;
using IntList = std::vector<int64_t>;
using ArgValue = c10::variant<
    tensorexpr::BufHandle,
    tensorexpr::VarHandle,
    double,
    int64_t,
    bool,
    BufList,
    IntList,
    ArgNone>;

using NNCLoweringFunction = std::function<Tensor(
    const std::vector<ArgValue>&,
    const std::vector<ExprHandle>&,
    const c10::optional<ScalarType>&,
    at::Device)>;

// Get the dimensions of a value.
std::vector<ExprHandle> valueShape(const ArgValue& v);

// If v is a tensor, broadcast it to match the shape of axes, or return
// directly if v is a constant.
ExprHandle tensorOrConstant(
    const ArgValue& v,
    const std::vector<ExprHandle>& axes);

int64_t normalizeAndCheckIndex(int64_t idx, int64_t list_size);

ExprHandle broadcast(BufHandle b, const std::vector<ExprHandle>& axes);

ExprHandle constant(const ArgValue& v);

std::vector<ExprHandle> computeIndicesToBroadcast(
    const std::vector<ExprHandle>& outputAxes,
    const std::vector<ExprHandle>& inputSizes);

void promoteInputs(
    std::vector<ExprHandle>& inputs,
    const int typeConstraints = kAllTypes);

ExprHandle promoteToDtype(ExprHandle e, ScalarType dt);

ExprHandle promoteIntegerToDefaultType(const ExprHandle& e);

ExprHandle demoteOutput(
    const ExprHandle& e,
    const c10::optional<ScalarType> type);

inline std::string getArgValueName(const ArgValue& a) {
  if (c10::get_if<tensorexpr::BufHandle>(&a)) {
    return "BufHandle";
  } else if (c10::get_if<tensorexpr::VarHandle>(&a)) {
    return "VarHandle";
  } else if (c10::get_if<double>(&a)) {
    return "double";
  } else if (c10::get_if<int64_t>(&a)) {
    return "int64_t";
  } else if (c10::get_if<bool>(&a)) {
    return "bool";
  } else if (c10::get_if<BufList>(&a)) {
    return "BufList";
  } else if (c10::get_if<IntList>(&a)) {
    return "IntList";
  } else if (c10::get_if<ArgNone>(&a)) {
    return "None";
  } else {
    throw std::runtime_error("ArgValue type not handled in string conversion");
  }
}

template <class T>
std::vector<T> convertVecArgValue(const std::vector<ArgValue>& v) {
  std::vector<T> res;
  for (auto& x : v) {
    auto val = c10::get_if<T>(&x);
    if (val) {
      res.push_back(*val);
    } else {
      throw std::runtime_error(
          "vector type not homogeneous - found " + getArgValueName(x) +
          ", expected " + getArgValueName(v[0]));
    }
  }
  return res;
}

struct TensorInfo {
  std::vector<int64_t> dims;
  c10::ScalarType dtype;
};

TORCH_API Tensor computeOperandValue(
    c10::Symbol op,
    const std::vector<ArgValue>& inputs,
    const std::vector<ExprHandle>& outputShape,
    const c10::optional<ScalarType>& outputType,
    at::Device = at::kCPU);

class TORCH_API TensorExprKernel {
  struct ConstantDescr {
    BufPtr buf;
    void* ptr;
  };

 public:
  explicit TensorExprKernel(
      const std::shared_ptr<Graph>& subgraph,
      std::unordered_map<c10::Symbol, NNCLoweringFunction> custom_lowerings =
          {},
      bool pre_alloc = false);

  void run(Stack& stack);
  void runFast(
      const std::vector<void*>& inputs,
      const std::vector<void*>& outputs);

  void fallback(Stack& stack) {
    InterpreterState(code_).run(stack);
  }

  StmtPtr getCodeGenStmt();

  std::string getCodeText(const std::string& attr = "") {
    return codegen_->getCodeText(attr);
  }

  const std::shared_ptr<Graph> graph() {
    return graph_;
  }

  const std::vector<ConstantDescr>& getConstantDescriptors() const {
    return constants_;
  }

  const std::vector<CodeGen::BufferArg>& getBufferArgs() const {
    return bufferArgs_;
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
  void genInputDebugNames();
  void runKernel(Stack& stack);

  std::vector<DimArg> dimsFromSizes(const std::vector<ExprHandle>& sizes);
  std::vector<ExprHandle> sizesForValue(const torch::jit::Value* v);
  std::vector<ExprHandle> inferSizesForValue(const torch::jit::Value* v);
  std::vector<ExprHandle> sizesFromVaryingShape(
      const c10::VaryingShape<int64_t>& shape);

  // These functions broadcast shape and also store a `hasBroadcast_` variable.
  std::vector<ExprHandle> broadcastShapesMut(
      const std::vector<ExprHandle>& a,
      const std::vector<ExprHandle>& b);
  std::vector<ExprHandle> broadcastShapesMut(
      std::vector<std::vector<ExprHandle>> shapes);

  ExprHandle chunk(
      BufPtr b,
      size_t chunkIdx,
      int64_t dim,
      int64_t chunks,
      const std::vector<ExprHandle>& axes);

  ArgValue toArg(const torch::jit::Value* v) const;
  ExprHandle constant(const torch::jit::Value* v);

  ExprHandle tensorOrConstant(
      const torch::jit::Value* v,
      const std::vector<ExprHandle>& axes);

  Tensor computeValue(const torch::jit::Value* v);

  void bindConstant(const torch::jit::Value* v);

  StmtPtr transformLoops(BackendType backendType, StmtPtr st);

  std::string getCodeGenName(BackendType backendType);

  std::vector<CodeGen::CallArg> prepareRunArgs(
      const at::ArrayRef<IValue>& inputs,
      std::vector<at::Tensor>& outputs);
  BackendType inferBackendTypeFromDevice(at::Device device);

  Tensor bindInput(const torch::jit::Value* input);

  Tensor convertOutputToCorrectStrides(torch::jit::Value* v);

  // Captures the information for reduction operation nodes.
  struct ReductionInfo {
    std::vector<DimArg> reductionDims;
    std::vector<DimArg> outputDims;
    std::vector<size_t> axes;
    bool keepdim;
    c10::optional<Dtype> dtype;
  };

  NNCLoweringFunction getCustomLoweringFor(c10::Symbol op) const;
  std::unordered_map<c10::Symbol, NNCLoweringFunction> getCustomLowerings()
      const {
    return custom_lowerings_;
  }

  // Allocate memory for intermediate buffers at compile time.
  // Specifically, we pre-allocate memory for intermediate buffers with static
  // size and manage these buffers in the way we manage JIT constant tensors:
  // push the buf args into the stack so NNC IR can access them at runtime.
  void preAllocIntermediateBufs(std::unordered_set<BufPtr>& interm_bufs);

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
  std::unordered_set<BufPtr> bufOutputs_;
  std::unordered_map<const torch::jit::Value*, BufPtr> bufs_;
  std::unordered_map<const torch::jit::Value*, VarHandle> scalars_;
  std::unordered_map<const torch::jit::Value*, std::string> input_name_map_;
  std::unique_ptr<CodeGen> codegen_;
  at::Device device_ = at::kCPU;
  std::shared_ptr<Graph> graph_;
  Code code_;
  bool allow_fallback_{false};
  bool use_fallback_{false};
  bool hasRandom_{false};
  bool hasBroadcast_{false};
  std::unordered_map<const torch::jit::Value*, std::vector<ExprHandle>>
      known_sizes_;

  std::vector<at::Tensor> unpacked_constant_tensors_;
  std::vector<ConstantDescr> constants_;

  std::unordered_map<c10::Symbol, NNCLoweringFunction> custom_lowerings_;
  bool pre_alloc_{false};
};

TORCH_API int& getTECudaPointwiseLoopLevels();
TORCH_API int& getTECudaPointwiseBlockCount();
TORCH_API int& getTECudaPointwiseBlockSize();
TORCH_API bool& getTEGenerateBlockCode();
TORCH_API bool& getTEMustUseLLVMOnCPU();
TORCH_API bool fallbackAllowed();
TORCH_API bool setFallbackAllowed(bool value);
TORCH_API bool& getCatWoConditionals();
TORCH_API bool& getOptConditionals();

TORCH_API c10::optional<at::Device> pickDeviceType(
    const at::ArrayRef<torch::jit::Value*>& inputs);

} // namespace tensorexpr
} // namespace jit
} // namespace torch
