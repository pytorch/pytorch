#include <torch/csrc/jit/codegen/onednn/graph_helper.h>
#include <torch/csrc/jit/codegen/onednn/kernel.h>

#include <ATen/core/functional.h>
#include <torch/csrc/jit/jit_log.h>

namespace torch {
namespace jit {
namespace fuser {
namespace onednn {

using namespace dnnl::graph;
using data_type = dnnl::graph::logical_tensor::data_type;

LlgaKernel::LlgaKernel(const Node* fusionNode)
    : fusionNode_(fusionNode),
      graph_(fusionNode->g(attr::Subgraph)),
      nGraphInputs_(graph_->inputs().size()),
      nOutputs_(graph_->outputs().size()),
      debugName_(genDebugName()) {
  // TODO: This is a workaround to recreate the partitions here.
  // The ideal way is to use the partition serialization API (not available from
  // LLGA now) to carry a serialized string representation from graph rewrite
  // and deserialize it here.
  auto llgaGraphHelper = LlgaGraphHelper(graph_);
  auto partitions = llgaGraphHelper.getPartitions();
  tensorIdToValue_ = llgaGraphHelper.getTensorIdToValue();
  TORCH_CHECK(
      partitions.size() == 1,
      "LLGA subgraph should contain only one partition");
  partition_ = partitions[0];
  nPartitionInputs_ = partition_.get_in_ports().size();
#ifdef GRAPH_DEBUG_ENABLED
  GRAPH_DEBUG("Initialized ", debugName(), "\n", graph_->toString());
#endif
}

bool LlgaKernel::useOpaqueLayout(size_t offset) const {
  return LlgaNodeWrapper(fusionNode_).useOpaqueLayout(offset);
}

void LlgaKernel::initializeConstantInputs() {
  for (auto& lt : partition_.get_in_ports()) {
    auto inputId = lt.get_id();
    if (initializedInputIds_.find(inputId) == initializedInputIds_.end()) {
      TORCH_CHECK(
          tensorIdToValue_.count(inputId) > 0,
          "inputs with inputId ",
          inputId,
          " is missing");
      auto* value = tensorIdToValue_[inputId];

      TORCH_CHECK(
          value->node()->kind() == prim::Constant &&
              value->type()->cast<TensorType>(),
          "inputs with inputId ",
          inputId,
          " should be a Constant tensor");
      constantValues_.emplace_back(value);

      auto const_tensor = toIValue(value)->toTensor();
      constantInputs_.emplace_back(const_tensor);
    }
  }
}

std::map<size_t, int64_t> LlgaKernel::initializeTensorIdToOccurence() const {
  std::map<size_t, int64_t> tensorIdToOccurence;
  for (auto& lt : partition_.get_in_ports()) {
    auto inputId = lt.get_id();
    std::map<size_t, int64_t>::iterator it(tensorIdToOccurence.find(inputId));
    if (it != tensorIdToOccurence.end()) {
      it->second++;
    } else {
      tensorIdToOccurence[inputId] = 1;
    }
  }
  return tensorIdToOccurence;
}

ArgSpecs LlgaKernel::initializeInputSpecs(const TensorArgs& inputs) {
  ArgSpecs inputSpecs;
  inputSpecs.reserve(nPartitionInputs_);
  GRAPH_DEBUG("Initializing graph input logical tensors");
  std::map<size_t, int64_t> tensorIdToOccurence =
      initializeTensorIdToOccurence();
  for (size_t i = 0; i < nGraphInputs_; i++) {
    auto spec = ArgSpec(graph_->inputs()[i]).supplementTensorInfo(inputs[i]);
    initializedInputIds_.insert(spec.tid());
    int64_t occurence = tensorIdToOccurence[spec.tid()];
    inputSpecs.insert(inputSpecs.end(), occurence, spec);
    runArgsIdx_.insert(runArgsIdx_.end(), occurence, i);
  }
  GRAPH_DEBUG("Initializing constant input tensors");
  initializeConstantInputs();

  TORCH_CHECK(
      inputSpecs.size() + constantValues_.size() == nPartitionInputs_,
      "Partition inputs are missing");
  GRAPH_DEBUG(
      "Concatenating constant input logical tensors to graph input "
      "logical tensors");
  for (Value* constant_value : constantValues_) {
    ArgSpec constantInputSpec(constant_value);
    inputSpecs.emplace_back(constantInputSpec);
    constantLogicalTensors_.emplace_back(constantInputSpec.logical_tensor());
  }
  return inputSpecs;
}

ArgSpecs LlgaKernel::initializeOutputSpecs() const {
  ArgSpecs outputSpecs;
  outputSpecs.reserve(nOutputs_);
  for (size_t i = 0; i < nOutputs_; i++) {
    auto spec = ArgSpec(graph_->outputs()[i]);
    if (useOpaqueLayout(i)) {
      spec = spec.any();
    }
    outputSpecs.emplace_back(spec);
  }
  return outputSpecs;
}

std::tuple<RunArgs, RunArgs> LlgaKernel::prepareRunArgs(
    const TensorArgs& inputs,
    TensorArgs& outputs) const {
  RunArgs runInputs, runOutputs;
  auto numInputs = runArgsIdx_.size();
  for (size_t i = 0; i < numInputs; i++) {
    auto spec = inputSpecs_[i];
    auto input = inputs[runArgsIdx_[i]];
    runInputs.push_back(
        {spec.logical_tensor(), Engine::getEngine(), input.data_ptr()});
  }
  auto numConstantInputs = constantInputs_.size();
  for (size_t i = 0; i < numConstantInputs; i++) {
    // constantInputSpecs are placed after graphInputSpecs
    auto constantInputSpecIdx = nGraphInputs_ + i;
    auto constantInputSpec = inputSpecs_[constantInputSpecIdx];
    runInputs.push_back(
        {constantLogicalTensors_[i],
         Engine::getEngine(),
         constantInputs_[i].data_ptr()});
  }

  for (size_t i = 0; i < nOutputs_; i++) {
    auto spec = outputSpecs_[i];
    auto opt = c10::TensorOptions(spec.aten_scalar_type()).device(device_);

    if (spec.reuses_input_tensor()) {
#ifdef GRAPH_DEBUG_ENABLED
      GRAPH_DEBUG("inplace computation - input tensor would be reused");
#endif
      auto inputTensor = inputs[spec.get_input_tensor_index()];
      if (inputTensor.is_mkldnn()) {
        auto dataType = spec.dtype();
        if (C10_UNLIKELY(!useOpaqueLayout(i))) {
          // If the input tensor was between two partitions, it would've been
          // wrapped with LlgaTensorImpl. But if it's being reused as the output
          // tensor, which is not between two partitions, then we'd have to
          // re-wrap it with a sub-class of TensorImpl, as it'd be fed into a
          // PyTorch op.
#ifdef GRAPH_DEBUG_ENABLED
          GRAPH_DEBUG("rewrap tensors");
#endif
          auto llgaImpl =
              static_cast<LlgaTensorImpl*>(inputTensor.unsafeGetTensorImpl());
          switch (dataType) {
            case data_type::f32:
            case data_type::bf16:
              inputTensor = LlgaTensorImpl::llga_to_aten_tensor(llgaImpl);
              break;
            case data_type::s32:
            default:
              TORCH_CHECK(
                  false, "Invalid data type ", static_cast<size_t>(dataType));
          }
        }
        outputs.push_back(inputTensor);
        runOutputs.push_back(
            {spec.logical_tensor(),
             Engine::getEngine(),
             inputTensor.data_ptr()});
        return std::make_tuple(runInputs, runOutputs);
      }
    }
    if (useOpaqueLayout(i)) {
      // Wrap tensors between partitions with LlgaTensorImpl wrapper, so that we
      // can bypass guard-check, as strides would be different than those
      // expected.
#ifdef GRAPH_DEBUG_ENABLED
      GRAPH_DEBUG("Between two oneDNN Graph partitions");
#endif
      auto tensor = empty_llga(spec, opt);
      outputs.push_back(tensor);
      runOutputs.push_back(llga_from_aten_tensor(tensor));
    } else {
#ifdef GRAPH_DEBUG_ENABLED
      GRAPH_DEBUG("Neither opaque to PyTorch nor inplace-computation");
#endif
      auto tensor = at::empty_strided(spec.sizes(), spec.strides(), opt);
      outputs.push_back(tensor);
      runOutputs.push_back(
          {spec.logical_tensor(), Engine::getEngine(), tensor.data_ptr()});
    }
  }

  return std::make_tuple(runInputs, runOutputs);
}

compiled_partition LlgaKernel::compile(const partition& partition) {
  auto inputs = fmap(inputSpecs_, toLogicalTensor);
  auto outputs = fmap(outputSpecs_, toLogicalTensor);
  auto compilation = partition.compile(inputs, outputs, Engine::getEngine());

  // Since layouts of opaque outputs would be known after compilation,
  // we need to query them out from compilation and update outputSpecs
  for (size_t i = 0; i < nOutputs_; i++) {
    auto tid = outputSpecs_[i].tid();
    outputSpecs_[i] = compilation.query_logical_tensor(tid);
  }

  // Build static mapping from output id to input offset
  // in accordance with available inplace options
  for (auto&& option : compilation.get_inplace_ports()) {
    size_t inputId = option.first;
    size_t outputId = option.second;
    auto inputSpecIter =
        std::find_if(inputSpecs_.begin(), inputSpecs_.end(), [&](auto& spec) {
          return spec.tid() == inputId;
        });
    TORCH_CHECK(inputSpecIter != inputSpecs_.end(), "In-place input not found");
    auto inputOffset = inputSpecIter - inputSpecs_.begin();
    auto outputSpecIter =
        std::find_if(outputSpecs_.begin(), outputSpecs_.end(), [&](auto& spec) {
          return spec.tid() == outputId;
        });
    auto outputOffset = outputSpecIter - outputSpecs_.begin();
    outputSpecs_[outputOffset].set_compute_inplace();
    outputSpecs_[outputOffset].set_input_tensor_index(inputOffset);
  }

  return compilation;
}

void LlgaKernel::run(Stack& stack) {
#ifdef GRAPH_DEBUG_ENABLED
  GRAPH_DEBUG("In ", debugName(), "\n");
#endif

  // Grab input values from stack
  auto stackInputs = last(stack, nGraphInputs_);
  auto inputs = fmap(stackInputs, [&](const IValue& v) {
    TORCH_CHECK(
        v.isTensor(), "Stack values for LLGA partition must be Tensor type");
    return v.toTensor();
  });

  // Even in case of concurrent threads, the kernel would be initialized once.
  // TODO: Try not using an atomic lock
  c10::call_once(
      initialized_flag,
      [&](const TensorArgs& inputs) {
        GRAPH_DEBUG("Initializing input logical tensors");
        inputSpecs_ = initializeInputSpecs(inputs);
        GRAPH_DEBUG("Initializing output logical tensors");
        outputSpecs_ = initializeOutputSpecs();
        GRAPH_DEBUG("Compiling partition");
        compilation_ = compile(partition_);
        is_initialized_ = true;
      },
      inputs);
#ifdef GRAPH_DEBUG_ENABLED
  GRAPH_DEBUG("Preparing runtime tensors");
#endif
  TensorArgs outputs;
  RunArgs runInputs, runOutputs;
  std::tie(runInputs, runOutputs) = prepareRunArgs(inputs, outputs);
#ifdef GRAPH_DEBUG_ENABLED
  GRAPH_DEBUG("Executing partition");
#endif
  compilation_.execute(Stream::getStream(), runInputs, runOutputs);
#ifdef GRAPH_DEBUG_ENABLED
  GRAPH_DEBUG("Partition executed");
#endif

  // Update the stack.
  drop(stack, nGraphInputs_);
  for (auto& o : outputs)
    push_one(stack, std::move(o));
#ifdef GRAPH_DEBUG_ENABLED
  GRAPH_DEBUG("Stack updated");
#endif
}

} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch
