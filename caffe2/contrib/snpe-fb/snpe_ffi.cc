#ifdef __ARM_NEON__

#include "DiagLog/IDiagLog.hpp"
#include "zdl/DlContainer/IDlContainer.hpp"
#include "zdl/DlSystem/ITensorFactory.hpp"
#include "zdl/SNPE/SNPE.hpp"
#include "zdl/SNPE/SNPEFactory.hpp"
#include <string>

// Stringify input.
#define S_(x) #x
#define S(x) S_(x)

#define SNPE_ENFORCE(condition)                                                             \
  do {                                                                                      \
    if (!(condition)) {                                                                     \
      throw std::runtime_error(std::string("Exception in SNPE: ") + std::string(__FILE__) + \
                               std::string(":") + std::string(S(__LINE__)));                \
    }                                                                                       \
  } while (0);

struct SNPEContext {
 public:
  SNPEContext(const std::vector<uint8_t>& buffer) {
    container_ = zdl::DlContainer::IDlContainer::Open(buffer);
    SNPE_ENFORCE(container_);
    dnn_ = zdl::SNPE::SNPEFactory::CreateInstance(
        container_.get(), {}, zdl::DlSystem::Runtime_t::GPU, {}, true);
    SNPE_ENFORCE(dnn_);
    inputDims_ = dnn_->GetInputDimensions();
    inputTensor_ = zdl::SNPE::SNPEFactory::GetTensorFactory().CreateTensor(inputDims_);
    SNPE_ENFORCE(dnn_->GetOutputLayerNames() && (*dnn_->GetOutputLayerNames()).size() >= 1);
  }

  const zdl::DlSystem::Optional<zdl::DlSystem::TensorShape>& getInputDims() const { return inputDims_; };

  const std::vector<std::vector<size_t>>& run(const float* inputData, size_t count) {
    std::copy(inputData, inputData + count, inputTensor_->begin());
    dnn_->Execute(inputTensor_.get(), outputTensors_);
    SNPE_ENFORCE(outputTensors_.Size() >= 1);
    for (auto name : outputTensors_.GetTensorNames()) {
      const auto& outputTensor = outputTensors_.GetTensor(name);
      auto dims = outputTensor->GetShape().GetDimensions();
      outputDims_.push_back(std::vector<size_t>(dims, dims + outputTensor->GetShape().Rank()));
    }
    return outputDims_;
  }

  void copyOutputTo(float* outputData) {
    const auto& outputTensor = outputTensors_.GetTensor(*outputTensors_.GetTensorNames().begin());
    std::copy(outputTensor->cbegin(), outputTensor->cend(), outputData);
  }

 private:
  std::shared_ptr<zdl::DlContainer::IDlContainer> container_;
  std::shared_ptr<zdl::SNPE::SNPE> dnn_;
  zdl::DlSystem::Optional<zdl::DlSystem::TensorShape> inputDims_;
  std::vector<std::vector<size_t>> outputDims_;
  std::shared_ptr<zdl::DlSystem::ITensor> inputTensor_;
  zdl::DlSystem::TensorMap outputTensors_;
};

extern "C" {

bool snpe_has_gpu() {
  return zdl::SNPE::SNPEFactory::IsRuntimeAvailable(zdl::DlSystem::Runtime_t::GPU);
}

void* snpe_create(const uint8_t* container, size_t size) {
  std::vector<uint8_t> buffer(container, container + size);
  return new SNPEContext(buffer);
}

void snpe_destroy(void* ctx) { delete ((SNPEContext*)ctx); }

void snpe_get_input_dims(void* ctx, size_t const** dims, size_t* size) {
  const auto& inputDims = ((SNPEContext*)ctx)->getInputDims();
  *dims = (*inputDims).GetDimensions();
  *size = (*inputDims).Rank();
}

void snpe_run(void* ctx,
              const float* inputData,
              size_t inputSize,
              size_t const** outputDims,
              size_t* outputSize) {
  const auto& outputDims_ = ((SNPEContext*)ctx)->run(inputData, inputSize);
  // TODO(bwasti): Support outputting more than one tensor.
  SNPE_ENFORCE(outputDims_.size() >= 1);
  *outputDims = outputDims_[0].data();
  *outputSize = outputDims_[0].size();
}

void snpe_copy_output_to(void* ctx, float* outputData) {
  ((SNPEContext*)ctx)->copyOutputTo(outputData);
}
}

#endif
