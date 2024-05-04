#include "snpe_ffi.h"

#include "DiagLog/IDiagLog.hpp"
#include "zdl/DlContainer/IDlContainer.hpp"
#include "zdl/DlSystem/ITensorFactory.hpp"
#include "zdl/DlSystem/DlError.hpp"
#include "zdl/SNPE/SNPE.hpp"
#include "zdl/SNPE/SNPEBuilder.hpp"
#include "zdl/SNPE/SNPEFactory.hpp"

// Stringify input.
#define S_(x) #x
#define S(x) S_(x)

#define SNPE_ENFORCE(condition)                                                             \
  do {                                                                                      \
    if (!(condition)) {                                                                     \
      throw std::runtime_error(std::string("Exception in SNPE: ") + std::string(__FILE__) + \
                               std::string(":") + std::string(S(__LINE__)) +                 \
                               zdl::DlSystem::getLastErrorString());                        \
    }                                                                                       \
  } while (0);

struct SNPEContext {
 public:
  SNPEContext(const std::vector<uint8_t>& buffer, const char* input_name, bool enable_logging=false) {
    container_ = zdl::DlContainer::IDlContainer::open(buffer);
    SNPE_ENFORCE(container_);

    zdl::SNPE::SNPEBuilder snpeBuilder(container_.get());
    SNPE_ENFORCE(zdl::SNPE::SNPEFactory::isRuntimeAvailable(zdl::DlSystem::Runtime_t::GPU));

    dnn_ = snpeBuilder.setOutputLayers({}) // Just the last one is fine.
                      .setRuntimeProcessor(zdl::DlSystem::Runtime_t::GPU)
											.setPerformanceProfile(zdl::DlSystem::PerformanceProfile_t::HIGH_PERFORMANCE)
                      .build();

    if (enable_logging) {
      auto logger_opt = dnn_->getDiagLogInterface();
      if (!logger_opt) throw std::runtime_error("SNPE failed to obtain logging interface");
      auto logger = *logger_opt;
      auto opts = logger->getOptions();
      opts.LogFileDirectory = "/data/local/tmp/";
      SNPE_ENFORCE(logger->setOptions(opts));
      SNPE_ENFORCE(logger->start());
    }

    SNPE_ENFORCE(dnn_);

    inputDims_ = dnn_->getInputDimensions(input_name);

    inputTensor_ = zdl::SNPE::SNPEFactory::getTensorFactory().createTensor(inputDims_);

    SNPE_ENFORCE(dnn_->getOutputLayerNames() && (*dnn_->getOutputLayerNames()).size() >= 1);
  }

  const zdl::DlSystem::Optional<zdl::DlSystem::TensorShape>& getInputDims() const { return inputDims_; };

  const std::vector<std::vector<size_t>>& run(const float* inputData, size_t count) {
    SNPE_ENFORCE(inputData);

    // Copy input data.
    memcpy(inputTensor_->begin().dataPointer(), inputData, (count * sizeof(float)));
    SNPE_ENFORCE(inputTensor_.get());

    // Execute graph in the SNPE runtime.
    SNPE_ENFORCE(dnn_->execute(inputTensor_.get(), outputTensors_));

    SNPE_ENFORCE(outputTensors_.size() >= 1);
    for (auto name : outputTensors_.getTensorNames()) {
      const auto& outputTensor = outputTensors_.getTensor(name);
      auto dims = outputTensor->getShape().getDimensions();
      outputDims_.push_back(std::vector<size_t>(dims, dims + outputTensor->getShape().rank()));
    }

    return outputDims_;
  }

  void copyOutputTo(float* outputData) {
    const auto& outputTensor = outputTensors_.getTensor(*outputTensors_.getTensorNames().begin());
    SNPE_ENFORCE(outputTensor);
    memcpy(outputData, outputTensor->begin().dataPointer(), (outputTensor->getSize() * sizeof(float)));
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
  return zdl::SNPE::SNPEFactory::isRuntimeAvailable(zdl::DlSystem::Runtime_t::GPU);
}

void* snpe_create(const uint8_t* container, size_t size, const char* input_name) {
  std::vector<uint8_t> buffer(container, container + size);
  return new SNPEContext(buffer, input_name);
}

void snpe_destroy(void* ctx) { delete ((SNPEContext*)ctx); }

void snpe_get_input_dims(void* ctx, size_t const** dims, size_t* size) {
  const auto& inputDims = ((SNPEContext*)ctx)->getInputDims();
  *dims = (*inputDims).getDimensions();
  *size = (*inputDims).rank();
}

void snpe_run(void* ctx,
              const float* inputData,
              size_t inputSize,
              size_t const** outputDims,
              size_t* outputSize) {

  const auto& outputDims_ = ((SNPEContext*)ctx)->run(inputData, inputSize);
  SNPE_ENFORCE(outputDims_.size() >= 1);

  *outputDims = outputDims_[0].data();
  *outputSize = outputDims_[0].size();
}

void snpe_copy_output_to(void* ctx, float* outputData) {
  ((SNPEContext*)ctx)->copyOutputTo(outputData);
}

} // extern "C"

