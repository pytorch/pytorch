#include <torch/nativert/executor/AOTInductorDelegateExecutor.h>

#include <ATen/record_function.h>

#include <torch/csrc/export/pt2_archive_constants.h>
#include <torch/nativert/executor/Weights.h>

namespace torch::nativert {

#ifndef NATIVERT_MSVC_TEST
C10_DEFINE_TYPED_REGISTRY(
    AOTIModelContainerRunnerRegistry,
    c10::DeviceType,
    torch::inductor::AOTIModelContainerRunner,
    std::unique_ptr,
    const std::string&,
    size_t,
    const std::string&,
    const std::string&,
    const bool)
#endif // NATIVERT_MSVC_TEST

namespace {
template <typename T>
std::optional<at::ScalarType> parse_precision(
    const std::optional<T>& precision) {
  if (precision) {
    return static_cast<at::ScalarType>(*precision);
  }
  return std::nullopt;
}

c10::Device infer_target_device(const Node& node) {
  std::vector<c10::Device> devices;

  const auto& tensorValuesMeta = node.owningGraph()->tensorValuesMeta();
  for (const auto* output : node.outputs()) {
    if (auto it = tensorValuesMeta.find(std::string{output->name()});
        it != tensorValuesMeta.end()) {
      devices.emplace_back(it->second.device());
    }
  }

  TORCH_CHECK(!devices.empty(), "AOTI node should have at least one output");
  for (const auto i : c10::irange(1, devices.size())) {
    if (!torch::nativert::isSameDevice(devices[0], devices[i])) {
      LOG(WARNING) << "Node " << node
                   << " has outputs on multiple devices: " << devices[0]
                   << " and " << devices[i];
    }
  }

  return devices[0];
}

std::unique_ptr<torch::inductor::AOTIModelContainerRunner>
create_aoti_model_container_runner_cpu(
    const std::string& model_so_path,
    size_t num_models,
    const std::string& device_str,
    const std::string& cubin_dir,
    const bool run_single_threaded) {
  return std::make_unique<torch::inductor::AOTIModelContainerRunnerCpu>(
      model_so_path,
      num_models,
      /* run_single_threaded= */ run_single_threaded);
}

} // namespace

C10_REGISTER_TYPED_CREATOR(
    AOTIModelContainerRunnerRegistry,
    at::kCPU,
    create_aoti_model_container_runner_cpu)

AOTIDelegateExecutor::AOTIDelegateExecutor(
    const Node& node,
    const std::shared_ptr<Weights>& weights,
    const ExecutorConfig& executorConfig,
    caffe2::serialize::PyTorchStreamReader* packageReader,
    const MakeProxyExecutorFn& makeProxyExecutorFunc)
    : ETDelegateExecutor(torch::_export::archive_spec::AOTINDUCTOR_DIR, node) {
  TORCH_CHECK(
      packageReader, "Package reader cannot be null for lowered modules");

  auto path = get_delegate_dir() + "/";

  LOG(INFO) << "Loading aotinductor model from archive path: " << path;

  std::optional<std::string> model_name = std::nullopt;
  for (const auto& record : packageReader->getAllRecords()) {
    if (c10::starts_with(record, path) && c10::ends_with(record, ".so")) {
      model_name = record.substr(record.find_last_of("/\\") + 1);
      break;
    }
  }

  TORCH_CHECK(model_name.has_value(), "missing model .so in archive: ", path);
  path.pop_back(); // remove trailing slash

  std::string tmp_dir = extractToTemporaryFolder(*packageReader, path);
  LOG(INFO) << "Extracted aot_inductor model to: " << tmp_dir;

  std::string model_path = tmp_dir + "/" + *model_name;

  LOG(INFO) << "Loading aotinductor model from model path: " << model_path;

  auto device = infer_target_device(node);
  LOG(INFO) << "Creating AOTI model container runner with device "
            << device.str();

  aoti_model_container_runner_ = AOTIModelContainerRunnerRegistry()->Create(
      device.type(),
      model_path,
      /* num_models= */ executorConfig.maxNumConcurrentThreads,
      device.str(),
      /*cubin_dir=*/tmp_dir,
      /*run_single_threaded=*/false);

  for (const auto& [name, original_fqn] :
       aoti_model_container_runner_->getConstantNamesToOriginalFQNs()) {
    if (weights->contains(original_fqn)) {
      weight_names_map_[original_fqn] = name;
    } else {
      LOG(WARNING)
          << "AOTI's Constant " << original_fqn
          << " is not found in weights, it's likely a constant created by AOTI constant folding. "
          << "Valid weight FQNs are " << weights->toString();
    }
  }

  // AOTI's DelegateExecutor doesn't need to call processWeights or
  // commitWeights here because it's invoked from Executor's ctor already.
}

void AOTIDelegateExecutor::initWeights(std::shared_ptr<Weights> weights) {
  // Do nothing for AOTI, as AOTI's .so already contains the weights.
  LOG(INFO)
      << "Skipping initWeights for AOTI to use original weights from .so file.";
}

void AOTIDelegateExecutor::processWeights(std::shared_ptr<Weights> weights) {
  LOG(INFO) << "AOTIDelegateExecutor processing weights";
  std::unordered_map<std::string, at::Tensor*> new_weights;
  for (const auto& [original_fqn, name] : weight_names_map_) {
    new_weights.emplace(name, &weights->at(original_fqn));
  }

  aoti_model_container_runner_->update_inactive_constant_buffer(new_weights);
  aoti_model_container_runner_->run_const_fold(/*use_inactive=*/true);
}

void AOTIDelegateExecutor::commitWeights() {
  LOG(INFO) << "AOTIDelegateExecutor committing weights";
  aoti_model_container_runner_->swap_constant_buffer();
}

std::vector<at::Tensor> AOTIDelegateExecutor::run(
    std::vector<at::Tensor>& inputs) {
  RECORD_USER_SCOPE("sigmoid::AOTIDelegateExecutor::run");
  std::vector<at::Tensor> outputs = aoti_model_container_runner_->run(inputs);
  return outputs;
}

} // namespace torch::nativert
