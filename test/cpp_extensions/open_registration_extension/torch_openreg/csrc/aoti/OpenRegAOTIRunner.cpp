#include <torch/csrc/inductor/aoti_runner/model_container_runner.h>

namespace torch::inductor {
namespace {

// OpenReg has no separate kernel artifacts (cubin_dir) and no raw stream
// handles; the base runner covers everything the generic PrivateUse1 AOTI
// runtime needs.
class AOTIModelContainerRunnerOpenReg : public AOTIModelContainerRunner {
 public:
  AOTIModelContainerRunnerOpenReg(
      const std::string& model_so_path,
      size_t num_models,
      const std::string& device_str,
      const std::string& cubin_dir,
      const bool run_single_threaded)
      : AOTIModelContainerRunner(
            model_so_path,
            num_models,
            device_str,
            cubin_dir,
            run_single_threaded) {}
};

std::unique_ptr<AOTIModelContainerRunner> create_aoti_runner_openreg(
    const std::string& model_so_path,
    size_t num_models,
    const std::string& device_str,
    const std::string& cubin_dir,
    const bool run_single_threaded) {
  return std::make_unique<AOTIModelContainerRunnerOpenReg>(
      model_so_path, num_models, device_str, cubin_dir, run_single_threaded);
}

} // namespace

static RegisterAOTIModelRunner register_openreg_runner(
    "openreg",
    &create_aoti_runner_openreg);

} // namespace torch::inductor
