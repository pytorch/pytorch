#include <chrono>
#include <string>

#include <torch/csrc/inductor/aoti_runner/model_container_runner_cpu.h>
#include <torch/script.h>
#include <torch/torch.h>

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr
        << "Usage: ./standalone_test <input file> [benchmark iter] [warm-up iter]"
        << std::endl;
    return 1;
  }
  size_t benchmark_iter = 10;
  size_t warmup_iter = 3;

  if (argc == 3) {
    benchmark_iter = std::stoul(argv[2]);
  } else if (argc == 4) {
    benchmark_iter = std::stoul(argv[2]);
    warmup_iter = std::stoul(argv[3]);
  }

  std::string data_path = argv[1];
  torch::jit::script::Module data_loader = torch::jit::load(data_path);
  torch::jit::script::Module model =
      data_loader.attr("script_module").toModule();
  const auto& model_so_path = data_loader.attr("model_so_path").toStringRef();
  const auto& script_input_tensors = data_loader.attr("inputs").toList().vec();
  const auto& input_tensors = data_loader.attr("inputs").toTensorList().vec();
  const auto& output_tensors = data_loader.attr("outputs").toTensorList().vec();

  std::unique_ptr<torch::inductor::AOTIModelContainerRunner> runner;
  runner = std::make_unique<torch::inductor::AOTIModelContainerRunnerCpu>(
      model_so_path);

  // Check results.
  auto actual_output_tensors = runner->run(input_tensors);
  assert(output_tensors.size() == actual_output_tensors.size());
  for (size_t i = 0; i < output_tensors.size(); i++) {
    assert(torch::allclose(output_tensors[i], actual_output_tensors[i]));
  }

  // Start benchmarking for scripted module.
  // Warm up
  for (size_t i = 0; i < warmup_iter; i++) {
    model.forward(script_input_tensors);
  }

  // Benchmark
  auto start = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < benchmark_iter; i++) {
    model.forward(script_input_tensors);
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> non_lowered_duration = end - start;

  // Start benchmarking for lowered module.
  // Warm up
  for (size_t i = 0; i < warmup_iter; i++) {
    runner->run(input_tensors);
  }

  // Benchmark
  start = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < benchmark_iter; i++) {
    runner->run(input_tensors);
  }
  end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> lowered_duration = end - start;

  std::cout << "[Non-lowered] Time for " << benchmark_iter
            << "iter(s): " << non_lowered_duration.count() << " sec(s)"
            << std::endl;
  std::cout << "[Lowered] Time for " << benchmark_iter
            << "iter(s): " << lowered_duration.count() << " sec(s)"
            << std::endl;

  return 0;
}
