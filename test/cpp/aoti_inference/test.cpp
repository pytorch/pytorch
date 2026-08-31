#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <torch/csrc/inductor/aoti_runner/model_container_runner_cpu.h>
#if defined(USE_CUDA)
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#endif
#if defined(USE_CUDA) || defined(USE_ROCM)
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/inductor/aoti_runner/model_container_runner_cuda.h>
#endif
#include <torch/script.h>
#include <torch/torch.h>

#define STR_VALUE(x) #x
#define STRINGIZE(x) STR_VALUE(x)

namespace {

// Function to check if test data files exist and are valid
bool testDataFilesExist() {
  std::string bindir = STRINGIZE(CMAKE_CURRENT_BINARY_DIR);
  std::array<std::string, 4> required_files = {
      "data.pt",
      "script_data.pt",
      "script_model_cpu.pt",
      "script_model_cuda.pt"};

  for (const auto& filename : required_files) {
    std::string filepath = bindir + "/" + filename;
    std::ifstream file(filepath);
    if (!file.good()) {
      return false;
    }
  }
  return true;
}

// Function to ensure test data files are generated at runtime
void ensureTestDataGenerated() {
  static std::once_flag generated_flag;
  std::call_once(generated_flag, []() {
    // Only generate if files don't exist or are placeholders
    if (testDataFilesExist()) {
      return;
    }

    std::string bindir = STRINGIZE(CMAKE_CURRENT_BINARY_DIR);

    // Calculate path to source directory: build/test_aoti_inference -> build ->
    // pytorch
    std::string pytorch_root = bindir.substr(0, bindir.find_last_of("/"));
    pytorch_root = pytorch_root.substr(0, pytorch_root.find_last_of("/"));
    std::string source_dir = pytorch_root + "/test/cpp/aoti_inference";

    // Generate test data files (data.pt, etc.) by running test.py directly
    std::string test_script = source_dir + "/test.py";
    std::string test_data_cmd = "cd " + bindir + " && python " + test_script;
    std::cout << "Generating test data: " << test_data_cmd << std::endl;
    int result1 = std::system(test_data_cmd.c_str());
    if (result1 != 0) {
      std::cerr << "Warning: Test data generation failed with code " << result1
                << std::endl;
    }

    // Generate model files (script_*.pt) by running compile_model.py directly
    std::string compile_script = source_dir + "/compile_model.py";
    std::string models_cmd = "cd " + bindir + " && python " + compile_script;
    std::cout << "Generating model files: " << models_cmd << std::endl;
    int result2 = std::system(models_cmd.c_str());
    if (result2 != 0) {
      std::cerr << "Warning: Model generation failed with code " << result2
                << std::endl;
    }
  });
}

const std::unordered_map<std::string, at::Tensor> derefTensorConstantMap(
    torch::inductor::TensorConstantMap tensor_constant_map) {
  std::unordered_map<std::string, at::Tensor> ret;
  for (const auto& pair : tensor_constant_map) {
    ret.emplace(pair.first, *(pair.second));
  }
  return ret;
}

bool compareConstantMap(
    const std::unordered_map<std::string, at::Tensor>& lhs,
    const std::unordered_map<std::string, at::Tensor>& rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }

  for (const auto& pair : lhs) {
    auto it = rhs.find(pair.first);
    if (it == rhs.end() || !torch::allclose(pair.second, it->second)) {
      return false;
    }
  }
  return true;
}

void test_aoti(const std::string& device, bool use_runtime_constant_folding) {
  torch::NoGradGuard no_grad;

  std::string data_path =
      (std::filesystem::path(STRINGIZE(CMAKE_CURRENT_BINARY_DIR)) / "data.pt")
           .string();
  torch::jit::script::Module data_loader = torch::jit::load(data_path);
  std::string suffix = use_runtime_constant_folding
      ? device + "_use_runtime_constant_folding"
      : device;
  std::string path_attr = "model_so_path_" + suffix;
  std::string inputs_attr = "inputs_" + suffix;
  std::string outputs_attr = "outputs_" + suffix;
  const auto& model_so_path = data_loader.attr(path_attr.c_str()).toStringRef();
  const auto& ref_output_tensors =
      data_loader.attr(outputs_attr.c_str()).toTensorList().vec();

  std::unique_ptr<torch::inductor::AOTIModelContainerRunner> runner;
  if (device == "cpu") {
    runner = std::make_unique<torch::inductor::AOTIModelContainerRunnerCpu>(
        model_so_path);
#if defined(USE_CUDA) || defined(USE_ROCM)
  } else if (device == "cuda") {
    runner = std::make_unique<torch::inductor::AOTIModelContainerRunnerCuda>(
        model_so_path);
#endif
  } else {
    testing::AssertionFailure() << "unsupported device: " << device;
  }
  auto actual_output_tensors =
      runner->run(data_loader.attr(inputs_attr.c_str()).toTensorList().vec());
  ASSERT_TRUE(torch::allclose(ref_output_tensors[0], actual_output_tensors[0]));
}

void test_aoti_script(const std::string& device) {
  torch::NoGradGuard no_grad;

  std::string script_model = "script_model_" + device + ".pt";
  std::string model_path =
      (std::filesystem::path(
           STRINGIZE(CMAKE_CURRENT_BINARY_DIR)) / script_model.c_str())
           .string();
  torch::jit::script::Module model = torch::jit::load(model_path);

  std::string sample_data_path =
      (std::filesystem::path(
           STRINGIZE(CMAKE_CURRENT_BINARY_DIR)) / "script_data.pt")
           .string();
  torch::jit::script::Module sample_data = torch::jit::load(sample_data_path);
  std::string inputs_attr = "inputs_" + device;
  std::string outputs_attr = "outputs_" + device;
  const auto& inputs = sample_data.attr(inputs_attr.c_str()).toList().vec();
  const auto& ref_output_tensors =
      sample_data.attr(outputs_attr.c_str()).toTensorVector();
  auto outputs = model.forward(inputs).toTuple()->elements();
  ASSERT_EQ(outputs.size(), ref_output_tensors.size());
  for (size_t i = 0; i < ref_output_tensors.size(); i++) {
    ASSERT_TRUE(torch::allclose(outputs[i].toTensor(), ref_output_tensors[i]));
  }
}

void test_aoti_package_loader(
    const std::string& device,
    bool use_runtime_constant_folding) {
  torch::NoGradGuard no_grad;

  std::string data_path =
      (std::filesystem::path(STRINGIZE(CMAKE_CURRENT_BINARY_DIR)) / "data.pt")
           .string();
  torch::jit::script::Module data_loader = torch::jit::load(data_path);
  std::string suffix = use_runtime_constant_folding
      ? device + "_use_runtime_constant_folding"
      : device;
  std::string path_attr = "pt2_package_path_" + suffix;
  std::string inputs_attr = "inputs_" + suffix;
  std::string outputs_attr = "outputs_" + suffix;
  const auto& pt2_package_path =
      data_loader.attr(path_attr.c_str()).toStringRef();
  const auto& ref_output_tensors =
      data_loader.attr(outputs_attr.c_str()).toTensorList().vec();

  torch::inductor::AOTIModelPackageLoader runner(pt2_package_path);
  auto actual_output_tensors =
      runner.run(data_loader.attr(inputs_attr.c_str()).toTensorList().vec());
  ASSERT_TRUE(torch::allclose(ref_output_tensors[0], actual_output_tensors[0]));
}

void test_aoti_package_loader_multi_gpu(
    const std::string& device,
    bool use_runtime_constant_folding) {
  torch::NoGradGuard no_grad;
  // Ensure that this test will reset the default CUDA device on exit.
  torch::DeviceGuard device_guard(c10::Device("cuda"));

  std::string data_path =
      (std::filesystem::path(STRINGIZE(CMAKE_CURRENT_BINARY_DIR)) / "data.pt")
           .string();
  torch::jit::script::Module data_loader = torch::jit::load(data_path);
  std::string suffix = use_runtime_constant_folding
      ? device + "_use_runtime_constant_folding"
      : device;
  std::string path_attr = "pt2_package_path_" + suffix;
  std::string inputs_attr = "inputs_" + suffix;
  std::string outputs_attr = "outputs_" + suffix;
  const auto& pt2_package_path =
      data_loader.attr(path_attr.c_str()).toStringRef();
  const auto& ref_output_tensors =
      data_loader.attr(outputs_attr.c_str()).toTensorList().vec();

  // For all available CUDA devices: Load PT2 package on this device, run
  // inference, and validate results
  auto input_tensors =
      data_loader.attr(inputs_attr.c_str()).toTensorList().vec();
  for (int i = 0; i < torch::cuda::device_count(); i++) {
    auto options = torch::TensorOptions().device(torch::kCUDA, i);
    torch::inductor::AOTIModelPackageLoader runner(
        pt2_package_path, "model", false, 1, i);
    std::vector<torch::Tensor> input_tensors_on_device;
    for (auto input_tensor : input_tensors) {
      input_tensors_on_device.push_back(input_tensor.clone().to(options));
    }
    // Run loaded PT2 package on device
    auto actual_output_tensors = runner.run(input_tensors_on_device);
    ASSERT_TRUE(torch::allclose(
        ref_output_tensors[0].cpu(), actual_output_tensors[0].cpu()));
  }
}

void test_aoti_constants_update(
    const std::string& device,
    bool use_runtime_constant_folding) {
  torch::NoGradGuard no_grad;

  std::string data_path =
      (std::filesystem::path(STRINGIZE(CMAKE_CURRENT_BINARY_DIR)) / "data.pt")
           .string();

  torch::jit::script::Module data_loader = torch::jit::load(data_path);
  std::string suffix = use_runtime_constant_folding
      ? device + "_use_runtime_constant_folding"
      : device;
  std::string path_attr = "model_so_path_" + suffix;
  std::string inputs_attr = "inputs_" + suffix;
  std::string outputs_attr = "outputs_" + suffix;
  std::string weights_attr = "w_pre_" + suffix;
  std::string add_attr = "w_add_" + suffix;
  const auto& model_so_path = data_loader.attr(path_attr.c_str()).toStringRef();
  auto input_tensors =
      data_loader.attr(inputs_attr.c_str()).toTensorList().vec();
  const auto& ref_output_tensors =
      data_loader.attr(outputs_attr.c_str()).toTensorList().vec();

  const auto& weight_tensors =
      data_loader.attr(weights_attr.c_str()).toTensor();
  const auto& add_tensors = data_loader.attr(add_attr.c_str()).toTensor();

  torch::inductor::TensorConstantMap missing_map, rand_map, real_map;
  missing_map.emplace("L__self___w_pre", new at::Tensor(at::randn({4, 4})));
  rand_map.emplace("L__self___w_pre", new at::Tensor(at::randn({10})));
  rand_map.emplace("L__self___w_add", new at::Tensor(at::randn({10})));
  real_map.emplace("L__self___w_pre", new at::Tensor(weight_tensors));
  real_map.emplace("L__self___w_add", new at::Tensor(add_tensors));

  std::unique_ptr<torch::inductor::AOTIModelContainerRunner> runner;
  if (device == "cpu") {
    runner = std::make_unique<torch::inductor::AOTIModelContainerRunnerCpu>(
        model_so_path);
#if defined(USE_CUDA) || defined(USE_ROCM)
  } else if (device == "cuda") {
    runner = std::make_unique<torch::inductor::AOTIModelContainerRunnerCuda>(
        model_so_path);
#endif
  } else {
    testing::AssertionFailure() << "unsupported device: " << device;
  }
  // By default, buffer #1 get loaded with burned in weights. Correct results.
  auto actual_output_tensors = runner->run(input_tensors);
  ASSERT_TRUE(torch::allclose(ref_output_tensors[0], actual_output_tensors[0]));

  // Update with missing map which should throw.
  // Somehow EXPECT_THROW doesn't work here when running tests in a row, but
  // works when running AotInductorTest.RuntimeUpdateConstantsCuda individually.
  try {
    runner->update_constant_buffer(missing_map, false, true);
  } catch (const std::runtime_error& e) {
    EXPECT_THAT(e.what(), ::testing::HasSubstr("API call failed at"));
  }

  // Update random weight to buffer #1.
  runner->update_constant_buffer(missing_map, false, false);
  actual_output_tensors = runner->run(input_tensors);
  ASSERT_FALSE(
      torch::allclose(ref_output_tensors[0], actual_output_tensors[0]));

  // Update with real map.
  runner->update_constant_buffer(real_map, false, false);
  actual_output_tensors = runner->run(input_tensors);
  if (use_runtime_constant_folding) {
    runner->run_const_fold(/* use_inactive = */ false);
  }
  actual_output_tensors = runner->run(input_tensors);
  ASSERT_TRUE(torch::allclose(ref_output_tensors[0], actual_output_tensors[0]));

  // Update with full random map.
  runner->update_constant_buffer(rand_map, false, false);
  if (use_runtime_constant_folding) {
    runner->run_const_fold(/* use_inactive = */ false);
  }
  actual_output_tensors = runner->run(input_tensors);
  ASSERT_FALSE(
      torch::allclose(ref_output_tensors[0], actual_output_tensors[0]));

  for (auto& pair : missing_map) {
    delete pair.second;
  }
  for (auto& pair : rand_map) {
    delete pair.second;
  }
  for (auto& pair : real_map) {
    delete pair.second;
  }
}

// Exercises the free_fold_input_only_constants release/re-arm cycle end to end
// through the C++ runtime: load, predict, update the released fold inputs,
// re-fold, predict again. Without the re-arm in update_constant_buffer, the
// second run_const_fold() either throws (inputs released) or silently returns,
// leaving the stale folded constant and an unchanged output.
#if defined(USE_CUDA) || defined(USE_ROCM)
// Live allocator bytes, the C++ equivalent of torch.cuda.memory_allocated().
// reserved_bytes (used elsewhere in this file) does not shrink on free, so it
// cannot show a release.
size_t cudaAllocatedBytes() {
  int device_idx = -1;
  if (cudaGetDevice(&device_idx) != cudaSuccess || device_idx == -1) {
    throw std::runtime_error("cudaGetDevice failed!");
  }
  return c10::cuda::CUDACachingAllocator::getDeviceStats(device_idx)
      .allocated_bytes[0]
      .current;
}

// Runs once and reports how many bytes the run gave back, which for this model
// is dominated by the fold input released after folding.
size_t runAndMeasureFreed(
    torch::inductor::AOTIModelContainerRunner* runner,
    const std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& out) {
  if (cudaDeviceSynchronize() != cudaSuccess) {
    throw std::runtime_error("cudaDeviceSynchronize failed!");
  }
  size_t before = cudaAllocatedBytes();
  out = runner->run(inputs);
  if (cudaDeviceSynchronize() != cudaSuccess) {
    throw std::runtime_error("cudaDeviceSynchronize failed!");
  }
  size_t after = cudaAllocatedBytes();
  return before > after ? before - after : 0;
}

void test_aoti_free_fold_constants_update() {
  torch::NoGradGuard no_grad;

  std::string data_path =
      (std::filesystem::path(STRINGIZE(CMAKE_CURRENT_BINARY_DIR)) / "data.pt")
           .string();
  torch::jit::script::Module data_loader = torch::jit::load(data_path);

  const auto& model_so_path =
      data_loader.attr("model_so_path_free_fold").toStringRef();
  auto input_tensors =
      data_loader.attr("inputs_free_fold").toTensorList().vec();
  const auto& ref_output_tensors =
      data_loader.attr("outputs_free_fold").toTensorList().vec();
  const auto& ref_updated_output_tensors =
      data_loader.attr("outputs_updated_free_fold").toTensorList().vec();
  const auto& w_pre_updated =
      data_loader.attr("w_pre_updated_free_fold").toTensor();
  const auto& b_updated = data_loader.attr("b_updated_free_fold").toTensor();
  size_t fold_input_bytes = static_cast<size_t>(
      data_loader.attr("fold_input_bytes_free_fold").toInt());

  auto runner = std::make_unique<torch::inductor::AOTIModelContainerRunnerCuda>(
      model_so_path);

  // First run folds the const graph, then releases w_pre/b.
  std::vector<at::Tensor> actual_output_tensors;
  size_t freed =
      runAndMeasureFreed(runner.get(), input_tensors, actual_output_tensors);
  ASSERT_TRUE(torch::allclose(ref_output_tensors[0], actual_output_tensors[0]));
  // Numerics alone cannot tell a release from a constant left resident, so
  // pin the release itself.
  ASSERT_GT(freed, fold_input_bytes * 9 / 10);

  // Folding is idempotent: a second fold on an already-folded buffer must be a
  // no-op rather than tripping the released-inputs assertion.
  runner->run_const_fold(/* use_inactive = */ false);
  actual_output_tensors = runner->run(input_tensors);
  ASSERT_TRUE(torch::allclose(ref_output_tensors[0], actual_output_tensors[0]));

  // Supplying the released constants again must restore them and re-arm the
  // buffer so the next fold actually recomputes.
  torch::inductor::TensorConstantMap updated_map;
  updated_map.emplace("L__self___w_pre", new at::Tensor(w_pre_updated));
  updated_map.emplace("L__self___b", new at::Tensor(b_updated));
  runner->update_constant_buffer(updated_map, false, false);

  // The re-fold has to release again, not just recompute.
  freed =
      runAndMeasureFreed(runner.get(), input_tensors, actual_output_tensors);
  ASSERT_TRUE(
      torch::allclose(ref_updated_output_tensors[0], actual_output_tensors[0]));
  // The perturbed weights must actually move the output; an unchanged result
  // means the re-fold was skipped and the stale folded constant survived.
  ASSERT_FALSE(
      torch::allclose(ref_output_tensors[0], actual_output_tensors[0]));
  ASSERT_GT(freed, fold_input_bytes * 9 / 10);

  for (auto& pair : updated_map) {
    delete pair.second;
  }
}

// A weight update has to re-fold correctly in every update flavor. Each one
// restores the released fold inputs by a different route: an owning clone
// (default), the caller's own pointer (user_managed), or a CPU tensor copied
// H2D into fresh owned storage (allow_h2d_copy, which the runtime reaches
// through update_constant_buffer_from_cpu). user_managed and allow_h2d_copy
// are mutually exclusive, so there is no fourth combination to cover.
enum class FoldInputUpdateMode { kDefault, kUserManaged, kAllowH2DCopy };

void test_aoti_free_fold_constants_update_mode(FoldInputUpdateMode mode) {
  torch::NoGradGuard no_grad;

  std::string data_path =
      (std::filesystem::path(STRINGIZE(CMAKE_CURRENT_BINARY_DIR)) / "data.pt")
           .string();
  torch::jit::script::Module data_loader = torch::jit::load(data_path);

  const auto& model_so_path =
      data_loader.attr("model_so_path_free_fold").toStringRef();
  auto input_tensors =
      data_loader.attr("inputs_free_fold").toTensorList().vec();
  const auto& ref_output_tensors =
      data_loader.attr("outputs_free_fold").toTensorList().vec();
  size_t fold_input_bytes = static_cast<size_t>(
      data_loader.attr("fold_input_bytes_free_fold").toInt());

  // Mirrors FoldInputHeavyNet.forward in test.py, so the reference tracks
  // whatever weights each round supplies rather than a value from data.pt.
  auto eager_forward =
      [](const at::Tensor& x, const at::Tensor& w_pre, const at::Tensor& b) {
        return x * (at::relu(w_pre).sum(0) + b);
      };

  auto runner = std::make_unique<torch::inductor::AOTIModelContainerRunnerCuda>(
      model_so_path);

  // First run folds the const graph, then releases w_pre/b. This always goes
  // through the load-time owned-storage path, whatever the update mode is.
  std::vector<at::Tensor> outputs;
  size_t freed = runAndMeasureFreed(runner.get(), input_tensors, outputs);
  at::Tensor previous = outputs[0];
  ASSERT_TRUE(torch::allclose(ref_output_tensors[0], previous));
  ASSERT_GT(freed, fold_input_bytes * 9 / 10);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA);
  // Twice, so the release/re-arm cycle is exercised past the first update.
  for (int round = 0; round < 2; round++) {
    at::Tensor w_pre =
        at::randn({static_cast<int64_t>(fold_input_bytes / 16), 4}, options);
    at::Tensor b = at::randn({4}, options);
    at::Tensor expected = eager_forward(input_tensors[0], w_pre, b);

    bool from_cpu = mode == FoldInputUpdateMode::kAllowH2DCopy;
    // Must outlive the update call; under user_managed the container keeps no
    // copy of them. The H2D source is deliberately non-contiguous (transposed
    // twice around a contiguous copy, so the values match but the strides do
    // not): the runtime has to copy it element-wise, where a raw byte copy off
    // data_ptr() would silently produce garbage.
    at::Tensor w_pre_arg = from_cpu ? w_pre.cpu().t().contiguous().t() : w_pre;
    at::Tensor b_arg = from_cpu ? b.cpu() : b;
    if (from_cpu) {
      ASSERT_FALSE(w_pre_arg.is_contiguous());
    }

    torch::inductor::TensorConstantMap update_map;
    update_map.emplace("L__self___w_pre", &w_pre_arg);
    update_map.emplace("L__self___b", &b_arg);

    if (from_cpu) {
      runner->update_constant_buffer_from_cpu(
          update_map,
          /* use_inactive = */ false,
          /* validate_full_update = */ false);
    } else {
      runner->update_constant_buffer(
          update_map,
          /* use_inactive = */ false,
          /* validate_full_update = */ false,
          /* user_managed = */ mode == FoldInputUpdateMode::kUserManaged);
    }

    freed = runAndMeasureFreed(runner.get(), input_tensors, outputs);
    at::Tensor actual = outputs[0];
    ASSERT_TRUE(torch::allclose(expected, actual));
    // An unchanged result would mean the re-fold was skipped and the stale
    // folded constant survived.
    ASSERT_FALSE(torch::allclose(previous, actual));

    if (mode == FoldInputUpdateMode::kUserManaged) {
      // The container never copied, so there is nothing of its own to reclaim;
      // w_pre above is still held by this scope. Asserting the release frees
      // (almost) nothing is what pins that user_managed forgoes the saving.
      ASSERT_LT(freed, fold_input_bytes / 2);
    } else {
      // default clones and allow_h2d_copy allocates owned device storage, so
      // both must hand that storage back at the next fold.
      ASSERT_GT(freed, fold_input_bytes * 9 / 10);
    }
    previous = actual;
  }
}

// The sequence production serving actually runs, from nativert's
// AOTIDelegateExecutor: publish new weights into the inactive buffer, fold it
// there while the active buffer keeps serving, then swap. This is the only
// coverage of run_const_fold(use_inactive = true) -- a shared lock, the
// temporary swap of the model's constants map, and
// inactive().drop_fold_input_only() -- none of which the active-buffer tests
// reach. update_inactive_constant_buffer also forces validate_full_update, so
// it doubles as the check that the released fold inputs are always re-supplied
// on the path that matters.
void test_aoti_free_fold_inactive_update_fold_swap() {
  torch::NoGradGuard no_grad;

  std::string data_path =
      (std::filesystem::path(STRINGIZE(CMAKE_CURRENT_BINARY_DIR)) / "data.pt")
           .string();
  torch::jit::script::Module data_loader = torch::jit::load(data_path);
  const auto& model_so_path =
      data_loader.attr("model_so_path_free_fold").toStringRef();
  auto input_tensors =
      data_loader.attr("inputs_free_fold").toTensorList().vec();
  const auto& ref_output_tensors =
      data_loader.attr("outputs_free_fold").toTensorList().vec();
  size_t fold_input_bytes = static_cast<size_t>(
      data_loader.attr("fold_input_bytes_free_fold").toInt());
  int64_t fold_rows = static_cast<int64_t>(fold_input_bytes / 16);

  auto eager_forward =
      [](const at::Tensor& x, const at::Tensor& w_pre, const at::Tensor& b) {
        return x * (at::relu(w_pre).sum(0) + b);
      };

  auto runner = std::make_unique<torch::inductor::AOTIModelContainerRunnerCuda>(
      model_so_path);
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA);

  // Load-time fold of the active buffer, which releases its fold inputs.
  at::Tensor previous = runner->run(input_tensors)[0];
  ASSERT_TRUE(torch::allclose(ref_output_tensors[0], previous));

  // Twice, so the second round updates a buffer whose fold inputs were
  // released by the first round's fold rather than at load.
  for (int round = 0; round < 2; round++) {
    at::Tensor w_pre = at::randn({fold_rows, 4}, options);
    at::Tensor b = at::randn({4}, options);
    at::Tensor expected = eager_forward(input_tensors[0], w_pre, b);

    torch::inductor::TensorConstantMap weight_map;
    weight_map.emplace("L__self___w_pre", &w_pre);
    weight_map.emplace("L__self___b", &b);
    runner->update_inactive_constant_buffer(weight_map);

    // Folding the inactive buffer must release its fold inputs too, not just
    // the active buffer's.
    if (cudaDeviceSynchronize() != cudaSuccess) {
      throw std::runtime_error("cudaDeviceSynchronize failed!");
    }
    size_t before = cudaAllocatedBytes();
    runner->run_const_fold(/* use_inactive = */ true);
    if (cudaDeviceSynchronize() != cudaSuccess) {
      throw std::runtime_error("cudaDeviceSynchronize failed!");
    }
    size_t after = cudaAllocatedBytes();
    ASSERT_GT(before > after ? before - after : 0, fold_input_bytes * 9 / 10);

    // Still unswapped: the active buffer keeps serving the old weights.
    ASSERT_TRUE(torch::allclose(previous, runner->run(input_tensors)[0]));

    runner->swap_constant_buffer();
    at::Tensor actual = runner->run(input_tensors)[0];
    ASSERT_TRUE(torch::allclose(expected, actual));
    ASSERT_FALSE(torch::allclose(previous, actual));
    previous = actual;
  }
}

// Folding the same weights repeatedly must be discarded, not repeated. Once
// the first fold has released the fold inputs there is nothing left to fold
// from, so every later call has to return without touching them and without
// disturbing the folded result. A weight update is the only thing that re-arms
// the buffer.
void test_aoti_free_fold_repeated_const_fold() {
  torch::NoGradGuard no_grad;

  std::string data_path =
      (std::filesystem::path(STRINGIZE(CMAKE_CURRENT_BINARY_DIR)) / "data.pt")
           .string();
  torch::jit::script::Module data_loader = torch::jit::load(data_path);
  const auto& model_so_path =
      data_loader.attr("model_so_path_free_fold").toStringRef();
  auto input_tensors =
      data_loader.attr("inputs_free_fold").toTensorList().vec();
  const auto& ref_output_tensors =
      data_loader.attr("outputs_free_fold").toTensorList().vec();
  size_t fold_input_bytes = static_cast<size_t>(
      data_loader.attr("fold_input_bytes_free_fold").toInt());
  int64_t fold_rows = static_cast<int64_t>(fold_input_bytes / 16);

  auto eager_forward =
      [](const at::Tensor& x, const at::Tensor& w_pre, const at::Tensor& b) {
        return x * (at::relu(w_pre).sum(0) + b);
      };

  auto runner = std::make_unique<torch::inductor::AOTIModelContainerRunnerCuda>(
      model_so_path);

  // First run folds and releases the fold inputs.
  std::vector<at::Tensor> outputs;
  size_t freed = runAndMeasureFreed(runner.get(), input_tensors, outputs);
  ASSERT_TRUE(torch::allclose(ref_output_tensors[0], outputs[0]));
  ASSERT_GT(freed, fold_input_bytes * 9 / 10);

  // Every subsequent fold of the same weights is discarded. None may throw on
  // the released inputs, and the prediction may not drift.
  for (int i = 0; i < 5; i++) {
    runner->run_const_fold(/* use_inactive = */ false);
    ASSERT_TRUE(
        torch::allclose(ref_output_tensors[0], runner->run(input_tensors)[0]));
  }

  // A weight update re-arms, so the next fold is honoured rather than
  // discarded -- and the extra folds after it are discarded again.
  at::Tensor w_pre = at::randn(
      {fold_rows, 4}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));
  at::Tensor b =
      at::randn({4}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));
  torch::inductor::TensorConstantMap update_map;
  update_map.emplace("L__self___w_pre", &w_pre);
  update_map.emplace("L__self___b", &b);
  runner->update_constant_buffer(update_map, false, false);

  at::Tensor expected = eager_forward(input_tensors[0], w_pre, b);
  for (int i = 0; i < 3; i++) {
    runner->run_const_fold(/* use_inactive = */ false);
    ASSERT_TRUE(torch::allclose(expected, runner->run(input_tensors)[0]));
  }
  ASSERT_FALSE(torch::allclose(ref_output_tensors[0], expected));
}

// Back-to-back weight updates, in the two shapes serving can produce: update
// then predict (each update arms a fold that releases again), and several
// updates with no fold in between (only the last one's values may survive).
// The prediction after each has to match eager.
void test_aoti_free_fold_repeated_updates() {
  torch::NoGradGuard no_grad;

  std::string data_path =
      (std::filesystem::path(STRINGIZE(CMAKE_CURRENT_BINARY_DIR)) / "data.pt")
           .string();
  torch::jit::script::Module data_loader = torch::jit::load(data_path);
  const auto& model_so_path =
      data_loader.attr("model_so_path_free_fold").toStringRef();
  auto input_tensors =
      data_loader.attr("inputs_free_fold").toTensorList().vec();
  size_t fold_input_bytes = static_cast<size_t>(
      data_loader.attr("fold_input_bytes_free_fold").toInt());
  int64_t fold_rows = static_cast<int64_t>(fold_input_bytes / 16);

  auto eager_forward =
      [](const at::Tensor& x, const at::Tensor& w_pre, const at::Tensor& b) {
        return x * (at::relu(w_pre).sum(0) + b);
      };

  auto runner = std::make_unique<torch::inductor::AOTIModelContainerRunnerCuda>(
      model_so_path);
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA);

  auto update = [&](const at::Tensor& w_pre, const at::Tensor& b) {
    torch::inductor::TensorConstantMap map;
    // TensorConstantMap borrows, so these must outlive the call; they do,
    // since the caller's tensors are alive for the whole round.
    map.emplace("L__self___w_pre", const_cast<at::Tensor*>(&w_pre));
    map.emplace("L__self___b", const_cast<at::Tensor*>(&b));
    runner->update_constant_buffer(
        map,
        /* use_inactive = */ false,
        /* validate_full_update = */ false);
  };

  // First run folds and releases.
  runner->run(input_tensors);

  // Phase 1: update, predict, repeat. Every round re-arms, re-folds and
  // releases again.
  for (int round = 0; round < 4; round++) {
    at::Tensor w_pre = at::randn({fold_rows, 4}, options);
    at::Tensor b = at::randn({4}, options);
    update(w_pre, b);
    ASSERT_TRUE(torch::allclose(
        eager_forward(input_tensors[0], w_pre, b),
        runner->run(input_tensors)[0]));
  }

  // Phase 2: several updates with no fold in between. Each one overwrites the
  // previous, still-unfolded values, so the single fold at the end must use
  // the last ones.
  std::vector<at::Tensor> w_pres;
  std::vector<at::Tensor> bs;
  for (int round = 0; round < 4; round++) {
    w_pres.push_back(at::randn({fold_rows, 4}, options));
    bs.push_back(at::randn({4}, options));
    update(w_pres.back(), bs.back());
  }
  ASSERT_TRUE(torch::allclose(
      eager_forward(input_tensors[0], w_pres.back(), bs.back()),
      runner->run(input_tensors)[0]));
}
#endif

void test_aoti_extract_constants_map(const std::string& device) {
  torch::NoGradGuard no_grad;

  std::string data_path =
      (std::filesystem::path(STRINGIZE(CMAKE_CURRENT_BINARY_DIR)) / "data.pt")
           .string();

  torch::jit::script::Module data_loader = torch::jit::load(data_path);
  std::string path_attr = "model_so_path_" + device;
  std::string inputs_attr = "inputs_" + device;
  std::string outputs_attr = "outputs_" + device;
  std::string weights_attr = "w_pre_" + device;
  std::string add_attr = "w_add_" + device;
  const auto& model_so_path = data_loader.attr(path_attr.c_str()).toStringRef();
  auto input_tensors =
      data_loader.attr(inputs_attr.c_str()).toTensorList().vec();
  const auto& ref_output_tensors =
      data_loader.attr(outputs_attr.c_str()).toTensorList().vec();

  const auto& weight_tensors =
      data_loader.attr(weights_attr.c_str()).toTensor();
  const auto& add_tensors = data_loader.attr(add_attr.c_str()).toTensor();

  torch::inductor::TensorConstantMap rand_map, real_map;
  at::Tensor rand_pre, rand_add;
  at::Tensor w_pre, w_add;
  at::DeviceType device_type = device == "cuda" ? at::kCUDA : at::kCPU;
  rand_pre = at::randn({4, 4}).to(device_type);
  rand_add = at::randn({4, 4}).to(device_type);
  w_pre = at::Tensor(weight_tensors).to(device_type);
  w_add = at::Tensor(add_tensors).to(device_type);

  rand_map.emplace("L__self___w_pre", &rand_pre);
  rand_map.emplace("L__self___w_add", &rand_add);
  real_map.emplace("L__self___w_pre", &w_pre);
  real_map.emplace("L__self___w_add", &w_add);

  std::unique_ptr<torch::inductor::AOTIModelContainerRunner> runner;
  if (device == "cpu") {
    runner = std::make_unique<torch::inductor::AOTIModelContainerRunnerCpu>(
        model_so_path);
#if defined(USE_CUDA) || defined(USE_ROCM)
  } else if (device == "cuda") {
    runner = std::make_unique<torch::inductor::AOTIModelContainerRunnerCuda>(
        model_so_path);
#endif
  } else {
    testing::AssertionFailure() << "unsupported device: " << device;
  }

  // By default, buffer #1 get loaded with burned in weights. Correct results.
  auto actual_output_tensors = runner->run(input_tensors);
  ASSERT_TRUE(torch::allclose(ref_output_tensors[0], actual_output_tensors[0]));

  // We update the weights to buffer #2 and activate it. This should still
  // produce correct result, as it's the real constant map.
  runner->update_inactive_constant_buffer(real_map);
  auto extracted_inactive_weight =
      runner->extract_constants_map(/* use_inactive = */ true);
  auto extracted_active_weight =
      runner->extract_constants_map(/* use_inactive = */ false);
  auto cmp_real_map = derefTensorConstantMap(real_map);
  auto cmp_rand_map = derefTensorConstantMap(rand_map);
  ASSERT_TRUE(compareConstantMap(extracted_active_weight, cmp_real_map));
  ASSERT_TRUE(compareConstantMap(extracted_inactive_weight, cmp_real_map));

  // We update random weights to buffer #1. But do not swap in the weight yet.
  runner->update_inactive_constant_buffer(rand_map);
  extracted_inactive_weight =
      runner->extract_constants_map(/* use_inactive = */ true);
  ASSERT_TRUE(compareConstantMap(extracted_inactive_weight, cmp_rand_map));

  // We swap and activate the weight to buffer #1.
  // Active weight now should be the new weight, while inactive should be the
  // previous one.
  runner->swap_constant_buffer();
  extracted_inactive_weight =
      runner->extract_constants_map(/* use_inactive = */ true);
  extracted_active_weight =
      runner->extract_constants_map(/* use_inactive = */ false);
  ASSERT_TRUE(compareConstantMap(extracted_active_weight, cmp_rand_map));
  ASSERT_TRUE(compareConstantMap(extracted_inactive_weight, cmp_real_map));
}

void test_aoti_double_buffering(
    const std::string& device,
    bool use_runtime_constant_folding) {
  torch::NoGradGuard no_grad;

  std::string data_path =
      (std::filesystem::path(STRINGIZE(CMAKE_CURRENT_BINARY_DIR)) / "data.pt")
           .string();

  torch::jit::script::Module data_loader = torch::jit::load(data_path);
  std::string suffix = use_runtime_constant_folding
      ? device + "_use_runtime_constant_folding"
      : device;
  std::string path_attr = "model_so_path_" + suffix;
  std::string inputs_attr = "inputs_" + suffix;
  std::string outputs_attr = "outputs_" + suffix;
  std::string weights_attr = "w_pre_" + suffix;
  std::string add_attr = "w_add_" + suffix;
  const auto& model_so_path = data_loader.attr(path_attr.c_str()).toStringRef();
  auto input_tensors =
      data_loader.attr(inputs_attr.c_str()).toTensorList().vec();
  const auto& ref_output_tensors =
      data_loader.attr(outputs_attr.c_str()).toTensorList().vec();

  const auto& weight_tensors =
      data_loader.attr(weights_attr.c_str()).toTensor();
  const auto& add_tensors = data_loader.attr(add_attr.c_str()).toTensor();

  torch::inductor::TensorConstantMap rand_map, real_map;
  rand_map.emplace("L__self___w_pre", new at::Tensor(at::randn({4, 4})));
  rand_map.emplace("L__self___w_add", new at::Tensor(at::randn({4, 4})));
  real_map.emplace("L__self___w_pre", new at::Tensor(weight_tensors));
  real_map.emplace("L__self___w_add", new at::Tensor(add_tensors));

  std::unique_ptr<torch::inductor::AOTIModelContainerRunner> runner;
  if (device == "cpu") {
    runner = std::make_unique<torch::inductor::AOTIModelContainerRunnerCpu>(
        model_so_path);
#if defined(USE_CUDA) || defined(USE_ROCM)
  } else if (device == "cuda") {
    runner = std::make_unique<torch::inductor::AOTIModelContainerRunnerCuda>(
        model_so_path);
#endif
  } else {
    testing::AssertionFailure() << "unsupported device: " << device;
  }
  // By default, buffer #1 get loaded with burned in weights. Correct results.
  auto actual_output_tensors = runner->run(input_tensors);
  ASSERT_TRUE(torch::allclose(ref_output_tensors[0], actual_output_tensors[0]));

  // We update the weights to buffer #2 and activate it. This should still
  // produce correct result, as it's the real constant map.
  runner->update_inactive_constant_buffer(real_map);
  if (use_runtime_constant_folding) {
    runner->run_const_fold(/* use_inactive = */ true);
  }
  runner->swap_constant_buffer();
  actual_output_tensors = runner->run(input_tensors);
  ASSERT_TRUE(torch::allclose(ref_output_tensors[0], actual_output_tensors[0]));

  // We update random weights to buffer #1. But do not swap in the weight yet.
  runner->update_inactive_constant_buffer(rand_map);
  if (use_runtime_constant_folding) {
    runner->run_const_fold(/* use_inactive = */ true);
  }
  actual_output_tensors = runner->run(input_tensors);
  ASSERT_TRUE(torch::allclose(ref_output_tensors[0], actual_output_tensors[0]));

  // We swap and activate the weight to buffer #1. This is random weight and
  // should produce incorrect results.
  runner->swap_constant_buffer();
  actual_output_tensors = runner->run(input_tensors);
  ASSERT_FALSE(
      torch::allclose(ref_output_tensors[0], actual_output_tensors[0]));

  // Swap back to buffer #2 which is the real constants.
  runner->swap_constant_buffer();
  actual_output_tensors = runner->run(input_tensors);
  ASSERT_TRUE(torch::allclose(ref_output_tensors[0], actual_output_tensors[0]));

  for (auto& pair : rand_map) {
    delete pair.second;
  }
  for (auto& pair : real_map) {
    delete pair.second;
  }
}

#if defined(USE_CUDA) || defined(USE_ROCM)
void test_aoti_double_buffering_with_tensor_constants() {
  torch::NoGradGuard no_grad;

  std::string data_path = (std::filesystem::path(
                               STRINGIZE(CMAKE_CURRENT_BINARY_DIR)) /
                               "data_with_tensor_constants.pt")
                               .string();

  torch::jit::script::Module data_loader = torch::jit::load(data_path);
  std::string path_attr = "model_so_path";
  std::string inputs_attr = "inputs";
  std::string w_attr = "w";
  std::string outputs_attr = "outputs";
  const auto& model_so_path = data_loader.attr(path_attr.c_str()).toStringRef();
  auto input_tensors =
      data_loader.attr(inputs_attr.c_str()).toTensorList().vec();
  const auto& w_tensors = data_loader.attr(w_attr.c_str()).toTensor();
  const auto& ref_output_tensors =
      data_loader.attr(outputs_attr.c_str()).toTensorList().vec();

  torch::inductor::TensorConstantMap real_map;
  real_map.emplace("L__self___w", new at::Tensor(w_tensors));

  std::unique_ptr<torch::inductor::AOTIModelContainerRunner> runner;
  runner = std::make_unique<torch::inductor::AOTIModelContainerRunnerCuda>(
      model_so_path.c_str());

  // By default, buffer #1 get loaded with burned in weights. Correct results.
  auto actual_output_tensors = runner->run(input_tensors);
  ASSERT_TRUE(torch::allclose(ref_output_tensors[0], actual_output_tensors[0]));

  // We update the weights to buffer #2 and activate it. This should still
  // produce correct result, since we would have copied the tensor_constants.
  runner->update_inactive_constant_buffer(real_map);
  runner->swap_constant_buffer();
  actual_output_tensors = runner->run(input_tensors);
  ASSERT_TRUE(torch::allclose(ref_output_tensors[0], actual_output_tensors[0]));

  for (auto& pair : real_map) {
    delete pair.second;
  }
}

void test_aoti_user_managed_buffer() {
  torch::NoGradGuard no_grad;

  std::string data_path =
      (std::filesystem::path(
           STRINGIZE(CMAKE_CURRENT_BINARY_DIR)) / "large_data.pt")
           .string();

  // Memory information variable
  size_t DATASIZE = 128 * 1024 * 1024; // We have 128MB of weight data.

  torch::jit::script::Module data_loader = torch::jit::load(data_path);
  std::string path_attr = "model_so_path";
  std::string inputs_attr = "inputs";
  std::string outputs_attr = "outputs";
  std::string weights_attr = "w_pre";
  std::string add_attr = "w_add";
  const auto& model_so_path = data_loader.attr(path_attr.c_str()).toStringRef();
  auto input_tensors =
      data_loader.attr(inputs_attr.c_str()).toTensorList().vec();
  const auto& ref_output_tensors =
      data_loader.attr(outputs_attr.c_str()).toTensorList().vec();

  const auto& weight_tensors =
      data_loader.attr(weights_attr.c_str()).toTensor();
  const auto& add_tensors = data_loader.attr(add_attr.c_str()).toTensor();

  torch::inductor::TensorConstantMap rand_map, real_map;
  at::Tensor rand_pre, rand_add;
  at::Tensor w_pre, w_add;
  rand_pre = at::randn({4096, 4096}).contiguous().to(at::kCUDA);
  rand_add = at::randn({4096, 4096}).contiguous().to(at::kCUDA);
  w_pre = at::Tensor(weight_tensors).contiguous().to(at::kCUDA);
  w_add = at::Tensor(add_tensors).contiguous().to(at::kCUDA);

  rand_map.emplace("L__self___w_pre", &rand_pre);
  rand_map.emplace("L__self___w_add", &rand_add);
  real_map.emplace("L__self___w_pre", &w_pre);
  real_map.emplace("L__self___w_add", &w_add);

  std::unique_ptr<torch::inductor::AOTIModelContainerRunner> runner;
  runner = std::make_unique<torch::inductor::AOTIModelContainerRunnerCuda>(
      model_so_path);

  // We extract the memory information starting from here.
  int device_idx = -1;
  cudaError_t cudaStatus;
  cudaStatus = cudaGetDevice(&device_idx);
  c10::cuda::CUDACachingAllocator::DeviceStats stats =
      c10::cuda::CUDACachingAllocator::getDeviceStats(device_idx);
  size_t initTorchReserved = stats.reserved_bytes[0].current;
  size_t torchReserved = stats.reserved_bytes[0].current;
  if (cudaStatus != cudaSuccess || device_idx == -1) {
    throw std::runtime_error("cudaGetDevice failed!");
  }
  // This should contain one set of weight (128MB) loaded from .so
  size_t initMemory = 0;
  size_t totalMemory = 0;
  size_t preFreeMemory = 0;
  cudaStatus = cudaMemGetInfo(&preFreeMemory, &totalMemory);
  if (cudaStatus != cudaSuccess) {
    throw std::runtime_error("cudaMemGetInfo failed!");
  }
  // At this point, no memory should be consumed since we freed them all.
  runner->swap_constant_buffer();
  runner->free_inactive_constant_buffer();
  runner->swap_constant_buffer();
  cudaStatus = cudaMemGetInfo(&initMemory, &totalMemory);
  if (cudaStatus != cudaSuccess) {
    throw std::runtime_error("cudaMemGetInfo failed!");
  }
  ASSERT_EQ(initMemory - DATASIZE, preFreeMemory);

  // We update the active buffer, but with user_managed = True. This shouldn't
  // add any memory consumption.
  runner->update_constant_buffer(
      real_map,
      /*use_inactive = */ false,
      /*validate_full_updates = */ true,
      /*user_managed = */ true);
  size_t updateMemory = 0;
  cudaStatus = cudaMemGetInfo(&updateMemory, &totalMemory);
  if (cudaStatus != cudaSuccess) {
    throw std::runtime_error("cudaMemGetInfo failed!");
  }
  ASSERT_EQ(initMemory, updateMemory);

  // Make sure the output is correct with user managed buffer.
  auto actual_output_tensors = runner->run(input_tensors);
  ASSERT_TRUE(torch::allclose(ref_output_tensors[0], actual_output_tensors[0]));

  // Update with rand_map and extract the output of rand_map.
  // We let user_managed = false for rand_map, this should increase memory
  // consumption.
  cudaStatus = cudaMemGetInfo(&initMemory, &totalMemory);
  if (cudaStatus != cudaSuccess) {
    throw std::runtime_error("cudaMemGetInfo failed!");
  }
  runner->update_constant_buffer(
      rand_map,
      /*use_inactive = */ true,
      /*validate_full_updates = */ true,
      /*user_managed = */ false);
  cudaStatus = cudaMemGetInfo(&updateMemory, &totalMemory);
  if (cudaStatus != cudaSuccess) {
    throw std::runtime_error("cudaMemGetInfo failed!");
  }
  ASSERT_EQ(initMemory - DATASIZE, updateMemory);

  runner->swap_constant_buffer();
  auto ref_rand_output_tensors = runner->run(input_tensors);
  ASSERT_FALSE(
      torch::allclose(ref_output_tensors[0], ref_rand_output_tensors[0]));

  // Free everything.
  runner->free_inactive_constant_buffer();
  runner->swap_constant_buffer();
  runner->free_inactive_constant_buffer();

  // Set buffer #1 user_managed, and #2 not user managed, and compare the
  // underlying data
  runner->update_constant_buffer(
      real_map,
      /*use_inactive = */ false,
      /*validate_full_updates = */ true,
      /*user_managed = */ false);
  runner->update_constant_buffer(
      real_map,
      /*use_inactive = */ true,
      /*validate_full_updates = */ true,
      /*user_managed = */ true);

  auto extracted_active_weight =
      runner->extract_constants_map(/* use_inactive = */ false);
  auto extracted_inactive_weight =
      runner->extract_constants_map(/* use_inactive = */ true);
  auto cmp_real_map = derefTensorConstantMap(real_map);
  // Value-wise all weights are equal
  ASSERT_TRUE(compareConstantMap(extracted_active_weight, cmp_real_map));
  ASSERT_TRUE(compareConstantMap(extracted_inactive_weight, cmp_real_map));
  // Only when user_managed has the same underlying if set to true.
  ASSERT_FALSE(
      extracted_active_weight["L__self___w_pre"].data_ptr() ==
      cmp_real_map["L__self___w_pre"].data_ptr());
  ASSERT_TRUE(
      extracted_inactive_weight["L__self___w_pre"].data_ptr() ==
      cmp_real_map["L__self___w_pre"].data_ptr());

  // From non user_managed
  actual_output_tensors = runner->run(input_tensors);
  ASSERT_TRUE(torch::allclose(ref_output_tensors[0], actual_output_tensors[0]));

  // From user_managed
  runner->swap_constant_buffer();
  actual_output_tensors = runner->run(input_tensors);
  ASSERT_TRUE(torch::allclose(ref_output_tensors[0], actual_output_tensors[0]));

  // We modify the buffer by the data's pointer outside of container.
  cudaMemcpy(
      real_map["L__self___w_add"]->data_ptr(),
      rand_map["L__self___w_add"]->data_ptr(),
      4096 * 4096 * sizeof(float),
      cudaMemcpyDeviceToDevice);
  cudaMemcpy(
      real_map["L__self___w_pre"]->data_ptr(),
      rand_map["L__self___w_pre"]->data_ptr(),
      4096 * 4096 * sizeof(float),
      cudaMemcpyDeviceToDevice);

  // We should get the result of the rand output.
  actual_output_tensors = runner->run(input_tensors);
  ASSERT_TRUE(
      torch::allclose(ref_rand_output_tensors[0], actual_output_tensors[0]));
}

void test_aoti_free_buffer(bool use_runtime_constant_folding) {
  torch::NoGradGuard no_grad;

  std::string data_path =
      (std::filesystem::path(
           STRINGIZE(CMAKE_CURRENT_BINARY_DIR)) / "large_data.pt")
           .string();

  // Memory information variable
  size_t DATASIZE = 128 * 1024 * 1024; // We have 128MB of weight data.
  size_t FOLDEDDATASIZE = use_runtime_constant_folding
      ? 64 * 1024 * 1024
      : 0; // We have 64MB of folded data.

  torch::jit::script::Module data_loader = torch::jit::load(data_path);
  std::string path_attr = "model_so_path";
  if (use_runtime_constant_folding) {
    path_attr += std::string("_use_runtime_constant_folding");
  }
  std::string inputs_attr = "inputs";
  std::string outputs_attr = "outputs";
  std::string weights_attr = "w_pre";
  std::string add_attr = "w_add";
  const auto& model_so_path = data_loader.attr(path_attr.c_str()).toStringRef();
  auto input_tensors =
      data_loader.attr(inputs_attr.c_str()).toTensorList().vec();
  const auto& ref_output_tensors =
      data_loader.attr(outputs_attr.c_str()).toTensorList().vec();

  const auto& weight_tensors =
      data_loader.attr(weights_attr.c_str()).toTensor();
  const auto& add_tensors = data_loader.attr(add_attr.c_str()).toTensor();

  torch::inductor::TensorConstantMap rand_map, real_map;
  rand_map.emplace("L__self___w_pre", new at::Tensor(at::randn({4096, 4096})));
  rand_map.emplace("L__self___w_add", new at::Tensor(at::randn({4096, 4096})));
  real_map.emplace("L__self___w_pre", new at::Tensor(weight_tensors));
  real_map.emplace("L__self___w_add", new at::Tensor(add_tensors));

  std::unique_ptr<torch::inductor::AOTIModelContainerRunner> runner;
  runner = std::make_unique<torch::inductor::AOTIModelContainerRunnerCuda>(
      model_so_path);

  // We extract the memory information starting from here.
  int device_idx = -1;
  cudaError_t cudaStatus;
  cudaStatus = cudaGetDevice(&device_idx);
  if (cudaStatus != cudaSuccess || device_idx == -1) {
    throw std::runtime_error("cudaGetDevice failed!");
  }
  c10::cuda::CUDACachingAllocator::DeviceStats stats =
      c10::cuda::CUDACachingAllocator::getDeviceStats(device_idx);
  size_t initTorchActive = stats.active_bytes[0].current;
  size_t initTorchReserved = stats.reserved_bytes[0].current;
  // This should contain one set of weight (128MB) loaded from .so
  size_t torchActive1, torchActive2;
  size_t torchReserved1, torchReserved2;
  size_t initMemory = 0;
  size_t totalMemory = 0;
  cudaStatus = cudaMemGetInfo(&initMemory, &totalMemory);
  if (cudaStatus != cudaSuccess) {
    throw std::runtime_error("cudaMemGetInfo failed!");
  }

  // We update inactive buffer, this should create one copy (128MB) at buffer #2
  runner->update_inactive_constant_buffer(real_map);
  size_t updateMemory2 = 0;
  cudaStatus = cudaMemGetInfo(&updateMemory2, &totalMemory);
  if (cudaStatus != cudaSuccess) {
    throw std::runtime_error("cudaMemGetInfo failed!");
  }
  ASSERT_EQ(initMemory - DATASIZE, updateMemory2);

  // Call run, this should run const_fold and create the folded constant in #2
  // (64MB).
  if (use_runtime_constant_folding) {
    runner->run_const_fold(/* use_inactive = */ true);
    stats = c10::cuda::CUDACachingAllocator::getDeviceStats(device_idx);
    torchActive1 = stats.active_bytes[0].current;
    torchReserved1 = stats.reserved_bytes[0].current;
    size_t constFoldMemory = 0;
    cudaStatus = cudaMemGetInfo(&constFoldMemory, &totalMemory);
    if (cudaStatus != cudaSuccess) {
      throw std::runtime_error("cudaMemGetInfo failed!");
    }
    ASSERT_EQ(
        initMemory - DATASIZE - (torchReserved1 - initTorchReserved),
        constFoldMemory);
    ASSERT_EQ(torchActive1 - initTorchActive, FOLDEDDATASIZE);
  }

  // We swap and free the inactive buffer. (Use #2 and free #1)
  // Note that buffer #1 does not include folded-const
  stats = c10::cuda::CUDACachingAllocator::getDeviceStats(device_idx);
  torchActive1 = stats.active_bytes[0].current;
  torchReserved1 = stats.reserved_bytes[0].current;
  runner->swap_constant_buffer();
  runner->free_inactive_constant_buffer();
  stats = c10::cuda::CUDACachingAllocator::getDeviceStats(device_idx);
  torchActive2 = stats.active_bytes[0].current;
  torchReserved2 = stats.reserved_bytes[0].current;
  size_t postFreeMemory = 0;
  cudaStatus = cudaMemGetInfo(&postFreeMemory, &totalMemory);
  if (cudaStatus != cudaSuccess) {
    throw std::runtime_error("cudaMemGetInfo failed!");
  }
  // We should only have one set of buffer (#2), available memory should equal
  // initial memory minus the folded constants.
  ASSERT_EQ(initMemory - (torchReserved2 - initTorchReserved), postFreeMemory);
  // Buffer #1 does not include folded-consts
  ASSERT_EQ(torchActive2 - torchActive1, 0);

  // We update random weights to buffer #1 and run const fold.
  // We will have 2 full set of data plus 2 set of const-folded data.
  runner->update_inactive_constant_buffer(rand_map);
  runner->run_const_fold(/* use_inactive = */ true);
  stats = c10::cuda::CUDACachingAllocator::getDeviceStats(device_idx);
  torchActive1 = stats.active_bytes[0].current;
  torchReserved1 = stats.reserved_bytes[0].current;
  size_t updateMemory1 = 0;
  cudaStatus = cudaMemGetInfo(&updateMemory1, &totalMemory);
  if (cudaStatus != cudaSuccess) {
    throw std::runtime_error("cudaMemGetInfo failed!");
  }
  ASSERT_EQ(
      initMemory - DATASIZE - (torchReserved1 - initTorchReserved),
      updateMemory1);
  ASSERT_EQ(torchActive1 - initTorchActive, 2 * FOLDEDDATASIZE);

  // We directly free the buffer #1. This would free the DATASIZE weight.
  // If folded constant exists, it will not directly free the cudaMalloc, but
  // decrease the active buffer in CachingAllocator instead.
  stats = c10::cuda::CUDACachingAllocator::getDeviceStats(device_idx);
  torchActive1 = stats.active_bytes[0].current;
  runner->free_inactive_constant_buffer();
  cudaStatus = cudaMemGetInfo(&updateMemory1, &totalMemory);
  if (cudaStatus != cudaSuccess) {
    throw std::runtime_error("cudaMemGetInfo failed!");
  }
  stats = c10::cuda::CUDACachingAllocator::getDeviceStats(device_idx);
  torchActive2 = stats.active_bytes[0].current;
  torchReserved2 = stats.reserved_bytes[0].current;
  ASSERT_EQ(initMemory - (torchReserved2 - initTorchReserved), updateMemory1);
  ASSERT_EQ(FOLDEDDATASIZE, torchActive1 - torchActive2);

  // Free buffer #1 again, since #1 is freed, nothing should change.
  stats = c10::cuda::CUDACachingAllocator::getDeviceStats(device_idx);
  torchActive1 = stats.active_bytes[0].current;
  runner->free_inactive_constant_buffer();
  stats = c10::cuda::CUDACachingAllocator::getDeviceStats(device_idx);
  torchActive2 = stats.active_bytes[0].current;
  cudaStatus = cudaMemGetInfo(&updateMemory1, &totalMemory);
  if (cudaStatus != cudaSuccess) {
    throw std::runtime_error("cudaMemGetInfo failed!");
  }
  ASSERT_EQ(initMemory - (torchReserved2 - initTorchReserved), updateMemory1);
  ASSERT_EQ(torchActive1 - torchActive2, 0);

  // Swap and free #2, no data should exist in memory now.
  // However, the folded constants might still occupies the CUDA memory in
  // CachedAllocator.
  stats = c10::cuda::CUDACachingAllocator::getDeviceStats(device_idx);
  torchActive1 = stats.active_bytes[0].current;
  torchReserved1 = stats.reserved_bytes[0].current;
  runner->swap_constant_buffer();
  runner->free_inactive_constant_buffer();
  stats = c10::cuda::CUDACachingAllocator::getDeviceStats(device_idx);
  torchActive2 = stats.active_bytes[0].current;
  torchReserved2 = stats.reserved_bytes[0].current;
  cudaStatus = cudaMemGetInfo(&updateMemory1, &totalMemory);
  if (cudaStatus != cudaSuccess) {
    throw std::runtime_error("cudaMemGetInfo failed!");
  }

  ASSERT_EQ(
      initMemory + DATASIZE - (torchReserved2 - initTorchReserved),
      updateMemory1);
  ASSERT_EQ(FOLDEDDATASIZE, torchActive1 - torchActive2);
  ASSERT_EQ(0, torchActive2 - initTorchActive);

  for (auto& pair : rand_map) {
    delete pair.second;
  }
  for (auto& pair : real_map) {
    delete pair.second;
  }
}

void test_cuda_alloc_test() {
  torch::NoGradGuard no_grad;

  std::string data_path =
      (std::filesystem::path(
           STRINGIZE(CMAKE_CURRENT_BINARY_DIR)) / "cuda_alloc_data.pt")
           .string();
  torch::jit::script::Module data_loader = torch::jit::load(data_path);
  std::string path_attr = "model_so_path";
  std::string inputs_attr = "inputs";
  std::string outputs_attr = "outputs";
  const auto& model_so_path = data_loader.attr(path_attr.c_str()).toStringRef();
  const auto& ref_output_tensors =
      data_loader.attr(outputs_attr.c_str()).toTensorList().vec();

  size_t DATASIZE = 128 * 1024 * 1024; // We have 128MB of weight data.

  int device_idx = -1;
  cudaError_t cudaStatus;
  cudaStatus = cudaGetDevice(&device_idx);
  if (cudaStatus != cudaSuccess || device_idx == -1) {
    throw std::runtime_error("cudaGetDevice failed!");
  }

  c10::cuda::CUDACachingAllocator::emptyCache();
  c10::cuda::CUDACachingAllocator::DeviceStats stats =
      c10::cuda::CUDACachingAllocator::getDeviceStats(device_idx);
  size_t initTorchActive = stats.allocated_bytes[0].current;
  auto runner = std::make_unique<torch::inductor::AOTIModelContainerRunnerCuda>(
      model_so_path);
  stats = c10::cuda::CUDACachingAllocator::getDeviceStats(device_idx);
  size_t torchActive = stats.allocated_bytes[0].current;

  ASSERT_EQ(initTorchActive + DATASIZE, torchActive);

  auto actual_output_tensors =
      runner->run(data_loader.attr(inputs_attr.c_str()).toTensorList().vec());
  ASSERT_TRUE(torch::allclose(ref_output_tensors[0], actual_output_tensors[0]));
}

#ifdef USE_CUDA
class ThreadPool {
 private:
  struct Task {
    int id;
    std::vector<torch::Tensor> inputs;
  };

  std::vector<std::thread> workers;
  std::vector<c10::cuda::CUDAStream> cuda_streams;
  std::queue<Task> tasks;
  std::mutex queue_mutex;
  std::condition_variable condition;
  std::condition_variable completion_condition;
  std::atomic<int> active_tasks{0};
  std::atomic<bool> stop;

 public:
  ThreadPool(size_t num_threads) : stop(false) {
    // Create CUDA streams
    cuda_streams.reserve(num_threads);
    for (size_t i = 0; i < num_threads; ++i) {
      cuda_streams.push_back(c10::cuda::getStreamFromPool());
    }

    // Create worker threads
    for (size_t i = 0; i < num_threads; ++i) {
      workers.emplace_back([this, i] {
        while (true) {
          Task task;
          {
            std::unique_lock<std::mutex> lock(this->queue_mutex);
            this->condition.wait(
                lock, [this] { return this->stop || !this->tasks.empty(); });

            if (this->stop && this->tasks.empty()) {
              return;
            }

            task = std::move(this->tasks.front());
            this->tasks.pop();
          }

          // Process the task with this thread's CUDA stream
          process_function(task.id, task.inputs, this->cuda_streams[i]);

          // Mark task as completed
          {
            std::unique_lock<std::mutex> lock(this->queue_mutex);
            active_tasks--;
            if (active_tasks == 0 && this->tasks.empty()) {
              completion_condition.notify_all();
            }
          }
        }
      });
    }
  }

  // Updated processing function for vector of tensors and CUDA stream
  std::function<
      void(int, const std::vector<torch::Tensor>&, c10::cuda::CUDAStream&)>
      process_function;

  // Enqueue task with vector of tensors as input
  void enqueue(int i, std::vector<torch::Tensor> inputs) {
    {
      std::unique_lock<std::mutex> lock(queue_mutex);
      tasks.push({i, std::move(inputs)});
      active_tasks++;
    }
    condition.notify_one();
  }

  // Wait for all tasks to complete
  void wait() {
    std::unique_lock<std::mutex> lock(queue_mutex);
    completion_condition.wait(
        lock, [this] { return active_tasks == 0 && tasks.empty(); });
  }

  ~ThreadPool() {
    {
      std::unique_lock<std::mutex> lock(queue_mutex);
      stop = true;
    }

    condition.notify_all();
    for (std::thread& worker : workers) {
      worker.join();
    }
  }
};

void test_multi_cuda_streams(const std::string& device) {
  c10::InferenceMode mode;
  std::string data_path =
      (std::filesystem::path(STRINGIZE(CMAKE_CURRENT_BINARY_DIR)) / "data.pt")
           .string();
  torch::jit::script::Module data_loader = torch::jit::load(data_path);
  std::string path_attr = "pt2_package_path_" + device;
  std::string inputs_attr = "inputs_" + device;
  std::string outputs_attr = "outputs_" + device;
  const auto& pt2_package_path =
      data_loader.attr(path_attr.c_str()).toStringRef();
  const auto& ref_output_tensors =
      data_loader.attr(outputs_attr.c_str()).toTensorList().vec();
  auto inputs = data_loader.attr(inputs_attr.c_str()).toTensorList().vec();

  constexpr int N = 16;
  constexpr int num_threads = 4;
  std::vector<std::vector<torch::Tensor>> all_outputs(N);
  // Create thread pool with desired number of threads
  torch::inductor::AOTIModelPackageLoader loader(
      pt2_package_path, "model", false, num_threads);
  ThreadPool pool(num_threads);
  std::mutex results_mutex;

  // Set the processing function
  pool.process_function = [&](int i,
                              const std::vector<torch::Tensor>& inputs,
                              c10::cuda::CUDAStream& stream) {
    // Run inference with the task-specific input
    std::vector<torch::Tensor> outputs = loader.run(inputs, stream.stream());
    // Store results safely
    {
      std::lock_guard<std::mutex> lock(results_mutex);
      all_outputs[i] = outputs;
    }
  };
  // Enqueue all tasks
  for (int i = 0; i < N; i++) {
    pool.enqueue(i, inputs);
  }
  // Wait for all tasks to complete
  pool.wait();

  for (int i = 0; i < N; i++) {
    ASSERT_TRUE(torch::allclose(ref_output_tensors[0], all_outputs[i][0]));
  }
}
#endif // USE_CUDA

void test_concurrent_run_with_const_fold(const std::string& device) {
  torch::NoGradGuard no_grad;

  std::string data_path =
      (std::filesystem::path(STRINGIZE(CMAKE_CURRENT_BINARY_DIR)) / "data.pt")
           .string();

  torch::jit::script::Module data_loader = torch::jit::load(data_path);
  std::string suffix = device + "_use_runtime_constant_folding";
  const auto& model_so_path =
      data_loader.attr(("model_so_path_" + suffix).c_str()).toStringRef();
  auto input_tensors =
      data_loader.attr(("inputs_" + suffix).c_str()).toTensorList().vec();
  const auto& ref_output_tensors =
      data_loader.attr(("outputs_" + suffix).c_str()).toTensorList().vec();

  // num_models=1 forces all threads to contend for the single model instance.
  std::unique_ptr<torch::inductor::AOTIModelContainerRunner> runner;
  if (device == "cuda") {
    runner = std::make_unique<torch::inductor::AOTIModelContainerRunnerCuda>(
        model_so_path, /* num_models = */ 1);
  } else {
    FAIL() << "unsupported device: " << device;
  }

  constexpr int num_threads = 4;
  constexpr int num_iters = 4;
  std::atomic<int> ready{0};
  std::atomic<bool> failed{false};

  std::vector<std::thread> threads;
  for (int t = 0; t < num_threads; t++) {
    threads.emplace_back([&]() {
      ready.fetch_add(1);
      while (ready.load() < num_threads) {
      }
      for (int i = 0; i < num_iters; i++) {
        auto outputs = runner->run(input_tensors);
        if (!torch::allclose(ref_output_tensors[0], outputs[0])) {
          failed.store(true);
        }
      }
    });
  }

  for (auto& t : threads) {
    t.join();
  }

  ASSERT_FALSE(failed.load())
      << "One or more threads produced incorrect output";
}

// S638065 regression: with runtime constant folding, update_constant_buffer
// copies the new weights into the (inactive) constant buffer on the default
// stream (async device-to-device cudaMemcpy), while run_const_fold reads those
// constants on a separate stream. On ROCm the default stream is not implicitly
// ordered with other streams, so without a barrier the fold can read
// not-yet-copied weights and bake stale values into the folded constants.
// This exercises that exact edge: fold on a dedicated pool stream, then check
// the folded result against an independently computed reference. Each iteration
// uses fresh random weights so a stale read produces a detectably wrong output.
// The race is timing-dependent (the copy usually wins), hence the loop; a clean
// run depends on the cudaStreamSynchronize(0) at the end of
// update_constant_buffer (model_container.h). On NVIDIA the legacy default
// stream implicitly serializes, so this passes regardless.
void test_aoti_const_fold_separate_stream() {
  torch::NoGradGuard no_grad;

  // Use the large (size=4096) model so the per-weight D2D copy is big enough to
  // still be in flight when the fold starts reading.
  std::string data_path =
      (std::filesystem::path(
           STRINGIZE(CMAKE_CURRENT_BINARY_DIR)) / "large_data.pt")
           .string();
  torch::jit::script::Module data_loader = torch::jit::load(data_path);
  const auto& model_so_path =
      data_loader.attr("model_so_path_use_runtime_constant_folding")
          .toStringRef();
  auto input_tensors = data_loader.attr("inputs").toTensorList().vec();
  const auto& x = input_tensors[0];
  const int64_t size = x.size(1);

  auto runner = std::make_unique<torch::inductor::AOTIModelContainerRunnerCuda>(
      model_so_path);

  // Dedicated non-blocking pool stream for folding, distinct from the default
  // stream that update_constant_buffer copies the weights on.
  c10::cuda::CUDAStream fold_stream = c10::cuda::getStreamFromPool();

  constexpr int kIters = 100;
  for (int i = 0; i < kIters; ++i) {
    at::Tensor w_pre = at::randn({size, size}, x.options());
    at::Tensor w_add = at::randn({size, size}, x.options());
    // Independent reference, mirroring Net.forward in test.py.
    at::Tensor expected =
        at::matmul(x, at::relu(at::transpose(w_pre, 0, 1)) + w_add);

    torch::inductor::TensorConstantMap weight_map;
    weight_map.emplace("L__self___w_pre", &w_pre);
    weight_map.emplace("L__self___w_add", &w_add);

    // Copy weights into the inactive buffer (default stream), then fold on a
    // separate stream. We deliberately do NOT synchronize between these two:
    // ordering the copy before the fold is exactly what the fix must guarantee.
    runner->update_inactive_constant_buffer(weight_map);
    runner->run_const_fold(
        /* use_inactive = */ true,
        reinterpret_cast<AOTInductorStreamHandle>(fold_stream.stream()));
    // Isolate the copy->fold edge (under test) from the fold->inference edge by
    // finishing the fold before swapping the freshly folded buffer in.
    fold_stream.synchronize();
    runner->swap_constant_buffer();

    auto actual = runner->run(input_tensors);
    ASSERT_TRUE(torch::allclose(
        expected, actual[0], /* rtol = */ 1e-2, /* atol = */ 1e-2))
        << "iter " << i
        << ": folded constants reflect stale weights (S638065); the const-fold "
           "read the constant buffer before the default-stream weight copy "
           "completed";
  }
}
#endif // USE_CUDA || USE_ROCM
} // namespace

namespace torch::aot_inductor {

// Test fixture that ensures test data is generated once for all tests
class AotInductorTest : public ::testing::Test {
 public:
  // This runs once before all tests in this test suite
  static void SetUpTestSuite() {
    ensureTestDataGenerated();
  }
};

TEST_F(AotInductorTest, BasicTestCpu) {
  test_aoti("cpu", false);
}

TEST_F(AotInductorTest, BasicScriptTestCpu) {
  test_aoti_script("cpu");
}

TEST_F(AotInductorTest, BasicPackageLoaderTestCpu) {
  test_aoti_package_loader("cpu", false);
}

TEST_F(AotInductorTest, ExtractConstantsMapCpu) {
  test_aoti_extract_constants_map("cpu");
}

#ifdef USE_CUDA
TEST_F(AotInductorTest, BasicTestCuda) {
  test_aoti("cuda", true);
  test_aoti("cuda", false);
}

TEST_F(AotInductorTest, BasicScriptTestCuda) {
  test_aoti_script("cuda");
}

TEST_F(AotInductorTest, BasicPackageLoaderTestCuda) {
  test_aoti_package_loader("cuda", false);
}

TEST_F(AotInductorTest, BasicPackageLoaderTestMultiGpuCuda) {
  test_aoti_package_loader_multi_gpu("cuda", false);
}

TEST_F(AotInductorTest, UpdateUserManagedConstantsCuda) {
  test_aoti_user_managed_buffer();
}

TEST_F(AotInductorTest, RuntimeUpdateConstantsCuda) {
  test_aoti_constants_update("cuda", true);
}

TEST_F(AotInductorTest, UpdateConstantsCuda) {
  test_aoti_constants_update("cuda", false);
}

TEST_F(AotInductorTest, FreeFoldInputOnlyConstantsUpdateCuda) {
  test_aoti_free_fold_constants_update();
}

TEST_F(AotInductorTest, FreeFoldInputOnlyConstantsUpdateDefaultCuda) {
  test_aoti_free_fold_constants_update_mode(FoldInputUpdateMode::kDefault);
}

TEST_F(AotInductorTest, FreeFoldInputOnlyConstantsUpdateUserManagedCuda) {
  test_aoti_free_fold_constants_update_mode(FoldInputUpdateMode::kUserManaged);
}

TEST_F(AotInductorTest, FreeFoldInputOnlyConstantsUpdateAllowH2DCopyCuda) {
  test_aoti_free_fold_constants_update_mode(FoldInputUpdateMode::kAllowH2DCopy);
}

TEST_F(AotInductorTest, FreeFoldInputOnlyConstantsInactiveUpdateFoldSwapCuda) {
  test_aoti_free_fold_inactive_update_fold_swap();
}

TEST_F(AotInductorTest, FreeFoldInputOnlyConstantsRepeatedConstFoldCuda) {
  test_aoti_free_fold_repeated_const_fold();
}

TEST_F(AotInductorTest, FreeFoldInputOnlyConstantsRepeatedUpdatesCuda) {
  test_aoti_free_fold_repeated_updates();
}

TEST_F(AotInductorTest, ExtractConstantsMapCuda) {
  test_aoti_extract_constants_map("cuda");
}

TEST_F(AotInductorTest, RuntimeUpdateInactiveConstantsCuda) {
  test_aoti_double_buffering("cuda", true);
}

TEST_F(AotInductorTest, UpdateInactiveConstantsCuda) {
  test_aoti_double_buffering("cuda", false);
}

TEST_F(AotInductorTest, UpdateInactiveConstantsWithTensorConstantsCuda) {
  test_aoti_double_buffering_with_tensor_constants();
}

TEST_F(AotInductorTest, FreeInactiveConstantBufferCuda) {
  test_aoti_free_buffer(false);
}

TEST_F(AotInductorTest, FreeInactiveConstantBufferRuntimeConstantFoldingCuda) {
  test_aoti_free_buffer(true);
}

TEST_F(AotInductorTest, MultiStreamTestCuda) {
  test_multi_cuda_streams("cuda");
}

TEST_F(AotInductorTest, CudaAllocTestCuda) {
  test_cuda_alloc_test();
}

TEST_F(AotInductorTest, ConcurrentRunConstFoldCuda) {
  test_concurrent_run_with_const_fold("cuda");
}
#endif

// Registered for ROCm as well as CUDA: the S638065 race only manifests on AMD
// (on NVIDIA the legacy default stream implicitly orders with the fold stream).
#if defined(USE_CUDA) || defined(USE_ROCM)
TEST_F(AotInductorTest, ConstFoldSeparateStreamCuda) {
  test_aoti_const_fold_separate_stream();
}
#endif // USE_CUDA || USE_ROCM

} // namespace torch::aot_inductor
