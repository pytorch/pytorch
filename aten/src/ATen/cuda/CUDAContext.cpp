#include <ATen/cuda/CUDAConfig.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAPowerDamper.h>

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/util/CallOnce.h>

#include <deque>
#include <fstream>
#include <iostream>
#include <vector>
#include <thread>

namespace at::cuda {

namespace {

DeviceIndex num_gpus = -1;
c10::once_flag init_flag;
std::deque<c10::once_flag> device_flags;
std::vector<cudaDeviceProp> device_properties;

const char* power_damper_cfg_path = "/tmp/torch_power_damper_cfg";
c10::once_flag power_damper_config_file_flag;
std::thread* powerDamperThread = nullptr;
std::atomic<bool> runPowerDamper = false;

void initCUDAContextVectors() {
  num_gpus = c10::cuda::device_count();
  device_flags.resize(num_gpus);
  device_properties.resize(num_gpus);
}

void initDeviceProperty(DeviceIndex device_index) {
  cudaDeviceProp device_prop{};
  AT_CUDA_CHECK(cudaGetDeviceProperties(&device_prop, device_index));
  device_properties[device_index] = device_prop;
}

void initPowerDamperConfig() {
  if (getenv("PYTORCH_NO_POWERPLANT_BLOWUP") != nullptr) {
    std::cout << "Power Damper Enabled via Env Var." << std::endl;
    std::ofstream power_damper_config_file(power_damper_cfg_path);
    if (power_damper_config_file.is_open()) {
      const char* damper_config_env = getenv("PYTORCH_POWER_DAMPER_CONFIG");
      if (damper_config_env != nullptr) {
        power_damper_config_file << damper_config_env << std::endl;
      } else {
        power_damper_config_file << "100 80 50 2" << std::endl;
      }
      power_damper_config_file.close();
    } else {
      std::cout << "Failed to create power damper config file." << std::endl;
    }
  } else {
    std::cout << "Power Damper Disabled." << std::endl;
  }
}

void powerDamper(uint32_t *gpuIdx) {
  uint32_t tdp_pct_start = 100;
  uint32_t tdp_pct_end = 80;
  uint32_t damper_ms = 50;
  uint32_t damper_steps = 2;

  std::ifstream power_damper_config_file(power_damper_cfg_path);
  if (power_damper_config_file.is_open()) {
    try {
      std::cout << "Power damper config file found." << std::endl;
      power_damper_config_file >> tdp_pct_start >> tdp_pct_end >> damper_ms >> damper_steps;
      power_damper_config_file.close();
    } catch (const std::exception& e) {
      std::cout << "Failed to read power damper config file. Not starting power damper." << std::endl;
      return;
    }
  } else {
    std::cout << "Power damper config file not found. Not starting power damper." << std::endl;
    return;
  }

  std::cout << "Initialize power damper GPU = " << *gpuIdx << " with params: " << tdp_pct_start << ", " << tdp_pct_end << ", " << damper_ms << ", " << damper_steps << std::endl;
  facebook::gpu_power_damper::cuda_power_damper damper(*gpuIdx, tdp_pct_start, tdp_pct_end, damper_ms, damper_steps);
  damper.initialize_power_gen_params();
  
  std::cout << "Draining power." << std::endl;
  while (runPowerDamper) {
    damper.gen_and_drain_power();
  }
}

void spawnPowerDamper(uint32_t gpuIdx) {
  std::cout << "Spawning power damper." << std::endl;

  // This one needs to be joined somewhere
  uint32_t *pGpuIdx = new uint32_t(gpuIdx);
  powerDamperThread = new std::thread(powerDamper, pGpuIdx);
}

} // anonymous namespace

// We need this function to force the linking against torch_cuda(_cpp) on Windows.
// If you need to modify this function, please specify a new function and apply
// the changes according to https://github.com/pytorch/pytorch/pull/34288.
// Related issue: https://github.com/pytorch/pytorch/issues/31611.
/* Device info */
int warp_size() {
  return getCurrentDeviceProperties()->warpSize;
}

cudaDeviceProp* getCurrentDeviceProperties() {
  auto device = c10::cuda::current_device();

  // HACK ALERT: Assume that the device properties check
  // is done also in the begining of the application so this is
  // where we spawn the power damper thread.

  // Create power damper config that we can use to control the power damper
  // behavior during runtime.
  c10::call_once(power_damper_config_file_flag, initPowerDamperConfig);

  // Throw a dice so that we don't check the existance of the config file
  // for every call. This function should be called quite frequently. With 
  // this approach we have about 1 second between checks.
  bool shouldCheckDamperChanges = std::rand() % 10000 < 20;
  if (shouldCheckDamperChanges) {
    std::ifstream power_damper_config_file(power_damper_cfg_path);
    bool shouldSpawnDamper = power_damper_config_file.good();

    if (shouldSpawnDamper) {
      runPowerDamper = true;
      if (powerDamperThread == nullptr) {
        spawnPowerDamper(device);
      }
    } else {
      // Check if the damper is already running and if so, kill it
      if (runPowerDamper == true) {
        std::cout << "Killing power damper. Config file was likely deleted." << std::endl;
        runPowerDamper = false;
        powerDamperThread->join();
        delete powerDamperThread;
        powerDamperThread = nullptr;
      }
    }
  }
  

  return getDeviceProperties(device);
}

cudaDeviceProp* getDeviceProperties(c10::DeviceIndex device) {
  c10::call_once(init_flag, initCUDAContextVectors);
  if (device == -1) device = c10::cuda::current_device();
  AT_ASSERT(device >= 0 && device < num_gpus, "device=", static_cast<int>(device), ", num_gpus=", num_gpus);
  c10::call_once(device_flags[device], initDeviceProperty, device);
  return &device_properties[device];
}

bool canDeviceAccessPeer(c10::DeviceIndex device, c10::DeviceIndex peer_device) {
  c10::call_once(init_flag, initCUDAContextVectors);
  if (device == -1) device = c10::cuda::current_device();
  AT_ASSERT(device >= 0 && device < num_gpus, "device=", static_cast<int>(device), ", num_gpus=", num_gpus);
  AT_ASSERT(peer_device >= 0 && peer_device < num_gpus, "peer_device=", static_cast<int>(peer_device), ", num_gpus=", num_gpus);
  int can_access = 0;
  AT_CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, device, peer_device));
  return can_access != 0;
}

Allocator* getCUDADeviceAllocator() {
  return c10::cuda::CUDACachingAllocator::get();
}

} // namespace at::cuda
