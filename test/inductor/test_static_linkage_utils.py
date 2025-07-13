# Owner(s): ["module: inductor"]
from torch.testing._internal.common_utils import run_tests


def get_static_linkage_main_cpp_file():
    return """
#include <dlfcn.h>
#include <iostream>
#include <memory>
#include <torch/torch.h>
#include <vector>

#include <cuda.h>
#include <cuda_runtime_api.h>
// Include the AOTInductor headers
#include "Minus.wrapper/data/aotinductor/model/Minus.h"
#include "Plus.wrapper/data/aotinductor/model/Plus.h"
#include <torch/csrc/inductor/aoti_runtime/model_container.h>
#include <torch/csrc/inductor/aoti_torch/tensor_converter.h>

using torch::aot_inductor::AOTInductorModelMinus;
using torch::aot_inductor::AOTInductorModelPlus;
using torch::aot_inductor::ConstantHandle;
using torch::aot_inductor::ConstantMap;


int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr
        << "Usage: ./main <path> <device>"
        << std::endl;
    return 1;
  }
  std::string path = argv[1];
  std::string device_str = argv[2];
  try {
    torch::Device device(device_str);

    // Create two input tensors (10x10)
    auto tensor1 = torch::ones({10, 10}, device);
    auto tensor2 = torch::ones({10, 10}, device);
    // Create two input tensors (10x10)
    auto tensor3 = torch::ones({10, 10}, device);
    auto tensor4 = torch::ones({10, 10}, device);

    std::vector<at::Tensor> input_tensors = {tensor1, tensor2};
    std::vector<at::Tensor> input_tensors2 = {tensor3, tensor4};

    // Create array of input handles
    auto input_handles1 =
        torch::aot_inductor::unsafe_alloc_new_handles_from_tensors(
            input_tensors);
    auto input_handles2 =
        torch::aot_inductor::unsafe_alloc_new_handles_from_tensors(
            input_tensors2);

    // Create array for output handle
    AtenTensorHandle output_handle1;
    AtenTensorHandle output_handle2;

    auto constants_map = std::make_shared<ConstantMap>();
    auto constants_array = std::make_shared<std::vector<ConstantHandle>>();
    auto model1 = AOTInductorModelPlus::Create(
        constants_map, constants_array, device_str,
        path + "Plus.wrapper/data/"
        "aotinductor/model/");
    model1->load_constants();

    auto constants_map2 = std::make_shared<ConstantMap>();
    auto constants_array2 = std::make_shared<std::vector<ConstantHandle>>();
    auto model2 = AOTInductorModelMinus::Create(
        constants_map2, constants_array2, device_str,
        path + "Minus.wrapper/data/"
        "aotinductor/model/");
    model2->load_constants();

    // Run the model
    torch::aot_inductor::DeviceStreamType stream1 = nullptr;
    torch::aot_inductor::DeviceStreamType stream2 = nullptr;
    model1->run(&input_handles1[0], &output_handle1, stream1, nullptr);
    model2->run(&input_handles2[0], &output_handle2, stream2, nullptr);

    // Convert output handle to tensor
    auto output_tensor1 =
        torch::aot_inductor::alloc_tensors_by_stealing_from_handles(
            &output_handle1, 1);
    auto output_tensor2 =
        torch::aot_inductor::alloc_tensors_by_stealing_from_handles(
            &output_handle2, 1);

    if (!(torch::all(output_tensor1[0] == 2).item<bool>())){
      std::cout << "Wrong Output for Plus Model: " << output_tensor1 << std::endl;
      throw std::runtime_error("Tensor does not contain only the expected value 2.");
    }
    if (!(torch::all(output_tensor2[0] == 0).item<bool>())){
      std::cout << "Wrong Output for Minus Model: " << output_tensor1 << std::endl;
      throw std::runtime_error("Tensor does not contain only the expected value 0.");
    }

    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}

"""


def get_static_linkage_makelist_file_cuda():
    return """
cmake_minimum_required(VERSION 3.10)
project(TestProject)

set(CMAKE_CXX_STANDARD 17)

find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)

add_subdirectory(Plus.wrapper/data/aotinductor/model/)
add_subdirectory(Minus.wrapper/data/aotinductor/model/)

# Create executable
add_executable(main main.cpp)

target_compile_definitions(main PRIVATE USE_CUDA)

target_link_libraries(main PRIVATE torch cuda
                    ${CUDA_LIBRARIES}
                    Plus
                    Minus)
"""


def get_static_linkage_makelist_file_cpu():
    return """
cmake_minimum_required(VERSION 3.10)
project(TestProject)

set(CMAKE_CXX_STANDARD 17)

find_package(Torch REQUIRED)

add_subdirectory(Plus.wrapper/data/aotinductor/model/)
add_subdirectory(Minus.wrapper/data/aotinductor/model/)

# Create executable
add_executable(main main.cpp)

target_link_libraries(main PRIVATE torch
                    Plus
                    Minus)
"""


if __name__ == "__main__":
    run_tests()
