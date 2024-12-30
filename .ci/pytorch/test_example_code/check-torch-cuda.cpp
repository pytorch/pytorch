#include <torch/torch.h>

int main(int argc, const char* argv[]) {
    std::cout << "Checking that CUDA archs are setup correctly" << std::endl;
    TORCH_CHECK(torch::rand({ 3, 5 }, torch::Device(torch::kCUDA)).defined(), "CUDA archs are not setup correctly");

    // These have to run after CUDA is initialized

    std::cout << "Checking that magma is available" << std::endl;
    TORCH_CHECK(torch::hasMAGMA(), "MAGMA is not available");

    std::cout << "Checking that CuDNN is available" << std::endl;
    TORCH_CHECK(torch::cuda::cudnn_is_available(), "CuDNN is not available");
    return 0;
}
