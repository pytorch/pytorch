#include <torch/torch.h>

int main(int argc, const char* argv[]) {
    TORCH_CHECK(torch::hasMKL(), "MKL is not available");
    return 0;
}
