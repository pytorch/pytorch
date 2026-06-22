#include <ATen/ATen.h>
#include <torch/torch.h>

int main(int argc, const char* argv[]) {
    TORCH_CHECK(at::globalContext().isXNNPACKAvailable(), "XNNPACK is not available");
    return 0;
}
