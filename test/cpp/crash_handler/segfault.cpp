#include <torch/torch.h>

// This is used in <INSERT FILE HERE> to test libtorch's error handling and
// crash dumping
int main() {
    // torch::utils::_enable_minidump_collection("/tmp");
    // torch::
    torch::crash_handler::_enable_minidump_collection("/tmp");

    volatile int* bad = nullptr;
    return *bad;
}
