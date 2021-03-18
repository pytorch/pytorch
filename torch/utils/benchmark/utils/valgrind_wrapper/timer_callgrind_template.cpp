/* C++ template for Timer.collect_callgrind

This template will be consumed by `cpp_jit.py`, and will replace:
    `GLOBAL_SETUP_TEMPLATE_LOCATION`,
    `SETUP_TEMPLATE_LOCATION`
      and
    `STMT_TEMPLATE_LOCATION`
sections with user provided statements.
*/

#include <string>

#include <callgrind.h>
#include <torch/torch.h>

// Global setup. (e.g. #includes)
// GLOBAL_SETUP_TEMPLATE_LOCATION

#if defined(NVALGRIND)
static_assert(false);
#endif


int main(int argc, char* argv[]) {
    // This file should only be called inside of `Timer`, so we can adopt a
    // very simple and rigid argument parsing scheme.
    TORCH_CHECK(argc == 9);
    TORCH_CHECK(std::string(argv[1]) == "--number");
    auto number = std::stoi(argv[2]);

    TORCH_CHECK(std::string(argv[3]) == "--number_warmup");
    auto number_warmup = std::stoi(argv[4]);

    TORCH_CHECK(std::string(argv[5]) == "--repeats");
    auto repeats = std::stoi(argv[6]);

    TORCH_CHECK(std::string(argv[7]) == "--number_threads");
    auto number_threads = std::stoi(argv[8]);
    torch::set_num_threads(number_threads);

    // Setup
    // SETUP_TEMPLATE_LOCATION

    // Warmup
    for (int i = 0; i < number_warmup; i++) {
        // STMT_TEMPLATE_LOCATION
    }

    // Main loop
    for (int repeat = 0; repeat < repeats; repeat++) {
        CALLGRIND_TOGGLE_COLLECT;

        for (int i = 0; i < number; i++) {
        // STMT_TEMPLATE_LOCATION
        }

        // NB: See note in Module.cpp
        CALLGRIND_TOGGLE_COLLECT;
        CALLGRIND_DUMP_STATS;
    }
}
