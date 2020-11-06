namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

// IO data structure for kernel code;
static auto code_template_tensor_struct = R"(
#include <nvfuser_runtime/tensor.cu>
)";

// Code support for FP16 __half type and intrinsics
#ifdef __HIP_PLATFORM_HCC__
static auto code_fp16_support = R"()";
#else
static auto code_fp16_support = R"(
#include <nvfuser_runtime/fp16_support.cu>
)";
#endif

// struct and code for functions that need random number generation
static auto code_random_number_gen = R"(
#include <nvfuser_runtime/random_numbers.cu>
)";

// Helper functions for Operations
static auto code_helper_funcs = R"(
#include <nvfuser_runtime/helpers.cu>
)";

static auto code_template_block_reduction = R"(
#include <nvfuser_runtime/block_reduction.cu>
)";

static auto code_template_grid_reduction = R"(
#include <nvfuser_runtime/grid_reduction.cu>
)";

static auto code_template_block_broadcast = R"(
#include <nvfuser_runtime/broadcast.cu>
)";

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
