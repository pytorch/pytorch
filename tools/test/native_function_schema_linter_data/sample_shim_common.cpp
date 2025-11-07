#include "torch/csrc/stable/c/shim.h"

namespace {

AOTITorchError _register_adapters() {
  if (auto err = register_schema_adapter(
          "aten::empty_like",
          TORCH_VERSION_2_11_0,
          adapt_empty_like_v1_to_v2_11)) {
    return err;
  }

  if (auto err = register_schema_adapter(
          "aten::transpose",
          TORCH_VERSION_2_10_0,
          adapt_transpose_v1_to_v2_10)) {
    return err;
  }

  if (auto err = register_schema_adapter(
          "aten::clone", TORCH_VERSION_2_9_0, adapt_clone_v1_to_v2_9)) {
    return err;
  }

  return AOTI_TORCH_SUCCESS;
}

} // namespace
