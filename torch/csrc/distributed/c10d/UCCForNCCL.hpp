#pragma once

#include <string>
#include <vector>
#include <cassert>
#include <memory>

#include <ATen/DynamicLibrary.h>

namespace c10d {

inline std::shared_ptr<at::DynamicLibrary> loadTorchUCC() {
  const char *path = std::getenv("TORCH_UCC_LIBRARY_PATH");
  if (path != nullptr) {
    try {
      return std::make_shared<at::DynamicLibrary>(path);
    } catch (const c10::DynamicLibraryError &e) {
      TORCH_WARN("TORCH_UCC_LIBRARY_PATH is set, "
                 "but the loading of torch_ucc.so failed with:", e.msg());
    }
  }
  return nullptr;
}

}  // namespace c10d
