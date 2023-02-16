#pragma once
#include <string>
#include <vector>

#include <c10/macros/Export.h>

namespace nvfuser {

//! Helper methods to faciliate moving data between data buffers and files based
//! on what type of data is being copied.

TORCH_CUDA_CU_API bool append_to_text_file(
    const std::string& file_path,
    const std::string& src);

TORCH_CUDA_CU_API bool copy_from_binary_file(
    const std::string& file_path,
    std::vector<char>& dst);
TORCH_CUDA_CU_API bool copy_from_text_file(
    const std::string& file_path,
    std::string& dst);

TORCH_CUDA_CU_API bool copy_to_binary_file(
    const std::string& file_path,
    const std::vector<char>& dst);
TORCH_CUDA_CU_API bool copy_to_text_file(
    const std::string& file_path,
    const std::string& src);

} // namespace nvfuser
