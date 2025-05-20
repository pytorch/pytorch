#include <ATen/native/vulkan/api/DynamicShaderInfo.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>
#include <fmt/format.h>

#include <unistd.h>
#include <cstdio>
#include <filesystem>
#include <fstream>

namespace at::native::vulkan::api {

namespace {
int count_layouts(std::string_view src) {
  int count = 0;
  if (src.empty()) {
    return count;
  }
  auto pos = src.find("layout(set");
  while (pos != std::string::npos) {
    if (pos == 0 || src[pos - 1] == '\n') {
      count++;
    }
    pos = src.find("layout(set", pos + 1);
  }
  return count;
}
} // namespace

DynamicShaderInfo compile_glsl(std::string name, std::string_view src) {
  // Prototype-quality compilation: shell out to glslc, which we just
  // assume is available and on PATH. This doesn't support Win32, but
  // we should be replacing this with libshaderc for a real
  // implementation anyway.
  char tmp_shader_filename[] = "/tmp/shadersrc.XXXXXX";
  int shader_fd = mkstemp(tmp_shader_filename);
  TORCH_CHECK(
      shader_fd != -1, "could not create temporary file", std::strerror(errno));
  auto bytes_written = write(shader_fd, src.data(), src.size());
  TORCH_CHECK(
      bytes_written == static_cast<decltype(bytes_written)>(src.size()),
      "could not write to temporary file");

  char tmp_spv_filename[] = "/tmp/output_spv.XXXXXX";
  int spv_fd = mkstemp(tmp_spv_filename);
  TORCH_CHECK(
      spv_fd != -1, "could not create temporary file", std::strerror(errno));
  auto glslc_cmd = fmt::format(
      "glslc -fshader-stage=compute {} -o {} --target-env=vulkan1.0 -Werror",
      tmp_shader_filename,
      tmp_spv_filename);
  int result = std::system(glslc_cmd.c_str());
  TORCH_CHECK(result == 0, "error code ", result, " shelling out to glslc");

  std::ifstream in_spv(tmp_spv_filename, std::ios::binary);
  const auto fsize = std::filesystem::file_size(tmp_spv_filename);
  const auto spv_num_blocks = fsize / 4 + (fsize % 4 != 0);
  auto spv_data = std::make_unique<std::uint32_t[]>(spv_num_blocks);
  in_spv.read(reinterpret_cast<char*>(spv_data.get()), fsize);
  auto read_chars = in_spv.gcount();
  TORCH_CHECK(
      read_chars == static_cast<decltype(read_chars)>(fsize),
      "read ",
      read_chars,
      " but file size was ",
      fsize);

  std::ofstream out_dbg("/tmp/dbg.spv");
  out_dbg.write(reinterpret_cast<char*>(spv_data.get()), fsize);

  std::vector<VkDescriptorType> layout;
  auto num_layouts = count_layouts(src);
  layout.reserve(num_layouts);
  // See calling convention in torch/csrc/vulkan/Module.cpp.
  layout.push_back(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE); // output image;
  for (const auto ii : c10::irange(1, num_layouts - 1)) {
    // input samplers
    layout.push_back(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
  }
  layout.push_back(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
  return DynamicShaderInfo(
      std::move(name),
      std::move(spv_data),
      spv_num_blocks * 4,
      std::move(layout));
}

} // namespace at::native::vulkan::api
