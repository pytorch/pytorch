#include <kernel_db/utils.h>
#include <fstream>

namespace nvfuser {

bool append_to_text_file(const std::string& file_path, const std::string& src) {
  bool status = false;
  std::ofstream file(file_path, std::ios::app | std::ios::binary);
  if (file) {
    file.write(src.data(), src.size());
    file.close();
    status = true;
  }
  return status;
}

bool copy_from_binary_file(
    const std::string& file_path,
    std::vector<char>& dst) {
  bool status = false;
  std::ifstream file(file_path, std::ios::in | std::ios::binary);
  if (file) {
    file.seekg(0, std::ios::end);
    dst.resize(file.tellg());
    file.seekg(0, std::ios::beg);
    file.read(dst.data(), dst.size());
    file.close();
    status = true;
  }
  return status;
}

bool copy_from_text_file(const std::string& file_path, std::string& dst) {
  bool status = false;
  std::ifstream file(file_path, std::ios::in | std::ios::binary);
  if (file) {
    file.seekg(0, std::ios::end);
    dst.resize(file.tellg());
    file.seekg(0, std::ios::beg);
    // Can't use non-const data() for strings without C++17
    file.read(&dst[0], dst.size());
    file.close();
    status = true;
  }
  return status;
}

bool copy_to_binary_file(
    const std::string& file_path,
    const std::vector<char>& src) {
  bool status = false;
  std::ofstream file(file_path, std::ios::out | std::ios::binary);
  if (file) {
    file.write(src.data(), src.size());
    file.close();
    status = true;
  }
  return status;
}

bool copy_to_text_file(const std::string& file_path, const std::string& src) {
  bool status = false;
  std::ofstream file(file_path, std::ios::out | std::ios::binary);
  if (file) {
    file.write(src.data(), src.size());
    file.close();
    status = true;
  }
  return status;
}

} // namespace nvfuser
