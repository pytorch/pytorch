#if !defined(C10_MOBILE) && !defined(ANDROID)

#include <c10/util/error.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <torch/csrc/inductor/aoti_runner/model_container_runner.h>
#include <torch/csrc/inductor/aoti_runner/model_container_runner_cpu.h>

#include <fmt/format.h>
#include <miniz.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>

#ifndef _WIN32
#include <dirent.h>
#include <sys/stat.h>
#else
#include <filesystem>
namespace fs = std::filesystem;
#endif

// TODO: C++17 has the filesystem header, which may replace these
#ifdef _WIN32
// On Windows, the POSIX implementations are considered deprecated. We simply
// map to the newer variant.
#include <direct.h>
#include <io.h>
#include <process.h>
#define access _access
#define F_OK 0
#else
#include <sys/types.h>
#include <unistd.h>
#endif

namespace {
bool file_exists(std::string& path) {
#ifdef _WIN32
  return fs::exists(path);
#else
  struct stat rc {};
  return lstat(path.c_str(), &rc) == 0;
#endif
}

std::string create_temp_dir() {
#ifdef _WIN32
  throw std::runtime_error("Not implemented");
#else
  std::string temp_dir = "/tmp/XXXXXX";
  if (mkdtemp(temp_dir.data()) == nullptr) {
    throw std::runtime_error(
        std::string("Failed to create temporary directory: ") +
        c10::utils::str_error(errno));
  }
  return temp_dir;
#endif
}

#ifdef _WIN32
const std::string k_separator = "\\";
#else
const std::string k_separator = "/";
#endif

} // namespace

namespace torch::inductor {

namespace {
const nlohmann::json& load_json_file(std::string json_path) {
  if (!file_exists(json_path)) {
    throw std::runtime_error("File found: " + json_path);
  }

  std::ifstream json_file(json_path);
  TORCH_CHECK(json_file.is_open());
  static nlohmann::json json_obj;
  json_file >> json_obj;

  return json_obj;
}

std::tuple<std::string, std::string> get_cpp_compile_command(
    const std::string& filename,
    const std::vector<std::string>& sources,
    const nlohmann::json& compile_options,
    const std::string& output_dir = "") {
  // Construct the cpp command

  std::string compiler = compile_options["compiler"].get<std::string>();
  bool compile_only = compile_options["compile_only"].get<bool>();

  std::string source_args = "";
  for (const std::string& source : sources) {
    source_args += source + " ";
  }

  std::string file_ext = compile_only ? ".o" : ".so";
  std::string target_file = output_dir + filename + file_ext;

  std::string cflags_args = "";
  for (auto& arg : compile_options["cflags"]) {
    cflags_args += "-" + arg.get<std::string>() + " ";
  }

  std::string definitions_args = "";
  for (auto& arg : compile_options["definitions"]) {
    definitions_args += "-D " + arg.get<std::string>() + " ";
  }

  std::string include_dirs_args = "";
  for (auto& arg : compile_options["include_dirs"]) {
    include_dirs_args += "-I" + arg.get<std::string>() + " ";
  }

  std::string ldflags_args = "";
  for (auto& arg : compile_options["ldflags"]) {
    ldflags_args += "-" + arg.get<std::string>() + " ";
  }

  std::string libraries_dirs_args = "";
  for (auto& arg : compile_options["libraries_dirs"]) {
    libraries_dirs_args += "-L" + arg.get<std::string>() + " ";
  }

  std::string libraries_args = "";
  for (auto& arg : compile_options["libraries"]) {
    libraries_args += "-l" + arg.get<std::string>() + " ";
  }

  std::string passthrough_parameters_args = "";
  for (auto& arg : compile_options["passthrough_args"]) {
    passthrough_parameters_args += arg.get<std::string>() + " ";
  }

  std::string compile_only_arg = compile_only ? "-c" : "";

  std::string cmd = fmt::format(
      "{} {} {} {} {} {} {} {} {} {} -o {}",
      compiler,
      source_args,
      definitions_args,
      cflags_args,
      include_dirs_args,
      passthrough_parameters_args,
      ldflags_args,
      libraries_args,
      libraries_dirs_args,
      compile_only_arg,
      target_file);

  return std::make_tuple(cmd, target_file);
}

bool recursive_mkdir(const std::string& dir) {
  // Creates directories recursively, copied from jit_utils.cpp
  // Check if current dir exists
  const char* p_dir = dir.c_str();
  const bool dir_exists = (access(p_dir, F_OK) == 0);
  if (dir_exists) {
    return true;
  }

  // Try to create current directory
#ifdef _WIN32
  int ret = _mkdir(dir.c_str());
#else
  int ret = mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);
#endif
  // Success
  if (ret == 0) {
    return true;
  }

  // Find folder separator and check if we are at the top
  auto pos = dir.find_last_of("/\\");
  if (pos == std::string::npos) {
    return false;
  }

  // Try to create parent directory
  if (!(recursive_mkdir(dir.substr(0, pos)))) {
    return false;
  }

  // Try to create complete path again
#ifdef _WIN32
  ret = _mkdir(dir.c_str());
#else
  ret = mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);
#endif
  return ret == 0;
}

bool recursive_rmdir(const std::string& path) {
#ifdef _WIN32
  std::error_code ec;
  return fs::remove_all(path, ec) != static_cast<std::uintmax_t>(-1);
#else
  DIR* dir = opendir(path.c_str());
  if (!dir) {
    return false;
  }

  struct dirent* entry = nullptr;
  struct stat statbuf {};
  bool success = true;

  // Iterate through directory entries
  while ((entry = readdir(dir)) != nullptr) {
    std::string name = entry->d_name;

    // Skip "." and ".."
    if (name == "." || name == "..") {
      continue;
    }

    std::string full_path = path;
    full_path.append("/").append(name);

    // Get file status
    if (stat(full_path.c_str(), &statbuf) != 0) {
      success = false;
      continue;
    }

    if (S_ISDIR(statbuf.st_mode)) {
      // Recursively delete subdirectory
      if (!recursive_rmdir(full_path)) {
        success = false;
      }
    } else {
      // Delete file
      if (unlink(full_path.c_str()) != 0) {
        success = false;
      }
    }
  }

  closedir(dir);

  // Remove the directory itself
  if (rmdir(path.c_str()) != 0) {
    success = false;
  }

  return success;
#endif
}

std::string compile_so(
    const std::string& cpp_filename,
    const std::string& consts_filename) {
  // Compile the cpp file into a .so

  size_t lastindex = cpp_filename.find_last_of('.');
  std::string filename = cpp_filename.substr(0, lastindex);

  std::string compile_flags_path = filename + "_compile_flags.json";
  const nlohmann::json compile_flags = load_json_file(compile_flags_path);

  auto [compile_cmd, output_o] =
      get_cpp_compile_command(filename, {cpp_filename}, compile_flags);

  std::string linker_flags_path =
      cpp_filename.substr(0, lastindex) + "_linker_flags.json";
  const nlohmann::json linker_flags = load_json_file(linker_flags_path);

  auto [link_cmd, output_so] = get_cpp_compile_command(
      filename, {output_o, consts_filename}, linker_flags);

  // Run the commands to generate a .so file
  int status = system(compile_cmd.c_str());
  if (status != 0) {
    throw std::runtime_error("Failed to compile cpp file.");
  }
  status = system(link_cmd.c_str());
  if (status != 0) {
    throw std::runtime_error("Failed to link files.");
  }

  // Move the mmapped weights onto the .so
  std::string serialized_weights_path = filename + "_serialized_weights.bin";
  if (file_exists(serialized_weights_path)) {
    std::ifstream serialized_weights_file(
        serialized_weights_path, std::ios::binary);
    if (!serialized_weights_file.is_open()) {
      throw std::runtime_error("Failed to open serialized weights file");
    }
    std::vector<char> serialized_weights(
        (std::istreambuf_iterator<char>(serialized_weights_file)),
        std::istreambuf_iterator<char>());
    serialized_weights_file.close();

    std::ofstream output_so_file(output_so, std::ios::binary | std::ios::app);
    if (!output_so_file.is_open()) {
      throw std::runtime_error("Failed to open output .so file");
    }
    // Page align the weights
    std::streampos so_size = output_so_file.tellp();
    std::vector<char> padding(16384 - so_size % 16384, ' ');
    output_so_file.write(
        padding.data(), static_cast<std::streamsize>(padding.size()));
    output_so_file.write(
        serialized_weights.data(),
        static_cast<std::streamsize>(serialized_weights.size()));
    output_so_file.close();
  }

  return output_so;
}
} // namespace

void AOTIModelPackageLoader::load_metadata(const std::string& cpp_filename) {
  // Parse metadata json file (if it exists) into the metadata_ map
  size_t lastindex = cpp_filename.find_last_of('.');
  std::string metadata_json_path =
      cpp_filename.substr(0, lastindex) + "_metadata.json";

  const nlohmann::json metadata_json_obj = load_json_file(metadata_json_path);

  for (auto& item : metadata_json_obj.items()) {
    metadata_[item.key()] = item.value().get<std::string>();
  }
}

AOTIModelPackageLoader::AOTIModelPackageLoader(
    const std::string& model_package_path)
    : AOTIModelPackageLoader(model_package_path, "model") {}

AOTIModelPackageLoader::AOTIModelPackageLoader(
    const std::string& model_package_path,
    const std::string& model_name = "model") {
  // Extract all files within the zipfile to a temporary directory
  mz_zip_archive zip_archive;
  memset(&zip_archive, 0, sizeof(zip_archive));

  if (!mz_zip_reader_init_file(&zip_archive, model_package_path.c_str(), 0)) {
    throw std::runtime_error(
        std::string("Failed to initialize zip archive: ") +
        mz_zip_get_error_string(mz_zip_get_last_error(&zip_archive)));
  }

  temp_dir_ = create_temp_dir();
  std::string so_filename = "";
  std::string cpp_filename = "";
  std::string consts_filename = "";
  std::string found_filenames = ""; // Saving for bookkeeping
  std::string model_directory =
      "data" + k_separator + "aotinductor" + k_separator + model_name;

  for (uint32_t i = 0; i < zip_archive.m_total_files; i++) {
    uint32_t filename_len =
        mz_zip_reader_get_filename(&zip_archive, i, nullptr, 0);
    if (filename_len == 0) {
      throw std::runtime_error("Failed to read filename");
    }
    char* filename = new char[filename_len + 1];
    if (!mz_zip_reader_get_filename(&zip_archive, i, filename, filename_len)) {
      throw std::runtime_error("Failed to read filename");
    }

    std::string filename_str(filename);
    found_filenames += filename_str;
    found_filenames += " ";

    // Only compile files in the specified model directory
    if (filename_str.length() >= model_directory.length() &&
        filename_str.substr(0, model_directory.length()) == model_directory) {
      std::string output_path_str = temp_dir_;
      output_path_str += k_separator;
      output_path_str += filename_str;

      // Create the parent directory if it doesn't exist
      size_t parent_path_idx = output_path_str.find_last_of("/\\");
      if (parent_path_idx == std::string::npos) {
        throw std::runtime_error(
            "Failed to find parent path in " + output_path_str);
      }
      std::string parent_path = output_path_str.substr(0, parent_path_idx);
      if (!recursive_mkdir(parent_path.c_str())) {
        throw std::runtime_error(fmt::format(
            "Failed to create directory {}: {}",
            parent_path,
            c10::utils::str_error(errno)));
      }

      // Extracts file to the temp directory
      mz_zip_reader_extract_file_to_file(
          &zip_archive, filename, output_path_str.c_str(), 0);

      // Save the file for bookkeeping
      size_t extension_idx = output_path_str.find_last_of('.');
      if (extension_idx != std::string::npos) {
        std::string filename_extension = output_path_str.substr(extension_idx);
        if (filename_extension == ".cpp") {
          cpp_filename = output_path_str;
        }
        if (filename_extension == ".o") {
          consts_filename = output_path_str;
        }
        if (filename_extension == ".so") {
          so_filename = output_path_str;
        }
      }
    }
  }

  // Close the zip archive as we have extracted all files to the temp
  // directory
  if (!mz_zip_reader_end(&zip_archive)) {
    throw std::runtime_error(
        std::string("Failed to close zip archive: {}") +
        mz_zip_get_error_string(mz_zip_get_last_error(&zip_archive)));
  }

  if (cpp_filename.empty() && so_filename.empty()) {
    throw std::runtime_error(
        "No AOTInductor generate cpp file or so file found in zip archive. Loaded the following:\n" +
        found_filenames);
  }

  // Compile the .so
  std::string so_path = !so_filename.empty()
      ? so_filename
      : compile_so(cpp_filename, consts_filename);

  // Load metadata which can be queried by user
  load_metadata(cpp_filename);

  // Construct the runner depending on the device information
  std::string device = metadata_["AOTI_DEVICE_KEY"];

  if (device.empty()) {
    throw std::runtime_error("No device information found.");
  }

  std::unordered_map<std::string, CreateAOTIModelRunnerFunc>
      registered_aoti_runner = getAOTIModelRunnerRegistry();

  if (registered_aoti_runner.find(device) == registered_aoti_runner.end()) {
    throw std::runtime_error("Unsupported device found: " + device);
  }

  std::string cubin_dir = temp_dir_ + k_separator + model_directory;
  runner_ = registered_aoti_runner[device](so_path, 1, device, cubin_dir);
}

AOTIModelPackageLoader::~AOTIModelPackageLoader() {
  // Clean up the temporary directory
  if (!temp_dir_.empty()) {
    recursive_rmdir(temp_dir_);
  }
}

AOTIModelContainerRunner* AOTIModelPackageLoader::get_runner() {
  return runner_.get();
}

std::vector<at::Tensor> AOTIModelPackageLoader::run(
    const std::vector<at::Tensor>& inputs,
    void* stream_handle) {
  return runner_->run(inputs, stream_handle);
}

std::vector<at::Tensor> AOTIModelPackageLoader::boxed_run(
    std::vector<at::Tensor>& inputs,
    void* stream_handle) {
  return runner_->boxed_run(inputs, stream_handle);
}

std::unordered_map<std::string, std::string> AOTIModelPackageLoader::
    get_metadata() {
  return metadata_;
}

std::vector<std::string> AOTIModelPackageLoader::get_call_spec() {
  return runner_->get_call_spec();
}

void AOTIModelPackageLoader::load_constants(
    std::unordered_map<std::string, at::Tensor>& constants_map,
    bool use_inactive,
    bool check_full_update) {
  std::unordered_map<std::string, std::string> constant_name_to_fqn =
      runner_->getConstantNamesToOriginalFQNs();
  std::unordered_map<std::string, at::string> fqn_to_constant_name;
  for (const auto& it : constant_name_to_fqn) {
    fqn_to_constant_name.emplace(it.second, it.first);
  }

  std::unordered_map<std::string, at::Tensor> updated_constants_map;
  for (const auto& it : constants_map) {
    if (fqn_to_constant_name.find(it.first) != fqn_to_constant_name.end()) {
      updated_constants_map.emplace(fqn_to_constant_name[it.first], it.second);
    } else {
      throw std::runtime_error("Constant not found: " + it.first);
    }
  }

  return runner_->update_constant_buffer(
      updated_constants_map, use_inactive, check_full_update);
}

std::vector<std::string> AOTIModelPackageLoader::get_constant_fqns() {
  std::unordered_map<std::string, std::string> constant_name_to_fqn =
      runner_->getConstantNamesToOriginalFQNs();
  std::vector<std::string> constant_fqns;
  constant_fqns.reserve(constant_name_to_fqn.size());
  for (const auto& it : constant_name_to_fqn) {
    constant_fqns.push_back(it.second);
  }
  return constant_fqns;
}

} // namespace torch::inductor
#endif
