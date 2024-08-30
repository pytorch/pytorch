#if !defined(C10_MOBILE) && !defined(ANDROID)

#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <torch/csrc/inductor/aoti_runner/model_container_runner.h>
#include <torch/csrc/inductor/aoti_runner/model_container_runner_cpu.h>
#ifdef USE_CUDA
#include <torch/csrc/inductor/aoti_runner/model_container_runner_cuda.h>
#endif

#include <fmt/format.h>
#include <miniz.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>

// TODO: Investigate why this is necessary, but fixes build problems in FRL
#if __has_include("filesystem")
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

#ifndef _WIN32
#include <sys/stat.h>
#endif

namespace {
bool file_exists(std::string& path) {
#ifdef _WIN32
  return fs::exists(path);
#else
  struct stat rc;
  return lstat(path.c_str(), &rc) == 0;
#endif
}
} // namespace

namespace torch::inductor {

const nlohmann::json& AOTIModelPackageLoader::load_json_file(
    std::string json_path) {
  if (!file_exists(json_path)) {
    throw std::runtime_error(fmt::format("File found: {}", json_path));
  }

  std::ifstream json_file(json_path);
  TORCH_CHECK(json_file.is_open());
  static nlohmann::json json_obj;
  json_file >> json_obj;

  return json_obj;
}

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

std::tuple<std::string, std::string> AOTIModelPackageLoader::
    get_cpp_compile_command(
        fs::path filename,
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
  fs::path target_file = output_dir / filename.replace_extension(file_ext);

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
      target_file.string());

  return std::make_tuple(cmd, target_file.string());
}

std::string AOTIModelPackageLoader::compile_so(
    const std::string& cpp_filename,
    const std::string& consts_filename) {
  // Compile the cpp file into a .so

  size_t lastindex = cpp_filename.find_last_of('.');
  std::string filename = cpp_filename.substr(0, lastindex);

  std::string compile_flags_path = filename + "_compile_flags.json";
  const nlohmann::json compile_flags = load_json_file(compile_flags_path);

  auto compile_result =
      get_cpp_compile_command(filename, {cpp_filename}, compile_flags);
  std::string compile_cmd = std::get<0>(compile_result);
  std::string output_o = std::get<1>(compile_result);

  std::string linker_flags_path =
      cpp_filename.substr(0, lastindex) + "_linker_flags.json";
  const nlohmann::json linker_flags = load_json_file(linker_flags_path);

  auto link_result = get_cpp_compile_command(
      filename, {output_o, consts_filename}, linker_flags);
  std::string link_cmd = std::get<0>(link_result);
  std::string output_so = std::get<1>(link_result);

  // Run the commands to generate a .so file
  system(compile_cmd.c_str());
  system(link_cmd.c_str());

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
    throw std::runtime_error(fmt::format(
        "Failed to initialize zip archive: {}",
        mz_zip_get_error_string(mz_zip_get_last_error(&zip_archive))));
  }

  fs::path temp_dir = fs::temp_directory_path() / std::tmpnam(nullptr);
  std::filesystem::create_directories(temp_dir);

  std::string cpp_filename = "";
  std::string consts_filename = "";
  std::string found_filenames = ""; // Saving for bookkeeping
  for (uint i = 0; i < zip_archive.m_total_files; i++) {
    uint filename_len = mz_zip_reader_get_filename(&zip_archive, i, nullptr, 0);
    if (filename_len == 0) {
      throw std::runtime_error("Failed to read filename");
    }
    char* filename = new char[filename_len + 1];
    if (!mz_zip_reader_get_filename(&zip_archive, i, filename, filename_len)) {
      throw std::runtime_error("Failed to read filename");
    }
    fs::path filepath(filename);

    if (filepath.parent_path() !=
        fmt::format("data/aotinductor/{}", model_name)) {
      continue;
    }
    found_filenames += filename;
    found_filenames += "\n";

    fs::path output_path = temp_dir / filename;
    fs::create_directories(output_path.parent_path());
    mz_zip_reader_extract_file_to_file(
        &zip_archive, filename, output_path.c_str(), 0);

    if (output_path.extension() == ".cpp") {
      cpp_filename = output_path;
    }
    if (output_path.extension() == ".o") {
      consts_filename = output_path;
    }
  }

  // Close the zip archive as we have extracted all files to the temp directory
  mz_zip_reader_end(&zip_archive);

  if (cpp_filename.empty()) {
    throw std::runtime_error(fmt::format(
        "No AOTInductor generate cpp file found in zip archive. Loaded the following:\n{}",
        found_filenames));
  }

  // Compile the .so
  std::string so_path = compile_so(cpp_filename, consts_filename);

  // Load metadata which can be queried by user
  load_metadata(cpp_filename);

  // Construct the runner depending on the device information
  std::string device = metadata_["AOTI_DEVICE_KEY"];

  if (device.empty()) {
    throw std::runtime_error("No device information found.");
#ifdef USE_CUDA
  } else if (device == "cuda") {
    runner_ = new AOTIModelContainerRunnerCuda(so_path);
#endif
  } else if (device == "cpu") {
    runner_ = new AOTIModelContainerRunnerCpu(so_path);
  } else {
    throw std::runtime_error(
        fmt::format("Unsupported device found: {}", device));
  }

  fs::remove_all(temp_dir);
}

AOTIModelContainerRunner* AOTIModelPackageLoader::get_runner() {
  return runner_;
}

std::vector<at::Tensor> AOTIModelPackageLoader::run(
    std::vector<at::Tensor>& inputs) {
  return runner_->run(inputs);
}

std::unordered_map<std::string, std::string> AOTIModelPackageLoader::
    get_metadata() {
  return metadata_;
}

std::vector<std::string> AOTIModelPackageLoader::get_call_spec() {
  return runner_->get_call_spec();
}

} // namespace torch::inductor
#endif
