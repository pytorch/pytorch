#if !defined(C10_MOBILE) && !defined(ANDROID)

#include <c10/util/error.h>
#include <c10/util/string_view.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <torch/csrc/inductor/aoti_runner/model_container_runner.h>
#include <torch/csrc/inductor/aoti_runner/model_container_runner_cpu.h>

#include <fmt/format.h>
#include <miniz.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <regex>

#ifndef _WIN32
#include <dirent.h>
#include <sys/stat.h>
#else
#include <filesystem>
namespace fs = std::filesystem;
#endif

// TODO: C++17 has the filesystem header, which may replace these
#ifdef _WIN32
#include <Windows.h>
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

const std::string k_separator = "/";

std::string remove_duplicate_separator_of_path(const std::string& path) {
  /*
  On Windows, temp file path maybe has duplicate separator.
  Need to remove the duplication:
  Origin: C:/Users/Xuhan/AppData/Local/Temp//tmpl10jfwef/filename
  Processed: C:/Users/Xuhan/AppData/Local/Temp/tmpl10jfwef/filename
  */
  std::string result = path;
  size_t pos = 0;

  while ((pos = result.find("//", pos)) != std::string::npos) {
    result.replace(pos, 2, "/");
  }

  return result;
}

std::string normalize_path_separator(const std::string& orig_path) {
  /*
  On Windows and Linux have different separator:
  On Windows use "\", and the path like: C:\Users\Test\file.txt
  On Linux use "/", and the path like: /home/user/file.txt

  In order to simplify the path operation, we can use this function to
  normalize path separator. It will convert Windows separator to Linux
  separator, and reuse the common code to handle both Windows and Linux
  path.
  On Windows, when we input: "C:\Users\Test\file.txt", the output should be:
  "C:/Users/Test/file.txt". And then, we can process the output like on Linux.
  */
  std::string normalized_path = orig_path;
#ifdef _WIN32
  std::replace(normalized_path.begin(), normalized_path.end(), '\\', '/');
#endif
  normalized_path = remove_duplicate_separator_of_path(normalized_path);
  return normalized_path;
}

bool file_exists(const std::string& path) {
#ifdef _WIN32
  return fs::exists(path);
#else
  struct stat rc{};
  return lstat(path.c_str(), &rc) == 0;
#endif
}

std::string create_temp_dir() {
#ifdef _WIN32
  try {
    fs::path temp_dir = fs::temp_directory_path();
    return temp_dir.string();
  } catch (const fs::filesystem_error& e) {
    throw std::runtime_error(
        "Failed to get temporary directory: " + std::string(e.what()));
  } catch (...) {
    throw std::runtime_error(
        "Unknown error occurred while getting temporary directory");
  }
#else
  std::string temp_dir = "/tmp/XXXXXX";
  TORCH_CHECK(
      mkdtemp(temp_dir.data()) != nullptr,
      "Failed to create temporary directory: ",
      c10::utils::str_error(errno));
  return temp_dir;
#endif
}

const char* object_file_ext() {
#ifdef _WIN32
  return ".obj";
#else
  return ".o";
#endif
}

const char* extension_file_ext() {
#ifdef _WIN32
  return ".pyd";
#else
  return ".so";
#endif
}

const char* get_output_flags(bool compile_only) {
  if (compile_only) {
#ifdef _WIN32
    return "/c /Fo"; // codespell:ignore
#else
    return "-c -o";
#endif
  }

#ifdef _WIN32
  return "/Fe";
#else
  return "-o";
#endif
}

bool _is_windows_os() {
#ifdef _WIN32
  return true;
#else
  return false;
#endif
}
} // namespace

namespace torch::inductor {

namespace {
const nlohmann::json& load_json_file(const std::string& json_path) {
  TORCH_CHECK(file_exists(json_path), "File not found: ", json_path);

  std::ifstream json_file(json_path);
  TORCH_CHECK(json_file.is_open());
  static nlohmann::json json_obj;
  json_file >> json_obj;

  return json_obj;
}

std::tuple<std::string, std::string> get_cpp_compile_command(
    const std::string& arg_filename,
    const std::vector<std::string>& sources,
    const nlohmann::json& compile_options,
    const std::string& output_dir = "") {
  // Construct the cpp command
  auto filename = normalize_path_separator(arg_filename);

  std::string compiler = compile_options["compiler"].get<std::string>();
  bool compile_only = compile_options["compile_only"].get<bool>();

  std::string source_args;
  for (const std::string& source : sources) {
    source_args += normalize_path_separator(source) + " ";
  }

  std::string file_ext =
      compile_only ? object_file_ext() : extension_file_ext();
  std::string target_file = output_dir + filename + file_ext;
  std::string target_dir = output_dir;
  if (target_dir.empty()) {
    size_t parent_path_idx = filename.find_last_of(k_separator);
    target_dir = filename.substr(0, parent_path_idx);
  }

  std::string cflags_args;
  for (auto& arg : compile_options["cflags"]) {
    // [Windows compiler need it] convert first char arg to std::string, for
    // following plus(+) strings.
    cflags_args += std::string(_is_windows_os() ? "/" : "-") +
        arg.get<std::string>() + " ";
  }

  std::string definitions_args;
  for (auto& arg : compile_options["definitions"]) {
    definitions_args += std::string(_is_windows_os() ? "/D" : "-D ") +
        arg.get<std::string>() + " ";
  }

  std::string include_dirs_args;
  for (auto& arg : compile_options["include_dirs"]) {
    include_dirs_args += std::string(_is_windows_os() ? "/I" : "-I") +
        arg.get<std::string>() + " ";
  }

  std::string ldflags_args;
  for (auto& arg : compile_options["ldflags"]) {
    ldflags_args += std::string(_is_windows_os() ? "/" : "-") +
        arg.get<std::string>() + " ";
  }

  std::string libraries_dirs_args;
  for (auto& arg : compile_options["libraries_dirs"]) {
    if (_is_windows_os()) {
      libraries_dirs_args +=
          fmt::format("/LIBPATH:\"{}\"", arg.get<std::string>()) + " ";
    } else {
      libraries_dirs_args += "-L" + arg.get<std::string>() + " ";
    }
  }

  std::string libraries_args;
  for (auto& arg : compile_options["libraries"]) {
    if (_is_windows_os()) {
      libraries_args += fmt::format("{}.lib", arg.get<std::string>()) + " ";
    } else {
      libraries_args += "-l" + arg.get<std::string>() + " ";
    }
  }

  std::string passthrough_parameters_args;
  std::regex script_regex(R"(--script=[^,]*script\.ld)");
  std::string replacement =
      "--script=" + target_dir + k_separator + "script.ld";
  for (auto& arg : compile_options["passthrough_args"]) {
    std::string arg_str =
        std::regex_replace(arg.get<std::string>(), script_regex, replacement);
    passthrough_parameters_args += arg_str + " ";
  }

  std::string output_flags = get_output_flags(compile_only);

  std::string cmd;
  /*
  Format command as python frontend cpp_builder:
  https://github.com/pytorch/pytorch/blob/3ef1bef36c73b4def0e1b71847e27fde1556c0fb/torch/_inductor/cpp_builder.py#L1780-L1790
  https://github.com/pytorch/pytorch/blob/3ef1bef36c73b4def0e1b71847e27fde1556c0fb/torch/_inductor/cpp_builder.py#L1959-L1976
  */
  if (_is_windows_os()) {
    cmd = fmt::format(
        "{} {} {} {} {} {} {}{}",
        compiler,
        include_dirs_args,
        definitions_args,
        cflags_args,
        source_args,
        passthrough_parameters_args,
        output_flags,
        target_file);
    if (compile_only == false) {
      cmd += fmt::format(
          " /LD /link {} {} {}",
          libraries_dirs_args,
          libraries_args,
          ldflags_args);
    }
    cmd = normalize_path_separator(cmd);
  } else {
    cmd = fmt::format(
        "{} {} {} {} {} {} {} {}",
        compiler,
        source_args,
        definitions_args,
        cflags_args,
        include_dirs_args,
        passthrough_parameters_args,
        output_flags,
        target_file);
    if (compile_only == false) {
      cmd += fmt::format(
          " {} {} {}", ldflags_args, libraries_args, libraries_dirs_args);
    }
  }

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
  auto pos = dir.find_last_of(k_separator);
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
  struct stat statbuf{};
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
    std::vector<std::string>& obj_filenames) {
  // Compile the cpp file into a .so

  size_t lastindex = cpp_filename.find_last_of('.');
  std::string filename = cpp_filename.substr(0, lastindex);

  std::string compile_flags_path =
      normalize_path_separator(filename + "_compile_flags.json");
  const nlohmann::json compile_flags = load_json_file(compile_flags_path);

  auto [compile_cmd, output_o] =
      get_cpp_compile_command(filename, {cpp_filename}, compile_flags);

  std::string linker_flags_path = normalize_path_separator(
      cpp_filename.substr(0, lastindex) + "_linker_flags.json");
  const nlohmann::json linker_flags = load_json_file(linker_flags_path);

  obj_filenames.push_back(output_o);
  auto [link_cmd, output_so] =
      get_cpp_compile_command(filename, obj_filenames, linker_flags);

  // Run the commands to generate a .so file
  TORCH_CHECK(system(compile_cmd.c_str()) == 0, "Failed to compile cpp file.");
  TORCH_CHECK(system(link_cmd.c_str()) == 0, "Failed to link files.");

  // Move the mmapped weights onto the .so
  std::string serialized_weights_path = filename + "_serialized_weights.bin";
  if (file_exists(serialized_weights_path)) {
    std::ifstream serialized_weights_file(
        serialized_weights_path, std::ios::binary);
    TORCH_CHECK(
        serialized_weights_file.is_open(),
        "Failed to open serialized weights file");

    std::vector<char> serialized_weights(
        (std::istreambuf_iterator<char>(serialized_weights_file)),
        std::istreambuf_iterator<char>());
    serialized_weights_file.close();

    std::ofstream output_so_file(output_so, std::ios::binary | std::ios::app);
    TORCH_CHECK(output_so_file.is_open(), "Failed to open output .so file");
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

std::unordered_set<std::string> find_model_names(
    const std::vector<std::string>& paths) {
  std::unordered_set<std::string> model_names;

  // Escape the separator if it's backslash (needed for regex)
  std::string sep = k_separator;

  std::string pattern =
      "data" + sep + "aotinductor" + sep + "([^" + sep + "]+)" + sep;
  std::regex re(pattern);

  for (const auto& path : paths) {
    std::smatch match;
    if (std::regex_search(path, match, re) && match.size() > 1) {
      model_names.insert(match[1].str());
    }
  }

  return model_names;
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

class RAIIMinizArchive {
 public:
  RAIIMinizArchive(const std::string& zip_path) {
    mz_zip_zero_struct(&_zip_archive);
    TORCH_CHECK(
        mz_zip_reader_init_file(
            &_zip_archive, normalize_path_separator(zip_path).c_str(), 0),
        "Failed to initialize zip archive: ",
        mz_zip_get_error_string(mz_zip_get_last_error(&_zip_archive)));
  }
  RAIIMinizArchive(const RAIIMinizArchive&) = delete;
  RAIIMinizArchive& operator=(const RAIIMinizArchive&) = delete;
  RAIIMinizArchive(RAIIMinizArchive&&) noexcept = delete;
  RAIIMinizArchive& operator=(RAIIMinizArchive&&) noexcept = delete;
  ~RAIIMinizArchive() {
    // Unconditionally close the file.  We can't handle any errors here without
    // terminating the program.
    mz_zip_reader_end(&_zip_archive);
  }

  std::vector<std::string> get_filenames() {
    const unsigned num_zip_files{mz_zip_reader_get_num_files(&_zip_archive)};
    std::vector<std::string> zip_filenames{};
    zip_filenames.reserve(num_zip_files);

    for (unsigned i{0}; i < num_zip_files; ++i) {
      // filename_buf_size == 0 returns the filename length, including null
      // terminator
      const auto zip_filename_len{
          mz_zip_reader_get_filename(&_zip_archive, i, nullptr, 0)};
      TORCH_CHECK(
          zip_filename_len, "Failed to read zip filename length at index ", i);

      // std::string implicitly appends a character for the null terminator
      std::string zip_filename(zip_filename_len - 1, '\0');
      TORCH_CHECK(
          mz_zip_reader_get_filename(
              &_zip_archive, i, zip_filename.data(), zip_filename_len),
          "Failed to read zip filename at index ",
          i);

      zip_filenames.emplace_back(std::move(zip_filename));
    }

    return zip_filenames;
  }

  void extract_file(
      const std::string& zip_filename,
      const std::string& dest_filename) {
    // Can't normalize_path_separator zip_filename, as it is zip index.
    std::string path_dest_filename = normalize_path_separator(dest_filename);
    if (!mz_zip_reader_extract_file_to_file(
            &_zip_archive,
            zip_filename.c_str(),
            path_dest_filename.c_str(),
            0)) {
#ifdef _WIN32
      DWORD dwErrCode = GetLastError();
      TORCH_CHECK(
          false,
          "Failed to extract zip file ",
          zip_filename,
          " to destination file ",
          path_dest_filename,
          ", error code: ",
          dwErrCode,
          " mz_zip error string: ",
          mz_zip_get_error_string(mz_zip_get_last_error(&_zip_archive)));
#else
      TORCH_CHECK(
          false,
          "Failed to extract zip file ",
          zip_filename,
          " to destination file ",
          path_dest_filename,
          ", mz_zip error string: ",
          mz_zip_get_error_string(mz_zip_get_last_error(&_zip_archive)));
#endif
    }
  }

 private:
  mz_zip_archive _zip_archive{};
};

std::unordered_map<std::string, std::string> AOTIModelPackageLoader::
    load_metadata_from_package(
        const std::string& model_package_path,
        const std::string& model_name) {
  // Open the zip archive
  RAIIMinizArchive zip_archive{model_package_path};
  auto found_filenames{zip_archive.get_filenames()};
  TORCH_CHECK(!found_filenames.empty(), "No files found in zip archive.");

  // Find the file prefix (similar to constructor logic)
  std::string file_prefix;
  if (found_filenames.size() >= 2) {
    size_t pos = found_filenames[0].find('/');
    std::string prefix0 = found_filenames[0].substr(0, pos);
    pos = found_filenames[1].find('/');
    std::string prefix1 = found_filenames[1].substr(0, pos);

    if (!prefix0.empty() && !prefix1.empty() && prefix0 == prefix1) {
      file_prefix = prefix0 + "/";
    }
  }

  // Construct the expected metadata file path within the zip
  std::string model_directory = normalize_path_separator(
      file_prefix + "data" + k_separator + "aotinductor" + k_separator +
      model_name);
  std::string metadata_suffix = "wrapper_metadata.json";

  std::string metadata_filename;

  for (auto const& zip_filename_str : found_filenames) {
    auto cur_filename = normalize_path_separator(zip_filename_str);

    if (c10::starts_with(cur_filename, model_directory) &&
        c10::ends_with(cur_filename, metadata_suffix)) {
      metadata_filename = cur_filename;
      break;
    }
  }

  if (metadata_filename.empty()) {
    std::string found_filenames_str;
    for (const std::string& filename : found_filenames) {
      found_filenames_str += filename + "\n";
    }
    std::string model_names_str;
    for (const std::string& model_name_tmp :
         find_model_names(found_filenames)) {
      model_names_str += model_name_tmp + "\n";
    }

    TORCH_CHECK(
        "Failed to find a generated cpp file or so file for model '",
        model_name,
        "' in the zip archive.\n\nAvailable models in the archive:\n",
        model_names_str,
        "\n\nTo load a specific model, please provide its name using the `model_name` parameter when calling AOTIModelPackageLoader() or torch._inductor.package.load_package.\n\n",
        "The following files were loaded from the archive:\n",
        found_filenames_str);
  }

  // Create temporary directory for extraction
  std::string temp_dir = normalize_path_separator(create_temp_dir());
  std::string output_path_str =
      normalize_path_separator(temp_dir + k_separator + metadata_filename);

  // Create the parent directory if it doesn't exist
  size_t parent_path_idx = output_path_str.find_last_of(k_separator);
  TORCH_CHECK(
      parent_path_idx != std::string::npos,
      "Failed to find parent path in " + output_path_str);
  std::string parent_path = output_path_str.substr(0, parent_path_idx);
  TORCH_CHECK(
      recursive_mkdir(parent_path),
      "Failed to create directory " + parent_path,
      ": ",
      c10::utils::str_error(errno));

  LOG(INFO) << "Extract file: " << metadata_filename << " to "
            << output_path_str;
  zip_archive.extract_file(metadata_filename, output_path_str);

  // Parse the metadata json file
  const nlohmann::json metadata_json_obj = load_json_file(output_path_str);

  std::unordered_map<std::string, std::string> metadata;
  for (auto& item : metadata_json_obj.items()) {
    metadata[item.key()] = item.value().get<std::string>();
  }
  // Clean up temporary directory
  recursive_rmdir(temp_dir);

  return metadata;
}

AOTIModelPackageLoader::AOTIModelPackageLoader(
    const std::string& model_package_path,
    const std::string& model_name,
    const bool run_single_threaded,
    const size_t num_runners,
    const c10::DeviceIndex device_index) {
  if (run_single_threaded) {
    TORCH_CHECK(
        num_runners == 1,
        "num_runners must be 1 when run_single_threaded is true");
  } else {
    TORCH_CHECK(
        num_runners >= 1,
        "num_runners must be >=1 when run_single_threaded is false");
  }

  // Extract all files within the zipfile to a temporary directory
  RAIIMinizArchive zip_archive{model_package_path};
  auto found_filenames{zip_archive.get_filenames()};
  TORCH_CHECK(!found_filenames.empty(), "No files found in zip archive.");

  // All the paths are prepended with a tmp/ directory. We need to find the
  // prefix.
  std::string file_prefix;
  size_t pos = found_filenames[0].find('/');
  std::string prefix0 = found_filenames[0].substr(0, pos);
  pos = found_filenames[1].find('/');
  std::string prefix1 = found_filenames[1].substr(0, pos);

  if (!prefix0.empty() && !prefix1.empty() && prefix0 == prefix1) {
    file_prefix = prefix0 + "/";
  } else {
    LOG(WARNING)
        << "You are using an outdated version of the pt2 archive which do not have a prefix in front of each filename. Example: \n"
        << found_filenames[0] << "\n"
        << found_filenames[1];
  }

  temp_dir_ = normalize_path_separator(create_temp_dir());

  std::string so_filename;
  std::string cpp_filename;
  std::string weight_blob_filename;
  std::vector<std::string> obj_filenames;
  std::string model_directory = normalize_path_separator(
      file_prefix + "data" + k_separator + "aotinductor" + k_separator +
      model_name);
  std::string const_directory = normalize_path_separator(
      file_prefix + "data" + k_separator + "constants");

  // zip_filename_str can't be normalize_path_separator, because it should be
  // as index for mz_zip_reader_extract_file_to_file.
  for (auto const& zip_filename_str : found_filenames) {
    auto cur_filename = normalize_path_separator(zip_filename_str);
    // Only compile files in the specified model directory
    if (c10::starts_with(cur_filename, model_directory) ||
        c10::starts_with(cur_filename, const_directory)) {
      std::string output_path_str = temp_dir_;

      if (c10::starts_with(cur_filename, model_directory)) {
        output_path_str += k_separator;
        output_path_str += cur_filename;
      } else { // startsWith(zip_filename_str, const_directory)
        // Extract constants to the same directory as the rest of the files
        // to be consistent with internal implementation
        size_t lastSlash = cur_filename.find_last_of(k_separator);
        std::string filename = cur_filename;
        if (lastSlash != std::string::npos) {
          filename = cur_filename.substr(lastSlash + 1);
        }
        output_path_str.append(k_separator)
            .append(model_directory)
            .append(k_separator)
            .append(filename);
      }

      std::string output_file_path = normalize_path_separator(output_path_str);
      LOG(INFO) << "Extract file: " << zip_filename_str << " to "
                << output_file_path;

      // Create the parent directory if it doesn't exist
      size_t parent_path_idx = output_file_path.find_last_of(k_separator);
      TORCH_CHECK(
          parent_path_idx != std::string::npos,
          "Failed to find parent path in " + output_file_path);

      std::string parent_path = output_file_path.substr(0, parent_path_idx);
      TORCH_CHECK(
          recursive_mkdir(parent_path),
          "Failed to create directory " + parent_path,
          ": ",
          c10::utils::str_error(errno));

      // Extracts file to the temp directory
      zip_archive.extract_file(zip_filename_str, output_path_str);

      // Save the file for bookkeeping
      size_t extension_idx = output_file_path.find_last_of('.');
      if (extension_idx != std::string::npos) {
        std::string filename_extension = output_file_path.substr(extension_idx);
        if (filename_extension == ".cpp") {
          cpp_filename = output_file_path;
        } else if (filename_extension == object_file_ext()) {
          obj_filenames.push_back(output_file_path);
        } else if (filename_extension == extension_file_ext()) {
          so_filename = output_file_path;
        } else if (filename_extension == ".blob") {
          weight_blob_filename = output_file_path;
        }
      }
    }
  }

  if (cpp_filename.empty() && so_filename.empty()) {
    std::string found_filenames_str;
    for (const std::string& filename : found_filenames) {
      found_filenames_str += filename + "\n";
    }
    std::string model_names_str;
    for (const std::string& model_name_tmp :
         find_model_names(found_filenames)) {
      model_names_str += model_name_tmp + "\n";
    }

    TORCH_CHECK(
        false,
        "Failed to find a generated cpp file or so file for model '",
        model_name,
        "' in the zip archive.\n\nAvailable models in the archive:\n",
        model_names_str,
        "\n\nTo load a specific model, please provide its name using the `model_name` parameter when calling AOTIModelPackageLoader() or torch._inductor.package.load_package.\n\n",
        "The following files were loaded from the archive:\n",
        found_filenames_str);
  }

  // Compile the .so
  std::string so_path = !so_filename.empty()
      ? so_filename
      : compile_so(cpp_filename, obj_filenames);

  // Load metadata which can be queried by user
  load_metadata(cpp_filename);

  // Construct the runner depending on the device information
  std::string device_key = metadata_["AOTI_DEVICE_KEY"];
  TORCH_CHECK(!device_key.empty(), "No device information found.");

  std::unordered_map<std::string, CreateAOTIModelRunnerFunc>
      registered_aoti_runner = getAOTIModelRunnerRegistry();

  TORCH_CHECK(
      registered_aoti_runner.find(device_key) != registered_aoti_runner.end(),
      "Unsupported device key found: ",
      device_key);

  c10::Device device = c10::Device(device_key);
  device.set_index(device_index);

  std::string cubin_dir = temp_dir_ + k_separator + model_directory;
  runner_ = registered_aoti_runner[device_key](
      so_path, num_runners, device.str(), cubin_dir, run_single_threaded);

  if (!weight_blob_filename.empty()) {
    runner_->update_constant_buffer_from_blob(weight_blob_filename);
  }
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
    std::vector<at::Tensor>&& inputs,
    void* stream_handle) {
  return runner_->boxed_run(std::move(inputs), stream_handle);
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
    bool check_full_update,
    bool user_managed) {
  std::unordered_map<std::string, std::string> constant_name_to_fqn =
      runner_->getConstantNamesToOriginalFQNs();
  std::unordered_map<std::string, std::string> fqn_to_constant_name;
  for (const auto& it : constant_name_to_fqn) {
    fqn_to_constant_name.emplace(it.second, it.first);
  }

  std::unordered_map<std::string, at::Tensor> updated_constants_map;
  for (const auto& it : constants_map) {
    if (fqn_to_constant_name.find(it.first) != fqn_to_constant_name.end()) {
      updated_constants_map.emplace(fqn_to_constant_name[it.first], it.second);
    } else {
      TORCH_CHECK(false, "Constant not found: ", it.first);
    }
  }

  return runner_->update_constant_buffer(
      updated_constants_map, use_inactive, check_full_update, user_managed);
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

void AOTIModelPackageLoader::update_constant_buffer(
    std::unordered_map<std::string, at::Tensor>& tensor_map,
    bool use_inactive,
    bool validate_full_updates,
    bool user_managed) {
  runner_->update_constant_buffer(
      tensor_map, use_inactive, validate_full_updates, user_managed);
}
} // namespace torch::inductor
#endif
