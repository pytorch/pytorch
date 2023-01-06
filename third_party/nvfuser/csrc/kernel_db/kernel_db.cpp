#include <mutex>
#include <regex>

#include <instrumentation.h>
#include <kernel_db/kernel_db.h>
#include <kernel_db/utils.h>
#include <utils.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

static std::mutex kernel_db_lock;

KernelDb::KernelDb(bool _disabled)
    : disabled_(_disabled),
      initialized_(false),
      kernel_map_(),
      kernel_db_path_(),
      kernel_db_txt_file_() {}

KernelDb& KernelDb::get() {
  const std::string kernel_db_dir = "nvfuser_kernel_db";
  const std::string kernel_db_file = "db.csv";

  return get(
      kernel_db_dir,
      kernel_db_file,
      true,
      !isOptionEnabled(EnableOption::KernelDb),
      false);
}

KernelDb& KernelDb::get(
    const std::string& kernel_db_dir,
    const std::string& kernel_db_file,
    bool use_temp_dir,
    bool disabled,
    bool reset) {
  std::lock_guard<std::mutex> guard(kernel_db_lock);

  // The KernelDb is minimally constructed to at least hold the disable and
  // initialized booleans
  static KernelDb singleton(disabled);

  if (reset) {
    singleton.disabled_ = true;
    singleton.initialized_ = false;
    singleton.kernel_map_.clear();
    singleton.kernel_db_path_.clear();
    singleton.kernel_db_txt_file_.clear();
  }

  singleton.disabled_ = disabled;

  // Intialize the Db if it isn't already disabled
  if (!singleton.disabled_ && !singleton.initialized_) {
    // If the appropriate files are not found or unable to be created, disable
    auto success = false;
    try {
      success = singleton.open(kernel_db_dir, kernel_db_file, use_temp_dir);
    } catch (const std::exception& e) {
      TORCH_WARN(
          "nvFuser's kernel_db had an unexpected exception while opening",
          e.what());
    }
    if (!success) {
      singleton.disabled_ = true;
    } else {
      singleton.initialized_ = true;
    }
  }
  return singleton;
}

bool KernelDb::open(
    const std::string& kernel_db_dir,
    const std::string& kernel_db_file,
    bool use_temp_dir) {
  FUSER_PERF_SCOPE("KernelDb::open");
  const std::string header(
      "kernel_signature,compile_args,kernel_code_file,cubin_file");

  // The KernelDb directory is queried and created if it doesn't exist
  {
    FUSER_PERF_SCOPE("KernelDb::open::create_directory");
    if (use_temp_dir) {
      kernel_db_path_ = fs::temp_directory_path() / kernel_db_dir;
    } else {
      kernel_db_path_ = fs::path(kernel_db_dir);
    }
    if (!fs::is_directory(kernel_db_path_)) {
      try {
        fs::create_directory(kernel_db_path_);
      } catch (const std::exception& e) {
        TORCH_WARN(
            "Unable to create nvFuser Kernel DB directory! ",
            kernel_db_path_.string(),
            e.what());
        return false;
      }
    }
  }

  // The CSV file that captures the db is read if it exists
  {
    FUSER_PERF_SCOPE("KernelDb::open::read_db_txt_file");

    kernel_db_txt_file_ = kernel_db_path_ / kernel_db_file;
    if (fs::is_regular_file(kernel_db_txt_file_)) {
      std::ifstream in_file(kernel_db_txt_file_.c_str(), std::ios::in);
      if (in_file) {
        bool matched_header = false;
        bool read_db_file = true;
        std::regex db_line_regex(
            "^([\\w-]+),([\\w -\\=]+),([\\w-\\/]+\\.cu),([\\w-\\/]+\\.cubin)$");
        for (std::string line; std::getline(in_file, line);) {
          if (!matched_header) {
            if (line.compare(header) == 0) {
              matched_header = true;
            } else {
              // Header is corrupted or badly formed
              TORCH_WARN(
                  "Kernel DB: CSV file header is corrupted or badly formed - Resetting!: ",
                  line);
              read_db_file = false;
              break;
            }
          } else {
            std::smatch db_line_match;
            if (std::regex_match(line, db_line_match, db_line_regex)) {
              if (db_line_match.size() == 5) {
                KernelDbEntry temp{
                    db_line_match[1],
                    db_line_match[2],
                    db_line_match[3],
                    db_line_match[4]};

                fs::path code_path = kernel_db_path_ / temp.kernel_code_file;
                std::string code;
                if (copy_from_text_file(code_path.string(), code)) {
                  kernel_map_[code] = temp;
                } else {
                  TORCH_WARN(
                      "Kernel DB: Unable to copy cuda file: ",
                      code_path.string());
                }
              }
            } else {
              TORCH_WARN("Kernel DB: CSV line Doesn't match: ", line);
            }
          }
        }
        if (read_db_file) {
          return true;
        }
      }
    }
  }

  // If reading of the CSV file was successful, the rest of this method
  // is skipped

  // Remove all files from directory if valid db txt file was not found
  for (const auto& dir_entry : fs::directory_iterator(kernel_db_path_)) {
    const fs::path& path = dir_entry.path();
    if (fs::is_regular_file(path)) {
      if (path.extension() == ".cubin" || path.extension() == ".cu" ||
          path.extension() == ".csv") {
        fs::remove(path);
      }
    }
  }

  // Create an empty db csv file
  {
    FUSER_PERF_SCOPE("KernelDb::open::create_db_txt_file");

    if (copy_to_text_file(kernel_db_txt_file_, header + "\n")) {
      return true;
    }
  }
  return false;
}

bool KernelDb::query(
    const std::string& kernel_code,
    const std::string& compile_args,
    std::string& kernel_signature,
    std::vector<char>& cubin) const {
  FUSER_PERF_SCOPE("KernelDb::query");
  bool status = false;
  auto db_entry = kernel_map_.find(kernel_code);

  // Kernel Match is found
  if (db_entry != kernel_map_.end()) {
    // Make sure the compilation args also match
    if (db_entry->second.compile_args == compile_args) {
      // Copy the cubin to a data buffer and record the kernel name for module
      // loading
      fs::path cubin_file_path = kernel_db_path_ / db_entry->second.cubin_file;
      if (copy_from_binary_file(cubin_file_path.string(), cubin)) {
        kernel_signature = db_entry->second.kernel_signature;
        status = true;
      }
    }
  }
  return status;
}

// This method will write a cubin and the kernel code to files as well as add
// an entry to the db csv file.
bool KernelDb::write(
    const std::string& kernel_code,
    const std::string& compile_args,
    const std::string& kernel_signature,
    const std::vector<char>& cubin) {
  FUSER_PERF_SCOPE("KernelDb::write");
  std::lock_guard<std::mutex> guard(kernel_db_lock);
  bool status = false;

  // If the kernel doesn't already exist in the hash map, add it.
  if (kernel_map_.count(kernel_code) == 0) {
    // The cubin and kernel code files are given a unique number based on the
    // size of the hash map.
    std::string kernel_num =
        std::to_string(static_cast<unsigned long>(kernel_map_.size()));

    // Kernel Code File path
    std::string code_file_name("kernel_" + kernel_num + ".cu");
    fs::path code_file_path = kernel_db_path_ / code_file_name;

    // Cubin File path
    std::string cubin_file_name("kernel_" + kernel_num + ".cubin");
    fs::path cubin_file_path = kernel_db_path_ / cubin_file_name;

    // Copy kernel code to file
    status = copy_to_text_file(code_file_path.string(), kernel_code);

    // If the kernel code copy was successful, copy the cubin to file
    if (status) {
      status = copy_to_binary_file(cubin_file_path.string(), cubin);
    }

    // If both files were created successfully, add an entry to the CSV file
    if (status) {
      std::string entry(kernel_signature);
      entry += "," + compile_args + "," + code_file_name + "," +
          cubin_file_name + "\n";
      status = append_to_text_file(kernel_db_txt_file_.string(), entry);
    }

    // If writing both files and adding an entry the CSV file was successful,
    // finally add an entry to the Kernel DB
    if (status) {
      KernelDbEntry tmp{
          kernel_signature, compile_args, code_file_name, cubin_file_name};
      kernel_map_[kernel_code] = tmp;
    }
  }
  return status;
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
