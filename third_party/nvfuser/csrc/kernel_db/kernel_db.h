#pragma once

#if defined(__cplusplus) && (__cplusplus >= 201703L)
#include <filesystem>
namespace fs = std::filesystem;
#elif defined(__cplusplus) && (__cplusplus >= 201402L)
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#error "C++14 or Higher is required for filesystem library!"
#endif

#include <unordered_map>
#include <vector>

#include <c10/macros/Export.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

//! KernelDbEntry captures information to be printed per fusion in a csv file
//! that is used to restore a hash map
struct KernelDbEntry {
  //! Cuda kernel function signature that is required to load the Cubin
  std::string kernel_signature;
  //! Compilation args supplied to NVRTC -- register usage and compute
  //! capability can be specific to a kernel instance
  std::string compile_args;
  //! Full file path to Cuda Kernel
  std::string kernel_code_file;
  //! Full file path to cubin
  std::string cubin_file;
};

//! KernelDb class is a singleton structure that is used to open, query, and
//! write to the the database that is held in a hash map.  The kernel code is
//! used as string key to the hash map.
class TORCH_CUDA_CU_API KernelDb {
  KernelDb(bool _disabled);

  KernelDb(const KernelDb&) = delete;
  KernelDb& operator=(const KernelDb&) = delete;

  //! Open is private because this method should only be called once by the
  //! singleton upon creation to create a new db or restore an existing one.
  bool open(
      const std::string& kernel_db_dir,
      const std::string& kernel_db_file,
      bool use_temp_dir);

 public:
  //! Thread-Safe method to get the Meyer's singleton -- Interface
  static KernelDb& get();
  //! Thread-Safe method to get the Meyer's singleton -- For testing
  static KernelDb& get(
      const std::string& kernel_db_dir,
      const std::string& kernel_db_file,
      bool use_temp_dir = true,
      bool disabled = false,
      bool reset = false);

  //! Enable is derived from two booleans
  bool enabled() const {
    return !disabled_ && initialized_;
  }
  //! Returns the number entries in the db
  size_t size() const {
    return kernel_map_.size();
  }

  //! Query uses the string of the kernel code to lookup whether a cubin already
  //! exists for the given kernel.  Additionally, the compile args are also
  //! matched.
  bool query(
      const std::string& kernel_code,
      const std::string& compile_args,
      std::string& kernel_signature,
      std::vector<char>& cubin) const;
  //! Write is used to write a new entry to the db upon compilation of a
  //! new fusion
  bool write(
      const std::string& kernel_code,
      const std::string& compile_args,
      const std::string& kernel_signature,
      const std::vector<char>& cubin);

 private:
  //! Disablement is specified by the user and can also be set by a
  //! failure to open the db
  bool disabled_ = true;
  //! Db is only initialized after it is successfully open
  bool initialized_ = false;
  //! Hash Map of kernel_string -> db_entry
  std::unordered_map<std::string, KernelDbEntry> kernel_map_;

  //! Full path to the db directory
  fs::path kernel_db_path_;
  //! Full path to csv file used to record and restore the db
  fs::path kernel_db_txt_file_;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
