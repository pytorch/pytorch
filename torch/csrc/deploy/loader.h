#pragma once
#include <c10/util/Optional.h>
#include <dlfcn.h>
#include <elf.h>
#include <memory>

namespace torch {
namespace deploy {

struct DeployLoaderError : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

struct TLSIndex {
  size_t module_id; // if module_id & TLS_LOCAL_FLAG, then module_id &
                    // ~TLS_LOCAL_FLAG is a TLSMemory*;
  size_t offset;
};

struct SymbolProvider {
  SymbolProvider() = default;
  virtual at::optional<Elf64_Addr> sym(const char* name) const = 0;
  virtual at::optional<TLSIndex> tls_sym(const char* name) const = 0;
  SymbolProvider(const SymbolProvider&) = delete;
  SymbolProvider& operator=(const SymbolProvider&) = delete;
  virtual ~SymbolProvider() = default;
};

// RAII wrapper around dlopen
struct SystemLibrary : public SymbolProvider {
  // create a wrapper around an existing handle returned from dlopen
  // if steal == true, then this will dlclose the handle when it is destroyed.
  static std::shared_ptr<SystemLibrary> create(
      void* handle = RTLD_DEFAULT,
      bool steal = false);
  static std::shared_ptr<SystemLibrary> create(const char* path, int flags);
};

struct CustomLibrary : public SymbolProvider {
  static std::shared_ptr<CustomLibrary> create(
      const char* filename,
      int argc = 0,
      const char** argv = nullptr);
  virtual void add_search_library(std::shared_ptr<SymbolProvider> lib) = 0;
  virtual void load() = 0;
};

using SystemLibraryPtr = std::shared_ptr<SystemLibrary>;
using CustomLibraryPtr = std::shared_ptr<CustomLibrary>;

} // namespace deploy
} // namespace torch
