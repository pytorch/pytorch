// Code in this file is a heavily modified version of the dynamic loader
// from android's bionic library. Here is the license for that project:

/*
 * Copyright (C) 2016 The Android Open Source Project
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the
 *    distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 * OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 * AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
 * OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

#include <dlfcn.h>
#include <elf.h>
#include <fcntl.h>
#include <libgen.h>
#include <link.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <atomic>
#include <cerrno>
#include <cinttypes>
#include <climits>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>
// Get PAGE_SIZE and PAGE_MASK.
#include <sys/user.h>

#include <torch/csrc/deploy/interpreter/Optional.hpp>
#include <torch/csrc/deploy/irange.h>

#include <fmt/format.h>
#include <torch/csrc/deploy/loader.h>
#include <torch/csrc/deploy/mem_file.h>

namespace torch {
namespace deploy {

#define DEPLOY_ERROR(msg_fmt, ...) \
  throw DeployLoaderError(fmt::format(msg_fmt, ##__VA_ARGS__))

#define DEPLOY_CHECK(cond, fmt, ...)  \
  if (!(cond)) {                      \
    DEPLOY_ERROR(fmt, ##__VA_ARGS__); \
  }

std::vector<std::string> split_path(const std::string& s, char delim) {
  const char* cur = s.c_str();
  const char* end = cur + s.size();
  if (cur == end) {
    return {};
  }
  std::vector<std::string> result;
  while (true) {
    // non-zero amount of chars
    const char* next = strchr(cur, delim);
    if (!next) {
      result.emplace_back(std::string(cur, end));
      break;
    }
    result.emplace_back(std::string(cur, next));
    cur = next + 1;
  }
  return result;
}

// https://stackoverflow.com/questions/23006930/the-shared-library-rpath-and-the-binary-rpath-priority/52647116#52647116
void replace_all(
    std::string& str,
    const std::string& from,
    const std::string& to) {
  if (from.empty())
    return;
  size_t start_pos = 0;
  while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
    str.replace(start_pos, from.length(), to);
    start_pos += to.length(); // In case 'to' contains 'from', like replacing
                              // 'x' with 'yx'
  }
}

std::string resolve_path(const std::string& origin, const std::string& t) {
  std::string result = t;
  replace_all(result, "$ORIGIN", origin);
  // NOLINTNEXTLINE
  char buf[PATH_MAX];
  char* resolved = realpath(result.c_str(), buf);
  if (!resolved) {
    return result;
  }
  return resolved;
}

std::string resolve_origin(const std::string& so_name) {
  // NOLINTNEXTLINE
  char origin[PATH_MAX];
  realpath(so_name.c_str(), origin);
  dirname(origin);
  return origin;
}

template <typename... Args>
std::string stringf(const char* format, Args... args) {
  int size_s = snprintf(nullptr, 0, format, args...);
  std::string result(size_s + 1, 0);
  snprintf((char*)result.data(), size_s + 1, format, args...);
  return result;
}
// Returns the address of the page containing address 'x'.
#define PAGE_START(x) ((x)&PAGE_MASK)

// Returns the offset of address 'x' in its page.
#define PAGE_OFFSET(x) ((x) & ~PAGE_MASK)

// Returns the address of the next page after address 'x', unless 'x' is
// itself at the start of a page.
#define PAGE_END(x) PAGE_START((x) + (PAGE_SIZE - 1))

// from bionic
// returns the size a shared library will take in memory
size_t phdr_table_get_load_size(
    const Elf64_Phdr* phdr_table,
    size_t phdr_count,
    Elf64_Addr* out_min_vaddr,
    Elf64_Addr* out_max_vaddr) {
  Elf64_Addr min_vaddr = UINTPTR_MAX;
  Elf64_Addr max_vaddr = 0;

  bool found_pt_load = false;
  for (const auto i : multipy::irange(phdr_count)) {
    const Elf64_Phdr* phdr = &phdr_table[i];

    if (phdr->p_type != PT_LOAD) {
      continue;
    }
    found_pt_load = true;

    if (phdr->p_vaddr < min_vaddr) {
      min_vaddr = phdr->p_vaddr;
    }

    if (phdr->p_vaddr + phdr->p_memsz > max_vaddr) {
      max_vaddr = phdr->p_vaddr + phdr->p_memsz;
    }
  }
  if (!found_pt_load) {
    min_vaddr = 0;
  }

  min_vaddr = PAGE_START(min_vaddr);
  max_vaddr = PAGE_END(max_vaddr);

  if (out_min_vaddr != nullptr) {
    *out_min_vaddr = min_vaddr;
  }
  if (out_max_vaddr != nullptr) {
    *out_max_vaddr = max_vaddr;
  }
  return max_vaddr - min_vaddr;
}

#define MAYBE_MAP_FLAG(x, from, to) (((x) & (from)) ? (to) : 0)
#define PFLAGS_TO_PROT(x)                 \
  (MAYBE_MAP_FLAG((x), PF_X, PROT_EXEC) | \
   MAYBE_MAP_FLAG((x), PF_R, PROT_READ) | \
   MAYBE_MAP_FLAG((x), PF_W, PROT_WRITE))

// holds a pre-computed hash for a string that is used in a GNU-style hash
// tables and also keeps track of the string length.
struct GnuHash {
  GnuHash(const char* name) {
    uint32_t h = 5381;
    const uint8_t* name_bytes = reinterpret_cast<const uint8_t*>(name);
#pragma unroll 8
    while (*name_bytes != 0) {
      h += (h << 5) +
          *name_bytes++; // h*33 + c = h + h * 32 + c = h + h << 5 + c
    }
    hash = h;
    name_len = reinterpret_cast<const char*>(name_bytes) - name;
  }
  uint32_t hash;
  uint32_t name_len;
};

// this is a special builtin in the libc++ API used for telling C++ execption
// frame unwinding about functions loaded from a pathway other than the libc
// loader. it is passed a pointer to where the EH_FRAME section was loaded,
// which appears to include frame information relative to that address.
extern "C" void __register_frame(void*);
extern "C" void __deregister_frame(void*);

typedef void (*linker_dtor_function_t)();
typedef void (*linker_ctor_function_t)(int, const char**, char**);

// https://refspecs.linuxfoundation.org/LSB_2.1.0/LSB-Core-generic/LSB-Core-generic/ehframehdr.html
// note that eh_frame_ptr can be different types based on eh_frame_ptr_enc but
// we only support one sepecific encoding that is stored in a int32_t and an
// offset relative to the start of this struct.
struct EH_Frame_HDR {
  char version;
  char eh_frame_ptr_enc;
  char fde_count_enc;
  char table_enc;
  int32_t eh_frame_ptr;
};

// this is the libc++ function called to lookup thread local state.
// It is passed a pointer to an object of the same shape as TLSEntry
// with the module_id and offset.
extern "C" void* __tls_get_addr(void*);

extern "C" int __cxa_thread_atexit_impl(
    void (*dtor)(void*),
    void* obj,
    void* dso_symbol);

struct CustomLibraryImpl;

struct TLSMemory {
  TLSMemory(std::shared_ptr<CustomLibraryImpl> file, size_t size)
      // NOLINTNEXTLINE
      : file_(std::move(file)), mem_(malloc(size)) {}
  std::shared_ptr<CustomLibraryImpl> file_;
  void* mem_;
  ~TLSMemory() {
    // NOLINTNEXTLINE
    free(mem_);
  }
};

static void delete_TLSMemory(void* obj) {
  delete ((TLSMemory*)obj);
}

// This object performs TLS emulation for modules not loaded by dlopen.
// Normally modules have a module_id that is used as a key in libc for the
// thread local data for that module. However, there is no public API for
// assigning this module id. Instead, for modules that we load, we set module_id
// to a pointer to a TLSSegment object, and replace __tls_get_addr with a
// function that calls `addr`.

// libc module_id's are sequential, so we use the top bit as a flag to see
// if we have a local TLSegment object instead. This will break if
// someone creates 2^63 sequential objects, but it is hard to imagine
// a system with enough RAM to do that.
constexpr size_t TLS_LOCAL_FLAG = (1ULL << 63);

static void* local__tls_get_addr(TLSIndex* idx);

/* LLDB puts a breakpoint in this function, and reads __deploy_module_info to
 * get debug info from library.  */
__attribute__((noinline)) void __deploy_register_code() {
  std::cout << ""; // otherwise the breakpoint doesn't get hit, not sure if
                   // there is a more stable way of doing this.
};

struct DeployModuleInfo {
  const char* name;
  Elf64_Addr file_addr;
  size_t file_size;
  Elf64_Addr load_bias;
};

extern "C" {
// NOLINTNEXTLINE
DeployModuleInfo __deploy_module_info;
}

// RAII wrapper around dlopen
struct __attribute__((visibility("hidden"))) SystemLibraryImpl
    : public SystemLibrary {
  SystemLibraryImpl(void* handle, bool steal)
      : handle_(handle), own_handle_(steal && handle != RTLD_DEFAULT) {}

  multipy::optional<Elf64_Addr> sym(const char* name) const override {
    void* r = dlsym(handle_, name);
    if (!r) {
      return multipy::nullopt;
    }
    return (Elf64_Addr)r;
  }

  multipy::optional<TLSIndex> tls_sym(const char* name) const override;

  ~SystemLibraryImpl() override {
    if (own_handle_) {
      dlclose(handle_);
    }
  }

 private:
  void* handle_;
  bool own_handle_;
};

std::shared_ptr<SystemLibrary> SystemLibrary::create(void* handle, bool steal) {
  return std::make_shared<SystemLibraryImpl>(handle, steal);
}
std::shared_ptr<SystemLibrary> SystemLibrary::create(
    const char* path,
    int flags) {
  void* handle = dlopen(path, flags);
  return SystemLibrary::create(handle, handle != nullptr);
}

// reads DT_NEEDED and DT_RUNPATH from an unloaded elf file so we can sort out
// dependencies before calling dlopen
std::pair<const char*, std::vector<const char*>> load_needed_from_elf_file(
    const char* filename,
    const char* data) {
  auto header_ = (Elf64_Ehdr*)data;
  auto program_headers = (Elf64_Phdr*)(data + header_->e_phoff);
  auto n_program_headers = header_->e_phnum;
  const Elf64_Dyn* dynamic = nullptr;
  for (const auto i : multipy::irange(n_program_headers)) {
    const Elf64_Phdr* phdr = &program_headers[i];
    if (phdr->p_type == PT_DYNAMIC) {
      dynamic = reinterpret_cast<const Elf64_Dyn*>(data + phdr->p_offset);
      break;
    }
  }
  DEPLOY_CHECK(
      dynamic,
      "{}: could not load dynamic section for looking up DT_NEEDED",
      filename);

  const char* runpath = "";
  std::vector<const char*> needed;

  auto segment_headers = (Elf64_Shdr*)(data + header_->e_shoff);
  size_t n_segments = header_->e_shnum;
  const char* strtab = nullptr;

  const char* segment_string_table =
      data + segment_headers[header_->e_shstrndx].sh_offset;

  for (const auto i : multipy::irange(n_segments)) {
    const Elf64_Shdr* shdr = &segment_headers[i];
    if (shdr->sh_type == SHT_STRTAB &&
        strcmp(".dynstr", segment_string_table + shdr->sh_name) == 0) {
      strtab = data + shdr->sh_offset;
      break;
    }
  }

  DEPLOY_CHECK(strtab, "{}: could not load dynstr for DT_NEEDED", filename);

  for (const Elf64_Dyn* d = dynamic; d->d_tag != DT_NULL; ++d) {
    switch (d->d_tag) {
      case DT_NEEDED:
        // std::cout << "NEEDED: '" << strtab + d->d_un.d_val << "'\n";
        needed.push_back(strtab + d->d_un.d_val);
        break;
      case DT_RPATH: /* not quite correct, because this is a different order
                        than runpath,
                        but better than not processing it at all */
      case DT_RUNPATH:
        // std::cout << "RUNPATH: '" << strtab + d->d_un.d_val << "'\n";
        runpath = strtab + d->d_un.d_val;
        break;
    }
  }
  return std::make_pair(runpath, std::move(needed));
}

// common mechanism for reading the elf symbol table,
// and other information in the PT_DYNAMIC segment.
struct ElfDynamicInfo {
  std::string name_;
  const Elf64_Dyn* dynamic_ = nullptr;
  Elf64_Addr load_bias_ = 0;
  const Elf64_Sym* symtab_ = nullptr;
  const char* strtab_ = nullptr;
  size_t strtab_size_ = 0;
  Elf64_Rela* plt_rela_ = nullptr;
  size_t n_plt_rela_ = 0;
  Elf64_Rela* rela_ = nullptr;
  size_t n_rela_ = 0;
  linker_ctor_function_t init_func_ = nullptr;
  linker_ctor_function_t* init_array_ = nullptr;
  linker_dtor_function_t fini_func_ = nullptr;
  linker_dtor_function_t* fini_array_ = nullptr;
  size_t n_init_array_ = 0;
  size_t n_fini_array_ = 0;
  size_t gnu_nbucket_ = 0;
  uint32_t* gnu_bucket_ = nullptr;
  uint32_t* gnu_chain_ = nullptr;
  uint32_t gnu_maskwords_ = 0;
  uint32_t gnu_shift2_ = 0;
  Elf64_Addr* gnu_bloom_filter_ = nullptr;
  std::string runpath_;
  std::vector<const char*> needed_;

  const char* get_string(int idx) {
    return strtab_ + idx;
  }

  void initialize_from_dynamic_section(
      std::string name,
      Elf64_Dyn* dynamic,
      Elf64_Addr load_bias,
      bool check_absolute) {
    name_ = std::move(name);
    load_bias_ = load_bias;
    dynamic_ = dynamic;
    for (const Elf64_Dyn* d = dynamic_; d->d_tag != DT_NULL; ++d) {
      void* addr = (check_absolute && d->d_un.d_ptr > load_bias_)
          ? reinterpret_cast<void*>(d->d_un.d_ptr)
          : reinterpret_cast<void*>(load_bias_ + d->d_un.d_ptr);
      auto value = d->d_un.d_val;

      switch (d->d_tag) {
        case DT_SYMTAB:
          symtab_ = (Elf64_Sym*)addr;
          break;
        case DT_STRTAB:
          strtab_ = (const char*)addr;
          break;

        case DT_STRSZ:
          strtab_size_ = value;
          break;

        case DT_JMPREL:
          plt_rela_ = (Elf64_Rela*)addr;
          break;
        case DT_PLTRELSZ:
          n_plt_rela_ = value / sizeof(Elf64_Rela);
          break;
        case DT_RELA:
          rela_ = (Elf64_Rela*)addr;
          break;
        case DT_RELASZ:
          n_rela_ = value / sizeof(Elf64_Rela);
          break;

        case DT_INIT:
          init_func_ = reinterpret_cast<linker_ctor_function_t>(
              load_bias_ + d->d_un.d_ptr);
          break;

        case DT_FINI:
          fini_func_ = reinterpret_cast<linker_dtor_function_t>(
              load_bias_ + d->d_un.d_ptr);
          break;

        case DT_INIT_ARRAY:
          init_array_ = reinterpret_cast<linker_ctor_function_t*>(
              load_bias_ + d->d_un.d_ptr);
          break;

        case DT_INIT_ARRAYSZ:
          n_init_array_ =
              static_cast<uint32_t>(d->d_un.d_val) / sizeof(Elf64_Addr);
          break;

        case DT_FINI_ARRAY:
          fini_array_ = reinterpret_cast<linker_dtor_function_t*>(
              load_bias_ + d->d_un.d_ptr);
          break;

        case DT_FINI_ARRAYSZ:
          n_fini_array_ =
              static_cast<uint32_t>(d->d_un.d_val) / sizeof(Elf64_Addr);
          break;

        case DT_HASH:
          break;

        case DT_GNU_HASH: {
          gnu_nbucket_ = reinterpret_cast<uint32_t*>(addr)[0];
          // skip symndx
          gnu_maskwords_ = reinterpret_cast<uint32_t*>(addr)[2];
          gnu_shift2_ = reinterpret_cast<uint32_t*>(addr)[3];
          gnu_bloom_filter_ =
              reinterpret_cast<Elf64_Addr*>((Elf64_Addr)addr + 16);
          gnu_bucket_ =
              reinterpret_cast<uint32_t*>(gnu_bloom_filter_ + gnu_maskwords_);
          // amend chain for symndx = header[1]
          gnu_chain_ =
              gnu_bucket_ + gnu_nbucket_ - reinterpret_cast<uint32_t*>(addr)[1];
          --gnu_maskwords_;
        } break;
      }
    }

    if (!gnu_bucket_) {
      std::cout << fmt::format(
          "{}: warning, no DT_GNU_HASH found, symbol lookups on this module will not find anything.\n",
          name_);
    }

    // pass 2 for things that require the strtab_ to be loaded
    for (const Elf64_Dyn* d = dynamic_; d->d_tag != DT_NULL; ++d) {
      switch (d->d_tag) {
        case DT_NEEDED:
          needed_.push_back(get_string(d->d_un.d_val));
          break;
        case DT_RPATH: /* not quite correct, because this is a different order
                          than runpath,
                          but better than not processing it at all */
        case DT_RUNPATH:
          runpath_ = get_string(d->d_un.d_val);
          break;
      }
    }
  }

  multipy::optional<Elf64_Addr> sym(
      const char* name,
      GnuHash* precomputed_hash = nullptr) const {
    if (!gnu_bucket_) {
      return multipy::nullopt; // no hashtable was loaded
    }
    GnuHash hash_obj = precomputed_hash ? *precomputed_hash : GnuHash(name);
    auto hash = hash_obj.hash;
    auto name_len = hash_obj.name_len;
    constexpr uint32_t kBloomMaskBits = sizeof(Elf64_Addr) * 8;

    const uint32_t word_num = (hash / kBloomMaskBits) & gnu_maskwords_;
    const Elf64_Addr bloom_word = gnu_bloom_filter_[word_num];
    const uint32_t h1 = hash % kBloomMaskBits;
    const uint32_t h2 = (hash >> gnu_shift2_) % kBloomMaskBits;

    if ((1 & (bloom_word >> h1) & (bloom_word >> h2)) != 1) {
      return multipy::nullopt;
    }

    uint32_t sym_idx = gnu_bucket_[hash % gnu_nbucket_];
    if (sym_idx == 0) {
      return multipy::nullopt;
    }

    uint32_t chain_value = 0;
    const Elf64_Sym* sym = nullptr;

    do {
      sym = symtab_ + sym_idx;
      chain_value = gnu_chain_[sym_idx];
      if ((chain_value >> 1) == (hash >> 1)) {
        if (static_cast<size_t>(sym->st_name) + name_len + 1 <= strtab_size_ &&
            memcmp(strtab_ + sym->st_name, name, name_len + 1) == 0) {
          // found the matching entry, is it defined?
          if (sym->st_shndx != 0) {
            return sym->st_value +
                ((ELF64_ST_TYPE(sym->st_info) == STT_TLS) ? 0 : load_bias_);
          }
          // symbol isn't defined
          return multipy::nullopt;
        }
      }
      ++sym_idx;
    } while ((chain_value & 1) == 0);
    return multipy::nullopt;
  }
};

// for resolving TLS offsets we need to look through
// libc's already loaded libraries. We do not have the whole
// ELF file mapped in this case just a pointer to the program headers and
// the load_bias (offset in memory) where the library was loaded.
struct AlreadyLoadedSymTable {
 private:
  ElfDynamicInfo dyninfo_;

 public:
  AlreadyLoadedSymTable(
      const char* name,
      Elf64_Addr load_bias,
      const Elf64_Phdr* program_headers,
      size_t n_program_headers) {
    Elf64_Dyn* dynamic = nullptr;
    for (const auto i : multipy::irange(n_program_headers)) {
      const Elf64_Phdr* phdr = &program_headers[i];

      // Segment addresses in memory.
      Elf64_Addr seg_start = phdr->p_vaddr + load_bias;
      if (phdr->p_type == PT_DYNAMIC) {
        dynamic = reinterpret_cast<Elf64_Dyn*>(seg_start);
        break;
      }
    }
    DEPLOY_CHECK(
        dynamic, "%s: couldn't find PT_DYNAMIC in already loaded table.", name);
    dyninfo_.initialize_from_dynamic_section(name, dynamic, load_bias, true);
  }

  multipy::optional<Elf64_Addr> sym(const char* name) {
    return dyninfo_.sym(name);
  }
};
static int iterate_cb(struct dl_phdr_info* info, size_t size, void* data) {
  auto fn = (std::function<int(struct dl_phdr_info * info, size_t size)>*)data;
  return (*fn)(info, size);
}

// we need to find a TLS offset / module_id pair for a symbol which we cannot do
// with a normal dlsym call. Instead we iterate through all loaded libraries and
// check their symbol tables for the symbol. The value of the symbol is the TLS
// offset. When we find the library we also get the module id.
multipy::optional<TLSIndex> slow_find_tls_symbol_offset(const char* sym_name) {
  multipy::optional<TLSIndex> result = multipy::nullopt;
  std::function<int(struct dl_phdr_info*, size_t)> cb =
      [&](struct dl_phdr_info* info, size_t size) {
        // std::cout << "SEARCHING .. " << info->dlpi_name << "\n";
        AlreadyLoadedSymTable symtable(
            info->dlpi_name,
            info->dlpi_addr,
            info->dlpi_phdr,
            info->dlpi_phnum);
        auto sym_addr = symtable.sym(sym_name);
        if (sym_addr) {
          // std::cout << "FOUND IT IN: " << info->dlpi_name << " it has modid:
          // " << info->dlpi_tls_modid << "\n";
          result = TLSIndex{info->dlpi_tls_modid, *sym_addr};
          return 1;
        }
        return 0;
      };

  dl_iterate_phdr(iterate_cb, (void*)&cb);
  return result;
}

multipy::optional<TLSIndex> SystemLibraryImpl::tls_sym(const char* name) const {
  if (!sym(name)) {
    return multipy::nullopt; // before we do a bunch of slow lookups to find the
                             // module_id, check that this even defines the
                             // symbol
  }
  if (handle_ == RTLD_DEFAULT) {
    return slow_find_tls_symbol_offset(name);
  }

  struct link_map* lm = nullptr;
  DEPLOY_CHECK(
      0 == dlinfo(handle_, RTLD_DI_LINKMAP, &lm), "failed to query dlinfo");
  std::cout << "TLS dlinfo LOOKUP " << lm->l_name << " " << name << " "
            << "\n";

  ElfDynamicInfo info;
  info.initialize_from_dynamic_section(lm->l_name, lm->l_ld, lm->l_addr, true);
  auto r = info.sym(name);
  if (r) {
    size_t module_id = 0;
    DEPLOY_CHECK(
        0 == dlinfo(handle_, RTLD_DI_TLS_MODID, &module_id),
        "failed to query dlinfo for module_id");
    return TLSIndex{module_id, *r};
  }
  return multipy::nullopt;
}

// dlopen does not accept additional search paths as an argument.
// however, normal DT_NEEDED library load inherits the runpath of parents.
// So we need to pre-find all the libraries and call dlopen on them directly to
// get the same behavior. We can find the dependencies by reading the libraries
// dynamic section for recursive DT_NEEED entries.
void resolve_needed_libraries(
    std::vector<std::shared_ptr<SymbolProvider>>& libraries,
    const std::string& origin_relative,
    std::vector<std::string>& search_path,
    const std::string& runpath_template,
    const std::vector<const char*>& needed) {
  size_t search_path_start_size = search_path.size();

  std::string origin = resolve_origin(origin_relative);
  std::vector<std::string> paths = split_path(runpath_template, ':');
  // backwards because we want paths to be search in order but we search
  // search_path backward
  for (size_t i = paths.size(); i > 0; --i) {
    search_path.emplace_back(resolve_path(origin, paths[i - 1]));
  }

  for (const char* name : needed) {
    // std::cout << "ATTEMPTING FIND " << name << "\n";
    if (strcmp(name, "libtorch_python.so") == 0) {
      // torchvision expects it...
      continue;
    }
    // find the library, either (1) it is already loaded,
    //                          (2) it is an absolute path that exists,
    //                          (3) we find it in the search path
    //                          (4) we can dlopen it

    // (1) the library is already loaded
    const int base_flags = RTLD_LAZY | RTLD_LOCAL;
    void* handle = dlopen(name, base_flags | RTLD_NOLOAD);
    if (handle) {
      // std::cout << "ALREADY LOADED " << name << "\n";
      libraries.emplace_back(SystemLibrary::create(handle, true));
      continue;
    }

    std::string library_path = "";
    // (2) it is an absolute path
    if (strchr(name, '/') != nullptr) {
      library_path = name;
    } else {
      // (3) find it in the search path
      for (size_t i = search_path.size(); i > 0; --i) {
        std::stringstream ss;
        ss << search_path[i - 1] << "/" << name;
        if (access(ss.str().c_str(), F_OK) == 0) {
          library_path = ss.str();
          break;
        }
      }
    }

    std::vector<std::shared_ptr<SymbolProvider>>
        sublibraries; // these need to say loaded until we open library_path
                      // otherwise we might dlclose a sublibrary

    if (library_path != "") {
      // std::cout << "LOOKING FOR SUBLIBRARIES FOR FILE AT PATH " <<
      // library_path << "\n"; we found the actual file, recursively load its
      // deps before opening it so we resolve their paths correctly
      MemFile image(library_path.c_str());
      auto search =
          load_needed_from_elf_file(library_path.c_str(), image.data());
      resolve_needed_libraries(
          sublibraries, library_path, search_path, search.first, search.second);
    } else {
      library_path = name;
    }

    // either we didn't find the file, or we have already loaded its deps
    // in both cases, we now try to call dlopen. In the case where we didn't
    // find the file, we hope that something like LD_LIBRARY_PATH knows where it
    // is. In the case where we found it, we know its deps are loaded and
    // resolved.

    // std::cout << "OPENING " << library_path << "\n";
    handle = dlopen(library_path.c_str(), base_flags);
    DEPLOY_CHECK(
        handle, "{}: could not load library, dlopen says: {}", name, dlerror());
    libraries.emplace_back(SystemLibrary::create(handle, true));
  }

  // unwind search_path stack
  search_path.erase(
      search_path.begin() + search_path_start_size, search_path.end());
}

// NOLINTNEXTLINE
extern "C" void* __dso_handle;

struct __attribute__((visibility("hidden"))) CustomLibraryImpl
    : public std::enable_shared_from_this<CustomLibraryImpl>,
      public CustomLibrary {
  CustomLibraryImpl(const char* filename, int argc, const char** argv)
      : contents_(filename),
        mapped_library_(nullptr),
        name_(filename),
        argc_(argc),
        argv_(argv) {
    pthread_key_create(&tls_key_, nullptr);
    data_ = contents_.data();
    header_ = (Elf64_Ehdr*)data_;
    program_headers_ = (Elf64_Phdr*)(data_ + header_->e_phoff);
    n_program_headers_ = header_->e_phnum;
  }
  void add_search_library(std::shared_ptr<SymbolProvider> lib) override {
    symbol_search_path_.emplace_back(std::move(lib));
  }

  void check_library_format() {
    DEPLOY_CHECK(
        0 == memcmp(header_->e_ident, ELFMAG, SELFMAG),
        "{}: not an ELF file",
        this->name_);
    DEPLOY_CHECK(
        header_->e_type == ET_DYN,
        "{}: is not shared object file",
        this->name_);
    DEPLOY_CHECK(
        header_->e_ident[EI_CLASS] == ELFCLASS64,
        "{}: is not ELF64 format",
        this->name_);
    DEPLOY_CHECK(
        header_->e_ident[EI_DATA] == ELFDATA2LSB,
        "{}: is not 2's complement, little endian",
        this->name_);
    DEPLOY_CHECK(
        header_->e_machine == EM_X86_64,
        "{}: is not in x86_64 format",
        this->name_);
  }

  void reserve_address_space() {
    Elf64_Addr min_vaddr = 0;
    Elf64_Addr max_vaddr = 0;
    mapped_size_ = phdr_table_get_load_size(
        program_headers_, n_program_headers_, &min_vaddr, &max_vaddr);
    mapped_library_ = mmap(
        nullptr, mapped_size_, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    load_bias_ =
        (const char*)mapped_library_ - reinterpret_cast<const char*>(min_vaddr);
  }

  void load_segments() {
    // from bionic
    for (const auto i : multipy::irange(n_program_headers_)) {
      const Elf64_Phdr* phdr = &program_headers_[i];

      // Segment addresses in memory.
      Elf64_Addr seg_start = phdr->p_vaddr + load_bias_;
      Elf64_Addr seg_end = seg_start + phdr->p_memsz;

      switch (phdr->p_type) {
        case PT_DYNAMIC:
          dynamic_ = reinterpret_cast<Elf64_Dyn*>(seg_start);
          break;
        case PT_GNU_EH_FRAME:
          eh_frame_hdr_ = reinterpret_cast<EH_Frame_HDR*>(seg_start);
          DEPLOY_CHECK(
              eh_frame_hdr_->eh_frame_ptr_enc == 0x1b,
              "unsupported eh_frame_pointer_enc {}",
              eh_frame_hdr_->eh_frame_ptr_enc);
          eh_frame_ =
              (void*)((int64_t)&eh_frame_hdr_->eh_frame_ptr + eh_frame_hdr_->eh_frame_ptr);
          break;
        case PT_TLS:
          tls_file_size_ = phdr->p_filesz;
          tls_mem_size_ = phdr->p_memsz;
          tls_initalization_image_ = (void*)seg_start;
          break;
      };

      if (phdr->p_type != PT_LOAD) {
        continue;
      }

      Elf64_Addr seg_page_start = PAGE_START(seg_start);
      Elf64_Addr seg_page_end = PAGE_END(seg_end);

      Elf64_Addr seg_file_end = seg_start + phdr->p_filesz;

      // File offsets.
      Elf64_Addr file_start = phdr->p_offset;
      Elf64_Addr file_end = file_start + phdr->p_filesz;

      Elf64_Addr file_page_start = PAGE_START(file_start);
      Elf64_Addr file_length = file_end - file_page_start;

      if (contents_.size() <= 0) {
        DEPLOY_ERROR(
            "\"{}\" invalid file size: {}", name_.c_str(), contents_.size());
      }

      if (file_end > contents_.size()) {
        DEPLOY_ERROR(
            "invalid ELF file \"{}\" load segment[{}]:"
            " p_offset ({}) + p_filesz ({}) ( = {}) past end of file "
            "({})",
            name_.c_str(),
            i,
            reinterpret_cast<void*>(phdr->p_offset),
            reinterpret_cast<void*>(phdr->p_filesz),
            reinterpret_cast<void*>(file_end),
            contents_.size());
      }

      if (file_length != 0) {
        int prot = PFLAGS_TO_PROT(phdr->p_flags);

        void* seg_addr = mmap64(
            reinterpret_cast<void*>(seg_page_start),
            file_length,
            prot | PROT_WRITE, // initially everything is writable to do
                               // relocations
            MAP_FIXED | MAP_PRIVATE,
            contents_.fd(),
            file_page_start);
        fixup_prot_.emplace_back([=]() {
          mprotect(reinterpret_cast<void*>(seg_page_start), file_length, prot);
        });
        if (seg_addr == MAP_FAILED) {
          DEPLOY_ERROR(
              "couldn't map \"{}\" segment {}: {}",
              name_.c_str(),
              i,
              strerror(errno));
        }
      }

      // if the segment is writable, and does not end on a page boundary,
      // zero-fill it until the page limit.
      if ((phdr->p_flags & PF_W) != 0 && PAGE_OFFSET(seg_file_end) > 0) {
        memset(
            reinterpret_cast<void*>(seg_file_end),
            0,
            PAGE_SIZE - PAGE_OFFSET(seg_file_end));
      }

      seg_file_end = PAGE_END(seg_file_end);

      // seg_file_end is now the first page address after the file
      // content. If seg_end is larger, we need to zero anything
      // between them. This is done by using a private anonymous
      // map for all extra pages.
      if (seg_page_end > seg_file_end) {
        size_t zeromap_size = seg_page_end - seg_file_end;
        int prot = PFLAGS_TO_PROT(phdr->p_flags);
        void* zeromap = mmap(
            reinterpret_cast<void*>(seg_file_end),
            zeromap_size,
            prot | PROT_WRITE,
            MAP_FIXED | MAP_ANONYMOUS | MAP_PRIVATE,
            -1,
            0);
        fixup_prot_.emplace_back([=]() {
          mprotect(reinterpret_cast<void*>(seg_file_end), zeromap_size, prot);
        });
        if (zeromap == MAP_FAILED) {
          DEPLOY_ERROR(
              "couldn't zero fill \"{}\" gap: {}",
              name_.c_str(),
              strerror(errno));
        }
      }
    }
  }
  size_t module_id() const {
    size_t this_as_number = (size_t)this;
    return this_as_number | TLS_LOCAL_FLAG;
  }

  void read_dynamic_section() {
    dyninfo_.initialize_from_dynamic_section(
        name_, dynamic_, load_bias_, false);
    std::vector<std::string> empty_search_path;
    resolve_needed_libraries(
        symbol_search_path_,
        name_,
        empty_search_path,
        dyninfo_.runpath_,
        dyninfo_.needed_);
  }

  multipy::optional<Elf64_Addr> lookup_symbol(Elf64_Xword r_info) {
    const uint32_t r_type = ELF64_R_TYPE(r_info);
    const uint32_t r_sym = ELF64_R_SYM(r_info);

    if (r_sym == 0) {
      return (Elf64_Addr)0;
    }
    auto sym_st = dyninfo_.symtab_[r_sym];
    const char* sym_name = dyninfo_.get_string(sym_st.st_name);
    if (r_type == R_X86_64_JUMP_SLOT) {
      if (strcmp(sym_name, "__tls_get_addr") == 0) {
        return (Elf64_Addr)local__tls_get_addr;
      }
      if (strcmp(sym_name, "__cxa_thread_atexit") == 0) {
        return (Elf64_Addr)__cxa_thread_atexit_impl;
      }
    }
    for (const auto& sys_lib : symbol_search_path_) {
      auto r = sys_lib->sym(sym_name);
      if (r) {
        return r;
      }
    }
    auto r = sym(sym_name);
    if (r) {
      return r;
    }
    if (ELF64_ST_BIND(sym_st.st_info) != STB_WEAK) {
      DEPLOY_ERROR(
          "{}: '{}' symbol not found in ElfFile lookup",
          name_.c_str(),
          sym_name);
    }
    return multipy::nullopt;
  }

  multipy::optional<TLSIndex> tls_lookup_symbol(Elf64_Xword r_info) {
    const uint32_t r_sym = ELF64_R_SYM(r_info);

    if (r_sym == 0) {
      return TLSIndex{
          module_id(),
          0}; // note: offset is not queried when the symbol is blank
    }

    auto sym_st = dyninfo_.symtab_[r_sym];
    const char* sym_name = dyninfo_.get_string(sym_st.st_name);
    for (const auto& sys_lib : symbol_search_path_) {
      auto r = sys_lib->tls_sym(sym_name);
      if (r) {
        return r;
      }
    }
    auto r = tls_sym(sym_name);
    if (r) {
      return r;
    }

    if (ELF64_ST_BIND(sym_st.st_info) != STB_WEAK) {
      DEPLOY_ERROR(
          "{}: '{}' symbol not found in ElfFile lookup",
          name_.c_str(),
          sym_name);
    }
    return multipy::nullopt;
  }

  void relocate_one(const Elf64_Rela& reloc) {
    const uint32_t r_type = ELF64_R_TYPE(reloc.r_info);

    if (r_type == 0) {
      return;
    }

    void* const rel_target =
        reinterpret_cast<void*>(reloc.r_offset + load_bias_);

    // TLS relocations need to lookup symbols differently so we can get the
    // module_id
    if (r_type == R_X86_64_DTPMOD64 || r_type == R_X86_64_DTPOFF64) {
      auto tls_index = tls_lookup_symbol(reloc.r_info);
      if (!tls_index) {
        return; // skip weak relocation that wasn't found
      }
      switch (r_type) {
        case R_X86_64_DTPMOD64:
          *static_cast<size_t*>(rel_target) = tls_index->module_id;
          break;
        case R_X86_64_DTPOFF64:
          *static_cast<Elf64_Addr*>(rel_target) =
              tls_index->offset + reloc.r_addend;
          break;
      }
      return;
    }

    auto sym_addr = lookup_symbol(reloc.r_info);
    if (!sym_addr) {
      return; // skip weak relocation that wasn't found
    }

    switch (r_type) {
      case R_X86_64_JUMP_SLOT:
      case R_X86_64_64:
      case R_X86_64_GLOB_DAT: {
        const Elf64_Addr result = *sym_addr + reloc.r_addend;
        *static_cast<Elf64_Addr*>(rel_target) = result;
      } break;
      case R_X86_64_RELATIVE: {
        // In practice, r_sym is always zero, but if it weren't, the linker
        // would still look up the referenced symbol (and abort if the symbol
        // isn't found), even though it isn't used.
        const Elf64_Addr result = load_bias_ + reloc.r_addend;
        *static_cast<Elf64_Addr*>(rel_target) = result;
      } break;
      case R_X86_64_32: {
        const Elf32_Addr result = *sym_addr + reloc.r_addend;
        *static_cast<Elf32_Addr*>(rel_target) = result;
      } break;
      case R_X86_64_PC32: {
        const Elf64_Addr target = *sym_addr + reloc.r_addend;
        const Elf64_Addr base = reinterpret_cast<Elf64_Addr>(rel_target);
        const Elf32_Addr result = target - base;
        *static_cast<Elf32_Addr*>(rel_target) = result;
      } break;
      default:
        DEPLOY_ERROR("unknown reloc type {} in \"{}\"", r_type, name_.c_str());
        break;
    }
  }

  void relocate() {
    for (const auto i : multipy::irange(dyninfo_.n_rela_)) {
      relocate_one(dyninfo_.rela_[i]);
    }
    for (const auto i : multipy::irange(dyninfo_.n_plt_rela_)) {
      relocate_one(dyninfo_.plt_rela_[i]);
    }
  }

  void initialize() {
    call_function(dyninfo_.init_func_);
    for (const auto i : multipy::irange(dyninfo_.n_init_array_)) {
      call_function(dyninfo_.init_array_[i]);
    }
    initialized_ = true;
  }

  void finalize() {
    for (size_t i = dyninfo_.n_fini_array_; i > 0; --i) {
      call_function(dyninfo_.fini_array_[i - 1]);
    }
    call_function(dyninfo_.fini_func_);
  }

  void register_debug_info() {
    // std::cout << "target modules add " << name_.c_str() << "\n";
    // std::cout << "target modules load -f " << name_.c_str() << " -s "
    //           << std::hex << "0x" << load_bias_ << "\n";
    __deploy_module_info.name = name_.c_str();
    __deploy_module_info.file_addr = (Elf64_Addr)contents_.data();
    __deploy_module_info.file_size = contents_.size();
    __deploy_module_info.load_bias = load_bias_;
    // debugger script sets a breakpoint on this function,
    // then reads __deploy_module_info to issue the target module commands.
    __deploy_register_code();
  }

  // remove the extra write flags from read-only sections
  void protect() {
    for (const auto& fixup : fixup_prot_) {
      fixup();
    }
  }

  void load() override {
    check_library_format();
    reserve_address_space();
    load_segments();
    read_dynamic_section();
    relocate();
    protect();
    __register_frame(eh_frame_);
    eh_frame_registered_ = true;
    register_debug_info();
    initialize();
  }

  ~CustomLibraryImpl() override {
    // std::cout << "LINKER IS UNLOADING: " << name_ << "\n";
    if (initialized_) {
      finalize();
    }
    if (eh_frame_registered_) {
      __deregister_frame(eh_frame_);
    }
    if (mapped_library_) {
      munmap(mapped_library_, mapped_size_);
    }
  }
  void call_function(linker_dtor_function_t f) {
    if (f == nullptr || (int64_t)f == -1)
      return;
    f();
  }
  void call_function(linker_ctor_function_t f) {
    if (f == nullptr || (int64_t)f == -1)
      return;
    f(argc_, argv_, environ);
  }

  multipy::optional<Elf64_Addr> sym(const char* name) const override {
    return dyninfo_.sym(name);
  }

  multipy::optional<TLSIndex> tls_sym(const char* name) const override {
    auto r = dyninfo_.sym(name);
    if (r) {
      return TLSIndex{module_id(), *r};
    }
    return multipy::nullopt;
  }

  void* tls_addr(size_t offset) {
    // this was a TLS entry for one of our modules, so we use pthreads to
    // emulate thread local state.
    void* start = pthread_getspecific(tls_key_);
    if (!start) {
      auto tls_mem = new TLSMemory(shared_from_this(), tls_mem_size_);
      __cxa_thread_atexit_impl(delete_TLSMemory, tls_mem, &__dso_handle);
      start = tls_mem->mem_;
      memcpy(start, tls_initalization_image_, tls_file_size_);
      memset(
          (void*)((const char*)start + tls_file_size_),
          0,
          tls_mem_size_ - tls_file_size_);
      pthread_setspecific(tls_key_, start);
    }
    return (void*)((const char*)start + offset);
  }

 private:
  MemFile contents_;
  const char* data_ = nullptr;
  const Elf64_Ehdr* header_ = nullptr;
  const Elf64_Phdr* program_headers_ = nullptr;
  const EH_Frame_HDR* eh_frame_hdr_ = nullptr;
  void* eh_frame_ = nullptr;
  size_t n_program_headers_ = 0;
  void* mapped_library_ = nullptr;
  size_t mapped_size_ = 0;
  Elf64_Addr load_bias_ = 0;
  Elf64_Dyn* dynamic_ = nullptr;
  ElfDynamicInfo dyninfo_;
  std::string name_;
  int argc_ = 0;
  const char** argv_ = nullptr;
  bool initialized_ = false;
  bool eh_frame_registered_ = false;

  pthread_key_t tls_key_ = 0;
  void* tls_initalization_image_ = nullptr;
  size_t tls_file_size_ = 0;
  size_t tls_mem_size_ = 0;

  std::vector<std::shared_ptr<SymbolProvider>> symbol_search_path_;
  std::vector<std::function<void(void)>> fixup_prot_;
};

std::shared_ptr<CustomLibrary> CustomLibrary::create(
    const char* filename,
    int argc,
    const char** argv) {
  return std::make_shared<CustomLibraryImpl>(filename, argc, argv);
}

static void* local__tls_get_addr(TLSIndex* idx) {
  if ((idx->module_id & TLS_LOCAL_FLAG) != 0) {
    return ((CustomLibraryImpl*)(idx->module_id & ~TLS_LOCAL_FLAG))
        ->tls_addr(idx->offset);
  }
  return __tls_get_addr(idx);
}

} // namespace deploy
} // namespace torch
