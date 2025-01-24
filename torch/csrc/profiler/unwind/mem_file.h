// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <c10/util/error.h>
#include <elf.h>
#include <fcntl.h>
#include <fmt/format.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <torch/csrc/profiler/unwind/lexer.h>
#include <torch/csrc/profiler/unwind/unwind_error.h>
#include <unistd.h>
#include <cerrno>
#include <cstdio>
#include <cstring>

namespace torch::unwind {

struct Section {
  char* data = nullptr;
  size_t size = 0;
  const char* string(size_t offset) {
    return lexer(offset).readCString();
  }
  CheckedLexer lexer(size_t offset) {
    return CheckedLexer(data + offset, data, data + size);
  }
};

/// Memory maps a file into the address space read-only, and manages the
/// lifetime of the mapping. Here are a few use cases:
/// 1. Used in the loader to read in initial image, and to inspect
// ELF files for dependencies before callling dlopen.
///
/// 2. Used in unity to load the elf file.
struct MemFile {
  explicit MemFile(const char* filename_)
      : fd_(open(filename_, O_RDONLY)), name_(filename_) {
    UNWIND_CHECK(
        fd_ != -1,
        "failed to open {}: {}",
        filename_,
        c10::utils::str_error(errno));
    struct stat s {};
    if (-1 == fstat(fd_, &s)) {
      close(fd_); // destructors don't run during exceptions
      UNWIND_CHECK(
          false,
          "failed to stat {}: {}",
          filename_,
          c10::utils::str_error(errno));
    }
    n_bytes_ = s.st_size;
    UNWIND_CHECK(
        n_bytes_ > sizeof(Elf64_Ehdr), "empty shared library: {}", filename_);
    mem_ = (char*)mmap(nullptr, n_bytes_, PROT_READ, MAP_SHARED, fd_, 0);
    if (MAP_FAILED == mem_) {
      close(fd_);
      UNWIND_CHECK(
          false,
          "failed to mmap {}: {}",
          filename_,
          c10::utils::str_error(errno));
    }
    ehdr_ = (Elf64_Ehdr*)mem_;
#define ELF_CHECK(cond) UNWIND_CHECK(cond, "not an ELF file: {}", filename_)
    ELF_CHECK(ehdr_->e_ident[EI_MAG0] == ELFMAG0);
    ELF_CHECK(ehdr_->e_ident[EI_MAG1] == ELFMAG1);
    ELF_CHECK(ehdr_->e_ident[EI_MAG2] == ELFMAG2);
    ELF_CHECK(ehdr_->e_ident[EI_MAG3] == ELFMAG3);
    ELF_CHECK(ehdr_->e_ident[EI_CLASS] == ELFCLASS64);
    ELF_CHECK(ehdr_->e_ident[EI_VERSION] == EV_CURRENT);
    ELF_CHECK(ehdr_->e_version == EV_CURRENT);
    ELF_CHECK(ehdr_->e_machine == EM_X86_64);
#undef ELF_CHECK
    UNWIND_CHECK(
        ehdr_->e_shoff + sizeof(Elf64_Shdr) * ehdr_->e_shnum <= n_bytes_,
        "invalid section header table {} {} {}",
        ehdr_->e_shoff + sizeof(Elf64_Shdr) * ehdr_->e_shnum,
        n_bytes_,
        ehdr_->e_shnum);
    shdr_ = (Elf64_Shdr*)(mem_ + ehdr_->e_shoff);
    UNWIND_CHECK(
        ehdr_->e_shstrndx < ehdr_->e_shnum, "invalid strtab section offset");
    auto& strtab_hdr = shdr_[ehdr_->e_shstrndx];
    strtab_ = getSection(strtab_hdr);
  }

  MemFile(const MemFile&) = delete;
  MemFile(MemFile&&) = delete;
  MemFile& operator=(const MemFile&) = delete;
  MemFile& operator=(MemFile&&) = delete;
  [[nodiscard]] const char* data() const {
    return (const char*)mem_;
  }

  /// Returns whether or not the file descriptor
  /// of the underlying file is valid.
  int valid() {
    return fcntl(fd_, F_GETFD) != -1 || errno != EBADF;
  }

  ~MemFile() {
    if (mem_) {
      munmap((void*)mem_, n_bytes_);
    }
    if (fd_ >= 0) {
      close(fd_);
    }
  }

  /// Returns the size of the underlying file defined by the `MemFile`
  size_t size() {
    return n_bytes_;
  }
  [[nodiscard]] int fd() const {
    return fd_;
  }

  Section getSection(const Elf64_Shdr& shdr) {
    UNWIND_CHECK(shdr.sh_offset + shdr.sh_size <= n_bytes_, "invalid section");
    return Section{mem_ + shdr.sh_offset, shdr.sh_size};
  }

  Section getSection(const char* name, bool optional) {
    for (int i = 0; i < ehdr_->e_shnum; i++) {
      if (strcmp(strtab_.string(shdr_[i].sh_name), name) == 0) {
        return getSection(shdr_[i]);
      }
    }
    UNWIND_CHECK(optional, "{} has no section {}", name_, name);
    return Section{nullptr, 0};
  }

  Section strtab() {
    return strtab_;
  }

 private:
  template <typename T>
  T* load(size_t offset) {
    UNWIND_CHECK(offset < n_bytes_, "out of range");
    return (T*)(mem_ + offset);
  }
  int fd_;
  char* mem_{nullptr};
  size_t n_bytes_{0};
  std::string name_;
  Elf64_Ehdr* ehdr_;
  Elf64_Shdr* shdr_;
  Section strtab_ = {nullptr, 0};
};

} // namespace torch::unwind
