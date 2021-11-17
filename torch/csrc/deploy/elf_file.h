#pragma once

#include <c10/util/Optional.h>
#include <elf.h>
#include <torch/csrc/deploy/mem_file.h>
#include <vector>

namespace torch {
namespace deploy {

struct Section {
  explicit Section(
      const char* _name = nullptr,
      const char* _start = nullptr,
      size_t _len = 0)
      : name(_name), start(_start), len(_len) {}
  const char* name;
  const char* start;
  size_t len;

  operator bool() const {
    return start != nullptr;
  }
};

/*
 * This class provie utilities to handle ELF file. Only support 64bit ELF file.
 */
// TODO: consolidate other ELF file related functions in loader.cpp to this file
class ElfFile {
 public:
  explicit ElfFile(const char* filename);
  at::optional<Section> findSection(const char* name) const;

 private:
  Section toSection(Elf64_Shdr* shdr) {
    auto nameOff = shdr->sh_name;
    auto shOff = shdr->sh_offset;
    auto len = shdr->sh_size;
    const char* name = "";

    if (strtabSection_) {
      TORCH_CHECK(nameOff >= 0 && nameOff < strtabSection_.len);
      name = strtabSection_.start + nameOff;
    }
    const char* start = memFile_.data() + shOff;
    return Section{name, start, len};
  }

  [[nodiscard]] const char* str(size_t off) const {
    TORCH_CHECK(off < strtabSection_.len, "String table index out of range");
    return strtabSection_.start + off;
  }
  void checkFormat() const;
  MemFile memFile_;
  Elf64_Ehdr* ehdr_;
  Elf64_Shdr* shdrList_;
  size_t numSections_;

  Section strtabSection_;
  std::vector<Section> sections_;
};

} // namespace deploy
} // namespace torch
