#include <torch/csrc/deploy/Exception.h>
#include <torch/csrc/deploy/elf_file.h>
#include <torch/csrc/deploy/irange.h>

namespace torch {
namespace deploy {

ElfFile::ElfFile(const char* filename) : memFile_(filename) {
  const char* fileData = memFile_.data();
  ehdr_ = (Elf64_Ehdr*)fileData;
  checkFormat();

  numSections_ = ehdr_->e_shnum;
  shdrList_ = (Elf64_Shdr*)(fileData + ehdr_->e_shoff);

  auto strtabSecNo = ehdr_->e_shstrndx;
  MULTIPY_CHECK(
      strtabSecNo >= 0 && strtabSecNo < numSections_,
      "e_shstrndx out of range");

  strtabSection_ = toSection(&shdrList_[strtabSecNo]);

  sections_.reserve(numSections_);
  for (const auto i : multipy::irange(numSections_)) {
    sections_.emplace_back(toSection(&shdrList_[i]));
  }
}

at::optional<Section> ElfFile::findSection(const char* name) const {
  MULTIPY_CHECK(name != nullptr, "Null name");
  at::optional<Section> found = at::nullopt;
  for (const auto& section : sections_) {
    if (strcmp(name, section.name) == 0) {
      found = section;
      break;
    }
  }

  return found;
}

void ElfFile::checkFormat() const {
  // check the magic numbers
  MULTIPY_CHECK(
      (ehdr_->e_ident[EI_MAG0] == ELFMAG0) &&
          (ehdr_->e_ident[EI_MAG1] == ELFMAG1) &&
          (ehdr_->e_ident[EI_MAG2] == ELFMAG2) &&
          (ehdr_->e_ident[EI_MAG3] == ELFMAG3),
      "Unexpected magic numbers");
  MULTIPY_CHECK(
      ehdr_->e_ident[EI_CLASS] == ELFCLASS64, "Only support 64bit ELF file");
}

} // namespace deploy
} // namespace torch
