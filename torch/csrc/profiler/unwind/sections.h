#pragma once
#include <cxxabi.h>
#include <elf.h>
#include <torch/csrc/profiler/unwind/dwarf_enums.h>
#include <torch/csrc/profiler/unwind/dwarf_symbolize_enums.h>
#include <torch/csrc/profiler/unwind/mem_file.h>
#include <torch/csrc/profiler/unwind/range_table.h>
#include <torch/csrc/profiler/unwind/unwind_error.h>
#include <cstdint>

namespace torch::unwind {

static std::string demangle(const std::string& mangled_name) {
  int status = 0;
  char* realname =
      abi::__cxa_demangle(mangled_name.c_str(), nullptr, nullptr, &status);
  if (status == 0) {
    std::string demangled_name(realname);
    // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
    free(realname);
    return demangled_name;
  } else {
    return mangled_name;
  }
}

struct Sections {
  Sections() = default;
  void parse(const char* name) {
    library_ = std::make_unique<MemFile>(name);
    strtab = library_->getSection(".strtab", false);

    symtab = library_->getSection(".symtab", true);
    debug_info = library_->getSection(".debug_info", true);
    if (debug_info.size > 0) {
      debug_abbrev = library_->getSection(".debug_abbrev", false);
      debug_str = library_->getSection(".debug_str", false);
      debug_line = library_->getSection(".debug_line", false);
      // dwarf 5
      debug_line_str = library_->getSection(".debug_line_str", true);
      debug_rnglists = library_->getSection(".debug_rnglists", true);
      debug_addr = library_->getSection(".debug_addr", true);
      // dwarf 4
      debug_ranges = library_->getSection(".debug_ranges", true);
    }
    parseSymtab();
  }

  Section debug_info;
  Section debug_abbrev;
  Section debug_str;
  Section debug_line;
  Section debug_line_str;
  Section debug_rnglists;
  Section debug_ranges;
  Section debug_addr;
  Section symtab;
  Section strtab;

  const char* readString(CheckedLexer& data, uint64_t encoding, bool is_64bit) {
    switch (encoding) {
      case DW_FORM_string: {
        return data.readCString();
      }
      case DW_FORM_strp: {
        return debug_str.string(readSegmentOffset(data, is_64bit));
      }
      case DW_FORM_line_strp: {
        return debug_line_str.string(readSegmentOffset(data, is_64bit));
      }
      default:
        UNWIND_CHECK(false, "unsupported string encoding {:x}", encoding);
    }
  }

  uint64_t readSegmentOffset(CheckedLexer& data, bool is_64bit) {
    return is_64bit ? data.read<uint64_t>() : data.read<uint32_t>();
  }

  std::optional<uint64_t> findDebugInfoOffset(uint64_t address) {
    return debug_info_offsets_.find(address);
  }
  size_t compilationUnitCount() {
    return debug_info_offsets_.size() / 2;
  }
  void addDebugInfoRange(
      uint64_t start,
      uint64_t end,
      uint64_t debug_info_offset) {
    debug_info_offsets_.add(start, debug_info_offset, false);
    debug_info_offsets_.add(end, std::nullopt, false);
  }
  std::optional<std::string> findSubprogramName(uint64_t address) {
    if (auto e = symbol_table_.find(address)) {
      return demangle(strtab.string(*e));
    }
    return std::nullopt;
  }

 private:
  void parseSymtab() {
    auto L = symtab.lexer(0);
    char* end = symtab.data + symtab.size;
    while (L.loc() < end) {
      auto symbol = L.read<Elf64_Sym>();
      if (symbol.st_shndx == SHN_UNDEF ||
          ELF64_ST_TYPE(symbol.st_info) != STT_FUNC) {
        continue;
      }
      symbol_table_.add(symbol.st_value, symbol.st_name, false);
      symbol_table_.add(symbol.st_value + symbol.st_size, std::nullopt, false);
    }
  }

  std::unique_ptr<MemFile> library_;
  RangeTable<uint64_t> debug_info_offsets_;
  RangeTable<uint64_t> symbol_table_;
};

} // namespace torch::unwind
