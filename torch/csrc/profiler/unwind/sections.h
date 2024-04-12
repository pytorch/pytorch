#pragma once
#include <torch/csrc/profiler/unwind/abbreviation_table.h>
#include <torch/csrc/profiler/unwind/dwarf_enums.h>
#include <torch/csrc/profiler/unwind/dwarf_symbolize_enums.h>
#include <torch/csrc/profiler/unwind/mem_file.h>
#include <torch/csrc/profiler/unwind/range_table.h>
#include <torch/csrc/profiler/unwind/unwind_error.h>
#include <cstdint>

namespace torch::unwind {

struct Sections {
  Sections() {}
  void parse(const char* name) {
    library_ = std::make_unique<MemFile>(name);
    debug_info = library_->getSection(".debug_info", false);
    debug_abbrev = library_->getSection(".debug_abbrev", false);
    debug_aranges = library_->getSection(".debug_aranges", false);
    debug_str = library_->getSection(".debug_str", false);
    debug_line = library_->getSection(".debug_line", false);
    // dwarf 5
    debug_line_str = library_->getSection(".debug_line_str", true);
    debug_rnglists = library_->getSection(".debug_rnglists", true);
    // dwarf 4
    debug_ranges = library_->getSection(".debug_ranges", true);

    parseAranges();
    parseDebugInfo();
  }

  Section debug_info;
  Section debug_abbrev;
  Section debug_aranges;
  Section debug_str;
  Section debug_line;
  Section debug_line_str;
  Section debug_rnglists;
  Section debug_ranges;

  const char* readString(CheckedLexer& data, int encoding, bool is_64bit) {
    switch (encoding) {
      case DW_FORM_string: {
        return data.readCString();
      }
      case DW_FORM_strp: {
        return debug_str.data + readSegmentOffset(data, is_64bit);
      }
      case DW_FORM_line_strp: {
        return debug_line_str.data + readSegmentOffset(data, is_64bit);
      }
      default:
        UNWIND_CHECK(false, "unsupported string encoding {}", encoding);
    }
  }

  int64_t readSegmentOffset(CheckedLexer& data, bool is_64bit) {
    return is_64bit ? data.read<uint64_t>() : data.read<uint32_t>();
  }
  AbbrevationTable* getAbbrev(uint64_t offset) {
    auto e = abbrev_ranges_.find(offset);
    UNWIND_CHECK(
        e.has_value(), "no abbrevation table found for offset {}", offset);
    return getAbbrevByTableOffset(e->first, e->second);
  }
  unwind::optional<uint64_t> findDebugInfoOffset(uint64_t address) {
    return debug_info_offsets_.find(address);
  }
  AbbrevationTable* getAbbrevByTableOffset(
      uint64_t offset,
      uint8_t section_offset_size) {
    auto it = abbrev_tables_.find(offset);
    if (it == abbrev_tables_.end()) {
      auto table = std::make_unique<AbbrevationTable>();
      it = abbrev_tables_.emplace(offset, std::move(table)).first;
      it->second->parse(debug_abbrev, offset, section_offset_size);
    }
    return it->second.get();
  }
  size_t compilationUnitCount() {
    return debug_info_offsets_.size() / 2;
  }
  std::vector<uint64_t> debugInfoOffsets() {
    return debug_infos_;
  }

 private:
  void parseAranges() {
    LOG_INFO("init debug_aranges\n");
    char* end = debug_aranges.data + debug_aranges.size;
    CheckedLexer data = debug_aranges.lexer(0);
    while (data.loc() < end) {
      auto length = data.readSectionLength();
      auto version = data.read<uint16_t>();
      UNWIND_CHECK(
          version == 2, "unexpected dwarf aranges version {}", version);
      auto debug_info_offset =
          length.second ? data.read<uint64_t>() : data.read<uint32_t>();
      auto address_size = data.read<uint8_t>();
      auto segment_size = data.read<uint8_t>();
      UNWIND_CHECK(
          segment_size == 0, "expected no segment but found {}", segment_size);
      UNWIND_CHECK(
          address_size == 8,
          "aranges: expected 64-bit dwarf but found address size {}",
          address_size);
      auto off = (char*)data.loc() - debug_aranges.data;
      if (off % 16 != 0) {
        data.skip(16 - (off % 16));
      }
      while (data.loc() < end) {
        auto start_address = data.read<uint64_t>();
        auto length = data.read<uint64_t>();
        if (start_address == 0 && length == 0) {
          break;
        }
        debug_info_offsets_.add(start_address, debug_info_offset, false);
        debug_info_offsets_.add(start_address + length, std::nullopt, false);
      }
    }
    // debug_info_offsets_.dump();
    LOG_INFO("read {} aranges entries\n", debug_info_offsets_.size() / 2);
  }
  void parseDebugInfo() {
    auto end = debug_info.data + debug_info.size;
    auto L = debug_info.lexer(0);
    while (L.loc() < end) {
      auto start = (char*)L.loc();
      debug_infos_.push_back(start - debug_info.data);
      auto length = L.readSectionLength();
      CheckedLexer next = L;
      next.skip(length.first);
      auto version = L.read<uint16_t>();
      UNWIND_CHECK(
          version == 4 || version == 5, "expected dwarf version {}", version);
      uint64_t debug_abbrev_offset;
      if (version == 5) {
        L.read<uint8_t>(); // unit_type
        L.read<uint8_t>(); // address_size
        debug_abbrev_offset =
            length.second ? L.read<uint64_t>() : L.read<uint32_t>();
      } else {
        debug_abbrev_offset =
            length.second ? L.read<uint64_t>() : L.read<uint32_t>();
        L.read<uint8_t>(); // address_size
      }
      abbrev_ranges_.add(
          start - debug_info.data,
          std::make_pair(debug_abbrev_offset, length.second ? 8 : 4),
          true);
      L = next;
    }
  }
  std::unique_ptr<MemFile> library_;
  RangeTable<uint64_t> debug_info_offsets_;
  RangeTable<std::pair<uint64_t, uint8_t>> abbrev_ranges_;
  std::unordered_map<uint64_t, std::unique_ptr<AbbrevationTable>>
      abbrev_tables_;
  std::vector<uint64_t> debug_infos_;
};

} // namespace torch::unwind
