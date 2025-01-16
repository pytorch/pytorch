#pragma once
#include <torch/csrc/profiler/unwind/dwarf_enums.h>
#include <torch/csrc/profiler/unwind/dwarf_symbolize_enums.h>
#include <torch/csrc/profiler/unwind/lexer.h>
#include <torch/csrc/profiler/unwind/sections.h>
#include <torch/csrc/profiler/unwind/unwind_error.h>
#include <cstdint>
#include <optional>

namespace torch::unwind {

struct DebugInfo {
  DebugInfo(Sections& s) : s_(s) {}

  void parse(uint64_t offset) {
    auto L = parseHeader(offset);
    parseCompileUnit(L);
  }
  std::optional<uint64_t> lineNumberProgramOffset() {
    return line_number_program_offset_;
  }
  uint64_t nextOffset() {
    return end_ - s_.debug_info.data;
  }
  std::vector<std::pair<uint64_t, uint64_t>> ranges() {
    if (range_ptr_) {
      auto offset = range_ptr_->first;
      if (range_ptr_->second == DW_FORM_rnglistx) {
        UNWIND_CHECK(rnglists_base_, "rnglistx but not rnglists_base_ set");
        LOG_INFO("index for rnglistx {:x} + {:x}\n", *rnglists_base_, offset);
        CheckedLexer L = s_.debug_rnglists.lexer(
            *rnglists_base_ + offset * sec_offset_size_);
        auto read = readSegmentOffset(L);
        offset = *rnglists_base_ + read;
      }
      return version_ == 4 ? readRanges4(offset) : readRanges5(offset);
    }
    if (!highpc_) {
      return {};
    }
    return {{lowpc_, lowpc_ + *highpc_}};
  }

  bool is64bit() {
    return is_64bit_;
  }

 private:
  CheckedLexer parseHeader(uint64_t offset) {
    offset_ = offset;
    CheckedLexer L = s_.debug_info.lexer(offset_);
    std::tie(length_, is_64bit_) = L.readSectionLength();
    sec_offset_size_ = is_64bit_ ? 8 : 4;
    end_ = (const char*)L.loc() + length_;
    version_ = L.read<uint16_t>();
    UNWIND_CHECK(
        version_ == 5 || version_ == 4,
        "unexpected dwarf version {}",
        version_);
    uint8_t address_size = 0;
    if (version_ == 5) {
      auto unit_type = L.read<uint8_t>();
      UNWIND_CHECK(unit_type == 0x1, "unexpected unit type {}", unit_type);
      address_size = L.read<uint8_t>();
      debug_abbrev_offset_ =
          is_64bit_ ? L.read<uint64_t>() : L.read<uint32_t>();
    } else {
      debug_abbrev_offset_ =
          is_64bit_ ? L.read<uint64_t>() : L.read<uint32_t>();
      address_size = L.read<uint8_t>();
    }
    LOG_INFO(
        "compilation unit at offset {:x} with length {:x} and debug_abbrev_offset {:x}\n",
        offset,
        length_,
        debug_abbrev_offset_);
    UNWIND_CHECK(
        address_size == 8,
        "expected 64-bit dwarf but found address size {}",
        address_size);
    return L;
  }

  uint64_t readSegmentOffset(CheckedLexer& L) {
    return s_.readSegmentOffset(L, is_64bit_);
  }

  uint64_t readEncoded(CheckedLexer& L, uint64_t encoding) {
    switch (encoding) {
      case DW_FORM_data8:
      case DW_FORM_addr:
        return L.read<uint64_t>();
      case DW_FORM_data4:
        return L.read<uint32_t>();
      case DW_FORM_addrx: {
        auto idx = L.readULEB128();
        return s_.debug_addr.lexer(address_base_ + sizeof(uint64_t) * idx)
            .read<uint64_t>();
      }
      case DW_FORM_sec_offset:
        return readSegmentOffset(L);
      case DW_FORM_rnglistx: {
        return L.readULEB128();
      }
      default:
        UNWIND_CHECK(false, "unexpected encoding");
    }
  }

  void parseCompileUnit(CheckedLexer& L) {
    auto entry = L.readULEB128();
    auto A = findAbbrev(debug_abbrev_offset_, entry);
    while (true) {
      auto attr = A.readULEB128();
      auto form = A.readULEB128();
      if (attr == 0 && form == 0) {
        break;
      }
      if (form == DW_FORM_implicit_const) {
        A.readSLEB128();
      }
      if (attr == DW_AT_low_pc) {
        lowpc_ = readEncoded(L, form);
        LOG_INFO("  lowpc {:x}\n", lowpc_);
      } else if (attr == DW_AT_high_pc) {
        highpc_ = readEncoded(L, form);
        range_ptr_ = std::nullopt;
        LOG_INFO("  highpc {:x}\n", *highpc_);
      } else if (attr == DW_AT_addr_base) {
        UNWIND_CHECK(form == DW_FORM_sec_offset, "unexpected addr_base form");
        address_base_ = readSegmentOffset(L);
        LOG_INFO("  address base {:x}\n", address_base_);
      } else if (attr == DW_AT_rnglists_base) {
        UNWIND_CHECK(
            form == DW_FORM_sec_offset, "unexpected rnglists_base form");
        rnglists_base_ = readSegmentOffset(L);
        LOG_INFO("  range base {:x}\n", *rnglists_base_);
      } else if (form == DW_FORM_string) {
        L.readCString();
      } else if (attr == DW_AT_stmt_list) {
        UNWIND_CHECK(form == DW_FORM_sec_offset, "unexpected stmt_list form");
        LOG_INFO("  program table offset {:x}\n", *line_number_program_offset_);
        line_number_program_offset_ = readSegmentOffset(L);
      } else if (form == DW_FORM_exprloc) {
        auto sz = L.readULEB128();
        L.skip(int64_t(sz));
      } else if (form == DW_FORM_block1) {
        auto sz = L.read<uint8_t>();
        L.skip(int64_t(sz));
      } else if (attr == DW_AT_ranges) {
        auto range_offset = readEncoded(L, form);
        LOG_INFO("setting range_ptr to {:x} {:x}\n", range_offset, form);
        range_ptr_.emplace(range_offset, form);
      } else if (
          form == DW_FORM_udata || form == DW_FORM_rnglistx ||
          form == DW_FORM_strx || form == DW_FORM_loclistx ||
          form == DW_FORM_addrx) {
        L.readULEB128();
      } else if (form == DW_FORM_sdata) {
        L.readSLEB128();
      } else {
        auto sz = formSize(form, sec_offset_size_);
        UNWIND_CHECK(sz, "unsupported form in compilation unit {:x}", form);
        L.skip(int64_t(*sz));
      }
    }
  }

  std::vector<std::pair<uint64_t, uint64_t>> readRanges4(uint64_t offset) {
    CheckedLexer L = s_.debug_ranges.lexer(offset);
    std::vector<std::pair<uint64_t, uint64_t>> ranges;
    uint64_t base = lowpc_;
    while (true) {
      auto start = L.read<uint64_t>();
      auto end = L.read<uint64_t>();
      if (start == 0 && end == 0) {
        break;
      }
      if (start == std::numeric_limits<uint64_t>::max()) {
        base = end;
      } else {
        ranges.emplace_back(base + start, base + end);
      }
    }
    return ranges;
  }

  std::vector<std::pair<uint64_t, uint64_t>> readRanges5(uint64_t offset) {
    CheckedLexer L = s_.debug_rnglists.lexer(offset);
    uint64_t base = 0;
    LOG_INFO("BEGIN RANGES {:x}\n", offset);
    std::vector<std::pair<uint64_t, uint64_t>> ranges;
    while (true) {
      auto op = L.read<uint8_t>();
      switch (op) {
        case DW_RLE_end_of_list:
          LOG_INFO("END RANGES\n");
          return ranges;
        case DW_RLE_base_addressx: {
          base = readEncoded(L, DW_FORM_addrx);
          LOG_INFO("BASE ADDRX {:x}\n", base);
        } break;
        case DW_RLE_startx_length: {
          auto s = readEncoded(L, DW_FORM_addrx);
          auto e = L.readULEB128();
          LOG_INFO("startx_length {:x} {:x}\n", s, e);
          ranges.emplace_back(s, s + e);
        } break;
        case DW_RLE_base_address:
          base = L.read<uint64_t>();
          LOG_INFO("BASE ADDR {:x}\n", base);
          break;
        case DW_RLE_offset_pair: {
          auto s = L.readULEB128();
          auto e = L.readULEB128();
          LOG_INFO("offset_pair {:x} {:x}\n", s, e);
          ranges.emplace_back(base + s, base + e);
        } break;
        case DW_RLE_start_length: {
          auto s = L.read<uint64_t>();
          auto e = L.readULEB128();
          LOG_INFO("start_length {:x} {:x}\n", s, e);
          ranges.emplace_back(s, s + e);
        } break;
        default:
          UNWIND_CHECK(false, "unknown range op: {}", op);
      }
    }
  }

  CheckedLexer findAbbrev(uint64_t offset, uint64_t entry) {
    CheckedLexer L = s_.debug_abbrev.lexer(offset);
    while (true) {
      auto abbrev_code = L.readULEB128();
      UNWIND_CHECK(
          abbrev_code != 0,
          "could not find entry {} at offset {:x}",
          entry,
          offset);
      auto tag = L.readULEB128();
      L.read<uint8_t>(); // has children
      if (abbrev_code == entry) {
        UNWIND_CHECK(
            tag == DW_TAG_compile_unit,
            "first entry was not a compile unit but {}",
            tag);
        return L;
      }
      while (true) {
        auto attr = L.readULEB128();
        auto form = L.readULEB128();
        if (attr == 0 && form == 0) {
          break;
        }
        if (form == DW_FORM_implicit_const) {
          L.readSLEB128();
        }
      }
    }
  }

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  Sections& s_;
  std::optional<uint64_t> line_number_program_offset_;
  uint64_t offset_ = 0;
  uint8_t sec_offset_size_ = 0;
  uint64_t length_ = 0;
  const char* end_ = nullptr;
  uint64_t debug_abbrev_offset_ = 0;
  bool is_64bit_ = false;

  std::optional<std::pair<uint64_t, uint8_t>> range_ptr_;
  uint64_t lowpc_ = 0;
  std::optional<uint64_t> highpc_;
  uint16_t version_ = 0;
  uint64_t address_base_ = 0;
  std::optional<uint64_t> rnglists_base_;
};

} // namespace torch::unwind
