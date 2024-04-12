#pragma once
#include <cxxabi.h>
#include <torch/csrc/profiler/unwind/dwarf_enums.h>
#include <torch/csrc/profiler/unwind/lexer.h>
#include <torch/csrc/profiler/unwind/sections.h>
#include <torch/csrc/profiler/unwind/unwind_error.h>
#include <cstdint>

namespace torch::unwind {

static std::string demangle(const std::string& mangled_name) {
  int status = 0;
  char* realname =
      abi::__cxa_demangle(mangled_name.c_str(), nullptr, nullptr, &status);
  if (status == 0) {
    std::string demangled_name(realname);
    // NOLINTNEXTLINE
    free(realname);
    return demangled_name;
  } else {
    return mangled_name;
  }
}

struct DebugInfo {
  DebugInfo(Sections& s) : s_(s) {}

  void parse(uint64_t offset) {
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
    abbrevation_table_ =
        s_.getAbbrevByTableOffset(debug_abbrev_offset_, sec_offset_size_);

    while (readOne(L, abbrevation_table_))
      ;
    std::sort(rows_.begin(), rows_.end(), [](const auto& a, const auto& b) {
      return a.lowpc < b.lowpc;
    });
    if (rows_.empty()) {
      return;
    }
    for (auto it = rows_.begin(); it != rows_.end(); it++) {
      if (it != rows_.begin() && it[-1].highpc < it->lowpc) {
        table_.add(it[-1].highpc, std::nullopt, true);
      }
      table_.add(it->lowpc, it->name, true);
    }
    table_.add(rows_.back().highpc, std::nullopt, true);
    rows_.clear();
  }

  unwind::optional<std::string> getSubprogramName(uint64_t pc) {
    if (auto e = table_.find(pc)) {
      return demangle(std::string(*e));
    }
    return std::nullopt;
  }
  unwind::optional<uint64_t> lineNumberProgramOffset() {
    return line_number_program_offset_;
  }

 private:
  const char* readString(CheckedLexer& L, int encoding) {
    return s_.readString(L, encoding, is_64bit_);
  }

  uint64_t readSegmentOffset(CheckedLexer& L) {
    return s_.readSegmentOffset(L, is_64bit_);
  }

  bool readOne(CheckedLexer& L, AbbrevationTable* abbrev) {
    auto& entries = abbrev->entries_;
    auto entry = L.readULEB128();
    if (entry == 0) {
      LOG_INFO("{:x}: DONE\n", (start - s_.debug_info.data));
      return false;
    }
    UNWIND_CHECK(entry < entries.size(), "missing abbrev_code {:x}", entry);
    const auto& e = entries[entry];
    LOG_INFO(
        "{:x}: entry {} tag {}\n", (start - s_.debug_info.data), entry, e.tag);
    auto pc = e.codes.begin();
    while (pc != e.codes.end()) {
      switch (*pc++) {
        case CODE_SET_LOWPC: {
          auto encoding = *pc++;
          lowpc_ = L.read<uint64_t>();
          LOG_INFO("  lowpc {:x}\n", lowpc_);
        } break;
        case CODE_SET_BASEPC: {
          auto encoding = *pc++;
          cu_base_pc_ = L.read<uint64_t>();
          LOG_INFO("  base {:x}\n", cu_base_pc_);
        } break;
        case CODE_SET_HIGHPC: {
          auto encoding = *pc++;
          auto highpc = L.read<uint64_t>();
          ranges_.clear();
          ranges_.emplace_back(std::make_pair(lowpc_, lowpc_ + highpc));
          LOG_INFO("  highpc {:x}\n", highpc);
        } break;
        case CODE_SET_LINE_NUMBER_PROGRAM_OFFSET: {
          auto encoding = *pc++;
          line_number_program_offset_ = readSegmentOffset(L);
          LOG_INFO(
              "  program table offset {:x}\n", *line_number_program_offset_);
        } break;
        case CODE_SET_SUBPROGRAM_NAME: {
          auto encoding = *pc++;
          subprogram_name_ = readString(L, encoding);
          LOG_INFO("  subprogram name {}\n", subprogram_name_);
        } break;
        case CODE_ADVANCE: {
          L.skip(*pc++);
        } break;
        case CODE_SKIP_STRING: {
          while (0 != L.read<uint8_t>()) {
          }
        } break;
        case CODE_APPEND_ROW: {
          UNWIND_CHECK(subprogram_name_ != nullptr, "NULL SUBPROGRAM NAME\n");
          for (auto& r : ranges_) {
            rows_.emplace_back(Row{r.first, r.second, subprogram_name_});
          }
          ranges_.clear();
          LOG_INFO("APPEND ROW\n");
        } break;
        case CODE_JUMP_SIBLING: {
          auto encode = *pc++;
          auto off = readSegmentOffset(L);
          L = s_.debug_info.lexer(offset_ + off);
          LOG_INFO("SIBLING JUMP\n");
        } break;
        case CODE_UNSUPPORTED: {
          UNWIND_CHECK(false, "hit an unsupported encoding kind");
        } break;
        case CODE_SKIP_SIZED_BLOCK: {
          auto sz = L.readULEB128();
          L.skip(sz);
        } break;
        case CODE_SKIP_SIZED_BLOCK1: {
          auto sz = L.read<uint8_t>();
          L.skip(sz);
        } break;
        case CODE_SKIP_ULEB: {
          L.readULEB128();
        } break;
        case CODE_CALL_ONE: {
          auto off = readSegmentOffset(L);
          auto sub = s_.debug_info.lexer(offset_ + off);
          auto abbrev = (sub.loc() < end_) ? abbrevation_table_
                                           : s_.getAbbrev(offset_ + off);
          LOG_INFO("ENTER CALL off={:x} abbrev={:x}\n", off, abbrev->offset_);
          readOne(sub, abbrev);
          LOG_INFO("EXIT CALL\n");
        } break;
        case CODE_READ_RANGES: {
          auto range_offset = readSegmentOffset(L);
          readRanges(range_offset);
        } break;
        default: {
          UNWIND_CHECK(false, "unknown code: {}", *(pc - 1));
        } break;
      }
    }
    return true;
  }

  void readRanges(uint64_t offset) {
    if (version_ == 4) {
      return readRanges4(offset);
    } else {
      return readRanges5(offset);
    }
  }

  void readRanges4(uint64_t offset) {
    CheckedLexer L = s_.debug_ranges.lexer(offset);
    uint64_t base = cu_base_pc_;
    while (true) {
      auto start = L.read<uint64_t>();
      auto end = L.read<uint64_t>();
      if (start == 0 && end == 0) {
        break;
      }
      if (start == std::numeric_limits<uint64_t>::max()) {
        base = end;
      } else {
        ranges_.emplace_back(std::make_pair(base + start, base + end));
      }
    }
  }
  void readRanges5(uint64_t offset) {
    CheckedLexer L = s_.debug_rnglists.lexer(offset);
    uint64_t base = 0;
    LOG_INFO("BEGIN RANGES {:x}\n", offset);
    while (true) {
      auto op = L.read<uint8_t>();
      switch (op) {
        case DW_RLE_end_of_list:
          LOG_INFO("END RANGES\n");
          return;
        case DW_RLE_base_address:
          base = L.read<uint64_t>();
          LOG_INFO("BASE ADDR {:x}\n", base);
          break;
        case DW_RLE_offset_pair: {
          auto s = L.readULEB128();
          auto e = L.readULEB128();
          LOG_INFO("offset_pair {:x} {:x}\n", s, e);
          ranges_.emplace_back(std::make_pair(base + s, base + e));
        } break;
        case DW_RLE_start_length: {
          auto s = L.read<uint64_t>();
          auto e = L.readULEB128();
          LOG_INFO("start_length {:x} {:x}\n", s, e);
          ranges_.emplace_back(std::make_pair(s, s + e));
        } break;
        default:
          UNWIND_CHECK(false, "unknown range op: {}", op);
      }
    }
  }
  Sections& s_;
  unwind::optional<uint64_t> line_number_program_offset_;
  uint64_t offset_;
  uint8_t sec_offset_size_;
  uint64_t length_;
  const char* end_;
  uint64_t debug_abbrev_offset_;
  bool is_64bit_;
  AbbrevationTable* abbrevation_table_;
  const char* subprogram_name_;
  std::vector<std::pair<uint64_t, uint64_t>> ranges_;
  uint64_t lowpc_;
  uint64_t cu_base_pc_ = 0;
  uint16_t version_;
  struct Row {
    uint64_t lowpc;
    uint64_t highpc;
    const char* name;
  };
  std::vector<Row> rows_;
  RangeTable<const char*> table_;
};

} // namespace torch::unwind
