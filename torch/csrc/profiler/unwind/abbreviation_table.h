#pragma once
#include <torch/csrc/profiler/unwind/dwarf_symbolize_enums.h>
#include <torch/csrc/profiler/unwind/lexer.h>
#include <torch/csrc/profiler/unwind/mem_file.h>
#include <torch/csrc/profiler/unwind/unwind_error.h>
#include <algorithm>
#include <memory>
#include <vector>

namespace torch::unwind {

enum {
  CODE_ADVANCE,
  CODE_SKIP_STRING,
  CODE_SET_LOWPC,
  CODE_SET_BASEPC,
  CODE_SET_HIGHPC,
  CODE_SET_LINE_NUMBER_PROGRAM_OFFSET,
  CODE_SET_SUBPROGRAM_NAME,
  CODE_APPEND_ROW,
  CODE_JUMP_SIBLING,
  CODE_UNSUPPORTED,
  CODE_SKIP_SIZED_BLOCK,
  CODE_SKIP_SIZED_BLOCK1,
  CODE_SKIP_ULEB,
  CODE_CALL_ONE,
  CODE_READ_RANGES,
};

struct AbbrevationTable {
  AbbrevationTable() {}
  void parse(Section debug_abbrev, uint64_t offset, uint8_t sec_offset_size) {
    offset_ = offset;
    sec_offset_size_ = sec_offset_size;
    CheckedLexer data = debug_abbrev.lexer(offset);
    while (true) {
      auto abbrev_code = data.readULEB128();
      if (abbrev_code == 0) {
        break;
      }
      if (entries_.size() <= abbrev_code) {
        entries_.resize(abbrev_code + 1);
      }
      auto& entry = entries_.at(abbrev_code);
      entry.tag = data.readULEB128();
      size_t length = 0;
      data.read<uint8_t>(); // has_children
      bool skip = false;

      auto emit_code = [&](std::initializer_list<int> ops) {
        if (length > 0) {
          entry.codes.push_back(CODE_ADVANCE);
          entry.codes.push_back(length);
          length = 0;
        }
        for (auto op : ops) {
          entry.codes.push_back(op);
        }
      };

      bool has_name = false;
      bool has_low = false;
      bool has_high = false;

      while (true) {
        auto attr = data.readULEB128();
        auto form = data.readULEB128();
        if (attr == 0 && form == 0) {
          break;
        }
        if (form == DW_FORM_implicit_const) {
          data.readSLEB128();
        }
        if (skip) {
          continue;
        }
        if (attr == DW_AT_sibling) {
          emit_code({CODE_JUMP_SIBLING, int(form)});
          skip = true;
        } else if (attr == DW_AT_low_pc) {
          if (entry.tag == DW_TAG_compile_unit) {
            emit_code({CODE_SET_BASEPC, int(form)});
          } else {
            emit_code({CODE_SET_LOWPC, int(form)});
            has_low = true;
          }
        } else if (attr == DW_AT_high_pc) {
          emit_code({CODE_SET_HIGHPC, int(form)});
          has_high = true;
        } else if (
            (attr == DW_AT_name && entry.tag == DW_TAG_subprogram) ||
            attr == DW_AT_linkage_name) {
          emit_code({CODE_SET_SUBPROGRAM_NAME, int(form)});
          has_name = true;
        } else if (form == DW_FORM_string) {
          emit_code({CODE_SKIP_STRING});
        } else if (attr == DW_AT_stmt_list) {
          emit_code({CODE_SET_LINE_NUMBER_PROGRAM_OFFSET, int(form)});
        } else if (form == DW_FORM_exprloc) {
          emit_code({CODE_SKIP_SIZED_BLOCK});
        } else if (form == DW_FORM_block1) {
          emit_code({CODE_SKIP_SIZED_BLOCK1});
        } else if (form == DW_FORM_sdata || form == DW_FORM_udata) {
          emit_code({CODE_SKIP_ULEB});
        } else if (
            (attr == DW_AT_specification || attr == DW_AT_abstract_origin) &&
            entry.tag == DW_TAG_subprogram) {
          emit_code({CODE_CALL_ONE});
          has_name = true;
        } else if (
            attr == DW_AT_ranges && entry.tag == DW_TAG_subprogram &&
            form == DW_FORM_sec_offset) {
          emit_code({CODE_READ_RANGES});
          has_low = true;
          has_high = true;
        } else {
          auto sz = formSize(form, sec_offset_size_);
          if (!sz) {
            emit_code({CODE_UNSUPPORTED});
            skip = true;
          } else {
            length += sz.value();
          }
        }
      }
      // flush any final skip
      emit_code({});
      if (entry.tag == DW_TAG_subprogram && has_low && has_high && has_name) {
        emit_code({CODE_APPEND_ROW});
      }
    }
  }

  struct Entry {
    int64_t tag;
    std::vector<int> codes;
  };

  std::vector<Entry> entries_;
  uint8_t sec_offset_size_;
  uint64_t offset_;
};

} // namespace torch::unwind
