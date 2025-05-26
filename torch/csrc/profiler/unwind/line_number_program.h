#include <c10/util/irange.h>
#include <torch/csrc/profiler/unwind/debug_info.h>
#include <torch/csrc/profiler/unwind/dwarf_enums.h>
#include <torch/csrc/profiler/unwind/dwarf_symbolize_enums.h>
#include <torch/csrc/profiler/unwind/lexer.h>
#include <torch/csrc/profiler/unwind/sections.h>
#include <torch/csrc/profiler/unwind/unwind_error.h>
#include <tuple>

namespace torch::unwind {

struct LineNumberProgram {
  LineNumberProgram(Sections& s, uint64_t offset) : s_(s), offset_(offset) {}

  uint64_t offset() {
    return offset_;
  }
  void parse() {
    if (parsed_) {
      return;
    }
    parsed_ = true;
    CheckedLexer L = s_.debug_line.lexer(offset_);
    std::tie(length_, is_64bit_) = L.readSectionLength();
    program_end_ = (char*)L.loc() + length_;
    auto version = L.read<uint16_t>();
    UNWIND_CHECK(
        version == 5 || version == 4,
        "expected version 4 or 5 but found {}",
        version);
    if (version == 5) {
      auto address_size = L.read<uint8_t>();
      UNWIND_CHECK(
          address_size == 8,
          "expected 64-bit dwarf but found address size {}",
          address_size);
      segment_selector_size_ = L.read<uint8_t>();
    }
    header_length_ = is_64bit_ ? L.read<uint64_t>() : L.read<uint32_t>();
    program_ = L;
    program_.skip(int64_t(header_length_));
    minimum_instruction_length_ = L.read<uint8_t>();
    maximum_operations_per_instruction_ = L.read<uint8_t>();
    default_is_stmt_ = L.read<uint8_t>();
    line_base_ = L.read<int8_t>();
    line_range_ = L.read<uint8_t>();
    opcode_base_ = L.read<uint8_t>();
    UNWIND_CHECK(line_range_ != 0, "line_range_ must be non-zero");
    standard_opcode_lengths_.resize(opcode_base_);
    for (size_t i = 1; i < opcode_base_; i++) {
      standard_opcode_lengths_[i] = L.read<uint8_t>();
    }
    // fmt::print("{:x} {:x} {} {} {} {} {}\n", offset_, header_length_,
    // minimum_instruction_length_, maximum_operations_per_instruction_,
    // line_base_, line_range_, opcode_base_);
    uint8_t directory_entry_format_count = L.read<uint8_t>();

    if (version == 5) {
      struct Member {
        uint64_t content_type;
        uint64_t form;
      };
      std::vector<Member> directory_members;
      directory_members.reserve(directory_entry_format_count);
      for (size_t i = 0; i < directory_entry_format_count; i++) {
        directory_members.push_back({L.readULEB128(), L.readULEB128()});
      }
      uint64_t directories_count = L.readULEB128();
      for (size_t i = 0; i < directories_count; i++) {
        for (auto& member : directory_members) {
          switch (member.content_type) {
            case DW_LNCT_path: {
              include_directories_.emplace_back(
                  s_.readString(L, member.form, is_64bit_));
            } break;
            default: {
              skipForm(L, member.form);
            } break;
          }
        }
      }

      for (auto i : c10::irange(directories_count)) {
        (void)i;
        LOG_INFO("{} {}\n", i, include_directories_[i]);
      }
      auto file_name_entry_format_count = L.read<uint8_t>();
      std::vector<Member> file_members;
      file_members.reserve(file_name_entry_format_count);
      for (size_t i = 0; i < file_name_entry_format_count; i++) {
        file_members.push_back({L.readULEB128(), L.readULEB128()});
      }
      auto files_count = L.readULEB128();
      for (size_t i = 0; i < files_count; i++) {
        for (auto& member : file_members) {
          switch (member.content_type) {
            case DW_LNCT_path: {
              file_names_.emplace_back(
                  s_.readString(L, member.form, is_64bit_));
            } break;
            case DW_LNCT_directory_index: {
              file_directory_index_.emplace_back(readData(L, member.form));
              UNWIND_CHECK(
                  file_directory_index_.back() < include_directories_.size(),
                  "directory index out of range");
            } break;
            default: {
              skipForm(L, member.form);
            } break;
          }
        }
      }
      for (auto i : c10::irange(files_count)) {
        (void)i;
        LOG_INFO("{} {} {}\n", i, file_names_[i], file_directory_index_[i]);
      }
    } else {
      include_directories_.emplace_back(""); // implicit cwd
      while (true) {
        auto str = L.readCString();
        if (*str == '\0') {
          break;
        }
        include_directories_.emplace_back(str);
      }
      file_names_.emplace_back("");
      file_directory_index_.emplace_back(0);
      while (true) {
        auto str = L.readCString();
        if (*str == '\0') {
          break;
        }
        auto directory_index = L.readULEB128();
        L.readULEB128(); // mod_time
        L.readULEB128(); // file_length
        file_names_.emplace_back(str);
        file_directory_index_.push_back(directory_index);
      }
    }
    UNWIND_CHECK(
        maximum_operations_per_instruction_ == 1,
        "maximum_operations_per_instruction_ must be 1");
    UNWIND_CHECK(
        minimum_instruction_length_ == 1,
        "minimum_instruction_length_ must be 1");
    readProgram();
  }
  struct Entry {
    uint32_t file = 1;
    int64_t line = 1;
  };
  std::optional<Entry> find(uint64_t address) {
    auto e = program_index_.find(address);
    if (!e) {
      return std::nullopt;
    }
    return all_programs_.at(*e).find(address);
  }
  std::string filename(uint64_t index) {
    return fmt::format(
        "{}/{}",
        include_directories_.at(file_directory_index_.at(index)),
        file_names_.at(index));
  }

 private:
  void skipForm(CheckedLexer& L, uint64_t form) {
    auto sz = formSize(form, is_64bit_ ? 8 : 4);
    UNWIND_CHECK(sz, "unsupported form {}", form);
    L.skip(int64_t(*sz));
  }

  uint64_t readData(CheckedLexer& L, uint64_t encoding) {
    switch (encoding) {
      case DW_FORM_data1:
        return L.read<uint8_t>();
      case DW_FORM_data2:
        return L.read<uint16_t>();
      case DW_FORM_data4:
        return L.read<uint32_t>();
      case DW_FORM_data8:
        return L.read<uint64_t>();
      case DW_FORM_udata:
        return L.readULEB128();
      default:
        UNWIND_CHECK(false, "unsupported data encoding {}", encoding);
    }
  }

  void produceEntry() {
    if (shadow_) {
      return;
    }
    if (ranges_.size() == 1) {
      start_address_ = address_;
    }
    PRINT_LINE_TABLE(
        "{:x}\t{}\t{}\n", address_, filename(entry_.file), entry_.line);
    UNWIND_CHECK(
        entry_.file < file_names_.size(),
        "file index {} > {} entries",
        entry_.file,
        file_names_.size());
    ranges_.add(address_, entry_, true);
  }
  void endSequence() {
    if (shadow_) {
      return;
    }
    PRINT_LINE_TABLE(
        "{:x}\tEND\n", address_, filename(entry_.file), entry_.line);
    program_index_.add(start_address_, all_programs_.size(), false);
    program_index_.add(address_, std::nullopt, false);
    all_programs_.emplace_back(std::move(ranges_));
    ranges_ = RangeTable<Entry>();
  }
  void readProgram() {
    while (program_.loc() < program_end_) {
      PRINT_INST("{:x}: ", (char*)program_.loc() - (s_.debug_line.data));
      uint8_t op = program_.read<uint8_t>();
      if (op >= opcode_base_) {
        auto op2 = int64_t(op - opcode_base_);
        address_ += op2 / line_range_;
        entry_.line += line_base_ + (op2 % line_range_);
        PRINT_INST(
            "address += {}, line += {}\n",
            op2 / line_range_,
            line_base_ + (op2 % line_range_));
        produceEntry();
      } else {
        switch (op) {
          case DW_LNS_extended_op: {
            auto len = program_.readULEB128();
            auto extended_op = program_.read<uint8_t>();
            switch (extended_op) {
              case DW_LNE_end_sequence: {
                PRINT_INST("end_sequence\n");
                endSequence();
                entry_ = Entry{};
              } break;
              case DW_LNE_set_address: {
                address_ = program_.read<uint64_t>();
                if (!shadow_) {
                  PRINT_INST(
                      "set address {:x} {:x} {:x}\n",
                      address_,
                      min_address_,
                      max_address_);
                }
                shadow_ = address_ == 0;
              } break;
              default: {
                PRINT_INST("skip extended op {}\n", extended_op);
                program_.skip(int64_t(len - 1));
              } break;
            }
          } break;
          case DW_LNS_copy: {
            PRINT_INST("copy\n");
            produceEntry();
          } break;
          case DW_LNS_advance_pc: {
            PRINT_INST("advance pc\n");
            address_ += program_.readULEB128();
          } break;
          case DW_LNS_advance_line: {
            entry_.line += program_.readSLEB128();
            PRINT_INST("advance line {}\n", entry_.line);

          } break;
          case DW_LNS_set_file: {
            PRINT_INST("set file\n");
            entry_.file = program_.readULEB128();
          } break;
          case DW_LNS_const_add_pc: {
            PRINT_INST("const add pc\n");
            address_ += (255 - opcode_base_) / line_range_;
          } break;
          case DW_LNS_fixed_advance_pc: {
            PRINT_INST("fixed advance pc\n");
            address_ += program_.read<uint16_t>();
          } break;
          default: {
            PRINT_INST("other {}\n", op);
            auto n = standard_opcode_lengths_[op];
            for (int i = 0; i < n; ++i) {
              program_.readULEB128();
            }
          } break;
        }
      }
    }
    PRINT_INST(
        "{:x}: end {:x}\n",
        ((char*)program_.loc() - s_.debug_line.data),
        program_end_ - s_.debug_line.data);
  }

  uint64_t address_ = 0;
  bool shadow_ = false;
  bool parsed_ = false;
  Entry entry_ = {};
  std::vector<std::string> include_directories_;
  std::vector<std::string> file_names_;
  std::vector<uint64_t> file_directory_index_;
  uint8_t segment_selector_size_ = 0;
  uint8_t minimum_instruction_length_ = 0;
  uint8_t maximum_operations_per_instruction_ = 0;
  int8_t line_base_ = 0;
  uint8_t line_range_ = 0;
  uint8_t opcode_base_ = 0;
  bool default_is_stmt_ = false;
  CheckedLexer program_ = {nullptr};
  char* program_end_ = nullptr;
  uint64_t header_length_ = 0;
  uint64_t length_ = 0;
  bool is_64bit_ = false;
  std::vector<uint8_t> standard_opcode_lengths_;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  Sections& s_;
  uint64_t offset_;
  uint64_t start_address_ = 0;
  RangeTable<uint64_t> program_index_;
  std::vector<RangeTable<Entry>> all_programs_;
  RangeTable<Entry> ranges_;
};

} // namespace torch::unwind
