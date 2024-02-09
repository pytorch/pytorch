#pragma once
#include <c10/util/irange.h>
#include <torch/csrc/profiler/unwind/action.h>
#include <torch/csrc/profiler/unwind/lexer.h>
#include <array>
#include <iostream>
#include <sstream>
#include <vector>

struct TableState {
  Action cfa;
  std::array<Action, D_REG_SIZE> registers;
  friend std::ostream& operator<<(std::ostream& out, const TableState& self) {
    out << "cfa = " << self.cfa << "; ";
    for (auto r : c10::irange(self.registers.size())) {
      if (self.registers.at(r).kind != A_UNDEFINED) {
        out << "r" << r << " = " << self.registers.at(r) << "; ";
      }
    }
    return out;
  }
};

// FDE - Frame Description Entry (Concept in ELF spec)
// This format is explained well by
// https://www.airs.com/blog/archives/460
// Details of different dwarf actions are explained
// in the spec document:
// https://web.archive.org/web/20221129184704/https://dwarfstd.org/doc/DWARF4.doc
// An overview of how DWARF unwinding works is given in
// https://dl.acm.org/doi/pdf/10.1145/3360572
// A similar implementation written in rust is:
// https://github.com/mstange/framehop/

template <bool LOG = false>
struct FDE {
  FDE(void* data, const char* library_name, uint64_t load_bias)
      : library_name_(library_name), load_bias_(load_bias) {
    Lexer L(data);
    auto length = L.read4or8Length();
    void* fde_start = L.loc();
    void* cie_data = (void*)((int64_t)fde_start - L.read<uint32_t>());
    Lexer LC(cie_data);
    auto cie_length = LC.read4or8Length();
    void* cie_start = LC.loc();
    auto zero = LC.read<uint32_t>();
    TORCH_INTERNAL_ASSERT(zero == 0, "expected 0 for CIE");
    auto version = LC.read<uint8_t>();
    TORCH_INTERNAL_ASSERT(
        version == 1 || version == 3, "non-1 version for CIE");
    augmentation_string_ = LC.readCString();
    if (hasAugmentation("eh")) {
      throw UnwindError("unsupported 'eh' augmentation string");
    }
    code_alignment_factor_ = LC.readULEB128();
    data_alignment_factor_ = LC.readSLEB128();
    if (version == 1) {
      ra_register_ = LC.read<uint8_t>();
    } else {
      ra_register_ = LC.readULEB128();
    }
    // we assume this in the state
    TORCH_INTERNAL_ASSERT(ra_register_ == 16, "unexpected number of registers");
    if (augmentation_string_ && *augmentation_string_ == 'z') {
      augmentation_length_ = LC.readULEB128();
      Lexer A(LC.loc());
      for (auto ap = augmentation_string_ + 1; *ap; ap++) {
        switch (*ap) {
          case 'L':
            lsda_enc = A.read<uint8_t>();
            break;
          case 'R':
            fde_enc = A.read<uint8_t>();
            break;
          case 'P': {
            uint8_t personality_enc = A.read<uint8_t>();
            A.readEncoded(personality_enc);
          } break;
          case 'S': {
            // signal handler
          } break;
          default: {
            throw UnwindError("unknown augmentation string");
          } break;
        }
      }
    }
    LC.skip(augmentation_length_);
    low_pc_ = L.readEncoded(fde_enc);
    high_pc_ = low_pc_ + L.readEncodedValue(fde_enc);

    if (hasAugmentation("z")) {
      augmentation_length_fde_ = L.readULEB128();
    }
    L.readEncodedOr(lsda_enc, 0);

    cie_begin_ = LC.loc();
    fde_begin_ = L.loc();
    cie_end_ = (void*)((const char*)cie_start + cie_length);
    fde_end_ = (void*)((const char*)fde_start + length);
  }

  // OP Code implementations

  void advance_raw(int64_t amount) {
    auto previous_pc = current_pc_;
    current_pc_ += amount;
    if (LOG) {
      (*out_) << (void*)(previous_pc - load_bias_) << "-"
              << (void*)(current_pc_ - load_bias_) << ": " << state() << "\n";
    }
  }

  void advance_loc(int64_t amount) {
    if (LOG) {
      (*out_) << "advance_loc " << amount << "\n";
    }
    advance_raw(amount * code_alignment_factor_);
  }

  void offset(int64_t reg, int64_t offset) {
    if (LOG) {
      (*out_) << "offset " << reg << " " << offset << "\n";
    }
    if (reg > (int64_t)state().registers.size()) {
      if (LOG) {
        (*out_) << "OFFSET OF BIG REGISTER " << reg << "ignored...\n";
      }
      return;
    }
    state().registers.at(reg) =
        Action{A_LOAD_CFA_OFFSET, -1, offset * data_alignment_factor_};
  }

  void restore(int64_t reg) {
    if (LOG) {
      (*out_) << "restore " << reg << "\n";
    }
    if (reg > (int64_t)state().registers.size()) {
      if (LOG) {
        (*out_) << "RESTORE OF BIG REGISTER " << reg << "ignored...\n";
      }
      return;
    }
    state().registers.at(reg) = initial_state_.registers.at(reg);
  }

  void def_cfa(int64_t reg, int64_t off) {
    if (LOG) {
      (*out_) << "def_cfa " << reg << " " << off << "\n";
    }
    last_reg_ = reg;
    last_offset_ = off;
    state().cfa = Action::regPlusData(reg, off);
  }
  void def_cfa_register(int64_t reg) {
    def_cfa(reg, last_offset_);
  }
  void def_cfa_offset(int64_t off) {
    def_cfa(last_reg_, off);
  }

  void remember_state() {
    if (LOG) {
      (*out_) << "remember_state\n";
    }
    state_stack_.push_back(state());
  }
  void restore_state() {
    if (LOG) {
      (*out_) << "restore_state\n";
    }
    state_stack_.pop_back();
  }

  void undefined(int64_t reg) {
    if (LOG) {
      (*out_) << "undefined " << reg << "\n";
    }
    state().registers.at(reg) = Action::undefined();
  }
  void register_(int64_t reg, int64_t rhs_reg) {
    if (LOG) {
      (*out_) << "register " << reg << " " << rhs_reg << "\n";
    }
    state().registers.at(reg) = Action::regPlusData(reg, 0);
  }

  TableState& state() {
    return state_stack_.back();
  }

  void dump(std::ostream& out) {
    out_ = &out;
    out << "FDE(augmentation_string=" << augmentation_string_
        << ", low_pc=" << (void*)(low_pc_ - load_bias_)
        << ",high_pc=" << (void*)(high_pc_ - load_bias_)
        << ",code_alignment_factor=" << code_alignment_factor_
        << ", data_alignment_factor=" << data_alignment_factor_
        << ", ra_register_=" << ra_register_ << ")\n";
    readUpTo(high_pc_);
    out_ = &std::cout;
  }

  TableState readUpTo(uint64_t addr) {
    if (addr < low_pc_ || addr > high_pc_) {
      throw UnwindError("Address not in range");
    }
    if (LOG) {
      (*out_) << "readUpTo " << (void*)addr << " for " << library_name_
              << " at " << (void*)load_bias_ << "\n";
    }
    state_stack_.emplace_back();
    current_pc_ = low_pc_;
    // parse instructions...
    Lexer LC(cie_begin_);
    while (LC.loc() < cie_end_ && current_pc_ <= addr) {
      readInstruction(LC);
    }
    if (current_pc_ > addr) {
      return state();
    }

    initial_state_ = state_stack_.back();

    if (LOG) {
      (*out_) << "--\n";
    }

    Lexer L(fde_begin_);
    while (L.loc() < fde_end_ && current_pc_ <= addr) {
      readInstruction(L);
    }
    // so that we print the full range in debugging
    if (current_pc_ <= addr) {
      advance_raw(addr - current_pc_);
    }
    return state();
  }

  void dumpAddr2Line() {
    std::cout << "addr2line -f -e " << library_name_ << " "
              << (void*)(low_pc_ - load_bias_) << "\n";
  }

  void readInstruction(Lexer& L) {
    uint8_t bc = L.read<uint8_t>();
    auto op = bc >> 6;
    auto lowbits = bc & 0x3F;
    switch (op) {
      case 0x0: {
        switch (lowbits) {
          case DW_CFA_nop: {
            return; // nop
          }
          case DW_CFA_advance_loc1: {
            auto delta = L.read<uint8_t>();
            return advance_loc(delta);
          }
          case DW_CFA_advance_loc2: {
            auto delta = L.read<uint16_t>();
            return advance_loc(delta);
          }
          case DW_CFA_advance_loc4: {
            auto delta = L.read<uint32_t>();
            return advance_loc(delta);
          }
          case DW_CFA_restore_extended: {
            auto reg = L.readULEB128();
            return restore(reg);
          }
          case DW_CFA_undefined: {
            auto reg = L.readULEB128();
            return undefined(reg);
          }
          case DW_CFA_register: {
            auto reg = L.readULEB128();
            auto rhs_reg = L.readULEB128();
            return register_(reg, rhs_reg);
          }
          case DW_CFA_def_cfa: {
            auto reg = L.readULEB128();
            auto off = L.readULEB128();
            return def_cfa(reg, off);
          }
          case DW_CFA_def_cfa_register: {
            auto reg = L.readULEB128();
            return def_cfa_register(reg);
          }
          case DW_CFA_def_cfa_offset: {
            auto off = L.readULEB128();
            return def_cfa_offset(off);
          }
          case DW_CFA_offset_extended_sf: {
            auto reg = L.readULEB128();
            auto off = L.readSLEB128();
            return offset(reg, off);
          }
          case DW_CFA_remember_state: {
            return remember_state();
          }
          case DW_CFA_restore_state: {
            return restore_state();
          }
          case DW_CFA_GNU_args_size: {
            // GNU_args_size, we do not need to know it..
            L.readULEB128();
            return;
          }
          case DW_CFA_expression: {
            auto reg = L.readULEB128();
            auto len = L.readULEB128();
            auto end = (void*)((uint64_t)L.loc() + len);
            auto op = L.read<uint8_t>();
            if ((op & 0xF0) == 0x70) { // DW_bregX
              auto rhs_reg = (op & 0xF);
              auto addend = L.readSLEB128();
              if (L.loc() == end) {
                state().registers.at(reg) =
                    Action::regPlusDataDeref(rhs_reg, addend);
                return;
              }
            }
            throw UnwindError("Unsupported dwarf expression");
          }
          case DW_CFA_def_cfa_expression: {
            auto len = L.readULEB128();
            auto end = (void*)((uint64_t)L.loc() + len);
            auto op = L.read<uint8_t>();
            if ((op & 0xF0) == 0x70) { // DW_bregX
              auto rhs_reg = (op & 0xF);
              auto addend = L.readSLEB128();
              if (L.loc() != end) {
                auto op2 = L.read<uint8_t>();
                if (op2 == DW_OP_deref && L.loc() == end) { // deref
                  state().cfa = Action::regPlusDataDeref(rhs_reg, addend);
                  return;
                }
              }
            }
            throw UnwindError("Unsupported def_cfa dwarf expression");
          }
          default: {
            std::stringstream ss;
            ss << "unknown op code " << (void*)(uint64_t)lowbits;
            throw UnwindError(ss.str());
          }
        }
      }
      case DW_CFA_advance_loc: {
        return advance_loc(lowbits);
      }
      case DW_CFA_offset: {
        auto off = L.readULEB128();
        return offset(lowbits, off);
      }
      case DW_CFA_restore: {
        return restore(lowbits);
      }
    }
  }
  // used for debug printing
  const char* library_name_;
  uint64_t load_bias_;

  // parsed from the eh_string data structures:
  const char* augmentation_string_ = nullptr;
  int64_t augmentation_length_ = 0;
  int64_t augmentation_length_fde_ = 0;

  int64_t code_alignment_factor_;
  int64_t data_alignment_factor_;
  void* cie_data_;

  int64_t ra_register_;
  uint8_t lsda_enc = DW_EH_PE_omit;
  uint8_t fde_enc = DW_EH_PE_absptr;
  uint64_t low_pc_ = UINT64_MAX;
  uint64_t high_pc_ = UINT64_MAX;

  void* cie_begin_;
  void* fde_begin_;
  void* cie_end_;
  void* fde_end_;

  // state accumulated while parsing instructions
  int64_t last_reg_ = 0;
  int64_t last_offset_ = 0;
  uint64_t current_pc_;

  TableState
      initial_state_; // state after the initial instructions, used by restore
  std::vector<TableState> state_stack_;

  std::ostream* out_ = &std::cout; // for debug dumping
 private:
  bool hasAugmentation(const char* s) {
    return strstr(augmentation_string_, s) != nullptr;
  }
};
