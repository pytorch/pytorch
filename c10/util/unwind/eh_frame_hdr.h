#pragma once
#include <stdint.h>
#include <iostream>

#include <c10/util/unwind/lexer.h>
#include <c10/util/unwind/unwind_error.h>

// Overview of the format described in
// https://refspecs.linuxfoundation.org/LSB_1.3.0/gLSB/gLSB/ehframehdr.html

struct EHFrameHdr {
  EHFrameHdr(void* base) : base_(base) {
    Lexer L(base, base);
    version_ = L.read<uint8_t>();
    eh_frame_ptr_enc_ = L.read<uint8_t>();
    fde_count_enc_ = L.read<uint8_t>();
    table_enc_ = L.read<uint8_t>();
    if (table_enc_ == DW_EH_PE_omit) {
      table_size_ = 0;
    } else {
      switch (table_enc_ & 0xF) {
        case DW_EH_PE_udata2:
        case DW_EH_PE_sdata2:
          table_size_ = 2;
          break;
        case DW_EH_PE_udata4:
        case DW_EH_PE_sdata4:
          table_size_ = 4;
          break;
        case DW_EH_PE_udata8:
        case DW_EH_PE_sdata8:
          table_size_ = 8;
          break;
        case DW_EH_PE_uleb128:
        case DW_EH_PE_sleb128:
          throw UnwindError("uleb/sleb table encoding not supported");
          break;
        default:
          throw UnwindError("unknown table encoding");
      }
    }
    eh_frame_ = (void*)L.readEncodedOr(eh_frame_ptr_enc_, 0);
    fde_count_ = L.readEncodedOr(fde_count_enc_, 0);
    table_start_ = L.loc();
  }
  size_t nentries() const {
    return fde_count_;
  }

  uint64_t lowpc(size_t i) const {
    return Lexer(table_start_, base_)
        .skip(2 * i * table_size_)
        .readEncoded(table_enc_);
  }
  void* fde(size_t i) const {
    return (void*)Lexer(table_start_, base_)
        .skip((2 * i + 1) * table_size_)
        .readEncoded(table_enc_);
  }

  void* entryForAddr(uint64_t addr) const {
    if (!table_size_ || !nentries()) {
      throw UnwindError("search table not present");
    }
    uint64_t low = 0;
    uint64_t high = nentries();
    while (low + 1 < high) {
      auto mid = (low + high) / 2;
      if (addr < lowpc(mid)) {
        high = mid;
      } else {
        low = mid;
      }
    }
    return fde(low);
  }

  friend std::ostream& operator<<(std::ostream& out, const EHFrameHdr& self) {
    out << "EHFrameHeader(version=" << self.version_
        << ",table_size=" << self.table_size_
        << ",fde_count=" << self.fde_count_ << ")";
    return out;
  }

 private:
  void* base_;
  void* table_start_;
  uint8_t version_;
  uint8_t eh_frame_ptr_enc_;
  uint8_t fde_count_enc_;
  uint8_t table_enc_;
  void* eh_frame_ = nullptr;
  int64_t fde_count_;
  uint32_t table_size_;
};
