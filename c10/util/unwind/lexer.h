#pragma once
#include <stdint.h>
#include <string.h>

#include <c10/util/unwind/dwarf_enums.h>
#include <c10/util/unwind/unwind_error.h>

struct Lexer {
  Lexer(void* data, void* base = nullptr)
      : next_((const char*)data), base_((int64_t)base) {}

  template <typename T>
  T read() {
    T result;
    memcpy(&result, next_, sizeof(T));
    next_ += sizeof(T);
    return result;
  }

  // SLEB/ULEB code adapted from LLVM equivalents
  int64_t readSLEB128() {
    int64_t Value = 0;
    unsigned Shift = 0;
    uint8_t Byte;
    do {
      Byte = read<uint8_t>();
      uint64_t Slice = Byte & 0x7f;
      if ((Shift >= 64 && Slice != (Value < 0 ? 0x7f : 0x00)) ||
          (Shift == 63 && Slice != 0 && Slice != 0x7f)) {
        throw UnwindError("sleb128 too big for int64");
      }
      Value |= Slice << Shift;
      Shift += 7;
    } while (Byte >= 128);
    // Sign extend negative numbers if needed.
    if (Shift < 64 && (Byte & 0x40)) {
      Value |= (-1ULL) << Shift;
    }
    return Value;
  }

  uint64_t readULEB128() {
    uint64_t Value = 0;
    unsigned Shift = 0;
    uint8_t p;
    do {
      p = read<uint8_t>();
      uint64_t Slice = p & 0x7f;
      if ((Shift >= 64 && Slice != 0) || Slice << Shift >> Shift != Slice) {
        throw UnwindError("uleb128 too big for uint64");
      }
      Value += Slice << Shift;
      Shift += 7;
    } while (p >= 128);
    return Value;
  }
  const char* readCString() {
    auto result = next_;
    next_ += strlen(next_) + 1;
    return result;
  }
  int64_t readEncoded(uint8_t enc) {
    int64_t r = 0;
    switch (enc & (~DW_EH_PE_indirect & 0xF0)) {
      case DW_EH_PE_absptr:
        break;
      case DW_EH_PE_pcrel:
        r = (int64_t)next_;
        break;
      case DW_EH_PE_datarel:
        r = base_;
        break;
      default:
        throw UnwindError("unknown encoding");
    }
    return r + readEncodedValue(enc);
  }
  int64_t readEncodedOr(uint8_t enc, int64_t orelse) {
    if (enc == DW_EH_PE_omit) {
      return orelse;
    }
    return readEncoded(enc);
  }
  int64_t read4or8Length() {
    int64_t length = read<uint32_t>();
    if (length == 0xFFFFFFFF) {
      length = read<int64_t>();
    }
    return length;
  }
  void* loc() const {
    return (void*)next_;
  }
  Lexer& skip(int64_t bytes) {
    next_ += bytes;
    return *this;
  }
  int64_t readEncodedValue(uint8_t enc) {
    switch (enc & 0xF) {
      case DW_EH_PE_udata2:
        return read<uint16_t>();
      case DW_EH_PE_sdata2:
        return read<int16_t>();
      case DW_EH_PE_udata4:
        return read<uint32_t>();
      case DW_EH_PE_sdata4:
        return read<int32_t>();
      case DW_EH_PE_udata8:
        return read<uint64_t>();
      case DW_EH_PE_sdata8:
        return read<int64_t>();
      case DW_EH_PE_uleb128:
        return readULEB128();
      case DW_EH_PE_sleb128:
        return readSLEB128();
      default:
        throw UnwindError("not implemented");
    }
  }

 private:
  const char* next_;
  int64_t base_;
};
