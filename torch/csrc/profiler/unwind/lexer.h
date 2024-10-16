#pragma once
#include <cstdint>
#include <cstring>
#include <utility>

#include <torch/csrc/profiler/unwind/dwarf_enums.h>
#include <torch/csrc/profiler/unwind/unwind_error.h>

namespace torch::unwind {

template <bool checked>
struct LexerImpl {
  LexerImpl(void* data, void* base = nullptr, void* end = nullptr)
      : next_((const char*)data),
        base_((int64_t)base),
        end_((const char*)end) {}

  template <typename T>
  T read() {
    T result;
    auto end = next_ + sizeof(T);
    UNWIND_CHECK(
        !checked || end <= end_,
        "read out of bounds {} >= {}",
        (void*)end,
        (void*)end_);
    memcpy(&result, next_, sizeof(T));
    next_ = end;
    return result;
  }

  // SLEB/ULEB code adapted from LLVM equivalents
  int64_t readSLEB128() {
    int64_t Value = 0;
    unsigned Shift = 0;
    uint8_t Byte = 0;
    do {
      Byte = read<uint8_t>();
      uint64_t Slice = Byte & 0x7f;
      if ((Shift >= 64 && Slice != (Value < 0 ? 0x7f : 0x00)) ||
          (Shift == 63 && Slice != 0 && Slice != 0x7f)) {
        throw UnwindError("sleb128 too big for int64");
      }
      Value |= int64_t(Slice << Shift);
      Shift += 7;
    } while (Byte >= 128);
    // Sign extend negative numbers if needed.
    if (Shift < 64 && (Byte & 0x40)) {
      Value |= int64_t((-1ULL) << Shift);
    }
    return Value;
  }

  uint64_t readULEB128() {
    uint64_t Value = 0;
    unsigned Shift = 0;
    uint8_t p = 0;
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
    if (!checked) {
      next_ += strlen(next_) + 1;
      return result;
    }
    while (next_ < end_) {
      if (*next_++ == '\0') {
        return result;
      }
    }
    UNWIND_CHECK(
        false, "string is out of bounds {} >= {}", (void*)next_, (void*)end_);
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
    return readSectionLength().first;
  }

  std::pair<int64_t, bool> readSectionLength() {
    int64_t length = read<uint32_t>();
    if (length == 0xFFFFFFFF) {
      return std::make_pair(read<int64_t>(), true);
    }
    return std::make_pair(length, false);
  }

  void* loc() const {
    return (void*)next_;
  }
  LexerImpl& skip(int64_t bytes) {
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
  const char* end_;
};

// using Lexer = LexerImpl<false>;
using CheckedLexer = LexerImpl<true>;
using Lexer = LexerImpl<false>;

} // namespace torch::unwind
