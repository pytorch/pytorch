#pragma once

#include <cstdint>
#include <ostream>
#include <string>

#include "lazy_tensors/computation_client/client_data.h"
#include "lazy_tensors/computation_client/ltc_logging.h"

namespace lazy_tensors {

inline std::string PrimitiveTypeName(PrimitiveType primitive_type) {
  switch (primitive_type) {
    case PrimitiveType::PRED: {
      return "pred";
    }
    case PrimitiveType::S8: {
      return "s8";
    }
    case PrimitiveType::S16: {
      return "s16";
    }
    case PrimitiveType::S32: {
      return "s32";
    }
    case PrimitiveType::S64: {
      return "s64";
    }
    case PrimitiveType::U8: {
      return "u8";
    }
    case PrimitiveType::U16: {
      return "u16";
    }
    case PrimitiveType::U32: {
      return "u32";
    }
    case PrimitiveType::U64: {
      return "u64";
    }
    case PrimitiveType::F16: {
      return "f16";
    }
    case PrimitiveType::F32: {
      return "f32";
    }
    case PrimitiveType::BF16: {
      return "bf16";
    }
    case PrimitiveType::F64: {
      return "f64";
    }
    case PrimitiveType::C64: {
      return "c64";
    }
    case PrimitiveType::C128: {
      return "c128";
    }
    case PrimitiveType::TUPLE: {
      return "tuple";
    }
    default: { return "invalid"; }
  }
}

inline std::ostream& operator<<(std::ostream& os,
                                PrimitiveType primitive_type) {
  os << PrimitiveTypeName(primitive_type);
  return os;
}

}  // namespace lazy_tensors
