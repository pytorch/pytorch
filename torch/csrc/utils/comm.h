#pragma once

#include <iostream>

// Enum classes used for inter- and intra- process communications, currently
// including distributed and CUDA comm.
namespace torch {
namespace utils {
namespace comm {

enum class ReduceOp : std::uint8_t {
  SUM = 0,
  PRODUCT,
  MIN,
  MAX,
  BAND, // Bitwise AND
  BOR, // Bitwise OR
  BXOR, // Bitwise XOR
  UNUSED,
};

inline std::ostream& operator<<(std::ostream & out, const ReduceOp& op) {
  switch (op) {
    case ReduceOp::SUM:
      out << "SUM";
      return out;
    case ReduceOp::PRODUCT:
      out << "PRODUCT";
      return out;
    case ReduceOp::MIN:
      out << "MIN";
      return out;
    case ReduceOp::MAX:
      out << "MAX";
      return out;
    case ReduceOp::BAND:
      out << "BAND";
      return out;
    case ReduceOp::BOR:
      out << "BOR";
      return out;
    case ReduceOp::BXOR:
      out << "BXOR";
      return out;
    default:
      throw std::runtime_error("Invalid ReduceOp");
  }
}

}}}
