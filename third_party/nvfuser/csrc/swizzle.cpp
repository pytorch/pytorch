#include <swizzle.h>

#include <arith.h>
#include <ir_builder.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {
namespace swizzles {

// ------------------------------------------------------------
// Swizzle Definitions
//   for each swizzle name:
// un(Swizzle Name) e.g. unZShape is the inverse of ZShape,
//  (unswizzle is needed for inlining and is currently not actively used.)
// ------------------------------------------------------------

// Unit Z swizzle:
//  Alternate directions of Y dimension:
//    1 2 3      1 2 3
//    4 5 6  =>  6 5 4
//    7 8 9      7 8 9
std::pair<Val*, Val*> ZShape(Val* x, Val* y, Val* size_y) {
  auto zero = x->fusion()->zeroVal();
  auto one = x->fusion()->oneVal();
  auto two = IrBuilder::create<Int>(2);
  return {x, where(eq(mod(x, two), zero), y, sub(sub(size_y, one), y))};
}

// ZShape is inverse of itself
std::pair<Val*, Val*> unZShape(Val* x, Val* y, Val* size_y) {
  return ZShape(x, y, size_y);
}

// Block cyclic Xor swizzle: (bank conflict removal)
//  Apply cyclic Xor within blocks:
//   Example: cyclic Xor
//    1   2  3  4       1   2   3  4
//    5   6  7  8       6   5   8  7
//    9  10 11 12  =>   11  12  9 10
//    13 14 15 16       16  15 14 13
std::pair<Val*, Val*> Xor(Val* x, Val* y) {
  // Need to validate in swizzle configuration:
  //  size_x == size_y
  return {x, bitwise_xor(x, y)};
}

// Xor is inverse of itself
std::pair<Val*, Val*> unXor(Val* x, Val* y) {
  return Xor(x, y);
}

// Block cyclic shift swizzle: (bank conflict removal)
//  Apply cyclic shift within blocks:
//   Example: cyclic shift
//    1   2  3  4       1   2   3   4
//    5   6  7  8       8   5   6   7
//    9  10 11 12  =>   11  12  9  10
//    13 14 15 16       14  15  16 13
std::pair<Val*, Val*> CyclicShift(Val* x, Val* y, Val* size_x) {
  return {x, mod(add(x, y), size_x)};
}

std::pair<Val*, Val*> unCyclicShift(Val* x, Val* y, Val* size_x) {
  return {x, mod(sub(add(size_x, y), x), size_x)};
}

// Scatter swizzle:
//   Corresponds to the data layout out of ldmatrix intrinsic.
//   supported dimensions are : 8x4, 16x4, 32x4
std::pair<Val*, Val*> Scatter(Val* x, Val* y, int size_x) {
  TORCH_CHECK(
      size_x == 8 || size_x == 16 || size_x == 32,
      "Unsupported Scatter swizzle size");
  Val* size_x_val = IrBuilder::create<Int>(size_x);
  auto four = IrBuilder::create<Int>(4);
  return {cpp_div(add(mul(y, size_x_val), x), four), mod(x, four)};
}

std::pair<Val*, Val*> unScatter(Val* x, Val* y, int size_x) {
  TORCH_CHECK(
      size_x == 8 || size_x == 16 || size_x == 32,
      "Unsupported Scatter swizzle size");
  Val* size_x_div_4 = IrBuilder::create<Int>(size_x / 4);
  auto four = IrBuilder::create<Int>(4);
  return {add(y, mul(mod(x, size_x_div_4), four)), cpp_div(x, size_x_div_4)};
}

} // namespace swizzles

std::pair<Val*, Val*> dispatchSwizzle(
    Swizzle2DType type,
    Val* x,
    Val* y,
    Val* maybe_size_x,
    Val* maybe_size_y) {
  switch (type) {
    case Swizzle2DType::ZShape:
      return swizzles::ZShape(x, y, maybe_size_y);
    case Swizzle2DType::XOR:
      return swizzles::Xor(x, y);
    case Swizzle2DType::CyclicShift:
      return swizzles::CyclicShift(x, y, maybe_size_x);
    case Swizzle2DType::Scatter:
      return swizzles::Scatter(x, y, maybe_size_x->evaluateInt());
    default:
      TORCH_INTERNAL_ASSERT(false, "Unsupported swizzle type");
  }
}

std::pair<Val*, Val*> dispatchUnSwizzle(
    Swizzle2DType type,
    Val* x,
    Val* y,
    Val* maybe_size_x,
    Val* maybe_size_y) {
  switch (type) {
    case Swizzle2DType::ZShape:
      return swizzles::unZShape(x, y, maybe_size_y);
    case Swizzle2DType::XOR:
      return swizzles::unXor(x, y);
    case Swizzle2DType::CyclicShift:
      return swizzles::unCyclicShift(x, y, maybe_size_x);
    case Swizzle2DType::Scatter:
      return swizzles::unScatter(x, y, maybe_size_x->evaluateInt());
    default:
      TORCH_INTERNAL_ASSERT(false, "Unsupported swizzle type");
  }
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
