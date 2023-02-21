// Utility macro for this file
#define DEVICE_INLINE __device__ inline

// Utility class for 2D swizzle:
template <typename index_t>
struct IndexGeneric {
  const index_t x = 0, y = 0;
  DEVICE_INLINE IndexGeneric(index_t x_, index_t y_) : x(x_), y(y_) {}
};

// Default type for integration
using Index2D = IndexGeneric<nvfuser_index_t>;

// Small type for unit computation
using Index2DInt = IndexGeneric<int>;

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
DEVICE_INLINE Index2D ZShape(Index2D in, Index2D unit_dim) {
  return Index2D(in.x, in.x % 2 == 0 ? in.y : (unit_dim.y - in.y - 1));
}

// ZShape is inverse of itself
DEVICE_INLINE Index2D unZShape(Index2D in, Index2D unit_dim) {
  return ZShape(in, unit_dim);
}

// Block cyclic Xor swizzle: (bank conflict removal)
//  Apply cyclic Xor within blocks:
//   Example: cyclic Xor
//    1   2  3  4       1   2   3  4
//    5   6  7  8       6   5   8  7
//    9  10 11 12  =>   11  12  9 10
//    13 14 15 16       16  15 14 13
// Note:
DEVICE_INLINE Index2D Xor(Index2D in, Index2DInt unit_dim) {
  // Need to validate in swizzle configuration:
  //  unit_dim.x == unit_dim.y
  return Index2D(in.x, (in.y ^ in.x));
}

// Inverse of Xor is itself
DEVICE_INLINE Index2D unXor(Index2D in, Index2DInt unit_dim) {
  return Xor(in, unit_dim);
}

// Scatter swizzle:
//   Corresponds to the data layout out of ldmatrix intrinsic.
//   supported dimensions are : 8x4, 16x4, 32x4
template <int row_size>
DEVICE_INLINE Index2D Scatter(Index2D in) {
  static_assert(row_size == 8 || row_size == 16 || row_size == 32);
  return Index2D((in.y * row_size + in.x) / 4, in.x % 4);
}

template <int row_size>
DEVICE_INLINE Index2D unScatter(Index2D in) {
  static_assert(row_size == 8 || row_size == 16 || row_size == 32);
  return Index2D(in.y + (in.x % (row_size / 4)) * 4, in.x / (row_size / 4));
}

#undef DEVICE_INLINE
