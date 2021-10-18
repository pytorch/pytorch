#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/codegen/cuda/type.h>

#include <bitset>
#include <map>
#include <unordered_map>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

//! Represents mapping to bool from BIDx, BIDy, BIDz, TIDx, TIDy and TIDz.
class ParallelTypeBitmap {
 public:
  static constexpr int num_p_type = 6;

  ParallelTypeBitmap() = default;

  //! Return true if pt is included
  bool get(ParallelType pt) const;
  //! Set the mapping of pt
  bool set(ParallelType pt, bool);
  //! Assign logical AND with other
  ParallelTypeBitmap operator&=(const ParallelTypeBitmap& other);
  //! Assign logical OR with other
  ParallelTypeBitmap operator|=(const ParallelTypeBitmap& other);
  //! Assign logical NOR with other
  ParallelTypeBitmap operator^=(const ParallelTypeBitmap& other);
  //! Return logical compliment
  ParallelTypeBitmap operator~() const;
  //! Return true if none of the mapppings is true
  bool none() const;
  //! Return true if any of the mapppings is true
  bool any() const;
  //! Return true if all of the mapppings is true
  bool all() const;
  //! Return true if the parallel type corresponding to a position
  //! defined in offset_to_pt_ is true
  bool operator[](size_t pos) const;
  //! Return an equivalent std::map
  std::map<ParallelType, bool> getMap() const;
  //! Return true if TIDx/y/z is included
  bool hasTID() const;
  //! Return true if BIDx/y/z is included
  bool hasBID() const;

 private:
  ParallelTypeBitmap(const std::bitset<num_p_type>& bs) : bitset_(bs) {}

 private:
  std::bitset<num_p_type> bitset_;
  //! Map of ParallelType to bit positions
  const static std::unordered_map<ParallelType, int, TypeHash> pt_to_offset_;
  //! Map of bit positions to ParallelType
  const static std::unordered_map<int, ParallelType> offset_to_pt_;
};

ParallelTypeBitmap operator&(
    const ParallelTypeBitmap& lhs,
    const ParallelTypeBitmap& rhs);

ParallelTypeBitmap operator|(
    const ParallelTypeBitmap& lhs,
    const ParallelTypeBitmap& rhs);

ParallelTypeBitmap operator^(
    const ParallelTypeBitmap& lhs,
    const ParallelTypeBitmap& rhs);

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
