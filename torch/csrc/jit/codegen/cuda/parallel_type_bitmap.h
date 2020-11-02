#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/codegen/cuda/type.h>

#include <bitset>
#include <map>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

// Represents mapping to bool from BIDx, BIDy, BIDz, TIDx, TIDy and TIDz.
class ParallelTypeBitmap {
 public:
  static constexpr int num_p_type = 6;

  ParallelTypeBitmap() = default;

  bool get(ParallelType pt) const;
  bool set(ParallelType pt, bool);
  ParallelTypeBitmap operator&=(const ParallelTypeBitmap& other);
  ParallelTypeBitmap operator|=(const ParallelTypeBitmap& other);
  ParallelTypeBitmap operator^=(const ParallelTypeBitmap& other);
  ParallelTypeBitmap operator~() const;
  bool none() const;
  bool any() const;
  bool all() const;
  bool operator[](size_t pos) const;
  std::map<ParallelType, bool> getMap() const;

 private:
  ParallelTypeBitmap(const std::bitset<num_p_type>& bs) : bitset_(bs) {}

 private:
  std::bitset<num_p_type> bitset_;
  const static std::unordered_map<ParallelType, int, TypeHash> pt_to_offset_;
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
