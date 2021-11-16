#include <torch/csrc/jit/codegen/cuda/parallel_type_bitmap.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

const std::unordered_map<ParallelType, int, TypeHash>
    ParallelTypeBitmap::pt_to_offset_{
        {ParallelType::BIDx, 0},
        {ParallelType::BIDy, 1},
        {ParallelType::BIDz, 2},
        {ParallelType::TIDx, 3},
        {ParallelType::TIDy, 4},
        {ParallelType::TIDz, 5}};

const std::unordered_map<int, ParallelType> ParallelTypeBitmap::offset_to_pt_ =
    {{0, ParallelType::BIDx},
     {1, ParallelType::BIDy},
     {2, ParallelType::BIDz},
     {3, ParallelType::TIDx},
     {4, ParallelType::TIDy},
     {5, ParallelType::TIDz}};

bool ParallelTypeBitmap::get(ParallelType pt) const {
  if (pt_to_offset_.find(pt) == pt_to_offset_.end()) {
    TORCH_INTERNAL_ASSERT(false, "Could not recognize parallel type.");
  }
  return bitset_[pt_to_offset_.at(pt)];
}

bool ParallelTypeBitmap::set(ParallelType pt, bool new_val) {
  if (pt_to_offset_.find(pt) == pt_to_offset_.end()) {
    TORCH_INTERNAL_ASSERT(false, "Could not recognize parallel type.");
  }
  bool old_val = bitset_[pt_to_offset_.at(pt)];
  bitset_[pt_to_offset_.at(pt)] = new_val;
  return old_val;
}

ParallelTypeBitmap ParallelTypeBitmap::operator&=(
    const ParallelTypeBitmap& other) {
  bitset_ &= other.bitset_;
  return *this;
}

ParallelTypeBitmap ParallelTypeBitmap::operator|=(
    const ParallelTypeBitmap& other) {
  bitset_ |= other.bitset_;
  return *this;
}

ParallelTypeBitmap ParallelTypeBitmap::operator^=(
    const ParallelTypeBitmap& other) {
  bitset_ ^= other.bitset_;
  return *this;
}

ParallelTypeBitmap ParallelTypeBitmap::operator~() const {
  return ParallelTypeBitmap(~bitset_);
}

bool ParallelTypeBitmap::none() const {
  return bitset_.none();
}

bool ParallelTypeBitmap::any() const {
  return bitset_.any();
}

bool ParallelTypeBitmap::all() const {
  return bitset_.all();
}

bool ParallelTypeBitmap::operator[](size_t pos) const {
  TORCH_INTERNAL_ASSERT(
      pos < num_p_type, "Invalid index to ParallelTypeBitset: ", pos);
  return bitset_[pos];
}

bool ParallelTypeBitmap::hasTID() const {
  for (auto pt : {ParallelType::TIDx, ParallelType::TIDy, ParallelType::TIDz}) {
    if (get(pt)) {
      return true;
    }
  }
  return false;
}

bool ParallelTypeBitmap::hasBID() const {
  for (auto pt : {ParallelType::BIDx, ParallelType::BIDy, ParallelType::BIDz}) {
    if (get(pt)) {
      return true;
    }
  }
  return false;
}

std::map<ParallelType, bool> ParallelTypeBitmap::getMap() const {
  std::map<ParallelType, bool> map;
  for (const auto& pt_offset : pt_to_offset_) {
    map.emplace(pt_offset.first, bitset_[pt_offset.second]);
  }
  return map;
}

ParallelTypeBitmap operator&(
    const ParallelTypeBitmap& lhs,
    const ParallelTypeBitmap& rhs) {
  auto x = lhs;
  x &= rhs;
  return x;
}

ParallelTypeBitmap operator|(
    const ParallelTypeBitmap& lhs,
    const ParallelTypeBitmap& rhs) {
  auto x = lhs;
  x |= rhs;
  return x;
}

ParallelTypeBitmap operator^(
    const ParallelTypeBitmap& lhs,
    const ParallelTypeBitmap& rhs) {
  auto x = lhs;
  x ^= rhs;
  return x;
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
