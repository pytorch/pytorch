#pragma once

#include <c10/macros/Export.h>
#include <type.h>

#include <array>
#include <bitset>
#include <map>
#include <unordered_map>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

constexpr int getParallelTypeBitMapOffset(ParallelType pt) {
  switch (pt) {
    case ParallelType::BIDx:
      return 0;
    case ParallelType::BIDy:
      return 1;
    case ParallelType::BIDz:
      return 2;
    case ParallelType::TIDx:
      return 3;
    case ParallelType::TIDy:
      return 4;
    case ParallelType::TIDz:
      return 5;
    default:
      return -1;
  }
}

//! Represents mapping to bool from BIDx, BIDy, BIDz, TIDx, TIDy and TIDz.
class ParallelTypeBitmap {
 public:
  static constexpr int kNumParallelTypes = 6;

  //! Iterator for ParallelTypeBitmap. Picks only set types.
  class Iterator {
   public:
    static Iterator begin(const ParallelTypeBitmap& map);

    static Iterator end(const ParallelTypeBitmap& map);

    bool operator==(const Iterator& other) const;

    bool operator!=(const Iterator& other) const;

    Iterator& operator++();

    Iterator operator++(int);

    ParallelType operator*() const;

   private:
    Iterator(const ParallelTypeBitmap& map, int offset);

    void skipToSetType();

   private:
    const ParallelTypeBitmap& map_;
    int offset_ = 0;

    static constexpr int kOffsetEnd = kNumParallelTypes;
  };

  ParallelTypeBitmap() = default;

  explicit ParallelTypeBitmap(ParallelType pt) {
    set(pt);
  }

  //! Return true if pt is included
  bool get(ParallelType pt) const {
    auto offset = getParallelTypeBitMapOffset(pt);
    TORCH_INTERNAL_ASSERT(
        offset != -1, "Could not recognize parallel type: ", pt);
    return bitset_[offset];
  }

  //! Set the flag of pt
  bool set(ParallelType pt, bool new_val = true) {
    auto offset = getParallelTypeBitMapOffset(pt);
    TORCH_INTERNAL_ASSERT(
        offset != -1, "Could not recognize parallel type: ", pt);
    bool old_val = bitset_[offset];
    bitset_[offset] = new_val;
    return old_val;
  }

  //! Clear the flag of pt
  bool clear(ParallelType pt) {
    return set(pt, false);
  }

  //! Assign logical AND with other
  ParallelTypeBitmap operator&=(const ParallelTypeBitmap& other) {
    bitset_ &= other.bitset_;
    return *this;
  }

  //! Assign logical OR with other
  ParallelTypeBitmap operator|=(const ParallelTypeBitmap& other) {
    bitset_ |= other.bitset_;
    return *this;
  }

  //! Assign logical NOR with other
  ParallelTypeBitmap operator^=(const ParallelTypeBitmap& other) {
    bitset_ ^= other.bitset_;
    return *this;
  }

  //! Return logical compliment
  ParallelTypeBitmap operator~() const {
    return ParallelTypeBitmap(~bitset_);
  }

  //! Return true if none of the mapppings is true
  bool none() const {
    return bitset_.none();
  }

  //! Return true if any of the mapppings is true
  bool any() const {
    return bitset_.any();
  }

  //! Return true if all of the mapppings is true
  bool all() const {
    return bitset_.all();
  }

  //! Return true if the parallel type corresponding to a position
  //! defined in offset_to_pt_ is true
  bool operator[](size_t pos) const {
    TORCH_INTERNAL_ASSERT(
        pos < kNumParallelTypes, "Invalid index to ParallelTypeBitset: ", pos);
    return bitset_[pos];
  }

  //! Return true if TIDx/y/z is included
  bool hasTID() const {
    return (bitset_ & kTIDBits).any();
  }

  //! Return true if BIDx/y/z is included
  bool hasBID() const {
    return (bitset_ & kBIDBits).any();
  }

  //! Set all of the TID flags
  void setAllTID() {
    *this |= ParallelTypeBitmap(kTIDBits);
  }

  //! Set all of the BID flags
  void setAllBID() {
    *this |= ParallelTypeBitmap(kBIDBits);
  }

  //! Clear all of the TID flags
  void clearAllTID() {
    auto tid_bits = ParallelTypeBitmap(kTIDBits);
    auto not_tid_bits = ~tid_bits;
    *this &= not_tid_bits;
  }

  //! Clear all of the BID flags
  void clearAllBID() {
    auto bid_bits = ParallelTypeBitmap(kBIDBits);
    auto not_bid_bits = ~bid_bits;
    *this &= not_bid_bits;
  }

  //! Get an iterator to traverse set types
  Iterator begin() const {
    return Iterator::begin(*this);
  }

  //! Get an end iterator to traverse set types
  Iterator end() const {
    return Iterator::end(*this);
  }

  bool operator==(const ParallelTypeBitmap& other) const {
    return bitset_ == other.bitset_;
  }

  std::string toString() const;

 private:
  explicit constexpr ParallelTypeBitmap(
      const std::bitset<kNumParallelTypes>& bs)
      : bitset_(bs) {}

 private:
  std::bitset<kNumParallelTypes> bitset_;

  static constexpr std::bitset<ParallelTypeBitmap::kNumParallelTypes> kTIDBits{
      (1u << getParallelTypeBitMapOffset(ParallelType::TIDx)) |
      (1u << getParallelTypeBitMapOffset(ParallelType::TIDy)) |
      (1u << getParallelTypeBitMapOffset(ParallelType::TIDz))};

  static constexpr std::bitset<ParallelTypeBitmap::kNumParallelTypes> kBIDBits{
      (1u << getParallelTypeBitMapOffset(ParallelType::BIDx)) |
      (1u << getParallelTypeBitMapOffset(ParallelType::BIDy)) |
      (1u << getParallelTypeBitMapOffset(ParallelType::BIDz))};
};

inline ParallelTypeBitmap operator&(
    const ParallelTypeBitmap& lhs,
    const ParallelTypeBitmap& rhs) {
  auto x = lhs;
  x &= rhs;
  return x;
}

inline ParallelTypeBitmap operator|(
    const ParallelTypeBitmap& lhs,
    const ParallelTypeBitmap& rhs) {
  auto x = lhs;
  x |= rhs;
  return x;
}

inline ParallelTypeBitmap operator^(
    const ParallelTypeBitmap& lhs,
    const ParallelTypeBitmap& rhs) {
  auto x = lhs;
  x ^= rhs;
  return x;
}

inline bool ParallelTypeBitmap::Iterator::operator==(
    const ParallelTypeBitmap::Iterator& other) const {
  return offset_ == other.offset_ && map_ == other.map_;
}

inline bool ParallelTypeBitmap::Iterator::operator!=(
    const ParallelTypeBitmap::Iterator& other) const {
  return !(*this == other);
}

inline ParallelTypeBitmap::Iterator& ParallelTypeBitmap::Iterator::
operator++() {
  ++offset_;
  skipToSetType();
  return *this;
}

inline ParallelTypeBitmap::Iterator ParallelTypeBitmap::Iterator::operator++(
    int) {
  const auto before_increment = *this;
  ++offset_;
  skipToSetType();
  return before_increment;
}

inline ParallelType ParallelTypeBitmap::Iterator::operator*() const {
  return kParallelTypeThreads[offset_];
}

inline ParallelTypeBitmap::Iterator::Iterator(
    const ParallelTypeBitmap& map,
    int offset)
    : map_(map), offset_(offset) {
  skipToSetType();
}

inline void ParallelTypeBitmap::Iterator::skipToSetType() {
  while (offset_ < kOffsetEnd && !map_[offset_]) {
    ++offset_;
  }
}

inline ParallelTypeBitmap::Iterator ParallelTypeBitmap::Iterator::begin(
    const ParallelTypeBitmap& map) {
  return Iterator(map, 0);
}

inline ParallelTypeBitmap::Iterator ParallelTypeBitmap::Iterator::end(
    const ParallelTypeBitmap& map) {
  return Iterator(map, kOffsetEnd);
}

//! Map from ParallelType to template type T
template <typename T>
class ParallelTypeMap {
 public:
  ParallelTypeMap() = default;

  ParallelTypeMap(const T& init) {
    std::fill(map_.begin(), map_.end(), init);
  }

  T& operator[](ParallelType pt) {
    return map_[getParallelTypeBitMapOffset(pt)];
  }

  const T& operator[](ParallelType pt) const {
    return map_[getParallelTypeBitMapOffset(pt)];
  }

  T& at(ParallelType pt) {
    return map_.at(getParallelTypeBitMapOffset(pt));
  }

  const T& at(ParallelType pt) const {
    return map_.at(getParallelTypeBitMapOffset(pt));
  }

  auto begin() {
    return map_.begin();
  }

  auto begin() const {
    return map_.begin();
  }

  auto end() {
    return map_.begin();
  }

  auto end() const {
    return map_.begin();
  }

 private:
  std::array<T, ParallelTypeBitmap::kNumParallelTypes> map_;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
