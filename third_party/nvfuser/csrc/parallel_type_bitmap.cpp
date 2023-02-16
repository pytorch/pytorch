#include <parallel_type_bitmap.h>

namespace nvfuser {

constexpr std::bitset<ParallelTypeBitmap::kNumParallelTypes>
    ParallelTypeBitmap::kTIDBits;
constexpr std::bitset<ParallelTypeBitmap::kNumParallelTypes>
    ParallelTypeBitmap::kBIDBits;

std::string ParallelTypeBitmap::toString() const {
  std::stringstream ss;
  ss << "(";
  bool is_first = true;
  for (ParallelType pt : *this) {
    if (!is_first) {
      ss << " ";
    }
    ss << pt;
    is_first = false;
  }
  ss << ")";
  return ss.str();
}

} // namespace nvfuser
