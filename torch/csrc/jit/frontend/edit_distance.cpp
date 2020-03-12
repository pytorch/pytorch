#include <torch/csrc/jit/frontend/edit_distance.h>
#include <algorithm>
#include <cstring>
#include <memory>

namespace torch {
namespace jit {

// computes levenshtein edit distance between two words
// returns maxEditDistance + 1 if the edit distance exceeds MaxEditDistance
// reference: http://llvm.org/doxygen/edit__distance_8h_source.html
size_t ComputeEditDistance(
    const char* word1,
    const char* word2,
    size_t maxEditDistance) {

  size_t m = strlen(word1);
  size_t n = strlen(word2);

  const unsigned small_buffer_size = 64;
  unsigned small_buffer[small_buffer_size];
  std::unique_ptr<unsigned[]> allocated;
  unsigned* row = small_buffer;
  if (n + 1 > small_buffer_size) {
    row = new unsigned[n + 1];
    allocated.reset(row);
  }

  for (unsigned i = 1; i <= n; ++i)
    row[i] = i;

  for (size_t y = 1; y <= m; ++y) {
    row[0] = y;
    unsigned best_this_row = row[0];

    unsigned previous = y - 1;
    for (size_t x = 1; x <= n; ++x) {
      int old_row = row[x];
      row[x] = std::min(
          previous + (word1[y - 1] == word2[x - 1] ? 0u : 1u),
          std::min(row[x - 1], row[x]) + 1);
      previous = old_row;
      best_this_row = std::min(best_this_row, row[x]);
    }

    if (maxEditDistance && best_this_row > maxEditDistance)
      return maxEditDistance + 1;
  }

  unsigned result = row[n];
  return result;
}

} // namespace jit
} // namespace torch
