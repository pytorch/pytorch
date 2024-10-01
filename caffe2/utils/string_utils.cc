#include "caffe2/utils/string_utils.h"

#include <algorithm>
#include <sstream>
#include <vector>
#include <cstdint>

namespace caffe2 {

std::vector<std::string>
split(char separator, const std::string& string, bool ignore_empty) {
  std::vector<std::string> pieces;
  std::stringstream ss(string);
  std::string item;
  while (getline(ss, item, separator)) {
    if (!ignore_empty || !item.empty()) {
      pieces.push_back(std::move(item));
    }
  }
  return pieces;
}

std::string trim(const std::string& str) {
  size_t left = str.find_first_not_of(' ');
  if (left == std::string::npos) {
    return str;
  }
  size_t right = str.find_last_not_of(' ');
  return str.substr(left, (right - left + 1));
}

size_t editDistance(
  const std::string& s1, const std::string& s2, size_t max_distance)
  {
    std::vector<size_t> current(s1.length() + 1);
    std::vector<size_t> previous(s1.length() + 1);
    std::vector<size_t> previous1(s1.length() + 1);

    return editDistanceHelper(
        s1.c_str(),
        s1.length(),
        s2.c_str(),
        s2.length(),
        current,
        previous,
        previous1,
        max_distance
    );
  }
  #define NEXT_UNSAFE(s, i, c) { \
      (c)=(uint8_t)(s)[(i)++]; \
  }

int32_t editDistanceHelper(const char* s1,
  size_t s1_len,
  const char* s2,
  size_t s2_len,
  std::vector<size_t> &current,
  std::vector<size_t> &previous,
  std::vector<size_t> &previous1,
  size_t max_distance) {
    if (max_distance) {
      if (std::max(s1_len, s2_len) - std::min(s1_len, s2_len) > max_distance) {
        return max_distance+1;
      }
    }

    for (size_t j = 0; j <= s1_len; ++j) {
      current[j] = j;
    }

    int32_t str2_offset = 0;
    char prev2 = 0;
    for (size_t i = 1; i <= s2_len; ++i) {
      swap(previous1, previous);
      swap(current, previous);
      current[0] = i;

      // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
      char c2 = s2[str2_offset];
      char prev1 = 0;
      int32_t str1_offset = 0;

      NEXT_UNSAFE(s2, str2_offset, c2);

      size_t current_min = s1_len;
      for (size_t j = 1; j <= s1_len; ++j) {
        size_t insertion = previous[j] + 1;
        size_t deletion = current[j - 1] + 1;
        size_t substitution = previous[j - 1];
        size_t transposition = insertion;
        // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
        char c1 = s1[str1_offset];

        NEXT_UNSAFE(s1, str1_offset, c1);

        if (c1 != c2) {
          substitution += 1;
        }


        if (prev1 == c2 && prev2 == c1 && j > 1 && i > 1) {
          transposition = previous1[j - 2] + 1;
        }
        prev1 = c1;

        current[j] = std::min(std::min(insertion, deletion),
                         std::min(substitution, transposition));
        current_min = std::min(current_min, current[j]);
      }


      if (max_distance != 0 && current_min > max_distance) {
        return max_distance+1;
      }

      prev2 = c2;
    }

    return current[s1_len];
  }
} // namespace caffe2
