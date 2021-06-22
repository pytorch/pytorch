#pragma once

#include <c10/util/string_view.h>

#include <algorithm>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

namespace lazy_tensors {

template <typename Range>
std::string StrJoin(const Range& range, c10::string_view separator) {
  auto b = std::begin(range);
  auto e = std::end(range);
  if (b == e) {
    return "";
  }
  std::vector<std::string> str_tokens;
  std::transform(b, e, std::back_inserter(str_tokens), [](auto val) {
    std::ostringstream os;
    os << val;
    return os.str();
  });
  std::ostringstream joined;
  std::copy(str_tokens.begin(), str_tokens.end() - 1,
            std::ostream_iterator<std::string>(joined, separator.data()));
  joined << str_tokens.back();
  return joined.str();
}

}  // namespace lazy_tensors
