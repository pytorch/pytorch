#include "caffe2/utils/string_utils.h"

#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <sstream>

namespace caffe2 {

std::vector<std::string> split(char separator, const std::string& string) {
  std::vector<std::string> pieces;
  std::stringstream ss(string);
  std::string item;
  while (getline(ss, item, separator)) {
    pieces.push_back(std::move(item));
  }
  return pieces;
}

} // namespace caffe2
