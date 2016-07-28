#pragma once

#include <sstream>
#include <string>
#include <vector>

namespace caffe2 {

std::vector<std::string> split(char separator, const std::string& string);
}
