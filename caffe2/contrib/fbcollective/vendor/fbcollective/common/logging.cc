#include "fbcollective/common/logging.h"

#include <numeric>

namespace fbcollective {

EnforceNotMet::EnforceNotMet(
    const char* file,
    const int line,
    const char* condition,
    const std::string& msg)
    : msg_stack_{MakeString(
          "[enforce fail at ",
          file,
          ":",
          line,
          "] ",
          condition,
          ". ",
          msg)} {
  full_msg_ = this->msg();
}

std::string EnforceNotMet::msg() const {
  return std::accumulate(msg_stack_.begin(), msg_stack_.end(), std::string(""));
}

const char* EnforceNotMet::what() const noexcept {
  return full_msg_.c_str();
}

} // namespace fbcollective
