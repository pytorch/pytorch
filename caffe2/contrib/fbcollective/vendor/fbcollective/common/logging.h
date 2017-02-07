#pragma once

#include <climits>
#include <exception>
#include <functional>
#include <limits>
#include <vector>

#include "fbcollective/common/string.h"

namespace fbcollective {

class EnforceNotMet : public std::exception {
 public:
  EnforceNotMet(
      const char* file,
      const int line,
      const char* condition,
      const std::string& msg);

  std::string msg() const;

  inline const std::vector<std::string>& msg_stack() const {
    return msg_stack_;
  }

  virtual const char* what() const noexcept override;

 private:
  std::vector<std::string> msg_stack_;
  std::string full_msg_;
};

#define FBC_ENFORCE(condition, ...)                 \
  do {                                              \
    if (!(condition)) {                             \
      throw ::fbcollective::EnforceNotMet(          \
          __FILE__,                                 \
          __LINE__,                                 \
          #condition,                               \
          ::fbcollective::MakeString(__VA_ARGS__)); \
    }                                               \
  } while (false)

#define FBC_THROW(...)                 \
  throw ::fbcollective::EnforceNotMet( \
      __FILE__, __LINE__, "", ::fbcollective::MakeString(__VA_ARGS__))

/**
 * Rich logging messages
 *
 * FBC_ENFORCE_THAT can be used with one of the "checker functions" that
 * capture input argument values and add it to the exception message. E.g.
 * `FBC_ENFORCE_THAT(Equals(foo(x), bar(y)), "Optional additional message")`
 * would evaluate both foo and bar only once and if the results are not equal -
 * include them in the exception message.
 *
 * Some of the basic checker functions like Equals or Greater are already
 * defined below. Other header might define customized checkers by adding
 * functions to fbcollective::enforce_detail namespace. For example:
 *
 *   namespace fbcollective { namespace enforce_detail {
 *   inline EnforceFailMessage IsVector(const vector<TIndex>& shape) {
 *     if (shape.size() == 1) { return EnforceOK(); }
 *     return MakeString("Shape ", shape, " is not a vector");
 *   }
 *   }}
 *
 * With further usages like `FBC_ENFORCE_THAT(IsVector(Input(0).dims()))`
 *
 * Convenient wrappers for binary operations like FBC_ENFORCE_EQ are provided
 * too. Please use them instead of CHECK_EQ and friends for failures in
 * user-provided input.
 */

namespace enforce_detail {

struct EnforceOK {};

class EnforceFailMessage {
 public:
  constexpr /* implicit */ EnforceFailMessage(EnforceOK) : msg_(nullptr) {}

  EnforceFailMessage(EnforceFailMessage&&) = default;
  EnforceFailMessage(const EnforceFailMessage&) = delete;
  EnforceFailMessage& operator=(EnforceFailMessage&&) = delete;
  EnforceFailMessage& operator=(const EnforceFailMessage&) = delete;

  /* implicit */ EnforceFailMessage(std::string&& msg) {
    msg_ = new std::string(std::move(msg));
  }

  inline bool bad() const {
    return msg_;
  }

  std::string get_message_and_free(std::string&& extra) const {
    std::string r;
    if (extra.empty()) {
      r = std::move(*msg_);
    } else {
      r = ::fbcollective::MakeString(std::move(*msg_), ". ", std::move(extra));
    }
    delete msg_;
    return r;
  }

 private:
  std::string* msg_;
};

#define BINARY_COMP_HELPER(name, op)                         \
  template <typename T1, typename T2>                        \
  inline EnforceFailMessage name(const T1& x, const T2& y) { \
    if (x op y) {                                            \
      return EnforceOK();                                    \
    }                                                        \
    return MakeString(x, " vs ", y);                         \
  }
BINARY_COMP_HELPER(Equals, ==)
BINARY_COMP_HELPER(NotEquals, !=)
BINARY_COMP_HELPER(Greater, >)
BINARY_COMP_HELPER(GreaterEquals, >=)
BINARY_COMP_HELPER(Less, <)
BINARY_COMP_HELPER(LessEquals, <=)
#undef BINARY_COMP_HELPER

#define FBC_ENFORCE_THAT_IMPL(condition, expr, ...)         \
  do {                                                      \
    using namespace ::fbcollective::enforce_detail;         \
    const EnforceFailMessage& r = (condition);              \
    if (r.bad()) {                                          \
      throw EnforceNotMet(                                  \
          __FILE__,                                         \
          __LINE__,                                         \
          expr,                                             \
          r.get_message_and_free(MakeString(__VA_ARGS__))); \
    }                                                       \
  } while (false)
}

#define FBC_ENFORCE_THAT(condition, ...) \
  FBC_ENFORCE_THAT_IMPL((condition), #condition, __VA_ARGS__)

#define FBC_ENFORCE_EQ(x, y, ...) \
  FBC_ENFORCE_THAT_IMPL(Equals((x), (y)), #x " == " #y, __VA_ARGS__)
#define FBC_ENFORCE_NE(x, y, ...) \
  FBC_ENFORCE_THAT_IMPL(NotEquals((x), (y)), #x " != " #y, __VA_ARGS__)
#define FBC_ENFORCE_LE(x, y, ...) \
  FBC_ENFORCE_THAT_IMPL(LessEquals((x), (y)), #x " <= " #y, __VA_ARGS__)
#define FBC_ENFORCE_LT(x, y, ...) \
  FBC_ENFORCE_THAT_IMPL(Less((x), (y)), #x " < " #y, __VA_ARGS__)
#define FBC_ENFORCE_GE(x, y, ...) \
  FBC_ENFORCE_THAT_IMPL(GreaterEquals((x), (y)), #x " >= " #y, __VA_ARGS__)
#define FBC_ENFORCE_GT(x, y, ...) \
  FBC_ENFORCE_THAT_IMPL(Greater((x), (y)), #x " > " #y, __VA_ARGS__)

} // namespace fbcollective
