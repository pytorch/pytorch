#pragma once

#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace torch { namespace jit {
// SourceLocation represents source code-level debug information for a node.
// It contains information about where a node got generated.
// In the case of tracing this will be a python stack trace.
// In the case of using the scripting frontend this will be backed
// by a SourceRange object
struct SourceLocation {
  virtual ~SourceLocation() = default;
  virtual void highlight(std::ostream & out) const = 0;

  std::string wrapException(const std::exception & e, const std::string & additional = "") {
    std::stringstream msg;
    msg << "\n" << e.what() << ":\n";
    if(!additional.empty()) {
      msg << additional << ":\n";
    }
    highlight(msg);
    return msg.str();
  }
  void wrapAndRethrowException(const std::exception & e, const std::string & additional = "") {
    throw std::runtime_error(wrapException(e, additional));
  }

};

inline std::ostream& operator<<(std::ostream& out, const SourceLocation& sl) {
  sl.highlight(out);
  return out;
}


// normally a python stack trace
struct StringSourceLocation : public SourceLocation {
  StringSourceLocation(std::string context)
  : context(std::move(context)) {}
  void highlight(std::ostream & out) const override {
    out << context;
  }
private:
  std::string context;
};

}}
