#pragma once
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace torch {
namespace jit {

// A template environment is a mapping from template variable names, e.g.,
// identifier (corresponding to $identifier) to their expansions.
//
// This template environment supports storing strings, numbers and lists
// of strings, and can be chained together (so that lookup proceeds in
// in the top level environment, and then recurses into a parent
// environment if the key is not found.)
struct TemplateEnv {
  TemplateEnv() : parent(nullptr) {}
  TemplateEnv(TemplateEnv& parent) : parent(&parent) {}

  using string_list = std::vector<std::string>;

  // Add a string 'v' to the map at key 'k'.
  void s(const std::string& k, const std::string& v) {
    strings_[k] = v;
    lists_.erase(k);
  }

  // Add a number 'v' to the map at key 'k'
  template <typename T>
  void d(const std::string& k, const T& v) {
    strings_[k] = std::to_string(v);
    lists_.erase(k);
  }

  // Retrieve the string representation of the value stored at 'k' from the map.
  // Raises an exception if the key is not found.
  const std::string& s(const std::string& k) const {
    if (strings_.count(k) == 0) {
      if (parent) {
        return parent->s(k);
      }
      notFound(k);
    }
    return strings_.at(k);
  }

  // Store a list of strings 'v' in the map at 'k'.
  void v(const std::string& k, const string_list& v) {
    lists_[k] = v;
    strings_.erase(k);
  }

  // Retrieve a list of strings stored at 'k' from the map.
  // Raises an exception if the key is not found.
  const string_list& v(const std::string& k) const {
    if (lists_.count(k) == 0) {
      if (parent) {
        return parent->v(k);
      }
      notFound(k);
    }
    return lists_.at(k);
  }

  // Test if a string 'k' is a string (as opposed to a list.)
  bool keyIsString(const std::string& k) const {
    if (strings_.count(k) > 0)
      return true;
    if (lists_.count(k) > 0)
      return false;
    if (parent)
      return parent->keyIsString(k);
    notFound(k);
  }

 private:
  [[noreturn]] void notFound(const std::string& k) const {
    std::stringstream ss;
    ss << "key not found: " << k;
    throw std::logic_error(ss.str());
  }
  std::unordered_map<std::string, std::string> strings_;
  std::unordered_map<std::string, string_list> lists_;
  TemplateEnv* parent;
};

/*
# Match $identifier or ${identifier} and replace with the value in env.
# If this identifier is at the beginning of whitespace on a line
# and its value is a list then it is treated as
# block substitution by indenting all lines of all elements.
# If the identifier is on a line starting with non-whitespace and a list
# then it is comma separated. ${,foo} will insert a comma before the list
# if this list is not empty and ${foo,} will insert one after.
*/
struct CodeTemplate {
  /* implicit */ CodeTemplate(std::string t) : template_text(std::move(t)) {}

  std::string format(const TemplateEnv& env) const {
    std::stringstream out;
    size_t pos = 0;
    size_t indent = 0;
    bool all_whitespace = true;
    while (pos < template_text.size()) {
      char c = template_text[pos];
      if (c == '$') {
        std::stringstream kss;
        bool comma_before;
        bool comma_after;
        size_t new_pos = parseKey(pos, kss, comma_before, comma_after);
        std::string k = kss.str();
        bool is_string = env.keyIsString(k);
        if (all_whitespace) {
          if (is_string)
            emitStringWithIndents(out, indent, env.s(k));
          else
            emitLinesIndented(out, indent, env.v(k));
        } else {
          if (is_string)
            out << env.s(k);
          else
            emitCommaSeparatedList(out, env.v(k), comma_before, comma_after);
        }
        all_whitespace = false;
        pos = new_pos;
      } else {
        out << c;
        if (!isspace(c))
          all_whitespace = false;
        indent++;
        if (c == '\n') {
          indent = 0;
          all_whitespace = true;
        }
        pos++;
      }
    }
    return out.str();
  }

 private:
  using string_list = std::vector<std::string>;
  char charAt(size_t p) const {
    if (p >= template_text.size())
      throw std::logic_error("EOS found in key");
    return template_text[p];
  }
  size_t parseKey(
      size_t pos,
      std::ostream& k,
      bool& comma_before,
      bool& comma_after) const {
    comma_before = false;
    comma_after = false;
    pos++;
    if (charAt(pos) == '{') {
      pos++;
      if (charAt(pos) == ',') {
        comma_before = true;
        pos++;
      }
      pos = parseIdent(pos, k);
      if (charAt(pos) == ',') {
        comma_after = true;
        pos++;
      }
      if (charAt(pos) != '}')
        throw std::logic_error("missing terminating '}'");
      pos++;
      return pos;
    } else {
      return parseIdent(pos, k);
    }
  }
  size_t parseIdent(size_t pos, std::ostream& k) const {
    while (pos < template_text.size() &&
           (isalnum(template_text[pos]) || template_text[pos] == '_')) {
      k << template_text[pos];
      pos++;
    }
    return pos;
  }
  void emitCommaSeparatedList(
      std::ostream& out,
      const string_list& strings,
      bool comma_before,
      bool comma_after) const {
    if (comma_before && strings.size() > 0)
      out << ", ";
    for (size_t i = 0; i < strings.size(); ++i) {
      if (i > 0)
        out << ", ";
      out << strings[i];
    }
    if (comma_after && strings.size() > 0)
      out << ", ";
  }
  // These indentation functions follow the convention that they never emit
  // leading or trailing newlines when the input string does not have leading
  // or trailing newlines. It's the responsibility of the calling function
  // to indent correctly in the context.
  void emitIndent(std::ostream& out, size_t indent) const {
    for (size_t i = 0; i < indent; ++i) {
      out << " ";
    }
  }
  void emitStringWithIndents(
      std::ostream& out,
      size_t indent,
      const std::string& str) const {
    for (auto c : str) {
      out << c;
      if (c == '\n') {
        emitIndent(out, indent);
      }
    }
  }
  void emitLinesIndented(
      std::stringstream& out,
      size_t indent,
      const string_list& strings) const {
    for (size_t i = 0; i < strings.size(); ++i) {
      if (i > 0)
        emitIndent(out, indent);
      emitStringWithIndents(out, indent, strings[i]);
      if (i + 1 != strings.size())
        out << "\n";
    }
  }
  std::string template_text;
};
static inline std::string format(const std::string& fmt, TemplateEnv& env) {
  return CodeTemplate(fmt).format(env);
}

} // namespace jit
} // namespace torch
