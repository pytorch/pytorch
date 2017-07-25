#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <sstream>

namespace torch { namespace jit {

struct TemplateEnv {
  TemplateEnv(TemplateEnv * parent = nullptr)
  : parent(parent) {}
  using string_list = std::vector<std::string>;
  // strings
  void s(const std::string & k, const std::string & v) {
    strings_[k] = v;
  }
  // numbers
  template<typename T>
  void d(const std::string & k, const T & v) {
    strings_[k] = std::to_string(v);
  }
  const std::string & s(const std::string & k) const {
    if(strings_.count(k) == 0) {
      if(parent) {
        return parent->s(k);
      }
      notFound(k);
    }
    return strings_.at(k);
  }
  // lists of strings
  void v(const std::string & k, const string_list & v) {
    lists_[k] = v;
  }
  const string_list & v(const std::string & k) const {
    if(lists_.count(k) == 0) {
      if(parent) {
        return parent->v(k);
      }
      notFound(k);
    }
    return lists_.at(k);
  }
  bool keyIsString(const std::string & k) const {
    if(strings_.count(k) > 0)
      return true;
    if(lists_.count(k) > 0)
      return false;
    if(parent)
      return parent->keyIsString(k);
    notFound(k);
  }
private:
  [[ noreturn ]]
  void notFound(const std::string & k) const {
    std::stringstream ss;
    ss << "key not found: " << k;
    throw std::logic_error(ss.str());
  }
  std::unordered_map<std::string,std::string> strings_;
  std::unordered_map<std::string,string_list> lists_;
  TemplateEnv * parent;
};

/*
# Match $identifier or ${identifier} and replace with value in env.
# If this identifier is at the beginning of whitespace on a line
# and its value is a list then it is treated as
# block subsitution by indenting all lines of all elements.
# If the identifier is on a line starting with non-whitespace and a list
# then it is comma separated. ${,foo} will insert a comma before the list
# if this list is not empty and ${foo,} will insert one after.
*/
struct CodeTemplate {
  /* implicit */ CodeTemplate(const std::string & t)
  : template_text(t) {}

  std::string format(const TemplateEnv & env) {
    std::stringstream out;
    size_t pos = 0;
    size_t indent = 0;
    bool all_whitespace = true;
    while(pos < template_text.size()) {
      char c = template_text[pos];
      if(c == '$') {
        std::stringstream kss;
        bool comma_before;
        bool comma_after;
        size_t new_pos = parseKey(pos,kss,comma_before,comma_after);
        std::string k = kss.str();
        bool is_string = env.keyIsString(k);
        if(all_whitespace) {
          if(is_string)
            emitStringWithIndents(out, indent, env.s(k));
          else
            emitLinesIndented(out, indent, env.v(k));
        } else {
          if(is_string)
            out << env.s(k);
          else
            emitCommaSeparatedList(out, env.v(k), comma_before, comma_after);
        }
        all_whitespace = false;
        pos = new_pos;
      } else {
        out << c;
        if(!isspace(c))
          all_whitespace = false;
        indent++;
        if(c == '\n') {
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
  char charAt(size_t p) {
    if (p >= template_text.size())
      throw std::logic_error("EOS found in key");
    return template_text[p];
  }
  size_t parseKey(size_t pos, std::ostream & k, bool & comma_before, bool & comma_after) {
    comma_before = false;
    comma_after = false;
    pos++;
    if(charAt(pos) == '{') {
      pos++;
      if(charAt(pos) == ',') {
        comma_before = true;
        pos++;
      }
      pos = parseIdent(pos, k);
      if(charAt(pos) == ',') {
        comma_after = true;
        pos++;
      }
      if(charAt(pos) != '}')
        throw std::logic_error("missing terminating '}'");
      pos++;
      return pos;
    } else {
      return parseIdent(pos, k);
    }
  }
  size_t parseIdent(size_t pos, std::ostream & k) {
    while(pos < template_text.size() &&
      (isalnum(template_text[pos]) || template_text[pos] == '_')) {
      k << template_text[pos];
      pos++;
    }
    return pos;
  }
  void emitCommaSeparatedList(std::ostream & out, const string_list & strings, bool comma_before, bool comma_after) {
    if(comma_before && strings.size() > 0)
      out << ", ";
    for(size_t i = 0; i < strings.size(); ++i) {
      if(i > 0)
        out << ", ";
      out << strings[i];
    }
    if(comma_after && strings.size() > 0)
      out << ", ";
  }
  void emitIndent(std::ostream & out, size_t indent) {
    for(size_t i = 0; i < indent; ++i) {
      out << " ";
    }
  }
  void emitStringWithIndents(std::ostream & out, size_t indent, const std::string & str) {
    for(auto c : str) {
      out << c;
      if(c == '\n') {
        emitIndent(out, indent);
      }
    }
  }
  void emitLinesIndented(std::stringstream & out, size_t indent, const string_list & strings) {
    for(size_t i = 0; i < strings.size(); ++i) {
      if(i > 0)
        emitIndent(out, indent);
      emitStringWithIndents(out,indent,strings[i]);
      if(i+1 != strings.size())
        out << "\n";
    }
  }
  std::string template_text;
};
static std::string format(const std::string & fmt, TemplateEnv & env) {
  return CodeTemplate(fmt).format(env);
}

}}
