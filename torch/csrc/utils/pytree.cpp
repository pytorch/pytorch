#include <torch/csrc/utils/pytree.h>

#include <ctype.h>
#include <cstring>
#include <stack>

namespace torch {
namespace pytree {

std::vector<size_t> pre_parse(const StrTreeSpec& spec) {
  std::vector<size_t> ret;
  std::stack<std::pair<size_t, size_t>> stack;
  size_t i = 0;
  const size_t size = spec.size();

  ret.resize(size);
  while (i < size) {
    const auto c = spec[i];
    switch (c) {
      case Config::kNodeDataBegin: {
        stack.push({i, i});
        break;
      }
      case Config::kNodeDataEnd: {
        auto& item = stack.top();
        size_t last_sep_idx = item.second;
        ret[last_sep_idx] = i;
        stack.pop();
        break;
      }
      case Config::kDictStrKeyQuote: {
        size_t idx = i;
        i++;
        while (spec[i] != Config::kDictStrKeyQuote) {
          i++;
        }
        ret[idx] = i;
        ret[i] = idx;
        break;
      }
      case Config::kChildrenSep: {
        auto& item = stack.top();
        size_t last_sep_idx = item.second;
        ret[last_sep_idx] = i;
        item.second = i;
        break;
      }
    }
    i++;
  }
  return ret;
}

size_t read_number(const StrTreeSpec& spec, size_t& read_idx) {
  size_t num = 0;
  while (isdigit(spec[read_idx])) {
    num = 10 * num + (spec[read_idx] - '0');
    read_idx++;
  }
  return num;
}

std::vector<size_t> read_node_layout(
    const StrTreeSpec& spec,
    size_t& read_idx) {
  const size_t child_num = read_number(spec, read_idx);
  std::vector<size_t> vec(child_num);

  size_t child_idx = 0;
  while (spec[read_idx] == Config::kChildrenDataSep) {
    ++read_idx;
    vec[child_idx++] = read_number(spec, read_idx);
  }
  return vec;
}

} // namespace pytree
} // namespace torch
