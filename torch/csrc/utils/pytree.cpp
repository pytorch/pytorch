#include <torch/csrc/utils/pytree.h>

#include <ctype.h>
#include <cstring>
#include <stack>

namespace torch {
namespace pytree {
namespace {
struct StrTreeSpecData {
  std::vector<size_t> idxs;
  size_t containers_num = 0;
};

StrTreeSpecData pre_parse(const StrTreeSpec& spec) {
  StrTreeSpecData ret;
  std::stack<std::pair<size_t, size_t>> stack;
  size_t i = 0;
  size_t containers_num = 0;
  const size_t size = spec.size();

  ret.idxs.resize(size);
  while (i < size) {
    const auto c = spec[i];
    switch (c) {
      case Config::kDict:
      case Config::kList:
      case Config::kLeaf: {
        ret.containers_num++;
        break;
      }
      case Config::kNodeDataBegin: {
        stack.push({i, i});
        break;
      }
      case Config::kNodeDataEnd: {
        auto& item = stack.top();
        size_t last_sep_idx = item.second;
        ret.idxs[last_sep_idx] = i;
        stack.pop();
        break;
      }
      case Config::kDictStrKeyQuote: {
        size_t idx = i;
        i++;
        while (spec[i] != Config::kDictStrKeyQuote) {
          i++;
        }
        ret.idxs[idx] = i;
        ret.idxs[i] = idx;
        break;
      }
      case Config::kChildrenSep: {
        auto& item = stack.top();
        size_t last_sep_idx = item.second;
        ret.idxs[last_sep_idx] = i;
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

TreeSpec from_str_internal(
    const StrTreeSpec& spec,
    size_t read_idx,
    const StrTreeSpecData& spec_data) {
  bool isTuple = false;
  switch (spec[read_idx]) {
    case Config::kTuple:
      isTuple = true;
      [[fallthrough]];
    case Config::kList: {
      read_idx++;
      auto layout = read_node_layout(spec, read_idx);
      const auto size = layout.size();
      auto c =
          new Container<TreeSpecLeaf>(isTuple ? Kind::Tuple : Kind::List, size);

      size_t child_idx = 0;
      size_t leaves_offset = 0;

      if (size > 0) {
        while (spec[read_idx] != Config::kNodeDataEnd) {
          // NOLINTNEXTLINE
          auto next_delim_idx = spec_data.idxs[read_idx];
          read_idx++;
          c->items[child_idx] = from_str_internal(spec, read_idx, spec_data);
          read_idx = next_delim_idx;
          leaves_offset += layout[child_idx++];
        }
      } else {
        read_idx++;
      }
      c->leaves_num = leaves_offset;
      return c;
    }

    case Config::kDict: {
      read_idx++;
      auto layout = read_node_layout(spec, read_idx);
      const auto size = layout.size();
      auto c = new Container<TreeSpecLeaf>(Kind::Dict, size);

      size_t child_idx = 0;
      size_t leaves_offset = 0;

      if (size > 0) {
        while (spec[read_idx] != Config::kNodeDataEnd) {
          // NOLINTNEXTLINE
          auto next_delim_idx = spec_data.idxs[read_idx];
          read_idx++;
          if (spec[read_idx] == Config::kDictStrKeyQuote) {
            auto key_delim_idx = spec_data.idxs[read_idx];
            read_idx++;
            const size_t key_len = key_delim_idx - read_idx;
            // NOLINTNEXTLINE
            c->dict.keys[child_idx] = spec.substr(read_idx, key_len);
            read_idx = key_delim_idx + 2;
          } else {
            TORCH_INTERNAL_ASSERT(isdigit(spec[read_idx]));
            size_t key = read_number(spec, read_idx);
            c->dict.keys[child_idx] = KeyInt(key);
            read_idx += 1;
          }

          c->items[child_idx] = from_str_internal(spec, read_idx, spec_data);
          read_idx = next_delim_idx;
          leaves_offset += layout[child_idx++];
        }
      } else {
        read_idx++;
      }
      c->leaves_num = leaves_offset;
      return c;
    }

    case Config::kLeaf:
      return new Container<TreeSpecLeaf>(nullptr);
  }
  TORCH_INTERNAL_ASSERT(false);
  return new Container<TreeSpecLeaf>(Kind::None);
}

StrTreeSpec to_str_internal(const TreeSpec& spec) {
  std::string s;
  switch (spec.kind()) {
    case Kind::List:
      s.push_back(Config::kList);
      break;
    case Kind::Tuple:
      s.push_back(Config::kTuple);
      break;
    case Kind::Dict:
      s.push_back(Config::kDict);
      break;
    case Kind::Leaf:
      s.push_back(Config::kLeaf);
      return s;
    case Kind::None:
      return s;
  }
  const size_t size = spec.size();
  s.append(std::to_string(size));
  for (size_t i = 0; i < size; ++i) {
    s.push_back(Config::kChildrenDataSep);
    s.append(std::to_string(spec[i].leaves_num()));
  }
  s.push_back(Config::kNodeDataBegin);
  if (spec.kind() == Kind::Dict) {
    for (size_t i = 0; i < size; ++i) {
      if (i) {
        s.push_back(Config::kChildrenSep);
      }
      const auto& key = spec.key(i);
      if (key.kind() == Key::Kind::Int) {
        s.append(std::to_string(key.as_int()));
      } else if (key.kind() == Key::Kind::Str) {
        s.push_back(Config::kDictStrKeyQuote);
        s.append(key.as_str());
        s.push_back(Config::kDictStrKeyQuote);
      } else {
        TORCH_INTERNAL_ASSERT(false);
      }
      s.push_back(Config::kDictKeyValueSep);
      s.append(to_str_internal(spec[i]));
    }
  } else {
    for (size_t i = 0; i < size; ++i) {
      if (i) {
        s.push_back(Config::kChildrenSep);
      }
      s.append(to_str_internal(spec[i]));
    }
  }
  s.push_back(Config::kNodeDataEnd);
  return s;
}

} // namespace

TreeSpec from_str(const StrTreeSpec& spec) {
  return from_str_internal(spec, 0u, pre_parse(spec));
}

StrTreeSpec to_str(const TreeSpec& spec) {
  return to_str_internal(spec);
}

} // namespace pytree
} // namespace torch
