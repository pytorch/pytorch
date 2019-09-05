#include <vector>
#include <unordered_set>
#include <algorithm>

#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <c10/util/order_preserving_flat_hash_map.h>
#include <gtest/gtest.h>
#include <ATen/Tensor.h>

namespace {

using dict_int_int = ska_ordered::order_preserving_flat_hash_map<int64_t, int64_t>;

dict_int_int test_dict(dict_int_int& dict) {
  for (int64_t i = 0; i < 100; ++i) {
    dict[i] = i + 1;
  }

  int64_t i = 0;
  for (auto entry: dict) {
    TORCH_INTERNAL_ASSERT(entry.first == i && entry.second == i + 1);
    ++i;
  }

  // erase a few entries by themselves
  std::unordered_set<int64_t> erase_set = {0, 2, 9, 71};
  for (auto erase: erase_set) {
    dict.erase(erase);
  }

  // erase via iterators
  auto begin = dict.begin();
  for (size_t i = 0; i < 20; ++i)
    begin++;

  auto end = begin;
  for (size_t i = 0; i < 20; ++i) {
    erase_set.insert(end->first);
    end++;
  }
  dict.erase(begin, end);

  std::vector<size_t> order;
  for (size_t i = 0; i < 100; ++i) {
    if (!erase_set.count(i)) {
      order.push_back(i);
    }
  }

  i = 0;
  for (auto entry: dict) {
    TORCH_INTERNAL_ASSERT(order[i] == entry.first);
    TORCH_INTERNAL_ASSERT(dict[order[i]] == entry.second);
    TORCH_INTERNAL_ASSERT(entry.second == order[i] + 1);
    i++;
  }
  TORCH_INTERNAL_ASSERT(dict.size() == order.size());
  return dict;
}

TEST(OrderedPreservingDictTest, InsertAndDeleteBasic) {
  dict_int_int dict;
  test_dict(dict);
  dict.clear();
  test_dict(dict);
}

TEST(OrderedPreservingDictTest, InsertExistingDoesntAffectOrder) {
  dict_int_int dict;
  dict[0] = 1;
  dict[1] = 2;

  TORCH_INTERNAL_ASSERT(dict.begin()->first == 0);
  dict[0] = 1;
  TORCH_INTERNAL_ASSERT(dict.begin()->first == 0);
  dict[0] = 2;
  TORCH_INTERNAL_ASSERT(dict.begin()->first == 0);

  dict.erase(0);
  TORCH_INTERNAL_ASSERT(dict.begin()->first == 1);
}


TEST(OrderedPreservingDictTest, testRefType) {
  std::shared_ptr<int64_t> t;
  using dict_references = ska_ordered::order_preserving_flat_hash_map<int64_t, std::shared_ptr<int64_t>>;

  dict_references dict;

  auto ptr = std::make_shared<int64_t>(1);
  dict[1] = ptr;
  TORCH_INTERNAL_ASSERT(ptr.use_count() == 2);
  dict.erase(1);
  TORCH_INTERNAL_ASSERT(ptr.use_count() == 1);

  dict[2] = ptr;
  dict.clear();
  TORCH_INTERNAL_ASSERT(ptr.use_count() == 1);
}


TEST(OrderedPreservingDictTest, DictCollisions) {
  struct BadHash {
    size_t operator()(const int64_t input) {
      return input % 2;
    };
  };

  using bad_hash_dict =
      ska_ordered::order_preserving_flat_hash_map<int64_t, int64_t, BadHash>;

  for (auto init_dict_size : {27, 34, 41}) {
    bad_hash_dict dict;
    for (int64_t i = 0; i < init_dict_size; ++i) {
      dict[i] = i + 1;
    }

    int64_t i = 0;
    for (auto entry : dict) {
      TORCH_INTERNAL_ASSERT(entry.first == i && entry.second == i + 1);
      ++i;
    }

    // erase a few entries;
    std::unordered_set<int64_t> erase_set = {0, 2, 9};
    for (auto erase : erase_set) {
      dict.erase(erase);
    }

    // erase a few entries via iterator
    auto begin = dict.begin();
    for (size_t i = 0; i < 10; ++i) {
      begin++;
    }
    auto end = begin;
    for (size_t i = 0; i < 7; ++i) {
      erase_set.insert(end->first);
      end++;
    }
    dict.erase(begin, end);

    std::vector<int64_t> order;
    for (int64_t i = 0; i < init_dict_size; ++i) {
      if (!erase_set.count(i)) {
        order.push_back(i);
      }
    }

    i = 0;
    for (auto entry : dict) {
      TORCH_INTERNAL_ASSERT(dict[entry.first] == entry.second);
      TORCH_INTERNAL_ASSERT(dict[entry.first] == order[i] + 1);
      TORCH_INTERNAL_ASSERT(order[i] == entry.first);
      i += 1;
    }
    TORCH_INTERNAL_ASSERT(dict.size() == order.size());
  }
}


}
