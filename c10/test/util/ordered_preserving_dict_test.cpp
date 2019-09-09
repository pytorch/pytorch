#include <vector>
#include <unordered_set>
#include <algorithm>

#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <c10/util/order_preserving_flat_hash_map.h>
#include <gtest/gtest.h>
#include <ATen/Tensor.h>

namespace {

#define ASSERT_EQUAL(t1, t2) ASSERT_TRUE(t1 == t2);

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


// Tests taken from https://github.com/Tessil/ordered-map/blob/master/tests/ordered_map_tests.cpp

TEST(OrderedPreservingDictTest, test_range_insert) {
    // insert x values in vector, range insert x-15 values from vector to map, check values
    const int nb_values = 1000;
    std::vector<std::pair<int, int>> values;
    for(int i = 0; i < nb_values; i++) {
        values.push_back(std::make_pair(i, i+1));
    }

    dict_int_int map = {{-1, 0}, {-2, 0}};
    map.insert(values.begin() + 10, values.end() - 5);

    TORCH_INTERNAL_ASSERT(map.size(), 987);

    ASSERT_EQUAL(map.at(-1), 0);

    ASSERT_EQUAL(map.at(-2), 0);

    for(int i = 10, j = 2; i < nb_values - 5; i++, j++) {
        ASSERT_EQUAL(map.at(i), i+1);
    }
}

TEST(OrderedPreservingDictTest, test_range_erase_all) {
    // insert x values, delete all
    const std::size_t nb_values = 1000;
    dict_int_int map;
    for (size_t i = 0; i < nb_values; ++i) {
      map[i] = i + 1;
    }
    auto it = map.erase(map.begin(), map.end());
    ASSERT_TRUE(it == map.end());
    ASSERT_TRUE(map.empty());
}

TEST(OrderedPreservingDictTest, test_range_erase) {
    // insert x values, delete all with iterators except 10 first and 780 last values
    using HMap = ska_ordered::order_preserving_flat_hash_map<std::string, std::int64_t>;

    const std::size_t nb_values = 1000;
    HMap map;
    for (size_t i = 0; i < nb_values; ++i) {
      map[std::to_string(i)] = i;
      auto begin = map.begin();
      for (size_t j = 0; j <= i; ++j, begin++) {
        TORCH_INTERNAL_ASSERT(begin->second == j);
      }
    }

    auto it_first = std::next(map.begin(), 10);
    auto it_last = std::next(map.begin(), 220);

    auto it = map.erase(it_first, it_last);
    ASSERT_EQUAL(std::distance(it, map.end()), 780);
    ASSERT_EQUAL(map.size(), 790);
    ASSERT_EQUAL(std::distance(map.begin(), map.end()), 790);

    for(auto& val: map) {
        ASSERT_EQUAL(map.count(val.first), 1);
    }

    // Check order
    it = map.begin();
    for(std::size_t i = 0; i < nb_values; i++) {
        if(i >= 10 && i < 220) {
            continue;
        }
        auto exp_it = std::pair<std::string, std::int64_t>(std::to_string(i), i);
        TORCH_INTERNAL_ASSERT(*it == exp_it);
        ++it;
    }
}




}
