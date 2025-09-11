#include <algorithm>
#include <unordered_set>
#include <vector>

#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>
#include <c10/util/order_preserving_flat_hash_map.h>
#include <gtest/gtest.h>

namespace {

#define ASSERT_EQUAL_PRIM(t1, t2) ASSERT_TRUE(t1 == t2);

using dict_int_int =
    ska_ordered::order_preserving_flat_hash_map<int64_t, int64_t>;

dict_int_int test_dict(dict_int_int& dict) {
  for (const auto i : c10::irange(100)) {
    dict[i] = i + 1;
  }

  int64_t entry_i = 0;
  for (auto entry : dict) {
    TORCH_INTERNAL_ASSERT(
        entry.first == entry_i && entry.second == entry_i + 1);
    ++entry_i;
  }

  // erase a few entries by themselves
  std::unordered_set<int64_t> erase_set = {0, 2, 9, 71};
  for (auto erase : erase_set) {
    dict.erase(erase);
  }

  // erase via iterators
  auto begin = dict.begin();
  for ([[maybe_unused]] const auto i : c10::irange(20)) {
    begin++;
  }

  auto end = begin;
  for ([[maybe_unused]] const auto i : c10::irange(20)) {
    erase_set.insert(end->first);
    end++;
  }
  dict.erase(begin, end);

  std::vector<int64_t> order;
  for (const auto i : c10::irange(100)) {
    if (!erase_set.count(i)) {
      order.push_back(i);
    }
  }

  entry_i = 0;
  for (auto entry : dict) {
    TORCH_INTERNAL_ASSERT(order[entry_i] == entry.first);
    TORCH_INTERNAL_ASSERT(dict[order[entry_i]] == entry.second);
    TORCH_INTERNAL_ASSERT(entry.second == order[entry_i] + 1);
    entry_i++;
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
  using dict_references = ska_ordered::
      order_preserving_flat_hash_map<int64_t, std::shared_ptr<int64_t>>;

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
    for (const auto i : c10::irange(init_dict_size)) {
      dict[i] = i + 1;
    }

    int64_t i = 0;
    for (const auto& entry : dict) {
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
    for ([[maybe_unused]] const auto j : c10::irange(10)) {
      begin++;
    }
    auto end = begin;
    for ([[maybe_unused]] const auto j : c10::irange(7)) {
      erase_set.insert(end->first);
      end++;
    }
    dict.erase(begin, end);

    std::vector<int64_t> order;
    for (const auto j : c10::irange(init_dict_size)) {
      if (!erase_set.count(j)) {
        order.push_back(j);
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

// Tests taken from
// https://github.com/Tessil/ordered-map/blob/master/tests/ordered_map_tests.cpp

TEST(OrderedPreservingDictTest, test_range_insert) {
  // insert x values in vector, range insert x-15 values from vector to map,
  // check values
  const int nb_values = 1000;
  std::vector<std::pair<int, int>> values;
  for (const auto i : c10::irange(nb_values)) {
    values.emplace_back(i, i + 1);
  }

  dict_int_int map = {{-1, 0}, {-2, 0}};
  map.insert(values.begin() + 10, values.end() - 5);

  ASSERT_EQUAL_PRIM(map.size(), 987);

  ASSERT_EQUAL_PRIM(map.at(-1), 0);

  ASSERT_EQUAL_PRIM(map.at(-2), 0);

  auto begin = map.begin();
  begin++;
  begin++;
  for (int i = 10; i < nb_values - 5; i++, begin++) {
    // Check range inserted kv pairs: map(i) = i + 1 for i = 10,....995
    ASSERT_EQUAL_PRIM(map.at(i), i + 1);
    // Check range inserted kv pairs are correctly indexed/ordered
    TORCH_INTERNAL_ASSERT(begin->first == i);
    TORCH_INTERNAL_ASSERT(begin->second == i + 1);
  }
}

TEST(OrderedPreservingDictTest, test_range_erase_all) {
  // insert x values, delete all
  const std::size_t nb_values = 1000;
  dict_int_int map;
  for (const int64_t i : c10::irange<int64_t>(nb_values)) {
    map[i] = i + 1;
  }
  auto it = map.erase(map.begin(), map.end());
  ASSERT_TRUE(it == map.end());
  ASSERT_TRUE(map.empty());
}

TEST(OrderedPreservingDictTest, test_range_erase) {
  // insert x values, delete all with iterators except 10 first and 780 last
  // values
  using HMap =
      ska_ordered::order_preserving_flat_hash_map<std::string, std::int64_t>;

  const int64_t nb_values = 1000;
  HMap map;
  for (const auto i : c10::irange(nb_values)) {
    map[std::to_string(i)] = i;
    auto begin = map.begin();
    for (int64_t j = 0; j <= i; ++j, begin++) {
      TORCH_INTERNAL_ASSERT(begin->second == j);
    }
  }

  auto it_first = std::next(map.begin(), 10);
  auto it_last = std::next(map.begin(), 220);

  auto it = map.erase(it_first, it_last);
  ASSERT_EQUAL_PRIM(std::distance(it, map.end()), 780);
  ASSERT_EQUAL_PRIM(map.size(), 790);
  ASSERT_EQUAL_PRIM(std::distance(map.begin(), map.end()), 790);

  for (auto& val : map) {
    ASSERT_EQUAL_PRIM(map.count(val.first), 1);
  }

  // Check order
  it = map.begin();
  for (std::size_t i = 0; i < nb_values; i++) {
    if (i >= 10 && i < 220) {
      continue;
    }
    auto exp_it = std::pair<std::string, std::int64_t>(std::to_string(i), i);
    TORCH_INTERNAL_ASSERT(*it == exp_it);
    ++it;
  }
}

TEST(OrderedPreservingDictTest, test_move_constructor_empty) {
  ska_ordered::order_preserving_flat_hash_map<std::string, int64_t> map(0);
  ska_ordered::order_preserving_flat_hash_map<std::string, int64_t> map_move(
      std::move(map));

  // NOLINTNEXTLINE(bugprone-use-after-move)
  TORCH_INTERNAL_ASSERT(map.empty());
  TORCH_INTERNAL_ASSERT(map_move.empty());

  // NOLINTNEXTLINE(clang-analyzer-cplusplus.Move)
  TORCH_INTERNAL_ASSERT(map.find("") == map.end());
  TORCH_INTERNAL_ASSERT(map_move.find("") == map_move.end());
}

TEST(OrderedPreservingDictTest, test_move_operator_empty) {
  ska_ordered::order_preserving_flat_hash_map<std::string, int64_t> map(0);
  ska_ordered::order_preserving_flat_hash_map<std::string, int64_t> map_move;
  map_move = (std::move(map));

  // NOLINTNEXTLINE(bugprone-use-after-move)
  TORCH_INTERNAL_ASSERT(map.empty());
  TORCH_INTERNAL_ASSERT(map_move.empty());

  // NOLINTNEXTLINE(clang-analyzer-cplusplus.Move)
  TORCH_INTERNAL_ASSERT(map.find("") == map.end());
  TORCH_INTERNAL_ASSERT(map_move.find("") == map_move.end());
}

TEST(OrderedPreservingDictTest, test_reassign_moved_object_move_constructor) {
  using HMap =
      ska_ordered::order_preserving_flat_hash_map<std::string, std::string>;

  HMap map = {{"Key1", "Value1"}, {"Key2", "Value2"}, {"Key3", "Value3"}};
  HMap map_move(std::move(map));

  ASSERT_EQUAL_PRIM(map_move.size(), 3);
  // NOLINTNEXTLINE(bugprone-use-after-move)
  ASSERT_TRUE(map.empty());

  map = {{"Key4", "Value4"}, {"Key5", "Value5"}};
  TORCH_INTERNAL_ASSERT(
      map == (HMap({{"Key4", "Value4"}, {"Key5", "Value5"}})));
}

TEST(OrderedPreservingDictTest, test_reassign_moved_object_move_operator) {
  using HMap =
      ska_ordered::order_preserving_flat_hash_map<std::string, std::string>;

  HMap map = {{"Key1", "Value1"}, {"Key2", "Value2"}, {"Key3", "Value3"}};
  HMap map_move = std::move(map);

  ASSERT_EQUAL_PRIM(map_move.size(), 3);
  // NOLINTNEXTLINE(bugprone-use-after-move)
  ASSERT_TRUE(map.empty());

  map = {{"Key4", "Value4"}, {"Key5", "Value5"}};
  TORCH_INTERNAL_ASSERT(
      map == (HMap({{"Key4", "Value4"}, {"Key5", "Value5"}})));
}

TEST(OrderedPreservingDictTest, test_copy_constructor_and_operator) {
  using HMap =
      ska_ordered::order_preserving_flat_hash_map<std::string, std::string>;

  const std::size_t nb_values = 100;
  HMap map;
  for (const auto i : c10::irange(nb_values)) {
    map[std::to_string(i)] = std::to_string(i);
  }

  HMap map_copy = map;
  HMap map_copy2(map);
  HMap map_copy3;
  map_copy3[std::to_string(0)] = std::to_string(0);

  map_copy3 = map;

  TORCH_INTERNAL_ASSERT(map == map_copy);
  map.clear();

  TORCH_INTERNAL_ASSERT(map_copy == map_copy2);
  TORCH_INTERNAL_ASSERT(map_copy == map_copy3);
}

TEST(OrderedPreservingDictTest, test_copy_constructor_empty) {
  ska_ordered::order_preserving_flat_hash_map<std::string, int> map(0);
  ska_ordered::order_preserving_flat_hash_map<std::string, int> map_copy(map);

  TORCH_INTERNAL_ASSERT(map.empty());
  TORCH_INTERNAL_ASSERT(map_copy.empty());

  TORCH_INTERNAL_ASSERT(map.find("") == map.end());
  TORCH_INTERNAL_ASSERT(map_copy.find("") == map_copy.end());
}

TEST(OrderedPreservingDictTest, test_copy_operator_empty) {
  ska_ordered::order_preserving_flat_hash_map<std::string, int> map(0);
  ska_ordered::order_preserving_flat_hash_map<std::string, int> map_copy(16);
  map_copy = map;

  TORCH_INTERNAL_ASSERT(map.empty());
  TORCH_INTERNAL_ASSERT(map_copy.empty());

  TORCH_INTERNAL_ASSERT(map.find("") == map.end());
  TORCH_INTERNAL_ASSERT(map_copy.find("") == map_copy.end());
}

/**
 * at
 */
TEST(OrderedPreservingDictTest, test_at) {
  // insert x values, use at for known and unknown values.
  const ska_ordered::order_preserving_flat_hash_map<std::int64_t, std::int64_t>
      map = {{0, 10}, {-2, 20}};

  ASSERT_EQUAL_PRIM(map.at(0), 10);
  ASSERT_EQUAL_PRIM(map.at(-2), 20);
  bool thrown = false;
  try {
    map.at(1);
  } catch (...) {
    thrown = true;
  }
  ASSERT_TRUE(thrown);
}

/**
 * equal_range
 */
TEST(OrderedPreservingDictTest, test_equal_range) {
  ska_ordered::order_preserving_flat_hash_map<std::int64_t, std::int64_t> map =
      {{0, 10}, {-2, 20}};

  auto it_pair = map.equal_range(0);
  ASSERT_EQUAL_PRIM(std::distance(it_pair.first, it_pair.second), 1);
  ASSERT_EQUAL_PRIM(it_pair.first->second, 10);

  it_pair = map.equal_range(1);
  TORCH_INTERNAL_ASSERT(it_pair.first == it_pair.second);
  TORCH_INTERNAL_ASSERT(it_pair.first == map.end());
}

/**
 * operator[]
 */
TEST(OrderedPreservingDictTest, test_access_operator) {
  // insert x values, use at for known and unknown values.
  ska_ordered::order_preserving_flat_hash_map<std::int64_t, std::int64_t> map =
      {{0, 10}, {-2, 20}};

  ASSERT_EQUAL_PRIM(map[0], 10);
  ASSERT_EQUAL_PRIM(map[-2], 20);
  ASSERT_EQUAL_PRIM(map[2], std::int64_t());

  ASSERT_EQUAL_PRIM(map.size(), 3);
}

/**
 * swap
 */
TEST(OrderedPreservingDictTest, test_swap) {
  ska_ordered::order_preserving_flat_hash_map<std::int64_t, std::int64_t> map =
      {{1, 10}, {8, 80}, {3, 30}};
  ska_ordered::order_preserving_flat_hash_map<std::int64_t, std::int64_t> map2 =
      {{4, 40}, {5, 50}};

  using std::swap;
  swap(map, map2);

  TORCH_INTERNAL_ASSERT(
      map ==
      (ska_ordered::order_preserving_flat_hash_map<std::int64_t, std::int64_t>{
          {4, 40}, {5, 50}}));
  TORCH_INTERNAL_ASSERT(
      map2 ==
      (ska_ordered::order_preserving_flat_hash_map<std::int64_t, std::int64_t>{
          {1, 10}, {8, 80}, {3, 30}}));

  map.insert({6, 60});
  map2.insert({4, 40});

  TORCH_INTERNAL_ASSERT(
      map ==
      (ska_ordered::order_preserving_flat_hash_map<std::int64_t, std::int64_t>{
          {4, 40}, {5, 50}, {6, 60}}));
  TORCH_INTERNAL_ASSERT(
      map2 ==
      (ska_ordered::order_preserving_flat_hash_map<std::int64_t, std::int64_t>{
          {1, 10}, {8, 80}, {3, 30}, {4, 40}}));
}

TEST(OrderedPreservingDictTest, test_swap_empty) {
  ska_ordered::order_preserving_flat_hash_map<std::int64_t, std::int64_t> map =
      {{1, 10}, {8, 80}, {3, 30}};
  ska_ordered::order_preserving_flat_hash_map<std::int64_t, std::int64_t> map2;

  using std::swap;
  swap(map, map2);

  TORCH_INTERNAL_ASSERT(
      // NOLINTNEXTLINE(readability-container-size-empty)
      map ==
      (ska_ordered::
           order_preserving_flat_hash_map<std::int64_t, std::int64_t>{}));
  TORCH_INTERNAL_ASSERT(
      map2 ==
      (ska_ordered::order_preserving_flat_hash_map<std::int64_t, std::int64_t>{
          {1, 10}, {8, 80}, {3, 30}}));

  map.insert({6, 60});
  map2.insert({4, 40});

  TORCH_INTERNAL_ASSERT(
      map ==
      (ska_ordered::order_preserving_flat_hash_map<std::int64_t, std::int64_t>{
          {6, 60}}));
  TORCH_INTERNAL_ASSERT(
      map2 ==
      (ska_ordered::order_preserving_flat_hash_map<std::int64_t, std::int64_t>{
          {1, 10}, {8, 80}, {3, 30}, {4, 40}}));
}

} // namespace
