#include <c10/util/IntrusiveList.h>
#include <c10/util/irange.h>

#include <gtest/gtest.h>

namespace {

class ListItem : public c10::IntrusiveListHook {};

template <typename TItem>
void check_containers_equal(
    c10::IntrusiveList<TItem>& c1,
    std::vector<std::unique_ptr<TItem>>& c2) {
  EXPECT_EQ(c1.size(), c2.size());
  {
    auto it = c1.begin();
    for (const auto i : c10::irange(c1.size())) {
      EXPECT_EQ(&*it, c2[i].get());
      EXPECT_EQ(it, c1.iterator_to(*c2[i]));
      ++it;
    }
  }
  {
    auto it = c1.rbegin();
    for (const auto i : c10::irange(c1.size())) {
      EXPECT_EQ(&*it, c2[c2.size() - 1 - i].get());
      ++it;
    }
  }
};

} // namespace

TEST(IntrusiveList, TestInsert) {
  c10::IntrusiveList<ListItem> l;
  std::vector<std::unique_ptr<ListItem>> v;

  auto size = 50;

  for ([[maybe_unused]] const auto i : c10::irange(size)) {
    v.push_back(std::make_unique<ListItem>());
    l.insert(l.end(), *v.back());
    check_containers_equal(l, v);
  }
}

TEST(IntrusiveList, TestUnlink) {
  c10::IntrusiveList<ListItem> l;
  std::vector<std::unique_ptr<ListItem>> v;

  auto size = 50;

  for ([[maybe_unused]] const auto i : c10::irange(size)) {
    v.push_back(std::make_unique<ListItem>());
    l.insert(l.end(), *v.back());
  }

  for ([[maybe_unused]] const auto i : c10::irange(size)) {
    auto first = l.begin();
    EXPECT_TRUE(first->is_linked());
    first->unlink();
    EXPECT_FALSE(first->is_linked());
    v.erase(v.begin());
    check_containers_equal(l, v);
  }
}

TEST(IntrusiveList, TestMoveElement) {
  c10::IntrusiveList<ListItem> l;
  std::vector<std::unique_ptr<ListItem>> v;

  auto size = 5;

  for ([[maybe_unused]] const auto i : c10::irange(size)) {
    v.push_back(std::make_unique<ListItem>());
    l.insert(l.end(), *v.back());
  }

  // move 3rd element to the end of the list
  {
    auto it = l.iterator_to(*v[2]);
    EXPECT_TRUE(it->is_linked());
    l.iterator_to(*v[2])->unlink();
    EXPECT_FALSE(it->is_linked());
    l.insert(l.end(), *v[2]);
  }
  {
    auto it = v.begin() + 2;
    std::rotate(it, it + 1, v.end());
  }

  check_containers_equal(l, v);
}

TEST(IntrusiveList, TestEmpty) {
  c10::IntrusiveList<ListItem> l;
  ListItem i;

  EXPECT_TRUE(l.empty());
  l.insert(l.end(), i);
  EXPECT_FALSE(l.empty());
  l.begin()->unlink();
  EXPECT_TRUE(l.empty());
}
TEST(IntrusiveList, TestUnlinkUnlinked) {
  EXPECT_ANY_THROW(ListItem().unlink());
}

TEST(IntrusiveList, TestInitializerListCtro) {
  ListItem i, j;
  c10::IntrusiveList<ListItem> l({i, j});

  EXPECT_EQ(l.size(), 2);
  EXPECT_EQ(l.iterator_to(i), l.begin());
  EXPECT_EQ(l.iterator_to(j), ++l.begin());
}

TEST(IntrusiveList, TestNullListIterator) {
  auto null_iter = c10::ListIterator<c10::IntrusiveListHook, ListItem>{nullptr};

  EXPECT_ANY_THROW(--null_iter);
  EXPECT_ANY_THROW(++null_iter);
}
