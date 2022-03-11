#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <c10/util/irange.h>
#include <torch/csrc/profiler/containers.h>

TEST(ProfilerTest, AppendOnlyList) {
    const int n = 4096;
    torch::profiler::impl::AppendOnlyList<int, 1024> list;
    for (const auto i : c10::irange(n)) {
        list.emplace_back(i);
        ASSERT_EQ(list.size(), i + 1);
    }

    int expected = 0;
    for (const auto i : list) {
        ASSERT_EQ(i, expected++);
    }
    ASSERT_EQ(expected, n);

    list.clear();
    ASSERT_EQ(list.size(), 0);
}

TEST(ProfilerTest, AppendOnlyList_ref) {
    const int n = 512;
    torch::profiler::impl::AppendOnlyList<std::pair<int, int>, 64> list;
    std::vector<std::pair<int, int>*> refs;
    for (const auto _ : c10::irange(n)) {
        refs.push_back(list.emplace_back());
    }

    for (const auto i : c10::irange(n)) {
        *refs.at(i) = {i, 0};
    }

    int expected = 0;
    for (const auto& i : list) {
        ASSERT_EQ(i.first, expected++);
    }
}
