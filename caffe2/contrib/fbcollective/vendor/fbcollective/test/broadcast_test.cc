#include <functional>
#include <thread>
#include <vector>

#include "fbcollective/broadcast_one_to_all.h"
#include "fbcollective/test/base_test.h"

namespace fbcollective {
namespace test {
namespace {

// Function to instantiate and run algorithm.
using Func = void(
    std::shared_ptr<::fbcollective::Context>&,
    float* dataPtr,
    int dataSize,
    int rootRank);

// Test parameterization.
using Param = std::tuple<int, int, std::function<Func>>;

// Test fixture.
class BroadcastTest : public BaseTest,
                      public ::testing::WithParamInterface<Param> {};

TEST_P(BroadcastTest, SinglePointer) {
  auto contextSize = std::get<0>(GetParam());
  auto dataSize = std::get<1>(GetParam());
  auto fn = std::get<2>(GetParam());

  spawnThreads(contextSize, [&](int contextRank) {
    auto context =
        std::make_shared<::fbcollective::Context>(contextRank, contextSize);
    context->connectFullMesh(*store_, device_);

    std::unique_ptr<float[]> ptr(new float[dataSize]);

    // Run with varying root
    for (int rootRank = 0; rootRank < contextSize; rootRank++) {
      for (int i = 0; i < dataSize; i++) {
        ptr[i] = contextRank;
      }
      fn(context, ptr.get(), dataSize, rootRank);
      for (int i = 0; i < dataSize; i++) {
        ASSERT_EQ(rootRank, ptr[i]) << "Mismatch at index " << i;
      }
    }
  });
}

std::vector<int> genMemorySizes() {
  std::vector<int> v;
  v.push_back(sizeof(float));
  v.push_back(100);
  v.push_back(1000);
  v.push_back(10000);
  return v;
}

static std::function<Func> broadcastOneToAll = [](
    std::shared_ptr<::fbcollective::Context>& context,
    float* dataPtr,
    int dataSize,
    int rootRank) {
  ::fbcollective::BroadcastOneToAll<float> algorithm(
      context, dataPtr, dataSize, rootRank);
  algorithm.Run();
};

INSTANTIATE_TEST_CASE_P(
    OneToAllBroadcast,
    BroadcastTest,
    ::testing::Combine(
        ::testing::Range(2, 16),
        ::testing::ValuesIn(genMemorySizes()),
        ::testing::Values(broadcastOneToAll)));

} // namespace
} // namespace test
} // namespace fbcollective
