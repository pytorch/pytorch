#include <functional>
#include <thread>
#include <vector>

#include "fbcollective/allreduce_ring.h"
#include "fbcollective/allreduce_ring_chunked.h"
#include "fbcollective/test/base_test.h"

namespace fbcollective {
namespace test {
namespace {

// Function to instantiate and run algorithm.
using Func = void(
    std::shared_ptr<::fbcollective::Context>&,
    std::vector<float*> dataPtrs,
    int dataSize);

// Test parameterization.
using Param = std::tuple<int, int, std::function<Func>>;

// Test fixture.
class AllreduceTest : public BaseTest,
                      public ::testing::WithParamInterface<Param> {};

TEST_P(AllreduceTest, SinglePointer) {
  auto contextSize = std::get<0>(GetParam());
  auto dataSize = std::get<1>(GetParam());
  auto fn = std::get<2>(GetParam());

  spawnThreads(contextSize, [&](int contextRank) {
    auto context =
        std::make_shared<::fbcollective::Context>(contextRank, contextSize);
    context->connectFullMesh(*store_, device_);

    std::unique_ptr<float[]> ptr(new float[dataSize]);
    for (int i = 0; i < dataSize; i++) {
      ptr[i] = contextRank;
    }

    fn(context, {ptr.get()}, dataSize);

    auto expected = (contextSize * (contextSize - 1)) / 2;
    for (int i = 0; i < dataSize; i++) {
      ASSERT_EQ(expected, ptr[i]) << "Mismatch at index " << i;
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

static std::function<Func> allreduceRing = [](
    std::shared_ptr<::fbcollective::Context>& context,
    std::vector<float*> dataPtrs,
    int dataSize) {
  ::fbcollective::AllreduceRing<float> algorithm(context, dataPtrs, dataSize);
  algorithm.Run();
};

INSTANTIATE_TEST_CASE_P(
    AllreduceRing,
    AllreduceTest,
    ::testing::Combine(
        ::testing::Range(2, 16),
        ::testing::ValuesIn(genMemorySizes()),
        ::testing::Values(allreduceRing)));

static std::function<Func> allreduceRingChunked = [](
    std::shared_ptr<::fbcollective::Context>& context,
    std::vector<float*> dataPtrs,
    int dataSize) {
  ::fbcollective::AllreduceRingChunked<float> algorithm(
      context, dataPtrs, dataSize);
  algorithm.Run();
};

INSTANTIATE_TEST_CASE_P(
    AllreduceRingChunked,
    AllreduceTest,
    ::testing::Combine(
        ::testing::Range(2, 16),
        ::testing::ValuesIn(genMemorySizes()),
        ::testing::Values(allreduceRingChunked)));

} // namespace
} // namespace test
} // namespace fbcollective
