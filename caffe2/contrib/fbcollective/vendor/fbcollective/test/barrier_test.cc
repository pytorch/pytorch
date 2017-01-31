#include <functional>
#include <thread>
#include <vector>

#include "fbcollective/barrier_all_to_all.h"
#include "fbcollective/test/base_test.h"

namespace fbcollective {
namespace test {
namespace {

// Function to instantiate and run algorithm.
using Func = void(std::shared_ptr<::fbcollective::Context>&);

// Test parameterization.
using Param = std::tuple<int, std::function<Func>>;

// Test fixture.
class BarrierTest : public BaseTest,
                    public ::testing::WithParamInterface<Param> {};

TEST_P(BarrierTest, SinglePointer) {
  auto contextSize = std::get<0>(GetParam());
  auto fn = std::get<1>(GetParam());

  spawnThreads(contextSize, [&](int contextRank) {
    auto context =
        std::make_shared<::fbcollective::Context>(contextRank, contextSize);
    context->connectFullMesh(*store_, device_);
    fn(context);
  });
}

static std::function<Func> barrierAllToAll =
    [](std::shared_ptr<::fbcollective::Context>& context) {
      ::fbcollective::BarrierAllToAll algorithm(context);
      algorithm.Run();
    };

INSTANTIATE_TEST_CASE_P(
    BarrierAllToAll,
    BarrierTest,
    ::testing::Combine(
        ::testing::Range(2, 16),
        ::testing::Values(barrierAllToAll)));

} // namespace
} // namespace test
} // namespace fbcollective
