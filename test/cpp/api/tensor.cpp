#include <catch.hpp>

#include <ATen/ATen.h>

TEST_CASE("tensor/device-placement") {
  SECTION("DeviceGuard") {
    // SECTION("On index zero by default") {
    //   auto tensor = at::ones({3, 3}, at::kCUDA);
    //   REQUIRE(tensor.get_device() == 0);
    // }

    // // right hand side is TensorOptions
    // torch::OptionGuard guard = torch::device(torch::kCUDA, 1);
    // // convenience wrapper over OptionGuard
    // torch::DeviceGuard guard(torch::kCUDA, 1);
    // /// default device is CUDA
    // torch::DeviceGuard guard(1);

    // note that this is separate from DeviceGuard. DeviceGuard should move into the
    // detail namespace and do the actual thing. OptionGuard just modifies a
    // global singleton of option defaults. It operates at a higher level.
  }
}
