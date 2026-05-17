
#include <c10/core/Device.h>
#include <gtest/gtest.h>
#include <unordered_map>

#include <torch/nativert/executor/Placement.h>

using namespace ::testing;

namespace torch::nativert {

TEST(PlacementTest, IsSameDevice) {
  c10::Device cpuDevice = c10::Device(c10::DeviceType::CPU);
  c10::Device cpuDevice1 = c10::Device(c10::DeviceType::CPU);
  cpuDevice1.set_index(1);

  EXPECT_TRUE(isSameDevice(cpuDevice, cpuDevice));
  EXPECT_TRUE(isSameDevice(cpuDevice, cpuDevice1));

  c10::Device cudaDevice = c10::Device(c10::DeviceType::CUDA);
  c10::Device cudaDevice0 = c10::Device(c10::DeviceType::CUDA, 0);
  c10::Device cudaDevice1 = c10::Device(c10::DeviceType::CUDA, 1);
  EXPECT_TRUE(isSameDevice(cudaDevice, cudaDevice0));
  EXPECT_FALSE(isSameDevice(cudaDevice0, cudaDevice1));

  EXPECT_FALSE(isSameDevice(cudaDevice0, cpuDevice));
}

TEST(PlacementTest, PlacementDefaultOnly) {
  Placement placement(c10::Device(c10::DeviceType::CUDA, 0));

  std::ostringstream os;
  os << placement;
  EXPECT_EQ(os.str(), "|cuda:0");

  c10::Device cuda0 = c10::Device(c10::DeviceType::CUDA, 0);
  c10::Device cuda1 = c10::Device(c10::DeviceType::CUDA, 1);
  c10::Device cuda2 = c10::Device(c10::DeviceType::CUDA, 2);

  EXPECT_EQ(placement.getMappedDevice(cuda0), cuda0);
  EXPECT_EQ(placement.getMappedDevice(cuda1), cuda0);
  EXPECT_EQ(placement.getMappedDevice(cuda2), cuda0);
}

TEST(PlacementTest, PlacementBasic) {
  Placement placement(
      {{c10::Device(c10::DeviceType::CPU), c10::Device(c10::DeviceType::CPU)},
       {c10::Device(c10::DeviceType::CUDA, 0),
        c10::Device(c10::DeviceType::CUDA, 1)},
       {c10::Device(c10::DeviceType::CUDA, 1),
        c10::Device(c10::DeviceType::CUDA, 2)}},
      c10::Device(c10::DeviceType::CUDA, 0));

  std::ostringstream os;
  os << placement;
  EXPECT_EQ(os.str(), "cpu|cpu,cuda:0|cuda:1,cuda:1|cuda:2,|cuda:0");

  c10::Device cpu = c10::Device(c10::DeviceType::CPU);
  c10::Device cuda0 = c10::Device(c10::DeviceType::CUDA, 0);
  c10::Device cuda1 = c10::Device(c10::DeviceType::CUDA, 1);
  c10::Device cuda2 = c10::Device(c10::DeviceType::CUDA, 2);
  c10::Device cuda3 = c10::Device(c10::DeviceType::CUDA, 3);

  EXPECT_EQ(placement.getMappedDevice(cpu), cpu);
  EXPECT_EQ(placement.getMappedDevice(cuda0), cuda1);
  EXPECT_EQ(placement.getMappedDevice(cuda1), cuda2);
  EXPECT_EQ(placement.getMappedDevice(cuda2), cuda0);
  EXPECT_EQ(placement.getMappedDevice(cuda3), cuda0);
}

TEST(PlacementTest, Placement) {
  std::unordered_map<c10::Device, c10::Device> deviceMap1 = {
      {c10::Device("cuda:0"), c10::Device("cuda:1")}};
  Placement p1(deviceMap1);
  EXPECT_EQ(p1.getMappedDevice(c10::Device("cpu")), c10::Device("cpu"));
  EXPECT_EQ(p1.getMappedDevice(c10::Device("cuda")), c10::Device("cuda"));
  EXPECT_EQ(p1.getMappedDevice(c10::Device("cuda:0")), c10::Device("cuda:1"));

  std::unordered_map<c10::Device, c10::Device> deviceMap2 = {
      {c10::Device("cpu"), c10::Device("cuda:0")}};
  Placement p2(deviceMap2);
  EXPECT_EQ(p2.getMappedDevice(c10::Device("cpu")), c10::Device("cuda:0"));
  EXPECT_EQ(p2.getMappedDevice(c10::Device("cuda:0")), c10::Device("cuda:0"));
  EXPECT_EQ(p2.getMappedDevice(c10::Device("cuda:1")), c10::Device("cuda:1"));
}

} // namespace torch::nativert
