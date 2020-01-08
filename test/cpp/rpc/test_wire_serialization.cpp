#include <gtest/gtest.h>

#include <torch/torch.h>
#include <torch/csrc/jit/pickler.h>
#include <torch/csrc/jit/unpickler.h>
#include <torch/csrc/distributed/rpc/utils.h>

#include <memory>
#include <string>
#include <vector>

using namespace torch::distributed::rpc;

TEST(WireSerialize, Base) {
  auto run = [](const std::string& payload,
                const std::vector<at::Tensor>& tensors) {
    std::string serialized;
    {
      std::vector<char> mpayload(payload.begin(), payload.end());
      std::vector<at::Tensor> mtensors = tensors;
      serialized = torch::distributed::rpc::wireSerialize(
          std::move(mpayload), std::move(mtensors));
    }
    auto deser = torch::distributed::rpc::wireDeserialize(
        serialized.data(), serialized.size());
    EXPECT_EQ(payload.size(), deser.first.size());
    EXPECT_EQ(tensors.size(), deser.second.size());
    if (payload.size() > 0) {
      EXPECT_TRUE(
          memcmp(deser.first.data(), payload.data(), payload.size()) == 0);
    }
    for (size_t i = 0; i < tensors.size(); ++i) {
      EXPECT_TRUE(torch::equal(tensors[i], deser.second[i]));
    }
  };
  run("", {});
  run("hi", {});
  run("", {torch::randn({5, 5})});
  run("hi", {torch::randn({5, 5})});
  run("more", {torch::randn({5, 5}), torch::rand({10, 10})});
}

TEST(WireSerialize, RecopySparseTensors) {
  // Take a 1K row of a 1M tensors, and make sure we don't send across 1M rows.
  constexpr size_t k1K = 1024;
  at::Tensor main = torch::randn({k1K, k1K});
  at::Tensor tiny = main.select(0, 2); // Select a row in the middle
  EXPECT_EQ(tiny.numel(), k1K);
  EXPECT_EQ(tiny.storage().numel(), k1K * k1K);
  auto ser = torch::distributed::rpc::wireSerialize({}, {tiny});
  auto deser = torch::distributed::rpc::wireDeserialize(ser.data(), ser.size());
  EXPECT_TRUE(torch::equal(tiny, deser.second[0]));
  EXPECT_LT(ser.size(), (tiny.element_size() * k1K) + k1K);
}


TEST(WireSerialize, TestPicklerUnpicklerTensor) {
  constexpr int kRandTensorSize = 4;
  std::array<float, 3> kFromBlobTensorBits = {1, 2, 3};
  for (auto& t :
       {torch::randn(kRandTensorSize),
        torch::from_blob(
            kFromBlobTensorBits.data(), kFromBlobTensorBits.size())}) {
    std::string data;
    torch::jit::Pickler pickler(
        [&](const char* bytes, size_t len) { data.append(bytes, len); },
        nullptr);
    pickler.protocol();
    pickler.pushIValue(t);
    pickler.stop();
    auto tdata = pickler.tensorData();
    EXPECT_EQ(tdata.size(), 1);
    // Use different sizes to distinguish between from_blob/randn
    const bool expectDeleter = t.numel() != kFromBlobTensorBits.size();
    EXPECT_EQ(tdata[0].storageHasDeleter(), expectDeleter);

    size_t pos = 0;
    torch::jit::Unpickler unpickler(
        [&](char* buf, size_t n) -> size_t {
          if (pos >= data.size())
            return 0;
          const size_t tocopy = std::min(pos + n, data.size()) - pos;
          memcpy(buf, data.data() + pos, tocopy);
          pos += tocopy;
          return tocopy;
        },
        nullptr,
        nullptr,
        [&](const std::string& fname) -> at::DataPtr {
          if (fname == "0") {
            auto dptr = at::getCPUAllocator()->allocate(tdata[0].sizeInBytes());
            memcpy(dptr.get(), tdata[0].data(), tdata[0].sizeInBytes());
            return dptr;
          }
          throw std::runtime_error("not found");
        },
        {});
    auto ival = unpickler.parse_ivalue();
    EXPECT_TRUE(torch::equal(t, ival.toTensor()));
  }
}
