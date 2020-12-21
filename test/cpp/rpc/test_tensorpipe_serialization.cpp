#include <gtest/gtest.h>

#include <tensorpipe/core/message.h>
#include <torch/csrc/distributed/rpc/tensorpipe_utils.h>
#include <torch/torch.h>

#include <memory>
#include <string>
#include <vector>

TEST(TensorpipeSerialize, Base) {
  // Sender serializes
  at::Tensor t1 = torch::ones({1024}, at::ScalarType::Int);
  at::Tensor t2 = torch::ones({1024}, at::ScalarType::Float);
  std::vector<at::Tensor> tensors{t1, t2};
  std::vector<char> payload = {'1', '2', '3'};
  std::vector<char> payloadCopy = payload; // for testing
  torch::distributed::rpc::MessageType mtype =
      torch::distributed::rpc::MessageType::UNKNOWN;
  int64_t mId = 100;
  torch::distributed::rpc::Message sendingRpcMessage(
      std::move(payload), std::move(tensors), mtype);
  sendingRpcMessage.setId(mId);
  tensorpipe::Message sendingTpMessage;
  torch::distributed::rpc::TensorpipeWriteBuffers sendingTpBuffers;
  std::tie(sendingTpMessage, sendingTpBuffers) =
      torch::distributed::rpc::tensorpipeSerialize(
          std::move(sendingRpcMessage));

  // Mimic receiving message descriptor: recvingTpMessage is a copy of
  // sendingTpMessage except for the data pointers which are left null.
  tensorpipe::Message recvingTpMessage;
  recvingTpMessage.metadata = sendingTpMessage.metadata;
  recvingTpMessage.payloads.reserve(sendingTpMessage.payloads.size());
  for (auto& tpPayload : sendingTpMessage.payloads) {
    tensorpipe::Message::Payload p;
    p.length = tpPayload.length;
    p.metadata = tpPayload.metadata;
    recvingTpMessage.payloads.push_back(std::move(p));
  }
  EXPECT_EQ(recvingTpMessage.payloads.size(), sendingTpMessage.payloads.size());
  recvingTpMessage.tensors.reserve(sendingTpMessage.tensors.size());
  for (auto& tpTensor : sendingTpMessage.tensors) {
    tensorpipe::Message::Tensor t;
    t.buffer = tensorpipe::CpuBuffer{nullptr, tpTensor.buffer.cpu.length};
    t.metadata = tpTensor.metadata;
    recvingTpMessage.tensors.push_back(std::move(t));
  }
  EXPECT_EQ(recvingTpMessage.tensors.size(), sendingTpMessage.tensors.size());

  // Mimic readDescriptor() callback:
  // - Allocate buffers
  // - Fill pointers in tensorpipe message
  torch::distributed::rpc::TensorpipeReadBuffers recvingTpBuffers =
      torch::distributed::rpc::tensorpipeAllocate(recvingTpMessage);

  // Mimic tensorpipe data transfer
  for (int i = 0; i < recvingTpMessage.payloads.size(); i++) {
    tensorpipe::Message::Payload& srcPayload = sendingTpMessage.payloads[i];
    tensorpipe::Message::Payload& dstPayload = recvingTpMessage.payloads[i];
    if (srcPayload.length) {
      // Empty vector's data() can return nullptr, use the length to avoid
      // coying into nullptr
      memcpy(dstPayload.data, srcPayload.data, srcPayload.length);
    }
  }
  for (int i = 0; i < recvingTpMessage.tensors.size(); i++) {
    tensorpipe::Message::Tensor& srcTensor = sendingTpMessage.tensors[i];
    tensorpipe::Message::Tensor& dstTensor = recvingTpMessage.tensors[i];
    memcpy(
        dstTensor.buffer.cpu.ptr,
        srcTensor.buffer.cpu.ptr,
        srcTensor.buffer.cpu.length);
  }

  // Mimic read() callback:
  // - Unpickle
  torch::distributed::rpc::Message recvingRpcMessage =
      torch::distributed::rpc::tensorpipeDeserialize(
          std::move(recvingTpMessage), std::move(recvingTpBuffers));

  // Data is ready
  EXPECT_EQ(mtype, recvingRpcMessage.type());
  EXPECT_EQ(payloadCopy, recvingRpcMessage.payload());
  EXPECT_EQ(mId, recvingRpcMessage.id());
  EXPECT_TRUE(torch::equal(t1, recvingRpcMessage.tensors()[0]));
  EXPECT_TRUE(torch::equal(t2, recvingRpcMessage.tensors()[1]));
}

TEST(TensorpipeSerialize, RecopySparseTensors) {
  // Take a 1K row of a 1M tensors, and make sure we don't send across 1M rows.
  constexpr size_t k1K = 1024;
  at::Tensor main = torch::randn({k1K, k1K});
  at::Tensor tiny = main.select(0, 2); // Select a row in the middle
  EXPECT_EQ(tiny.numel(), k1K);
  EXPECT_EQ(tiny.storage().nbytes() / tiny.itemsize(), k1K * k1K);

  std::vector<at::Tensor> tensors{main, tiny};
  std::vector<char> payload = {'1', '2', '3'};
  torch::distributed::rpc::MessageType mtype =
      torch::distributed::rpc::MessageType::UNKNOWN;
  torch::distributed::rpc::Message sendingRpcMessage(
      std::move(payload), std::move(tensors), mtype);

  tensorpipe::Message sendingTpMessage;
  torch::distributed::rpc::TensorpipeWriteBuffers tpBuffers;
  std::tie(sendingTpMessage, tpBuffers) =
      torch::distributed::rpc::tensorpipeSerialize(
          std::move(sendingRpcMessage));

  EXPECT_EQ(tpBuffers.tensors.size(), 2);
  EXPECT_EQ(sendingTpMessage.tensors.size(), 2);
  EXPECT_TRUE(torch::equal(main, tpBuffers.tensors[0]));
  EXPECT_TRUE(torch::equal(tiny, tpBuffers.tensors[1]));
  // Test cloned storage
  EXPECT_EQ(main.storage().data(), sendingTpMessage.tensors[0].buffer.cpu.ptr);
  EXPECT_NE(tiny.storage().data(), sendingTpMessage.tensors[1].buffer.cpu.ptr);
  EXPECT_EQ(
      tiny.element_size() * k1K, sendingTpMessage.tensors[1].buffer.cpu.length);
}

TEST(TensorpipeSerialize, NoDeleterTensors) {
  std::vector<float> blob1{.8, .2};
  std::vector<float> blob2{.7, .5, .9};
  at::Tensor t1 = torch::from_blob((float*)(blob1.data()), blob1.size());
  at::Tensor t2 = torch::from_blob((float*)(blob2.data()), blob2.size());
  std::vector<at::Tensor> tensors{t1, t2};
  std::vector<char> payload = {'1', '2', '3'};
  torch::distributed::rpc::MessageType mtype =
      torch::distributed::rpc::MessageType::UNKNOWN;
  torch::distributed::rpc::Message sendingRpcMessage(
      std::move(payload), std::move(tensors), mtype);

  tensorpipe::Message sendingTpMessage;
  torch::distributed::rpc::TensorpipeWriteBuffers tpBuffers;
  std::tie(sendingTpMessage, tpBuffers) =
      torch::distributed::rpc::tensorpipeSerialize(
          std::move(sendingRpcMessage));

  EXPECT_EQ(tpBuffers.copiedTensors.size(), 2);
  EXPECT_EQ(sendingTpMessage.tensors.size(), 2);
  EXPECT_EQ(
      tpBuffers.copiedTensors[0].size(),
      sendingTpMessage.tensors[0].buffer.cpu.length);
  EXPECT_EQ(
      tpBuffers.copiedTensors[1].size(),
      sendingTpMessage.tensors[1].buffer.cpu.length);
  EXPECT_EQ(
      tpBuffers.copiedTensors[0].data(),
      sendingTpMessage.tensors[0].buffer.cpu.ptr);
  EXPECT_EQ(
      tpBuffers.copiedTensors[1].data(),
      sendingTpMessage.tensors[1].buffer.cpu.ptr);
  EXPECT_TRUE(
      memcmp(
          tpBuffers.copiedTensors[0].data(),
          t1.storage().data(),
          sendingTpMessage.tensors[0].buffer.cpu.length) == 0);
  EXPECT_TRUE(
      memcmp(
          tpBuffers.copiedTensors[1].data(),
          t2.storage().data(),
          sendingTpMessage.tensors[1].buffer.cpu.length) == 0);
}
