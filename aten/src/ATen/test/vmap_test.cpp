#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/core/op_registration/op_registration.h>
#include <torch/csrc/jit/operator.h>
#include <torch/torch.h>
#include <ATen/Batching.h>

using namespace at;

namespace {

TEST(VmapTest, TestBatchedTensor) {
  {
    // batched + batched
    Tensor x = makeBatched(ones({2, 3}), 0, 1);
    Tensor y = makeBatched(ones({2, 3}), 0, 1);
    Tensor out_wrapped = x * y;
    Tensor output = unwrapBatched(out_wrapped);
    std::vector<int64_t> expected_size = {2, 3};
    ASSERT_EQ(output.sizes(), expected_size);
  }
  {
    // batched + unbatched
    Tensor x = makeBatched(ones({2, 3}), 0, 1);
    Tensor y = ones({3});
    Tensor out_wrapped = x * y;
    Tensor output = unwrapBatched(out_wrapped);
    std::vector<int64_t> expected_size = {2, 3};
    ASSERT_EQ(output.sizes(), expected_size);
  }
  {
    // nesting: batched (outer) + batched (inner)
    Tensor x = makeBatched(ones({2, 3}), 0, 1);
    Tensor y = makeBatched(ones({5, 3}), 0, 2);
    Tensor out_wrapped = x * y;
    // We get a doubly wrapped BatchTensor...
    Tensor output = unwrapBatched(out_wrapped, 2);
    std::vector<int64_t> expected_size = {2, 5, 3};
    ASSERT_EQ(output.sizes(), expected_size);
  }
  {
    // some crazy nesting
    // vmap(l.w: vmap(l.x: vmap(l.y: vmap(l.z: (w * y) * (x * z))(z))(y))(x))(w)
    Tensor w = makeBatched(ones({2, 3}), 0, 1);
    Tensor x = makeBatched(ones({5, 3}), 0, 2);
    Tensor y = makeBatched(ones({7, 3}), 0, 3);
    Tensor z = makeBatched(ones({11, 3}), 0, 4);
    Tensor out_wrapped = (w * y) * (x * z);
    Tensor output = unwrapBatched(out_wrapped, 4);
    std::vector<int64_t> expected_size = {2, 5, 7, 11, 3};
    ASSERT_EQ(output.sizes(), expected_size);
  }
  {
    // Send a BatchTensor in to autograd (mul backward)
    auto grad_output = makeBatched(torch::eye(3), 0, 1);

    Variable x = torch::randn({3}, torch::requires_grad());
    Variable y = torch::randn({3});
    auto res = x * y;
    backward({res}, {grad_output});
    auto grad = unwrapBatched(x.grad(), 1);

    ASSERT_TRUE(torch::allclose(grad, torch::diagflat(y)));
    // FYI: probably going to need to change InputBuffer::accumulate
    // because that can do in-place things.
    // TODO: audit in-place ops in autograd engine
  }
}

}
