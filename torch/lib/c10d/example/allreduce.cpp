#include <ProcessGroupGloo.hpp>
#include <FileStore.hpp>

#include <gloo/transport/tcp/device.h>

using namespace ::c10d;

int main(int argc, char** argv) {
  int rank = atoi(getenv("RANK"));
  int size = atoi(getenv("SIZE"));
  auto store = std::make_shared<FileStore>("/tmp/c10d_example");
  std::vector<std::shared_ptr<::gloo::transport::Device>> devices;
  devices.push_back(::gloo::transport::tcp::CreateDevice("localhost"));
  ProcessGroupGloo pg(store, devices, rank, size);
  pg.initialize();

  // Create some tensors
  const auto ntensors = 10;
  std::vector<at::Tensor> tensors;
  for (auto i = 0; i < ntensors; i++) {
    auto x = at::ones(at::CPU(at::kFloat), {1000, 16 * (i + 1)});
    tensors.push_back(x);
  }

  // Kick off work
  std::vector<std::shared_ptr<ProcessGroup::Work>> pending;
  for (auto i = 0; i < ntensors; i++) {
    std::vector<at::Tensor> tmp = { tensors[i] };
    pending.push_back(pg.allreduce(tmp));
  }

  // Wait for work to complete
  for (auto& work : pending) {
    work->wait();
  }

  // Explicitly destroy process group
  // This is not done from its destructor
  pg.destroy();
}
