#pragma once

#include <string>
#include <thread>

#include <c10/util/intrusive_ptr.h>
#include <torch/csrc/distributed/c10d/control_plane/Handlers.hpp>

C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wdeprecated-literal-operator")
#include <httplib.h>
C10_DIAGNOSTIC_POP()

namespace c10d::control_plane {

class TORCH_API WorkerServer : public c10::intrusive_ptr_target {
 public:
  WorkerServer(const std::string& hostOrFile, int port = -1);
  ~WorkerServer() override;

  void shutdown();

 private:
  httplib::Server server_;
  std::thread serverThread_;
};

} // namespace c10d::control_plane
