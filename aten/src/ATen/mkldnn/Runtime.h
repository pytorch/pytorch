#pragma once

#include <mkldnn.hpp>

namespace at { namespace native {

// MKLDNNEngine singleton
struct MKLDNNEngine {
  static MKLDNNEngine& Instance() {
    static MKLDNNEngine myInstance;
    return myInstance;
  }
  mkldnn::engine& get_engine() { return _engine; }
  MKLDNNEngine(MKLDNNEngine const&) = delete;
  MKLDNNEngine& operator=(MKLDNNEngine const&) = delete;

protected:
  MKLDNNEngine():_engine(mkldnn::engine::cpu, 0) {}
  ~MKLDNNEngine() {}

private:
  mkldnn::engine _engine;
};

#define MKLDNN_EXEC(_primitive)                                   \
  std::vector<mkldnn::primitive> net;                             \
  net.push_back(_primitive);                                      \
  mkldnn::stream(mkldnn::stream::kind::eager).submit(net).wait(); \

}}  // namespace at::native
