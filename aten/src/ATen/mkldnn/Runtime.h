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

template<typename prim_t>
struct MKLDNNPrimitive {
  std::shared_ptr<prim_t> prim_;
  MKLDNNPrimitive() : prim_(nullptr) {}
  const prim_t& get_primitive() const { return *prim_; }
};

#define PRIMITIVE_DESC(mem) mem##_primitive_desc()
#define MREG(mem) mem{pd.PRIMITIVE_DESC(mem)}
#define ZREG(mem) mem{{zero_md(), MKLDNNEngine::Instance().get_engine()}, nullptr}

#define MKLDNN_EXEC(_primitive)                                   \
  std::vector<mkldnn::primitive> net;                             \
  net.push_back(_primitive);                                      \
  mkldnn::stream(mkldnn::stream::kind::eager).submit(net).wait(); \

}}  // namespace at::native
