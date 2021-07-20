#pragma once

#ifdef USE_C10D_UCC

#include <memory>
#include <stdexcept>
#include <ucp/api/ucp.h>

class UCXError : public std::runtime_error {
  using runtime_error::runtime_error;
};

// Singleton object holding UCP objects
class UCPContext {
  static std::unique_ptr<UCPContext> instance;
  UCPContext();
public:
  ucp_context_h context;
  ucp_worker_h worker;
  static UCPContext *get();
  ~UCPContext();
};

#endif
