#pragma once

#include <memory>

namespace torch { namespace autograd {

struct VariableVersion {
  VariableVersion() {
    saved_ref = false;
    version_block = new int[3];
    version_block[0] = 0; // version
    version_block[1] = 1; // refcount
    version_block[2] = 1; // number of variables currently using the counter
  };

  int operator++(int) { return version_block[0]++; }

  int operator*() { return *version_block; }

  int var_refcnt() { return version_block[2]; }

  void join_with(VariableVersion &other) {
    if (this == &other) {
      return;
    }
    cleanup();
    version_block = other.version_block;
    version_block[1]++;
    version_block[2]++;
  }

  std::unique_ptr<VariableVersion> new_saved_ref() {
    auto new_ver = new VariableVersion();
    new_ver->cleanup();
    new_ver->version_block = version_block;
    version_block[1]++;
    new_ver->saved_ref = true;
    return std::unique_ptr<VariableVersion>(new_ver);
  }

  void cleanup() {
    auto vb = version_block;
    version_block = nullptr;
    if (!saved_ref) --vb[2];
    if (--vb[1]) return;
    delete[] vb;
  }

  ~VariableVersion() { cleanup(); }

  int *version_block;
  bool saved_ref;
};

}} // namespace torch::autograd
