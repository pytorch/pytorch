#pragma once
#include <ATen/ATen.h>
#include <assert.h>
#include <dlfcn.h>
#include <unistd.h>
#include <experimental/filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include "interpreter_impl.h"

class Model {
 public:
  Model();
  ~Model();

  at::Tensor forward(at::Tensor input);

  int id;

 private:
  static std::atomic<int> s_id;
};

class Interpreter : public InterpreterImpl {
 private:
  std::string library_name_;
  void* handle_;

 public:
  size_t id;
  Interpreter() : handle_(nullptr), id(s_interpreter_id++) {
    char library_name[L_tmpnam];
    library_name_ = library_name;
    std::tmpnam(library_name);
    {
      std::ifstream src("build/lib/libinterpreter.so", std::ios::binary);
      std::ofstream dst(library_name, std::ios::binary);
      dst << src.rdbuf();
    }
    handle_ = dlopen(library_name, RTLD_LOCAL | RTLD_LAZY);
    if (!handle_) {
      throw std::runtime_error(dlerror());
    }

    // technically, we can unlike the library right after dlopen, and this is
    // better for cleanup because even if we crash the library doesn't stick
    // around. However, its crap for debugging because gdb can't find the
    // symbols if the library is no longer present.
    unlink(library_name_.c_str());

    void* initialize_interface = dlsym(handle_, "initialize_interface");
    assert(initialize_interface);
    ((void (*)(InterpreterImpl*))initialize_interface)(this);
    // the actual torch loading process is not thread safe, by doing it
    // in the constructor before we have multiple worker threads, then we
    // ensure it doesn't race.
    run_some_python("import torch");
  }
  ~Interpreter() {
    if (handle_) {
      this->teardown();

      // it segfaults its face off trying to unload, but it's not clear
      // if this is something we caused of if libtorch_python would also do the
      // same if it were opened/closed a lot...
      dlclose(handle_);
    }
  }
  Interpreter(const Interpreter&) = delete;
};
